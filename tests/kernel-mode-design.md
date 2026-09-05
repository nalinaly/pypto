# kernel 模式设计（封板版 v9）

**基线**：simpler `3e08ede7` · pypto `8ccd97de`（2026-08-29）。前 8 节结论在 `a64147b7`/`a4e33f49` 上核对；此后 simpler #2057/#2068、pypto #2545/#2559/#2560/#2372 **均未推翻任何结论，也未触及任何落点**，五个 HBG 锚点符号每轮复核均存活。
**参考**：`nalinaly/simpler@b6f905f6` · `nalinaly/pypto@9cece0b7`（两个 runtime 都已实现 kernel 模式并有参数化 ST）
**迭代记录与撤销过程**：见 `kernel-mode-steps.md` 与 `backup/`（18 个时间戳目录，含 v1–v8、v9 各阶段快照与 **codex 八轮原始意见**）

**状态：设计已收口，可以开工做 Phase 0 probe。** codex 第八轮结论：「架构设计已收口，可以开始 Phase 0 probe」——无架构级阻塞项，剩余均为 production ABI 收边（已在 9.8／9.8b 补齐）。

**明确不再写进本文档的**（属实现期决定，codex 建议，采纳）：`terminal_` 与 effective tuple 的具体存放位置、各入口 guard 的代码组织、错误日志结构、owner abandon 的具体封装。**不要再把实现细节堆进设计文档。** —— codex 第五轮判定"不能直接开工"，四个架构空缺见 9.4，我核实成立。前 8 节不因此重开。

---

## 0 · 决策终值表（**本表具有覆盖力**，与后文任何早期表述冲突时以本表为准）

文档正文按发现顺序生长，因此有六处早期结论被后面的章节取代。**实现时只看本表**；正文保留原文与修正过程仅供追溯（"为什么改成这样"）。

| # | 决策 | **终值** | 曾经的说法（已作废） | 依据 |
| - | ---- | -------- | -------------------- | ---- |
| 1 | close/teardown 失败语义 | **`CLOSING` ＋ 显式重试**：cleanup 失败停在 `CLOSING`、拒绝 dispatch、允许重试直到成功。**且判定重试结果时不得用单个 first-wins 错误槽**，须另设"意外 teardown 错误"专用槽，决策顺序 runtime status → unexpected teardown → 受控结果 | ~~通过前置后任何 teardown 错误 → 终态 unusable，只加一个 `terminal_` 位~~ | §11.1（参考 E.1 的状态机）＋ §10.1（过程记录 10.49.3 的 first-wins 教训） |
| 2 | AICPU stream | **不保留 private AICPU stream（连 bootstrap 用途也不要）**：init/register 迁到 **prepare 期在 caller stream 上异步 enqueue**，错误由 warmup ＋ 调用方同步暴露。`device_runner_base.cpp:583` 的 `aclrtSynchronizeStreamWithTimeout` 在 kernel 模式下**去掉** | ~~保留 bootstrap/control stream，其 init 期 synchronize 靠"限定在非-launch 路径"自洽~~ | §15.1（**A.4.11 是已确认决策，明确覆盖 prepare**）＋ §14.3（H.6 的形状） |
| 3 | 非目标 #8（内部 synchronize） | **kernel 模式 launch／prepare／close 任何路径都不做 PyPTO 内部 synchronize** | ~~收窄为"非-launch control 路径允许同步"~~ | §15.1（A.4.11） |
| 4 | freeze 守卫谓词 | **只放行本来就不 mutate 的调用**：每个 region 满足「已提交且 `requested ≤ cached`」**或**「未提交且请求 0」；否则在任何 mutation 之前整体拒绝 | ~~拒绝增长，也拒绝非零→零~~（会误伤 HBG 每次 bind 都传 `gm_sm_size=0`） | §9.8b（读 `commit_region` 三分支定出） |
| 5 | callable unregister | **允许 unregister（仅失效 host 句柄）；禁止 cid 复用；pinned device state 一律不随 unregister 释放** | ~~一律禁止 unregister~~ | §12.1（UT-027）＋ §13.1（D.5 悬空地址危害）＋ §15.3（A.6.7）——三处独立出处 |
| 6 | 换 caller stream | **三分支**：同 stream 不 wait 不 query 旧 tail（靠 FIFO）／换 stream 且旧 tail not-ready 则在任何新 enqueue 前失败／换 stream 且旧 tail complete 则 host query 后继续但**不 enqueue 旧 tail wait** | ~~只写"跨 stream 未 quiesce 切换不支持"~~ | §12.2（UT-061/062/062A） |
| 7 | binary 卸载 | **永不卸载**：process pin，`close()` 不释放，不调 `BinaryUnLoad`。**D.6 第 8 步"卸载 binary"不采用** | ~~（参考 D.6 第 8 步说 close 时卸载）~~ | §13.3（§6.4 当前态 ＋ `FinalizeL1Pinned` 的 "no BinaryUnLoad API"）——两份参考文档冲突，以代码为准 |
| 8 | DFX | **v1 全关**：args dump、PMU、dep-gen、scope stats、L2 swimlane 等一切增加 workspace／collector／回读／同步的通道；并有测试断言 kernel 模式下确实未启用 | ~~（原文档完全没有这一条）~~ | §15.2（A.8.6）——注意 dep-gen 是 #2057、swimlane 是 #2031，都是主线近期改动 |

**未决项（唯一一条，需你裁决）**：C21 的功能范围 —— 见 §7。TMR 可保留 device 侧数据依赖塑形；HBG 只能 device 侧塑形／分桶＋多图／明确不支持。

---

## 0.1 · 一句话定位

给同一颗 chip 上的 PyPTO 增加第二种资源所有权模式：放弃设备所有权，只借用调用方的 stream 提交一段有界异步 task 序列，使一个 PyPTO program 在 PyTorch 眼中与普通 AscendC/Triton 算子无法区分，并能被 ACLGraph 当普通节点 capture/replay。

命名：同一个 `Worker(level=2)` 的两种 **mode**（`ChipOwned` / `ChipBorrowed`），不是阶梯上新增层级。依据：`worker_level.py:53`「`L1` is absent because no such level exists」，以及主线 #1893 刚因"名字取自处理器/部署位置"把 L3 改名为 `node`——同一条理由。（注意：分析文档原写"host(3)"是**错的**，主线是 `node=3`。）

---

## 1 · 主线现状：核对结论

**逐字成立、可直接依赖**（每条都在 `a64147b7` 上核对过原文）：

| 事实 | 锚点 |
| ---- | ---- |
| `bytes_per_copy` 是保留字段，必须为 0 | `runtime_c_api.h:154`；校验在 `pipeline_contract.h:47` |
| TMR 契约 depth 2／6 项；HBG depth 2／4 项（不声明 GM_SM） | 各 `runtime_maker.cpp` 的 `get_pipeline_contract()` |
| 契约有自带准入校验 | `chip_worker.cpp:263` 调 `is_valid_pipeline_contract` ＋ `has_serviceable_arena_topology` |
| owned 模式确实占设备 | `device_runner.h:149`「Symmetric with finalize(): aclrtResetDevice + aclFinalize run there.」 |
| TMR 容量常量（256MB×4 ring＝1GB 等） | `tensormap_and_ringbuffer/runtime/runtime_types.h:65-87` |
| HBG mirror 按最坏上限定维，**为的是让 `prepare_task` 免容量检查** | `common/host_build_graph/shared_memory.h:300` |
| HBG ready queue 已改为 bind 期可达性定容，上限 32768，超限带状态失败 | `common/host_build_graph/runtime_types.h:132`（#1982） |
| 已有只读 committed-memory 查询 | `runtime_c_api.h:252` `committed_device_memory_ctx`、`chip_worker.h:249` |
| kernel 模式在主线**不存在**（从零新增） | 两仓 `DeviceExecutionMode`/`L1Borrowed`/`simpler_l1_*`/`ACLGraph` 等各 0 命中 |
| 上一代的三个越界 API 在主线**也不存在** | `rtStreamAddToModel`/`GetCaptureInfo`/`launchEarlyMode`/`RunPreSync` 各 0 命中 |

**两条必须修正的原文档判断**：

1. **§2.5/§2.6「slot pool ＋ lease 是可直接接入的地基」要加限定。** `pipeline_slot_pool.h:22` 自述「This is **capability only**」——纯 host 侧 generation 池，对 device 执行与 graph 生命周期无感知，释放点 `simpler_finalize_run` 也在 host 侧，而 **replay 不回 host**。→ lease 能管住 host 侧两次调用不串味（原则4），**不能表达 captured node 的所有权**。因此原则6「宁可 pin」不是保守选择，而是**唯一可选项**。
2. **两个 stream kind 目前是惰性声明。** `find_pipeline_resource` 全仓只有一个调用点（`chip_worker.cpp:590` 的 `arena_bank_for_slot`），只看三个 arena kind。→ "kernel 模式不声明 AICPU_STREAM"今天是 no-op；要让它有执法力，core 必须真正消费 stream 声明。

**主线活跃热点（按符号锚定，不按行号）**：HBG `runtime_maker.cpp`（14 个新 commit 里 4 个动它，#2060 已把公共部分迁入 `src/common`）、pypto `jit/{cache,decorator,specializer}.py`（#2539）。

---

## 2 · 表一：资源所有权（以参考 §6.4 当前态为准）

> ⚠️ 早期计划附录 D.1 已被 §6.4 取代。**最关键的差异：binary 与 graph-visible code handle 是进程 pin，`close()` 不释放、永不调用 `BinaryUnLoad`**（`load_aicpu_op.cpp` 的 `FinalizeL1Pinned` 注释：「graph-visible function handle remain process-pinned; **no BinaryUnLoad API**」；硬不变量 #11 同）。

| 资源 | owner | 创建 | graph 可见 | 释放条件 |
| ---- | ----- | ---- | ---------- | -------- |
| caller stream | PyTorch／调用方 | 外部 | 是 | 外部管理，PyPTO 不销毁 |
| hidden AICore stream | kernel context | init | 是 | **外部 quiescence 后** `close()` |
| Start／AicoreDone／PrepareTail／SerialTail event | kernel context | init | 是 | 同上 |
| **AICPU binary/function handle** | **进程 pin** | init | 是 | **进程结束；shutdown 不调 BinaryUnLoad** |
| **AICore binary/function handle** | **进程 pin** | prepare | 是 | **进程结束；不 unregister/不复用** |
| Runtime／KernelArgs | kernel context | prepare | 是 | `close()` |
| HBG callable-local 函数表 | CANN-owned 自包含 package | 每次 launch | 是 | package 由 task/graph owner 管，**code 仍进程 pin** |
| TRB append-only code entry | resident AICPU registry | 新 callable prepare | 是 | **不回收；进程结束** |
| workspace／arena／register window | kernel context | prepare | 是 | `close()` |
| AICore report 区 | kernel context | prepare | 是 | `close()` |
| queue call snapshot | taskQueue entry | 每次 Host 调用 | Host 队列可见 | callback 完成后 |
| CANN HostArgs copy | **CANN task/graph node** | enqueue/capture | 是 | 由 task 或 graph owner 管，**PyPTO 不访问** |
| HBG host GraphPlan | callable-local **单条** cache | 首次调用或参数语义变化 | 否 | 新 identity **事务替换**旧 entry；close 时释放 |
| HBG serialized launch blob | 单次 Host launch | 每次调用 | 被 CANN **深拷贝** | API 返回后可释放 |
| HBG working slot／ContextRegistry | kernel context（**设备内存**） | prepare | 是 | `close()` |
| input/output tensor storage | 调用方（默认 allocator 可 `recordStream`） | 外部 | 是 | task/graph 使用完毕；**v1 不自动分配纯 Out** |

**teardown 顺序**（由上表推出，不可交换）：外部 quiescence → `graph.reset()`／销毁全部 graph → `close()` context → 进程结束才动 code。
**`close()` 内部的十步释放顺序见 §13.2**（其中第 8 步"卸载 binary"**不采用**，理由见 §13.3）。
**`close()` 不同步、不猜 graph 生命周期**；未 `close()` 的 runner **拒绝销毁**（析构不做任何 runtime 调用），否则 captured graph 仍可能引用这些句柄。

---

## 3 · 表二：容量冻结 vs 准入（两件事，必须分开）

codex 第四轮指出我 v8 把两者混成"容量全关、新 callable 半开"过宽。定稿分开写：

| 维度 | 规则 | 依据 |
| ---- | ---- | ---- |
| **execution capacity**（workspace／arena／HBG working slot／register window 等 graph 可见执行容量） | **全关**。prepare/init 阶段一次性定容；**从第一个 package 可能被 capture 起**：禁止 `setup_static_arena` 释放或重新 commit 任何被绑定 region；更大的图 → **capacity error**；真要扩容 → **新 context generation ＋ 新 stable slot，旧 slot 保留到旧 graph 全部销毁**；**禁止更新"current base"全局变量让旧 captured package 静默指向新分配** | 参考 N.5 |
| **callable／code admission** | **capture 之外 append-only**。TRB code registry 与进程 pin 的 binary/code 可在 capture 外的 prepare 中 append/grow；当前实现明确接受**无固定 callable 上限**。capture 之内一律不允许 lazy register | §2.2 #4、§5.4、`l1_execution_state.h::seal()`（只是兼容 hook，**不表达容量冻结**） |
| **定容方法** | 收集全部已知 program 的需求，**仅对"可共享的同类资源"取最大值而非总和**；**不能盲目逐字段 `max()`**——含相对地址与内部布局的 arena 必须按最大 sizing **重新构建一个合法 image** | 参考 D.2 ＋ **D.4** |
| **launch 期** | **零分配**。不允许 launch-time fallback allocation；不暴露外部 workspace（v1 显式推迟项，非遗漏） | 硬不变量 #2、F.10、D.2 |
| **申报通道** | **复用主线已有的只读 `committed_device_memory_ctx`**，不新增公共 workspace-size API。若框架要在建 context 前扣常数，另定"配置推出的 capacity estimate"内部契约。**必须区分 `capacity / committed / used` 三种语义，不用一个数承担三者** | 主线 `runtime_c_api.h:252`；D.2 禁止的是"调用方据此分配"的规划 API，不是 telemetry |
| **restore** | 每次 eager/replay **都必须**从当前 node 的 pristine source 恢复 mutable working slot（`attach_populated` 不重置已填充内容，**只钉住地址绝不够**）。恢复前比对 `HbgWorkingBinding`（device_id／generation／各 base+capacity／binary identity），**任一不匹配在放行 scheduler/AICore 之前 fail-closed，不得"先 copy 一部分再报错"** | N.2.2、N.5、N.6 |
| **规范约束** | 每次 host build 的 graph 必须作为**该次 task/captured node 的 payload**管理，**不得实现为 context-wide、可被下一次 build 原地覆盖的 `current_graph`** | N.3 |

**已封板的决策**（不再列为待裁决）：内存**首期内部持有 ＋ prepare 期 pin ＋ launch 期零分配**。原分析文档 §4.2「报尺寸、不自行分配」与内存专题 D1「自持内存」的矛盾，按参考 F.10 裁决为后者，外部 workspace 明确推迟。

---

## 4 · 表三：非目标（12 条，全部有依据）

| # | 非目标 | 依据 |
| - | ------ | ---- |
| 1 | host 建图期读/写外部 device tensor 内容不支持；任何依赖外部 tensor 运行期数值决定拓扑的 host 行为不支持。**允许**：tensor metadata、device address 作为不解引用参数、host 已知且 graph 生命周期内固定的 scalar、编译后拓扑与 function binding | §15.4、N.11；机制上 device tensor 走 pass-through 不注册 host view；语义上 caller stream 前序写可能未执行 |
| 2 | 同一 device 不允许多个 live borrowed context（v1 single-live-context） | `hbg_context_registry.h` |
| 3 | 同一 context 内**任何** invocation/replay 不得重叠（不分 callable、不分 eager/replay、不分两张 graph）；PyPTO 仍占用全部 AICore | §2.2 #2 |
| 4 | 跨 stream 未 quiesce 切换不支持；换 stream 前须外部静默（**三分支精确规则见 §12.2**） | §2.2 #3；根因是 CANN capture 隔离 **107024**，参考因此把 SerialTail gate 关掉（`wait_for_serial_tail = false`） |
| 5 | 不提供运行期动态扩容 workspace／working slot／package capacity（**例外**：TRB code registry 允许 capture 外 append immutable callable、不回收旧 entry） | §2.2 #4 |
| 6 | capture 前必须完成 warm；capture 内不允许 lazy specialization、H2D staging、arena 增长、registry 变更 | 硬不变量 #8 |
| 7 | capture 后不提供修改 tensor 地址／scalar／拓扑的 graph update API（**scalar 在 capture 后固化，replay 不回 host 读新值**） | §2.2 #5、§9.3、§9.4 |
| 8 | 不提供 kernel 模式内部的 stream/device synchronize；不做"保险同步"（**该收窄已按 §15.1 正式撤回**：kernel 模式 launch／prepare／close 任何路径都不做 PyPTO 内部 synchronize，依据 A.4.11） | §2.2 #6、硬不变量 #3、F.13 |
| 9 | 不承诺外部 `from_blob`／自定义 deleter storage 在调用方提前销毁时安全；owner 必须持有到 graph 销毁且最后一次真实 device use 完成 | §2.2 #7、D.1 |
| 10 | `close()` 前调用方必须 quiesce 并销毁相关 graph；不 shutdown 时资源安静 pin 到进程结束 | 硬不变量 #9 |
| 11 | 完全未 report 的硬件 core 失联恢复不纳入算子内协议 | §2.2 #8 |
| 12 | **验收范围 A2/A3**；A5 保持源码同构 ＋ 导出相同 symbols ＋ stub，但"没有同等真实硬件证据，不能泛化为已上板"，**不作为完成条件**。但不得由 A2/A3 通过推断 A5 通过 | §2.2 #9、N.13；A5 若要上需独立证据 |

**上一代实现的硬禁止项**（防止重新引入，主线目前各 0 命中）：不提供 `launch_early_mode` 或等价开关；不把 AICPU orchestrator 放私有 stream；不用 capture query ＋ `rtStreamAddToModel` 构造 capture-only 拓扑（**连回退方案都不许**）；不在 prepare 时预启动一个等待未来 invocation 的 kernel；不以"性能更好"为由破坏 entry/exit 闭包。若目标 CANN 不支持纯 event 闭环捕获 → **Phase 0 门槛未通过，不是授权降级**。

---

## 5 · 单算子闭包：三条判据（原则2 的可执行形式）

1. **入口**：`start_event` 是所有内部分支的传递入口，任何分支都不得早于它。
2. **闭包内**：启动竞态**允许**（TMR 的 orchestrator/AICore 协作依赖它），但**消费动态字段前必须有 device 侧 happens-before**——AICore 启动只读 persistent 字段，本次 payload 只在 AICPU 开窗后读；必要时才加 `invocation_generation/ready`（release/acquire）。**不得用 host 同步或 per-run 分配掩盖竞态。**
3. **出口**：`done_event` 必须代表全部内部工作在**成功与失败两条路径**上都已完成或安全退出——AICPU 的 deinit/ack **不能**替代 hidden stream 上 AICore launch 之后 record 的 `aicore_done_event`。

> "AICore-first" 指的是 **Host enqueue 顺序**，不表示设备越过算子边界提前执行——两条分支都受 caller stream 的 Start event 约束。
> "Host 返回"只表示序列成功提交；设备可能仍在运行，tensor/context/graph 生命周期不能据此结束。

**launch 序列**（权威表述见参考 §8.2，与实现 `l1_launch_sequence.h` 一致）：

```text
caller stream                                 hidden AICore stream
[可选 consume PrepareTail]
[stream-switch gate；不做 graph 外等待]
async clear launch state / handshake / report
record Start
                                              wait Start
                                              launch 已注册的 AICore handle
                                              record AicoreDone
launch AICPU with HostArgs
wait AicoreDone
record SerialTail
```

**AICore-first 的两个理由**：① custom AICPU scheduler 等 AICore startup report 时可能占住调度资源，若 AICore SQE 排在后面，真实设备上可能形成**调度环或长 stall**；② AICore-first 有**更好的失败闭包**——hidden wait、AICore launch、done-record 三个 Host API 提交检查都在 AICPU task 入队**之前**完成。

**cancel 通道**：`AICORE_PRE_WINDOW_HOST_CANCEL = UINT32_MAX`，因为**一次 `aclrtMemsetAsync(..., 0xFF, ..., stream)` 就能发布它，无需分配 pinned host scalar**；稀疏轮询间隔 256（2 的幂 ＋ static_assert），安全性论证是"CANCEL 一直发布到整个 AICore launch 完成，稀疏轮询不会漏"。整块覆盖只在 AICPU 任务入队**前**合法。
**失败收敛**：三条返回路径都以 `wait(AicoreDone) + record(SerialTail)` 收尾；cancel 发布后**重试一次** completion record；补偿再失败时 caller stream 上没有可 join 的节点——这是**设计已接受的终局，必须显式写在注释里**。

**"干净拒绝" vs "poison" 的分界**：任何 enqueue **之前**校验失败 → 不提交调用 metadata、不改 warmed/layout 状态；**已部分 enqueue** 后失败 → **必须 poison context** ＋ 完成可表达的异步错误闭包。

**两张受限词汇表**（原则11 的执法形式，也是无硬件 CI 的可测性策略）：context 生命周期 5 ops（get_current_device／create+destroy hidden stream／create+destroy event）、launch 6 ops（wait_event／memset_handshake／record_event／launch_aicpu／launch_aicore／cancel_waiting_aicore）。同步、分配、stream/event 创建、capture 查询、model attach 在这两张表里**无法表达**——「Keeping those operations unrepresentable makes the host-only lifecycle tests an architectural guard instead of a mock that can silently exercise forbidden behavior.」

---

## 6 · 实施顺序（一条线性提交序列，按参考附录 K）

前面的"阶段 A–N"只作主题索引，**不再作为实施顺序**。可执行序列如下，每步可独立评审与回退：

| # | 提交 | 内容 | 门槛／判据 |
| - | ---- | ---- | ---------- |
| **1** | **Probe only** | 六个 Phase-0 onboard probe，不改正式 API：**A** 参数快照（launch 后立刻覆盖 host struct，连续 enqueue 多份不同参数不同步，最后外部 sync，验证每份 task 消费自己的快照）／**B** 双 stream capture（probe shim 把 capture query、model handle、`rtStreamAddToModel` 设为**禁止 API，任一调用立即失败**；图前后各放一个可观察的普通 NPU op）／**C** mixed launch API（记录 capture/instantiate/replay/sync 的准确 error code）／**D** event generation 复用（含两图共享 context event、warmup stream ≠ capture stream）／**E** handshake 最小失效区（**禁止以"全量清零能跑"结束分析**）／**F** entry-exit 边界（可控延迟 predecessor 写 entry marker，hidden task 启动时读、未就绪即报错；延迟 exit marker，caller 后继必须观察到；eager 与 replay 的 task/event 序列除 graph 节点容器外不得分叉） | 七条通过条件全满足；**A2/A3 为门槛**；`.stream()`/simulator **不能**作为语义证据；失败 → **停止主线实现并记录失败原语**，"不要先写一半 state 再用同步规避" |
| **2** | **ABI ＋ mode skeleton** | 五入口 C ABI（`supported/init/prepare_callable/launch/finalize_device`，caller stream 显式入参，只收 POD／blob／device pointer）；`ChipOwned`/`ChipBorrowed` 命名与互斥 claim（**一个标识符的全部出现，一个 commit 完成**）；unsupported stub；两张受限词汇表；phase 机含 **Poisoned**；no-reset UT。**不新建资源** | owned 路径一行不改；kernel 路径不复用 `OnboardNativeRunState`（参考 F.1/F.2 明确否掉"只删 sync"与"把 native-run token 交给 ACLGraph"） |
| **3** | **契约声明 ＋ 校验** | 两个 runtime 各出 kernel 契约（depth 按 context 固定，**不做运行期按路径自适应**——原则3 禁止 capture 查询，运行期没有合法选择信号）；放开 `bytes_per_copy != 0`（kernel 模式要求非 0，full 仍为 0，fail-closed 方向不变）；**让 core 真正消费两个 stream kind 的声明** | 都过 `is_valid_pipeline_contract` ＋ `has_serviceable_arena_topology`（depth=1 时三个 arena kind 一起变 1，仍一致）；去掉 AICPU_STREAM 声明会**导致行为改变**（今天不会） |
| **4** | **capability gate 先行** | orchestration SO 的 `requirements_v1` 独立可选导出符号；**未知 bit／缺元数据／host tensor-data 位一律拒；缺失即拒绝**（"absence cannot prove…"）。**必须在建 plan、分配执行资源、形成可 capture launch 之前 fail-closed**。按 N.11 **静态标记** orchestration 是否使用 `get_tensor_data/set_tensor_data`，**不允许因某个 example 只依赖 shape 就推断全部** | 老 SO 在 kernel 模式被拒、full 模式照旧；只生成 device predicate metadata 的 tensor read **保持允许**（N.13） |
| **5** | **persistent state** | hidden stream ＋ 4 个 event；prepare-once allocator；析构不做 runtime 调用、拒绝销毁未 close 的 runner；fault-injection UT | 无硬件可跑；用 fake ops 表驱动完整生命周期 |
| **6** | **resource prepare ＋ capacity freeze** | 表二全部规则：一次性定容（**max 非 sum，且 arena 按最大 sizing 重建合法 image**）、地址冻结、超限 capacity error、扩容走新 generation ＋ 新 slot、`HbgWorkingBinding` 校验；**独立的 capacity-freeze 状态，不用 `seal()` 表达**；输入 gate（**fail-closed 拒绝非 DEVICE tensor**，并断言 host-tensor 暂存路径零触发） | 五条：每槽 base/capacity 不变、底层 alloc/free 计数为零、恰好容量成功、容量+1 明确错误、失败不替换旧块且旧 plan 仍可用。另：**禁止 `acquire_graph_definition_block` 在 freeze 后再换块**（这是主线 #1988 新引入的按需增长路径，参考里不存在，必须显式禁掉） |
| **7** | **AICPU invocation 快照** | 每次调用一份 task-owned 不可变参数快照（复用 `WithHostArgs`，**不自建 device task-args pool**——CANN 已有 completion-aware pool）；两级 identity（invocation 含 `buffer.addr` ＋ scalar；structural 去掉这两者；**不 hash 原始 Tensor 字节**，padding 与未用维度不参与）；单条目 plan cache，事务替换，失败候选不破坏上一份可用条目 | Probe A 的判据在正式路径上复现；host 内存不随调用次数增长 |
| **8** | **persistent AICore args** | 可变字段全部移出，只留 `runtime_args`／`regs`／`ffts_base`／`workspace_base`；**prepare 期一次 H2D**，每次 launch 只传同一 device pointer；动态 tensor/scalar 由 AICPU 编码进 child task payload | AICore 不需要 per-call args pool；不存在"下次调用覆盖本次 KernelArgs" |
| **9** | **binder ＋ launch 协议** | 10 条前置校验 ＋ **§11.2 的逐位置处置表**（8 行，只有前两行"context 可继续用"，其余一律 poison）；§8.2 序列（含 **AICore-first** 与两个理由的注释）；memset cancel；三条失败路径的收尾与重试；每次 replay 的 pristine restore | 用 fake ops trace **逐项比对实际调用序**（不是"注释里写了顺序"）；每条失败路径有 fault injection；用错 event flag 的变体能被捕获（**`ACL_EVENT_SYNC`**——默认 flag 的 event 在 capture 中可能被直接拒绝） |
| **10** | **Python 入口** | mode 维度接到**现有** `runtime="tensormap_and_ringbuffer"/"host_build_graph"` 机制上（pypto main 上 `execution=` 不存在）；cache key 保留 `closure_constants`／`runtime` 等既有维度 | 不与现有 runtime 选择正交冲突；specialization 增长有界 |
| **11** | **torch adapter** | 只消费第 2 步冻结的 ABI，**不另定义接口**。六条：`.stream(false)`（**不是 `.stream()`，它会 drain taskQueue**）／`OpCommand::RunOpApiV2` 入队／callback 只捕 POD＋lease＋`at::Tensor`／**每个唯一的默认 allocator storage** `recordStream`／callback 前保留 tensor handle／**callback 与 close 共享 dispatch mutex**。tensor 生命周期**两个窗口都要**（host queue 等待窗口捕 `at::Tensor`；device 执行窗口 `recordStream`）——只做一项不够 | 判据写成反例：pending taskQueue 不被隐式 drain；删 Python tensor 后 callback 仍安全；callback 返回后 storage 不被复用；close/callback 互斥；默认 storage 去重；**外部 storage 分"可识别 → fail-fast"与"不可识别 → 只能形成调用方保活契约"两类** |
| **12** | **eager ＋ graph ST**（故障矩阵用 **§10.2 的七个阶段**作验收清单，并补 **§10.3** 参考未覆盖的六项） | 负向验收为主：未 quiesce 的切换与**任何重叠 invocation/replay 必须被拒**；外部 quiescence 后串行切换成功；重复 replay、两图交替、specialization 超限、close/callback race。CANN API trace 证明"不问不拿不分叉"（用 §8.3 的正反两栏当断言表），**并同时 trace allocator/runtime API 证明 execution 期零分配、graph 可见地址不变** | TMR 与 HBG **各自**验收；full 模式回归有可枚举集合 |

**基线测量**（可与第 1 步并行，产出的是**档位选取规则**而非一张图）：HBG image 体积与 host orchestration 耗时随图规模的曲线；TMR arena build 各档耗时；HBG Graph Definition block 的尺寸曲线与 grow 次数（**每槽 warm 后清零 allocator 计数，重复/换序 bind，要求地址与容量不变、alloc/free 为零**）。每条曲线须指定 workload、单位、采样点与判定阈值。

**两个 runtime 的专属工作**：

- **TMR**：prepare-time 静态状态 9 项；定长 invocation snapshot；`Start` 前只清**最小**区域（launch state／handshake／report 三类分别列，不是全量清零）；completion gate 的**四相汇合**（arrive → 唯一 finalizer → 全部线程读取最终状态 → depart），exactly-once arrive，leader 发布 init verdict，orchestrator 进 program 前等 verdict。有参考代码可读（`tensormap_and_ringbuffer/host/runtime_maker.cpp:958`、`l1_aicpu_args.h`）。
- **HBG**：按参考 N.9 的 H1–H7 细分（拆分 host build 与 H2D → variable launch blob ＋ placeholder bridge → stable execution slot ＋ capacity freeze → AICPU leader per-replay restore → 独立 registration 路径 → external tensor 数据契约 → ACLGraph/lifetime/回归），并采用 N.8 fallback 决策树与 N.10 P0/ST 矩阵。
- **平台**：a5 与 a2a3 的 HBG 文件同构，**不得只改一个 arch 就假定另一个成立**。
- **direct-AIV 快路径**：**明确推迟**，不阻塞通用路径（若要对齐参考性能水平再单列）。

---

## 7 · 唯一剩余的待裁决：C21 的功能范围

其余决策已封板。这一条必须由你定，因为它决定 kernel 模式**能覆盖什么**：

**问题**：主线 HBG 明确支持 host 建图期读控制张量来塑形（`runtime_maker.cpp:1217` 注释直接点名 **paged_attention 的 `context_lens` 与 `block_table`**），且仓内真实工作负载在用——`paged_attention_unroll_manual_scope` 读 `context_lens` 算出 `bn_this_batch`（**循环上界，直接决定发多少 task**）、`deepseek_v4_flash_decode` 读 `ext_num_tokens_per_owner`（`qwen3_14b_decode` 不用）。而 kernel 模式禁止它：device tensor 不注册 host view，且**即使能读，caller stream 上前序 torch 写也可能尚未执行**。这与分析文档 §1 边界条件二「只要和 vLLM 一起跑就必须走 kernel 模式」冲突。

**关键非对称**：**TMR 的编排在 device 上执行期建图，可以保留数据依赖塑形；被卡住的只有 HBG。**

**建议的封板范围**（codex 第四轮建议，我同意）：

> **首期明确：TMR 支持 device-side 数据依赖塑形；HBG 仅支持不依赖 replay-time device tensor 数值的拓扑。CPU control vector 留作后续独立设计。**

HBG 若要覆盖 paged attention，可选：**device 侧塑形**（拓扑静态化＋device 侧掩码/边界）或**分桶＋多图**（每个块数档位 capture 一张，replay 时按档选——vLLM 对 batch size 的既有做法）。
**已否决**：把控制张量当 "prepare 期 Host scalar" 传（scalar 在 capture 后固化，`replay 不重新进入 Python 读新 scalar`；且"Host scalar"不涵盖控制向量，参考 N.11 把 CPU control tensor 列为**待定**）；允许一次 caller-stream 同步（破坏 capture 透明）。

裁决后要做的两件事：改写 §1 边界条件二使其与范围一致；**分 runtime 给出"能覆盖哪些 orchestration"的清单**。

---

## 8 · 与原两份文档的差异索引

| 原文档 | 需要的修改 |
| ------ | ---------- |
| 分析 §2 层级表 | `host(3)` → `node(3)`，改引 #1893 |
| 分析 §2.5／§2.6 地基表 | 加一列"是否覆盖 capture/replay"；lease 不能表达 captured node 所有权 |
| 分析 §2.5 depth 结论 | "eager 2／capture 1 随路径可配"**不可实现**（无合法选择信号）→ 改为按 context 固定 |
| 分析 §4.2 | 三处形状 → **新开 kernel 专用路径**（F.1/F.2 否掉改造 run 家族）；"报尺寸不自行分配"与内存 D1 的矛盾按 F.10 裁决为内部持有 |
| 分析 §4.3 | 两个 stream kind 目前无消费者，"差异只是数据"对它们不成立 |
| 原则2 | 补 §5 的三条判据 ＋ 上一代取舍审计的五条理由与硬禁止项 |
| 原则5 | "归零" → **"复用前恢复到该 runtime 定义的 pristine 状态；若恢复可能部分失败，必须事务发布并隔离失败槽位"**；附 N.2.2 的可变字段清单与最小失效区域 |
| 原则6 | 从"保守选择"升为**"唯一可选项"**；并覆盖 stream/event 句柄，不只 code/binary |
| 原则9 | 补两级 identity 的字段划分；代际＝context 与 slot 共用一个值、binary identity 独立、resident DSO 只存 registry 地址 |
| 内存 D2 | `bytes_per_copy` 承载不了真实规模；改用已有 `committed_device_memory_ctx`，区分 capacity/committed/used |
| 内存 D3 四段 | 补 **Poisoned** 失败态；execution capacity 全关与 callable append-only 分开写 |
| 内存 D4 ready queue 数字 | HBG 已改为 bind 期定容、上限 32768 |
| 两份文档 §4.6 待定 | 待定一（depth）→ 按 context 固定；待定二（VMM）→ 不需要，固定容量＋冻结＋新 generation 扩容；待定三（归零验证）→ CANN API trace ＋ allocator trace |

---

## 9 · 落点索引：12 步在今天主线上改哪里

第 6 节说"做什么"，本节说"改哪个文件的哪个符号"。全部在 `a64147b7` / `a4e33f49` 上核对过。**按符号锚定，不按行号**（HBG `runtime_maker.cpp` 与 pypto `jit/*` 都是活跃热点）。

### 9.1 三条硬性结构约束（先看这三条，它们决定改动的粒度）

**① 凡进入 `ChipWorker` 强制 dlsym 表的符号，所有可加载构件都必须导出。**
`chip_worker.cpp:55` 的 `load_symbol<T>()` 在 dlsym 失败时**抛异常**，`:203` 起的装载块**全部强制加载**。
**范围要说准**（本节初稿写"全仓没有 optional-symbol 路径"过宽，codex 第五轮指正）：仓内**确实有**非抛出的可选 dlsym（例如 `a5/platform/sim/host/device_runner.cpp:129` 的本地 `load_sym` lambda，用于 AICPU SO）。受此约束的只是 **`ChipWorker` 的公共 host-runtime ABI 装载表**。
**构件数是 8 不是 4**：`tests/ut/py/test_host_runtime_abi.py` 用 `_SIM_CASES` ＋ `_ONBOARD_CASES` 枚举 **2 arch × {onboard, sim} × 2 runtime**。实现上通常在 shared onboard／sim C API 各写一次即可，**不是复制四份实现**；但导出面覆盖 8 个构件。
→ 新符号必须：8 个构件都导出；不支持者导出**显式 stub**（既有先例：sim 的 `supports_concurrent_native_prepare_ctx` 恒返 0，`sim/host/c_api_shared.cpp:888`）；`runtime_c_api.h:14` 本身也要求 unsupported variant 提供 stub。**并同步更新 `test_host_runtime_abi.py` 的符号集**，否则某个构件漏导出不会被发现。
→ 因此**平台范围的答案**：符号与契约覆盖全部 8 构件；只有**执行路径**可以 a2a3 先行，其余返回 unsupported。

**② requirements 走独立导出符号——但要分清"哪个 ABI 不能动"。**
不能动的是 **orchestration SO 对外导出的函数 ABI**：`pypto/src/codegen/orchestration/orchestration_codegen.cpp:169` 生成的 `OrchestrationConfig aicpu_orchestration_config(const ChipTaskArgs&)`。给它加返回字段会破坏所有既有 L2 产物，所以 requirements 必须是**另一个标量返回符号**（`pypto_orchestration_requirements_v1`）。
**可以动的是** simpler 侧的 host 内部结构 `HostOrchEntryPoints`（`hbg/host/runtime_maker.cpp:407`）——参考正是往它里面加了 `requirements_v1` 与 `requirements_v1_available` 两个字段。它是 simpler 私有的、随 dlopen 一起构造的，扩展它没有兼容问题。
（**这条修正了本节初稿**：初稿写"不动 `HostOrchEntryPoints`"，把 simpler 内部结构与 orchestration 导出 ABI 混为一谈了。）

**③ 第 4 步跨两个仓，缺一半就不成立。**

- **生产端（pypto）**：`src/codegen/orchestration/orchestration_codegen.cpp`（＋ `include/pypto/codegen/orchestration/orchestration_codegen.h`、`python/bindings/modules/codegen.cpp`）要**新增导出** requirements 符号，并根据编译期是否真的生成了 host `get_tensor_data/set_tensor_data` 置位。
- **消费端（simpler）**：dlsym **紧跟在解析 `entry`/`bind` 之后**（主线的 orchestration dlopen 块在 `hbg/host/runtime_maker.cpp:1024-1029` 附近），把结果存进 `HostOrchEntryPoints`；**校验点另设**，参考有**两处**调用 `validate_hbg_l1_requirements`。
- 顺序上生产端要先落或同时落：**消费端一旦要求"缺失即拒绝"，没有该符号的旧 orchestration 产物在 kernel 模式下全部被拒**——这是有意的，但必须与 pypto 侧的发布节奏对齐。

### 9.2 逐步落点

| 步 | 主要落点 | 说明 |
| -- | -------- | ---- |
| **1** Probe | 新增 `tests/st/`（参考用 `tests/st/runtime/l1/`）＋ 必要的 native helper（参考放 `tests/st/l1/host_args_probe/`，含独立 AICPU 探针 ＋ loader contract ＋ self-test） | probe 的禁止-API shim 是关键资产；现有 ST 目录见 `tests/st/{a2a3,a5,...}` |
| **2** ABI ＋ mode skeleton | 声明：`src/common/worker/runtime_c_api.h`（现有 run 家族在 `:388-436`）；dlsym：`src/common/worker/chip_worker.cpp:205-232`（**新符号加在这里，四份 runtime 必须同时导出**）；onboard 实现：`src/common/platform/onboard/host/c_api_shared.cpp`（现有 `simpler_launch_run` 在 `:831`）；sim 侧对应 stub：`src/common/platform/sim/host/c_api_shared.cpp`（先例：`supports_concurrent_native_prepare_ctx` 在此恒返 0） | mode 状态机与两张词汇表放 `src/common/platform/onboard/host/`（参考放 `l1_execution_state.{h,cpp}`、`l1_launch_sequence.h`） |
| **3** 契约 ＋ 校验 | 四份 `get_pipeline_contract()`：`src/{a2a3,a5}/runtime/{tmr,hbg}/host/runtime_maker.cpp`；放开 `bytes_per_copy != 0`：`src/common/worker/pipeline_contract.h:47`；准入点：`src/common/worker/chip_worker.cpp:263`；**让 core 消费 stream 声明**：`chip_worker.cpp:590` 的 `arena_bank_for_slot()` 是目前**唯一**读声明的地方，需要新增 stream 侧的对应消费；UT：`tests/ut/cpp/hierarchical/test_pipeline_contract.cpp`（已存在，header-only 可无硬件跑） | depth 按 context 固定，不做运行期自适应 |
| **4** capability gate | **跨两仓，见 9.1③**。pypto：`src/codegen/orchestration/orchestration_codegen.cpp` 新增导出符号（勿改 `aicpu_orchestration_config` 的返回 ABI，`:169`）。simpler：dlsym 紧跟 `entry`/`bind` 解析之后（orchestration dlopen 块在 `hbg/host/runtime_maker.cpp:1024-1029` 附近），结果存入 `HostOrchEntryPoints`（`:407`，**可扩展**）；校验点须在 `run_host_orchestration`（`:595`）与 `bind_callable_to_runtime_impl`（`:1320`）**之前**生效，参考有两处校验调用 | 只生成 device predicate metadata 的 tensor read 保持允许；缺符号即拒 → 与 pypto 发布节奏对齐 |
| **5** persistent state | `src/common/platform/onboard/host/`（hidden stream ＋ 4 event ＋ prepare-once allocator）；析构/销毁拒绝逻辑挂在 device context 销毁路径 `destroy_device_context` | 无硬件 UT 用 fake ops 表 |
| **6** capacity freeze | 定容与提交声明：`host_api.h:96` `setup_static_arena`（wrapper `c_api_shared.cpp:219`）；**但真正的增长 owner 是 `src/common/platform/onboard/host/device_runner_base.cpp:356` 附近的 arena allocator**（sim 有对应实现）——**freeze guard 必须落在这里**，不能只挡 `HostApi` 声明与上层调用点（codex 第五轮指正）。；必须禁掉的按需增长：`acquire_graph_definition_block`（`host_api.h:68`，调用点 `hbg/.../runtime_maker.cpp:476`，#1988 引入）、HBG host-tensor staging（`hbg/.../runtime_maker.cpp:1187`）、`RetainedTempBump::begin`（`tmr/.../runtime_maker.cpp:284`）。；`tmr/.../runtime_maker.cpp:537` 是 `bump == nullptr` 的**防御性 fallback，当前不可达**（正式 bind 总传非空 bump）——要求它 fail-closed，但不与活跃增长点并列。；`acquire_sm_mirror`（`host_api.h:77`）是 **host `new[]`**（`device_runner_base.cpp:230`），不占 HBM | 见第 6 节判据 |
| **7** AICPU 快照 ＋ identity | 参数快照与 plan cache 放 `src/common/worker/`（参考：`hbg_argument_snapshot.h`、`hbg_graph_plan_cache.h`）；tensor 字段来源 `ChipStorageTaskArgs`（`task_interface/task_args.h`） | 不自建 device task-args pool |
| **8** persistent AICore args | **修正**（初稿写的 `src/common/platform/include/common/kernel_args.h` **不存在**）：实际是 per-arch 的 `src/a2a3/platform/include/common/kernel_args.h` 与 `src/a5/platform/include/common/kernel_args.h`；共用 host helper 是 `src/common/platform/onboard/host/device_runner_helpers.{h,cpp}`。**注意现有实现走 allocator ＋ 直接 `rtMemcpy`（`device_runner_helpers.cpp:60`），不是 public `copy_to_device_ctx`** → 必须先定：扩展 `KernelArgsHelper` 持有 persistent storage，还是把所有权上移到 `ChipWorker` 走 C ABI | 未决，见 9.4 第 3 项 |
| **9** binder ＋ launch | 序列实现放 `src/common/platform/onboard/host/`；owned 路径 `DeviceRunner::launch_run`（`a2a3/platform/onboard/host/device_runner.cpp:581`）**保持不动**，kernel 走新路径 | event 创建须用 `aclrtCreateEventExWithFlag(..., ACL_EVENT_SYNC)`——默认 flag 可能在 capture 中被拒 |
| **10** Python 入口 | **修正**：`ir/distributed_compiled_program.py:84` 是 `DistributedConfig.runtime` 默认值，**只管 distributed 的 runtime flavor，不是 mode 的落点**（codex 指正）。`RuntimeKind`/`runtime_kind_to_name` 表示 **runtime flavor 而非 ownership mode**（定义跨 `include/pypto/ir/transforms/pass_context.h`、`src/ir/transforms/pass_context.cpp`、`python/bindings/modules/passes.cpp`、`pypto_core/passes.pyi`）——**mode 不得做成第三种 `RuntimeKind`**。；真正要串的分发链：`RunConfig` → `python/pypto/runtime/runner.py` → `python/pypto/runtime/worker.py` → `python/pypto/ir/compiled_program.py` → simpler 新 ABI。；**cache key 是否拆分尚未确定**：只有当 mode 改变编译产物时才拆；若同一 orchestration 产物可同时用于 owned/kernel、mode 只改运行分发，则**不应**进 `jit/cache.py` 的 `compile_opts` | 见 9.4 第 4 项 |
| **11** torch adapter | `python/bindings/`（已有 `bindings.cpp` ＋ `modules/`；参考新增的是 `python/bindings/torch_npu_l1_adapter.cpp`）；torch_npu 探测参考用 `cmake/detect_torch_npu.py` | 只消费第 2 步冻结的 ABI；不得改 native ABI |
| **12** ST | `tests/st/` 下按 runtime 参数化（参考：`tests/st/runtime/l1/test_l1_aclgraph.py` 用 `@pytest.mark.parametrize("runtime", [...])` 覆盖两个 runtime）；C++ UT `tests/ut/cpp/{common,types,hierarchical}/` | 负向验收为主 |

### 9.3 不要碰的东西（回归面控制）

- a2a3 的 `DeviceRunner::launch_run`（`device_runner.cpp:581`）、**a5 的 `DeviceRunner::launch_execution`（`a5/.../device_runner.h:94`）**、sim 的同名路径、共享 `DeviceRunnerBase` 的 owned prepare/cleanup；以及 `simpler_{prepare,launch,poll,wait,finalize}_run` 与 **`OnboardNativeRunContext`**（初稿误写 `OnboardNativeRunState`）：owned 路径，kernel 模式**另开路径**，不改造（参考 F.1/F.2 已否掉改造方案）。
- **callable_id 复用**：kernel 模式下禁止（**注意：`unregister` 本身可允许，见 §12.1 的修正**）（会触发 AICPU orch SO 的 `dlclose` + reload，`runtime_c_api.h:462`）。现有"native run 期间拒绝"的守卫是 per-run 的，**挡不住 captured graph**。
- **进程内切换 host runtime SO**：存在 captured graph 时禁止（`chip_worker.cpp:190` 的 dlclose+reload）。
- **orchestration SO 的导出函数 ABI**（`aicpu_orchestration_config`，pypto `orchestration_codegen.cpp:169`）：requirements 走独立符号，不改它的返回类型。（simpler 内部的 `HostOrchEntryPoints` **可以**扩展，见 9.1②。）
- L2/L3 的 `DeviceRuntimeLaunchDesc` ABI：kernel 模式**新建独立 context struct**，"不要强求一个 struct 同时服务两套生命周期"。
- `worker_level.py` 的 `WorkerLevel`：kernel 是 **mode** 不是 level，该文件不应因此改动。

### 9.4 原先阻塞开工的四个空缺 —— 落点已定，但其中两条我判断错过一次

codex 第五轮列的四条空缺，本轮全部有了代码依据的落点。**其中 ② 与 ③ 我第一次判断是错的，已按核实结果改正**（②「kernel 模式不建 AICPU stream」忽略了它是 bootstrap/control stream；③「扩展 `KernelArgsHelper`」忽略了它是 per-run 对象）。下面是改正后的版本。

codex 第五轮判定"不能开工"并列出四条。**四条现已全部有代码依据的落点**（下面每条都在 `a64147b7`＋`35f195bd` 上读过）。

**① close/finalize 控制点：不需要新增 ABI，`finalize_device` 本来就能返回失败。**
`runtime_c_api.h:312` 是 `int finalize_device(DeviceContextHandle ctx)`，`chip_worker.h:278` 的 typedef 也是 `int (*)(void *)`。问题**只在于调用方丢弃了返回值**——`chip_worker.cpp:498`：

```cpp
if (device_ctx_ != nullptr && finalize_device_fn_ != nullptr && initialized_) {
    finalize_device_fn_(device_ctx_);      // ← 返回值被丢弃
}
if (device_ctx_ != nullptr && destroy_device_context_fn_ != nullptr) {
    destroy_device_context_fn_(device_ctx_);   // ← 无条件继续
    device_ctx_ = nullptr;
}
if (lib_handle_) { dlclose(lib_handle_); }    // ← 无条件 dlclose
```

而 `destroy_device_context` 是 `void`（`c_api_shared.cpp:330`），只能日志＋早退（`:333` 已有"refusing to destroy…"分支）。
→ **落点**：(a) kernel 模式的 `finalize_device` 在 context 仍持有 graph 可见资源时返回错误；(b) 改 `ChipWorker::finalize()` **尊重该返回码**，非 0 时**不执行** destroy 与 `dlclose`，并把 context 留在可重试状态。
→ 这是**两处小改动 ＋ 一条语义**，不是新 ABI。`runtime_c_api.h:230` 的既有注释（"…`finalize_device()` first"）本来就把它当作先行的可失败步骤，与此一致。

**但这个 gate 必要而不充分** —— 我自查 worker 层全部 `dlclose` 站点后发现**另有三条卸载代码的路径**，其中两条必须一并禁掉：

| 路径 | 位置 | kernel 模式下的处置 |
| ---- | ---- | ------------------- |
| `~ChipWorker()` → `finalize()` | `chip_worker.cpp:163` | `finalize()` 返回 `void`，析构里无处报错。**但"拒绝时不 dlclose"正好等于故意泄漏句柄**，这与参考的做法一致（"refuses to destroy an unclosed runner and **conservatively leaves it alive**"）。可接受，需在注释里写明是有意泄漏 |
| **init 失败回滚** | `chip_worker.cpp:377`（`init_rc != 0`）＋ 作用域 guard `:41` | **安全**：发生在 `initialized_ = true` 之前，此时不可能有 captured graph |
| **切换 runtime 时卸载前一个 host runtime SO** | `chip_worker.cpp:190` 注释：「Cross-runtime isolation relies on `-fno-gnu-unique` (#453) allowing **dlclose to actually unload the previous runtime's SO** before loading the next one」 | **必须禁止**：进程内一旦存在 captured graph，就不得切换 runtime——否则卸载掉的正是图引用的代码 |
| **callable_id 复用触发 AICPU orch SO 的 dlclose + reload** | `runtime_c_api.h:462`：「reclaimed lazily when the cid is reused (the next `launch_device_register` triggers **`dlclose` + reload**)」 | **必须禁止**：kernel 模式下 callable 只能 append，**不得 unregister、不得复用 cid**。这正是表一里"TRB append-only code entry：不回收；进程结束"与"callable ID 只追加"的由来 |

**最后一条还暴露了一个与 D13 同形的缺陷**：`simpler_unregister_callable` 现有的守卫是「Rejected while a native run is prepared or executing on this context」——**这是 per-run 的，不是 per-graph 的**。captured graph 不占任何 native run，所以这条守卫**挡不住"图还活着但 cid 被复用"**。与 `PipelineSlotLease` 挡不住 captured node 是同一个根因（D13）。
→ 因此 ① 的完整落点是：**可失败的 finalize gate ＋ 禁止进程内 runtime 切换 ＋ 禁止 unregister/cid 复用**，三者缺一不可。

**还有两点我漏了（codex 第六轮补，核实成立）**：

- **"非零就跳过 destroy/dlclose"不足以描述状态**：`finalize()` 在调 `finalize_device` **之前**已经关掉并销毁 `run_lane_`、清了 native runs／global domains／comm sessions。若此时 finalize 返错，状态是 `initialized_=true`、`finalized_=false`、**但 `run_lane_==nullptr`**——既不是正常运行态也不是终态，**现有两个 bool 表达不了"只允许重试 close"**。→ 需要一个显式的 `CLOSING_RETRYABLE`（或等价）状态。
- **上层会把重试语义破坏掉**：Python 包装器在 `finally` 里**无论成败**都清 callable registry（`python/simpler/task_interface.py`），更上层成功返回后丢掉 `_chip_worker`（`python/simpler/worker.py`）。→ 失败必须**显式传播（返回或抛）**，且 native／nanobind／Python wrapper／`Worker.close()` **只在成功时才丢弃所有权**。
- 另外两条销毁路径的处置：**析构必须 non-throwing**，失败时只能保留 handle/context 并记录泄漏（不得继续 dlclose）；**DMA provisioning 失败**会在 `initialized_=true` 之后调 `finalize()`（`chip_worker.cpp:438` 附近），也要适配新语义；**init 期 context 创建后失败**走 `DlHandleGuard` 的 dlclose（`:268` 附近）——必须保证 kernel 可见资源**不可能在这些回滚点之前发布**，否则回滚也要过同一个 gate。

**② arena freeze owner 定了；stream 那条我判断错了 —— 要分 control 与 execution 两类。**

- **arena freeze owner**＝`DeviceRunnerBase::setup_static_arena`：onboard `device_runner_base.cpp:356`、**sim `src/common/platform/sim/host/device_runner_base.cpp:128`（两边都要加守卫）**。它自己的注释写明了要禁的行为：「If a caller asks for a larger layout on any region, **redo just that region**」。
  **守卫不能只拒"更大"**（codex 第六轮指正）：当前 `requested_size == 0` 会**释放**已有区域，且增长失败还会回滚释放 peer。→ 冻结后的正确语义是：**统一预检三块区域后再做任何 mutation**；允许"保持已有布局"，**拒绝增长，也拒绝非零→零的释放**。
- **stream：我原来写"kernel 模式不建 AICPU stream"是错的**（核实成立）。`ensure_device_initialized`（`:499`）建的 `stream_aicpu_` 是 **bootstrap/control stream**：`:576` 用它跑 AICPU `InitName`、`:583` 还对它做 `aclrtSynchronizeStreamWithTimeout`、`:600`/`:622` 用于 callable 注册与（a5）topology query。**不建它会在任何 launch 之前就打断这些控制路径。**
  → 正确划分是 **control/bootstrap stream vs execution stream**：
  - **control/bootstrap**：保留。它只在 init／注册期使用，是 PyPTO 内部的，**其 synchronize 发生在 init 而非 launch**——所以非目标 #8「不做 stream/device synchronize」要**限定在 launch 路径**，不适用于 init。
  - **execution**：kernel 模式下 AICPU task 排在 **caller stream**，AICore 走 hidden stream。注意 a5 的 owned 执行**复用**这对 bootstrap stream（`a5/.../device_runner.cpp:505`），而 a2a3 的执行另有 `RunStreamPair`（`a2a3/.../device_runner.cpp:581`）——**两个 arch 的改法不同**，不能只改 `ensure_device_initialized`。
- **分层缺口仍然成立**：`PipelineContract` 在 `src/common/platform/` **0 命中**，只存在于 `src/common/worker/`。
  → 但我原来建议的"经 `simpler_init` 传进去"**签名承载不了**（codex 指正）：`simpler_init` 没有保留字段，`prewarm_config` 是 nullable 的 per-task `CallConfig`（`call_config.h:111`，还是共享内存 wire POD），塞不进 context 级 mode。
  → **定案**：新增一个**版本化的 context-config ABI**，在 `simpler_init` **之前**传入 mode，由 platform 自己推导 control/execution stream 策略；容量建立完成后再经一个**显式的 freeze transition** 锁定；sim 侧在 `SimDeviceRunnerBase` 存 mode/freeze 并同样守卫 `setup_static_arena`（sim 无真实 stream，故没有"少建一条"的对应落点）。

**③ persistent `KernelArgs`：不上移到 `ChipWorker` 是对的，但"扩展 `KernelArgsHelper`"是错的 —— 它是 per-run 的。**
核实（codex 第六轮指正，成立）：`KernelArgsHelper` **嵌在 `PreparedExecution` 里**（`device_runner_base.h:545` 起的 struct），**每次 prepare 创建、每次 cleanup 释放**（a2a3 cleanup 路径引用 `prepared.kernel_args`，`device_runner.cpp:460` 附近）。
→ 直接扩展它去承载 context 生命周期，等于**让一个对象同时跑 owned per-run 与 kernel persistent 两套状态机**——正是表二／C.4 明令避免的。
→ **定案**：新增一个独立的 **`PersistentKernelArgs` owner（或包装层）**，挂在 `DeviceRunner`／专用 kernel context 上；**复用** `KernelArgsHelper` 的分配逻辑与 `runtime_device_copy_size()`（后者保证 runtime-agnostic：trb 只拷 `dev` 描述符、hbg 拷整个对象），但**不复用它的生命周期**。
→ 两个配套细节：helper **没有自动释放的析构**，依赖显式 `finalize_*()`，所以新 owner 必须在 `mem_alloc_.finalize()` **之前**显式释放；sim 侧目前是 runner 级的 `KernelArgs kernel_args_`（`sim/host/device_runner_base.h:426`），要单独保持语义一致。
→ allocator 与 arch-specific `KernelArgs`（`src/{a2a3,a5}/platform/include/common/kernel_args.h`）继续留在 platform 层。

**④ mode 不进 compile cache key —— 但要写成规范性不变量，不是从"参考没加"推出来的。**
codex 第六轮的方法论批评成立：参考的 cache key 里没有 mode，**不能证明**它不该加（主线还没有这个 mode）；是否入 key 只取决于 **mode 是否改变 pass／codegen／artifact metadata／编译得到的 facade**。（另：我引的 `python/pypto/runtime/l1.py` 只存在于**参考**，不能当主线证据。）
→ **定为规范性约束**：**mode 只影响 context 与运行分发，不影响编译产物**；因此 (a) 不得成为第三种 `RuntimeKind`（`pass_context.h:247` 明确它只表示两种 runtime ABI flavor），(b) 不进 `jit/cache.py` 的 `compile_opts`（`:253` 已按 runtime flavor 分键）。**并用测试锁定"两种 mode 共用同一 compile cache/artifact"**，否则这条约束会被后来的改动悄悄破坏。
→ requirements 符号：**存在性无条件新增**，内容只由编译后的程序需求决定，与 mode 无关，故同样不拆 cache——这条也写成不变量。
→ **分发链（补全 codex 指出的枢纽）**：`CompiledProgram` → `runner.execute_compiled`（`runtime/runner.py`）→ **`device_runner.execute_on_device`（`runtime/device_runner.py`）** → active/one-shot Worker。而 **active Worker 的 binding 目前是 `(level, platform, device_id, runtime)`**（`runtime/worker.py:455`）——既然一个 context 不混 mode，**binding 与复用查找必须加入 mode**。

> **主线漂移检查（本轮）**：simpler 新增 1 个 commit `35f195bd`「Fix: preserve explicit dependency flags in dep_gen replay (#2057)」，改的是 dep_gen replay／TMR orchestrator／platform_config／dfx 文档与 ST，**未触及本节任何落点**（`chip_worker.*`、`pipeline_contract.h`、两个 `runtime_maker.cpp` 的 host 侧、`device_runner_base.cpp` 均未改）。pypto 无新增。

**其余仍需补细的（不阻塞架构，但索引要写全）**：Probe 目录是接入现有 `tests/st/a2a3/{hbg,tmr}` 还是新建（含测试注册方式）；第 2 步要列 `chip_worker.h:314` 附近的函数指针类型/成员并扩展 8 构件 ABI 测试；第 4 步的 requirements dlsym 必须是**可选解析**（缺符号记 `available=false`，旧产物仍可走 owned，**只有 kernel mode 才拒**——不能复用 `load_symbol` 的强制语义），且 requirements bit 应在**真正调用普通 tensor-op emitter 时置位**（`pypto/src/codegen/tensor_op_codegen.cpp` 附近），不能靠 IR 文本扫描，还要定 mode 如何传到 `bind_callable_to_runtime_impl` 的校验点（**当前签名没有 mode**）；第 7 步要说明 HBG-specific plan cache 是否放 `common/worker` 的依赖边界；第 11 步要补 `python/bindings/CMakeLists.txt` 的 source/link 落点、module 注册、`.pyi`/facade、以及**无 `torch_npu` 时的 build/package 行为**；第 12 步的测试路径要加 `[simpler]`/`[pypto]` 前缀并写具名验收项。

**一条表述纪律**：`ACL_EVENT_SYNC` 在已安装 ACL header 中存在，但主线只有 `ACL_EVENT_TIME_LINE` 的用例——"默认 flag 会被 capture 拒绝"**必须作为第 1 步 probe 的硬验收项**，不能写成仓内已证实的事实。

### 9.5 开工前仍需定死的四项（codex 第六轮判定，我认同）

9.4 给出的是**落点**；下面四项是**还没定的设计决定**，每项都必须在动第一行代码前定死，否则会写出返工代码。

| # | 要定什么 | 为什么不能推迟 |
| - | -------- | -------------- |
| 1 | **finalize 失败的完整语义**：显式传播（返回/抛）＋ `CLOSING_RETRYABLE` 之类的中间态 ＋ 析构 non-throwing 只泄漏 ＋ init 回滚与 DMA 失败路径一致 ＋ **Python/nanobind 各层只在成功时丢弃所有权** | 现有两个 bool 表达不了中间态；上层 `finally` 会无条件清状态。语义不定，gate 形同虚设 |
| 2 | **control stream 与 execution stream 的划分** ＋ **版本化 context-config ABI**（在 `simpler_init` 之前传 mode）＋ **显式 freeze transition** ＋ **sim 对称落点** | a2a3／a5 的 execution stream 结构不同；`simpler_init` 签名承载不了 mode；freeze 必须能拒绝"增长"和"非零→零" |
| 3 | **platform context 里 persistent `KernelArgs` 的真正 owner**（独立 `PersistentKernelArgs`，不沿用 per-run 的 `KernelArgsHelper`），含释放顺序与 sim 语义 | 沿用 per-run owner 会让一个对象跑两套生命周期；helper 无自动析构，释放顺序错了就泄漏 |
| 4 | **mode 的完整分发链**（补 `runtime/device_runner.py` 枢纽）＋ **把 mode 加进 active Worker 的 binding**（现为 `(level, platform, device_id, runtime)`）＋ **用测试锁定两种 mode 共用同一 compile cache/artifact** | 一个 context 不混 mode，binding 不带 mode 会错误复用；"不进 cache key"这条约束需要测试守住 |

> 这四项**不需要**再读参考材料，也不需要重开前 8 节 —— 都是在主线代码上可以直接定的设计决定。

### 9.6 §9.5 四项的定案

**基线已推进到 simpler `3e08ede7` / pypto `8b5ac325`**（本轮 pull）。其中 **#2068「retire hbg logic that no longer fits host orchestration」改了两份 HBG `runtime_maker.cpp`**（9 insert／14 delete）——我核对了本节全部锚点符号，`acquire_graph_definition_block`、`run_host_orchestration`、`bind_callable_to_runtime_impl`、`HostOrchEntryPoints`、`bind_graph_definitions` **全部存活**，行号约下移 5 行。这正是"按符号锚定"的价值；后续每轮仍需重跑这个检查。

---

**定案 1 · 关闭是"可选的、调用方担保的、幂等的"，不是"由我们判定安全"—— 这消掉了中间态问题。**

我原来想的是"前移一个无副作用的许可查询"。**自查后这条要修正**：许可**根本不是我们能算出来的**。原则10 已经承认 host 侧无法观测 CANN 内部的 graph/replay 生命周期，C30／D13 也确认主线不存在"captured package 可回收"的信号。参考 §12.1 把这件事写得很直白：

> 普通 JIT 用户**可以完全不调用** close/shutdown，hidden owner 会**安静 pin 到进程结束**。若用户选择调用……caller 必须：① 停止新的 enqueue ② 等所有 eager task 完成 ③ 等所有 graph replay 完成 ④ reset/destroy 持有 L1 node 的 ACLGraph ⑤ 保持 context／binary／tensor storage／HBG source owner 到上述步骤完成。
> **shutdown/close 自身不做 synchronize，也不查询 graph owner。它重复调用幂等；失败时保留 owner 供显式重试。GC/atexit 不调用 runtime close。** 销毁某一张 graph 不会自动触发 device-wide shutdown。

于是定案是：

- **默认不关**：kernel 模式下 **`~ChipWorker()` 不执行设备侧 teardown** —— 资源按设计 pin 到进程结束。这直接消掉了"non-throwing 析构如何决策"的整个问题（比我原来的"析构时被拒就泄漏"更干净），也与 **GC/atexit 不得触发 close** 一致。
- **只在显式调用时关**：新增 `int close()`（或等价显式入口），语义为：
  - **幂等**：重复调用安全；
  - **只校验我们能观测的**：没有 in-flight／已 prepare 的 native run、phase 允许关闭；**不 synchronize、不查询 graph owner**；
  - **失败时保留 owner 供显式重试** —— 这就是"可重试"的全部含义，**不需要新的状态位**，因为失败路径什么都没拆。
- **因此许可检查前移仍然成立，但性质变了**：它不是"安全性判定"，而是"**本地可观测前置条件校验**"。可失败的一步因此天然在最前面，`finalize()` 现有的"先拆 `run_lane_` 再调 `finalize_device`"顺序问题随之消失。
- **调用方担保写进契约**（B2 非目标 #10 已有雏形）：上面五条前置条件是**调用方的义务，我们不验证**。
- **上层只在成功时丢所有权**：`python/simpler/task_interface.py` 现在无论成败都在 `finally` 清 callable registry、`python/simpler/worker.py` 成功返回后丢 `_chip_worker` —— 两处都要改成**成功才清**；且 **GC/atexit 路径不得调 close**。
- **其余销毁路径**：init 期 context 创建后失败走 `DlHandleGuard` 的 dlclose（`chip_worker.cpp:268` 附近）—— 约束"kernel 可见资源不得在该回滚点之前发布"即天然安全；DMA provisioning 失败（`:438` 附近）在 `initialized_=true` 之后调 `finalize()`，按 kernel 模式"默认不关"处理；**禁止进程内切换 runtime**、**禁止 unregister/cid 复用**（9.4①）。

**定案 2 · 一个带 action 的版本化 context-control ABI，承载 configure(mode) 与 freeze 两类操作。**

`simpler_init` 承载不了（无保留字段；`prewarm_config` 是 nullable 的 per-task `CallConfig` wire POD，`call_config.h:111`）。因此**新增一个版本化的 context-config ABI**，在 `simpler_init` **之前**调用，一个符号解决三件事：

| 操作 | 方向 | 时点 |
| ---- | ---- | ---- |
| 传入 mode（owned／kernel）与 kernel 侧容量意图 | in | `simpler_init` 之前 |
| ~~关闭许可查询~~ **不需要**：定案 1 修正后，关闭是调用方担保 ＋ 本地前置条件校验，无需向 platform 查询"是否安全" | — | — |
| 显式 **freeze transition**（锁定容量） | in | 初始容量建立完成后 |

- **stream 策略由 platform 自己按 mode 推导**，并且必须**区分两类 stream**（9.4② 的修正）：
  - **control/bootstrap**：`ensure_device_initialized`（`device_runner_base.cpp:499`）建的 `stream_aicpu_`／`stream_aicore_`。**两种 mode 都要保留** —— 它跑 AICPU `InitName`（`:576`）、对它 `aclrtSynchronizeStreamWithTimeout`（`:583`）、callable 注册（`:857`，`RegisterCallableName` 经 `launch_aicpu_payload`）也用它；`:600`/`:622` 是 **binary bootstrap**（`BootstrapDispatcher`），不是注册——**这处引用我原来标错了**（codex 第七轮指正），a5 topology query 实际在 `a5/.../device_runner.cpp:167`。**其 synchronize 在 init 期，不违反非目标 #8**（该条限定 launch 路径）。
  - **execution**：kernel 模式下 AICPU task 排 caller stream、AICore 走 hidden stream。**a5 的 owned 执行复用 bootstrap 这对**（`a5/.../device_runner.cpp:505`），**a2a3 另有 `RunStreamPair`**（`a2a3/.../device_runner.cpp:581`）→ **两个 arch 的改动不同，必须分别做**。
- **freeze 守卫**落在 `DeviceRunnerBase::setup_static_arena`：onboard `device_runner_base.cpp:356`、**sim `src/common/platform/sim/host/device_runner_base.cpp:128`**。语义：**先统一预检三块区域再做任何 mutation**；放行条件见 **§9.8b** 的两条精确谓词（不是简单的"拒绝增长＋拒绝非零→零"——那会误伤 HBG 每次 bind 都传 `gm_sm_size=0` 的正常调用）。
- **sim 对称**：在 `SimDeviceRunnerBase` 存 mode/freeze 并同样守卫 `setup_static_arena`；sim 无真实 stream，**没有"少建一条 stream"的对应落点**。

**定案 3 · 新增独立的 `PersistentKernelArgs` owner，挂在 kernel context 上。**

不沿用 `KernelArgsHelper`：它嵌在 `PreparedExecution`（`device_runner_base.h:545` 起）里，**per prepare 创建、per cleanup 释放**，扩展它会让一个对象同时跑 owned per-run 与 kernel persistent 两套状态机（表二／C.4 明令禁止）。

- **复用**其分配逻辑与 `runtime_device_copy_size()`（保证 runtime-agnostic：trb 只拷 `dev` 描述符、hbg 拷整个对象）。
- **不复用**其生命周期：新 owner 挂在 `DeviceRunner`／专用 kernel context，prepare 期一次 H2D，之后每次 launch 只传同一 device pointer。
- **释放顺序**：helper 无自动析构、依赖显式 `finalize_*()`，所以新 owner **必须在 `mem_alloc_.finalize()` 之前**显式释放。
- **sim**：目前是 runner 级 `KernelArgs kernel_args_`（`sim/host/device_runner_base.h:426`），需单独对齐语义。
- allocator 与 arch-specific `KernelArgs`（`src/{a2a3,a5}/platform/include/common/kernel_args.h`）继续留 platform 层。

**定案 4 · mode 进 Worker binding，不进 compile cache key，并用测试锁住。**

- **binding**：`ChipWorker._binding`（pypto `python/pypto/runtime/worker.py:455`）现在返回 `(level, platform, device_id, runtime)` → **加 mode**。同一处注释（`:97-102`）说明**复用查找在 `device_runner.execute_on_device`**，那里的匹配也要带 mode。**必须加**：一个 context 不混 mode（非目标 #2/#3），binding 不带 mode 会错误复用一个 owned context 去跑 kernel 调用。
- **分发链**：`CompiledProgram` → `runner.execute_compiled`（`runtime/runner.py`）→ **`device_runner.execute_on_device`（`runtime/device_runner.py`）** → active／one-shot Worker。
- **不进 cache key**（规范性不变量，不是"参考没加"推出来的）：mode 不改变 pass／codegen／artifact metadata／facade，因此不进 `jit/cache.py` 的 `compile_opts`（`:253` 已按 runtime flavor 分键），也**不得**成为第三种 `RuntimeKind`（`pass_context.h:247` 明确它只表示两种 runtime ABI flavor）。
- **用测试锁定**：一个显式测试证明"同一 program 在两种 mode 下命中同一 compile cache entry／同一 artifact"。这条约束没有测试守着，会被后续改动悄悄破坏。
- requirements 符号：**存在性无条件新增**，内容只由编译后的程序需求决定，与 mode 无关，故同样不拆 cache——同为不变量。

### 9.7 第七轮补正：三处事实修正 ＋ 三条尚未定死的语义

**修正 A（引用错误）**：`device_runner_base.cpp:600/622` 是 **binary bootstrap**（`BootstrapDispatcher`），不是 callable 注册；注册在 **`:857`**（`RegisterCallableName` 经 `launch_aicpu_payload`，同样用 `stream_aicpu_`），a5 topology query 在 **`a5/.../device_runner.cpp:167`**。我原来的结论（`stream_aicpu_` 是被多条 control 路径共用的 bootstrap/control stream）不变，但引用要改对。
→ 顺带把非目标 #8 的收窄说得更准（codex 建议）：不是"限定 `init()` 内"，而是 **"非-launch 的 control 路径允许同步"** —— 因为 callable 注册与 a5 首次 topology prepare 也会同步，且并不都发生在 `init()` 里。

**修正 B（定案 1 不完整）**：让 `~ChipWorker()` 不调 `finalize()` **不够**。`run_lane_` 是成员（`chip_worker.h:416`），而 **`ChipRunLane::~ChipRunLane()` 里就是 `try { close(); } catch (...) {}`**（`chip_run_lane.cpp:353`）——成员析构照样会做 runtime 调用。
→ **kernel 模式必须显式 abandon／leak 这个 owner**（例如 `run_lane_.release()` 之类的放弃语义），否则"析构零 runtime 调用"兑现不了。

**修正 C（还有一条 GC close 路径）**：pypto 侧 `python/pypto/runtime/worker.py:183` 附近用 **`weakref.finalize` 挂了一个 impl-only finalizer，"armed until a successful close"**，即不可达的 wrapper 会被 GC 触发 `impl.close()`。
→ 与 simpler 侧一样，**这条路径在 kernel 模式下必须按 mode 禁掉**（参考 §12.1：「GC/atexit 不调用 runtime close」）。

---

### 尚未定死的三条语义（本轮定，codex 第七轮要求）

**① post-admission teardown 失败：定为终态，不承诺重试。**
codex 指出我"两个 bool 足够"只对**前置拒绝**成立：一旦通过前置检查进入 teardown，`device_runner_base.cpp:1362` 及 a2a3 `:1073`／a5 `:903` 的公共/各 arch teardown 会在 stream destroy、allocator finalize、device reset 出错后**继续清状态并返回错误**，此时 `initialized_` 仍为 true 却面对一个半终结的 runner。核实成立。
→ **二选一里我选"终态"**：

- **前置拒绝**（本地可观测前置条件不满足）→ **零 mutation，可重试**；
- ⚠ **本条已被 §11.1 取代**：改为参考的 `CLOSING` ＋ 显式重试。以下原文保留以便追溯。
  ~~通过前置后任何 teardown 错误 → **终态 unusable**~~：不再承诺重试、不再允许 dispatch、**所有尚存资源及其 host owner 保持 pin 且不再发起任何 cleanup**（codex 措辞修正：teardown 前半段已成功销毁的资源无法"重新 pin"）、记录泄漏。
  **⚠ 实现约束见 §10.1**：终态判定**不得用单个 first-wins 错误槽**（会遮住真实 teardown 失败），须另设"意外 teardown 错误"专用槽，决策顺序 runtime status → unexpected teardown → 受控结果。
  **状态位组合固定为三种**（codex 建议，采纳）：live = `initialized_=true, finalized_=false, terminal_=false`；干净终结 = `false, true, false`；**终态/pinned = `true, false, true`**。所有入口先查 `terminal_`。实现上必须在**第一个 post-admission 错误**处进入终态、停止后续 runtime cleanup、并 abandon `run_lane_`。
  理由：承诺重试需要按资源记 cleanup debt，与"析构零 runtime 调用"和原则6（无可靠释放信号就 pin）都冲突；而"终态 + 全部 pin"正是原则7"在不 reset 的前提下自己收敛"的形态。需要的状态位只有一个 `terminal_`，不是完整的 cleanup-debt 机。

**② 版本化 context-control ABI 的 wire contract（要定到可实现）**：固定 `abi_version`、`struct_size`、`action`（至少 `CONFIGURE` / `FREEZE`）、payload、错误码；定义 **重复调用与越序调用**的语义（`CONFIGURE` 幂等且必须在 `simpler_init` 之前；`FREEZE` 幂等、`FREEZE` 之后再 `CONFIGURE` 一律拒）；按 C/POD 规则加 `is_trivially_copyable` / `is_standard_layout` 静态断言（仓内先例：`HbgContextRegistry` 的 `alignas(64)` ＋ static_assert）。

**③ 异 mode fallback 与 kernel one-shot 生命周期**：

- **同设备已有异 mode active context 时，明确拒绝**（不允许第二 context）。依据非目标 #2（v1 单 live context）＋ #3（同 context 内不得重叠）。
- **kernel 模式不支持静默 one-shot**：`ChipWorker.current()` 在 binding 加了 mode 之后，遇到异 mode 会返回 `None`，从而**静默落到 one-shot**，而 one-shot 目前在 `finally` 里**无条件 `close()`**（`runtime/device_runner.py:807` 附近）——这与"默认不关、pin 到进程结束"直接冲突。
  → **定案：kernel 模式要求显式 `ChipWorker`**；无 active worker 时**报错而不是造 one-shot**。
- 别忘同步四元组的其它使用点：`ChipWorker.current()` 的目标构造与比较（`worker.py:498` 附近）、`__enter__` 的重复检测与解包（`:655` 附近）。

**④ persistent owner 还缺的三条生命周期**（定案 3 的补充）：`runtime_args` 成功而 `device_k_args` 失败时的**部分初始化回滚**；fatal reset/quarantine 之后的 **`abandon()`**（不得再经 allocator free，先例见 `device_runner_base.cpp:1421`）；以及**明确 a5 `regs`、a2a3 `regs`/`pmu_reg_addrs` 与 collector backing 是否属于 persistent owner** —— 它们当前由各 arch 的 cleanup 单独释放（a2a3 `device_runner.cpp:464`、a5 `:614`），而不是 `KernelArgsHelper::finalize_*()`。sim 侧则要逐字段说明 kernel 模式下哪些 `kernel_args_` 保持 persistent、哪些仍由 active-run cleanup 清零。

### 9.8 context-control ABI 的具体 wire contract

按主线既有的跨 DSO 版本化结构惯例写（先例：`src/common/log/include/common/host_log_state.h:47` 的 `SimplerHostLogState`，校验风格见 `host_log.cpp:270`；`PipelineContract` 同为精确版本匹配）。

**结构**（POD，C linkage）：

```c
enum {
    SIMPLER_KERNEL_CTX_CONTROL_ABI_VERSION = 1,
};

typedef enum SimplerKernelCtxAction {
    SIMPLER_KERNEL_CTX_CONFIGURE = 1,   /* 传 mode ＋ 容量意图；必须在 simpler_init 之前 */
    SIMPLER_KERNEL_CTX_FREEZE    = 2,   /* 锁定容量；初始容量建立完成后 */
} SimplerKernelCtxAction;

typedef enum SimplerKernelMode {
    SIMPLER_MODE_CHIP_OWNED    = 0,     /* 现状语义，默认值 */
    SIMPLER_MODE_CHIP_BORROWED = 1,
} SimplerKernelMode;

typedef struct SimplerKernelCtxControl {
    uint32_t abi_version;   /* 必须等于 SIMPLER_KERNEL_CTX_CONTROL_ABI_VERSION */
    uint32_t struct_size;   /* 调用方 sizeof；消费方要求 >= 自己的 sizeof */
    uint32_t action;        /* SimplerKernelCtxAction */
    uint32_t mode;          /* SimplerKernelMode；仅 CONFIGURE 时有意义 */
    /* CONFIGURE 时的容量意图：0 表示"用配置默认值" */
    uint64_t gm_heap_bytes;
    uint64_t gm_sm_bytes;
    uint64_t runtime_arena_bytes;
    uint64_t reserved[4];   /* 必须全 0，否则拒 —— 未知非零即 fail-closed */
} SimplerKernelCtxControl;

int simpler_kernel_ctx_control(DeviceContextHandle ctx, const SimplerKernelCtxControl *control);
```

**校验规则（全部 fail-closed）**：

- `control == nullptr`、`abi_version != ..._ABI_VERSION`（**精确匹配**，与主线一致）、`struct_size < sizeof(SimplerKernelCtxControl)`、`action` 不在枚举内、`mode` 不在枚举内、**`reserved[]` 任一非 0** → 拒。
- 静态守卫：照主线真实先例 `src/common/worker/device_memory_info.h:24` 的双守卫写法 —— `static_assert(std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T>)` **再加一条固定布局断言** `static_assert(sizeof(SimplerKernelCtxControl) == ...)`，并对每个字段加 `offsetof` 断言（wire ABI 必须钉死布局）。
  （注：我 §9.7 引的 `HbgContextRegistry` 只存在于**参考**，主线没有；已改引上面这个真实先例。）

**状态转换与幂等**（这是之前最缺的一块）：

| 当前 | 收到 `CONFIGURE` | 收到 `FREEZE` |
| ---- | ---------------- | ------------- |
| 新建（未 `simpler_init`） | **接受**；**同 effective tuple** 重复调用**幂等**；tuple 不同则拒（不允许改主意） | 拒（容量还没建立） |
| 已 `simpler_init`、**容量尚未建立** | **拒**（越序） | **拒**（codex 补：否则空状态先 freeze，之后第一次非零 `setup_static_arena` 必被 §9.8b 拒——等于把 context 锁死） |
| 已 `simpler_init`、容量已建立、未 freeze | **拒**（越序：mode 必须在 init 之前定） | **接受**；重复调用**幂等** |
| 已 freeze | 拒 | **幂等接受**（同一状态重复确认） |

- **"同参数"的比较域**（codex 要求定死，采纳）：**不要 `memcmp` 整个结构**。首次 `CONFIGURE` 时先把 0 解析成配置默认值，保存 **effective tuple** `(mode, gm_heap_bytes, gm_sm_bytes, runtime_arena_bytes)`；之后只比较这个 tuple。`abi_version` / `struct_size` / `action` / `reserved` **不参与**比较。
- **未使用 payload 规则**：`FREEZE` 时 `mode` 与三个容量字段**必须为 0**，否则拒（避免"看起来在改配置其实被忽略"）。
- **默认行为不变**：完全不调用这个 ABI 时，mode 保持 `CHIP_OWNED`，一切与今天一致 —— 这是 owned 路径零回归的保证。
- **错误码：必须新增，不能"沿用"。** 我原先写的 `PTO_RUNTIME_ERR_INVALID_ARGUMENT` **不存在**（codex 指正，已核实）；主线 host 侧只有 `PTO_RUNTIME_ERR_{BASE, INTERNAL, UNSUPPORTED, PREPARED_INCOMPATIBLE}`（`runtime_c_api.h:95` 起）。而且**不存在可沿用的 host 侧 capacity error**。
  → 需要在该 band 内**新增并钉死数值**（band 规则：`PTO_RUNTIME_ERR_BASE = -(PTO_RUNTIME_LATCHED_CODE_MAX + 1) = -1000`，host 侧自此**递降**，与 device-latched 段**互不相交**，且各 `runtime_maker.cpp` 里有 static_assert 把两段隔开——新码必须遵守这条）。建议至少新增：
  - `INVALID_STATE`（**越序调用**——不要归进 `INTERNAL`，codex 建议，采纳）；
  - `CAPACITY_EXCEEDED`（freeze 后的增长/释放请求，供 §9.8b 的谓词使用）。
    结构/枚举/`reserved` 非法沿用 `INTERNAL`（与主线校验失败的既有用法一致）；合法但不支持的 borrowed 请求返回既有的 **`UNSUPPORTED`**。

**导出面**：这是进入 `ChipWorker` 强制 dlsym 表的符号 → **8 个 host-runtime 构件全部导出**（9.1①），不支持 kernel 模式的构件导出**显式 stub**，但 **stub 必须执行与真实实现完全相同的结构校验、时序检查与幂等状态机**（codex 指正我原来的写法与统一状态表冲突，采纳）——只在**校验全部通过之后**才按能力差异分流：

- `CONFIGURE(mode=CHIP_OWNED)` 且容量意图全 0 → 0；**带非零容量意图 → 不得无条件成功**（否则等于虚假声明已履约），返 `UNSUPPORTED`；
- `CONFIGURE(mode=CHIP_BORROWED)` → `UNSUPPORTED`；
- `FREEZE` → 仅在时序合法且该构件确无待冻结对象时返 0，否则按统一状态表拒。并**同步扩展 `tests/ut/py/test_host_runtime_abi.py` 的符号集**。

> **命名待定**：`simpler_kernel_ctx_control` 只是占位。落地时按仓内 `*_ctx` 后缀惯例定名（先例：`committed_device_memory_ctx`、`supports_concurrent_native_prepare_ctx`）。

### 9.8b freeze 守卫的精确谓词（读 `commit_region` 定出来的）

我前面写的"拒绝增长，也拒绝非零→零"**不够精确，而且会误伤 HBG**。`DeviceRunnerBase::setup_static_arena` 里的 `commit_region` lambda（`device_runner_base.cpp:376` 附近）只有三条分支：

```text
if (requested_size == 0)                              → 仅当已提交且 cached≠0 才 release  // 未提交时是 no-op
if (arena.is_committed() && requested_size <= cached) → return 0                    // 短路，零 mutation
otherwise                                             → arena.release(); reserve()  // 换基址！
```

三个 region 依次走它（`gm_heap` / `gm_sm` / `runtime_pool`）：onboard **`device_runner_base.cpp:412-414`**、sim **`src/common/platform/sim/host/device_runner_base.cpp:170-172`**（我原来写的"`:57-59`"是 awk 相对行号，已改为绝对行号）。于是：

**freeze 后的准入谓词 —— 只接受"本来就不会 mutate"的调用**：对三个 region 每一个，满足下列任一即可放行，否则**在任何 mutation 之前整体拒绝**：

1. `arena.is_committed() && requested_size <= cached_size` —— 走短路分支；或
2. `requested_size == 0 && !arena.is_committed()` —— 对一个从未提交的 region 请求 0，是真正的 no-op。

**为什么第 2 条必须有**：**HBG 每次 bind 都传 `gm_sm_size = 0`**（`a2a3/.../host_build_graph/host/runtime_maker.cpp:869` 附近）——它从来不用独立的 GM SM 区（image 挂在 runtime arena 尾部）。若照我原来的"拒绝非零→零"一刀切，**HBG 的每一次正常 bind 都会在 freeze 后被拒**。这就是"设计note"与"可实现"的差别。

**因此被拒的恰好是三类危险调用**：任一 region 请求变大（走 release+reserve → **换基址**，captured graph 持的是旧地址）；对已提交 region 请求 0（**真释放**）；region 未提交而请求非 0（首次提交 —— freeze 之后不应再有首次提交）。

→ 把 9.4② 与 9.6 定案 2 里"拒绝增长，也拒绝非零→零"的表述**替换为上面两条放行条件**。sim 侧 `src/common/platform/sim/host/device_runner_base.cpp:128` 的同名函数按同样谓词处理。

### 9.9 本轮漂移复核

- **simpler**：`3e08ede7`，本轮**无新提交**。
- **pypto**：新增 2 个 —— `62d75b44` fix(codegen): PTOAS v0.60 sort32 on A5、`6d02567e` fix(language): preserve tensor subclasses through loop carries。**均未触及任何落点**（`jit/`、`runtime/{runner,worker,device_runner}`、`codegen/orchestration`、`ir/compiled_program` 全部未改）。
- **锚点复核**：`acquire_graph_definition_block`(1)、`run_host_orchestration`(3)、`bind_callable_to_runtime_impl`(2)、`HostOrchEntryPoints`(3)、`bind_graph_definitions`(3) 全部存活；`pipeline_contract.h`、`chip_worker.cpp`、`device_runner_base.cpp`、`device_runner_helpers.h` 均在。

> 这个复核每轮都要重跑 —— HBG `runtime_maker.cpp` 与 pypto `jit/*` 是已知热点（#1982 #2040 #2051 #2060 #2068 #2539 都动过它们）。

---

## 10 · 参考的实证边界，与由此得到的两条补正

前九轮我对参考的 7385 行 `实现过程记录.md` 只取了 codex 的转述。本轮自己读了 N.13 指名为权威的 **10.48–10.62**，得到两条应当写进设计的东西，以及一份可直接用于 Phase 0 范围界定的"未覆盖清单"。

### 10.1 补正一：first-wins 错误槽会遮住真实的 teardown 失败

我在 §9.7① 把 post-admission 规则写成"**第一个** teardown 错误处进入终态"——**这是 first-wins 语义，参考踩过这个坑**（过程记录 10.49.3）：

> 审查还发现 first-wins `run_error_` 可能**先被合成 stage 错误占用，从而遮住另一 participant 真实的 `shutdown()` 失败**。最终代码增加独立 `hbg_unexpected_teardown_error_`：每个 participant 无论当前已有何种合成错误，都把**首个真实 shutdown 错误**汇合到该槽；所有线程完成 finalize 后，**返回决策先检查 runtime status 和 unexpected teardown，再考虑 controlled success**。于是只有**全部真实 shutdown 成功**时测试钩子才可能返回 0。

→ **补正**：终态判定**不能用单个 first-wins 错误槽**。必须另设一个**"意外 teardown 错误"专用槽**，任何 participant 的首个真实 teardown 失败都汇合到它；最终决策顺序是 **runtime status → unexpected teardown → 才考虑 controlled/预期结果**。否则一个预期内的受控错误会把真正的 teardown 失败吃掉，我们会以错误的理由进入终态，甚至误报成功。
→ 这条对 §9.7① 的"终态"选择本身没有影响（选择仍成立），但**改变了它的实现约束**。

### 10.2 补正二：device 侧 teardown 有七个必须各自收尾的阶段

我原来只写了"fault injection 覆盖每条失败路径"。参考给出了具体的**七个注入点**，且每个都列明"**必须经过的真实收尾**"（10.49.3）：`restore_copy`、`restore_publish`、`after_scheduler_init`、`before_classify`、`before_dispatch`、`shutdown`、`runtime_destroy`。共同要求是——无论在哪一阶段失败，都必须走完 **peer 统一跳过 classify/dispatch → 逐线程 shutdown → completion gate → deinit**，且 working slot 可以被部分改写但**不得发布 commit**。
→ 这七个阶段应直接作为阶段 12（ST）里 TMR/HBG 故障矩阵的**验收清单**，而不是笼统的"每条路径有测试"。
→ 另注一个惯例：注入错误用**专属值 `-1700 - stage`**，与自然错误分离；这与 §9.8 新增错误码要遵守 band 规则是同一思路。

### 10.3 参考**未覆盖**的清单 —— 直接用作 Phase 0 的范围下限

参考自己在 10.48.6 明确列出"仍不能勾选"的项。这些正是我们**不能假设已被证明**、必须自己验的：

| 未覆盖项 | 对我们的含义 |
| -------- | ------------ |
| 所有真实 CANN args base 恰好 `mod64 == 0`；unaligned parser 只有纯 Host 反例 | **不能声称 device backend 曾给出未对齐地址** → 若我们依赖对齐假设，要么自己验，要么显式写成假设 |
| 64 MiB generic payload **不等于**真实 HBG 最大 image，且**没有扫描首个失败 size** | 容量档位（表二／K.4）不能引用这个数字当上界；需要自己做失败 size 扫描 |
| 只有 A2/A3 device1；**A5 没有同等级的硬件时延、错误码与 cache 证据** | 印证非目标 #12（A5 不作完成条件）与 J.1 告警（不得由 A2/A3 推断 A5） |
| production HBG restore **尚未显式 poison** ready queue／wake list／completion／task state／mailbox | §9.8b 的 restore 谓词与原则5 的"最小失效区域"**缺少毒值验证** → 应进阶段 12 |
| wrong slot／callable／blob／affinity／KernelArgs 与四个 teardown 阶段**仍缺 test-build-only fault hook 与 hidden AICore tail 证据** | 与 10.2 的七阶段清单合并成同一份矩阵 |
| **graph owner 在 device 尚未 external quiescent 时的拒绝/保活，尚未单独注入验证** | 这正是 §9.7①「调用方担保、我们不验证」那条契约的**负向测试**，参考也没做过 → 我们必须做 |

> 一条方法论纪律也值得抄：参考记录明确说明本轮全部使用 device1、**未调用任何 device reset**，且结束后 `npu-smi` 仍显示历史 AICore 100% 与固定 HBM、process table 为空，与运行前一致，因此**不能据此归因为本轮资源泄漏**。我们的 Phase 0 也应这样记录基线，避免把环境既有状态误读成自己的泄漏。

### 10.4 吸纳状态结论

到此，参考侧我**亲自读过**的部分覆盖：设计文档正文（§1–§15）、附录 B.9／C／D／F／G.0／J／K／N（N.1–N.3、N.5、N.6、N.9、N.11、N.13）、过程记录 10.48–10.62、以及实现代码的关键头文件与 launch/restore/registry 路径。
**未逐字读**的是：附录 A／E／H／I 全文与过程记录的其余章节。codex 第四轮扫过并报告"不推翻 v9 现有结论，且含大量已被实现覆盖的历史状态"——这一条我保留为**转述**，未独立核实；但由于它们都属"更早的历史快照"（N.13 已声明前面几节不得作为当前状态），风险可接受。

---

## 11 · 附录 E 自读结果：一处**推翻我自己决定**的发现

codex 第四轮扫过附录 A/E/H/I 后报告"没有推翻 v8 的当前架构结论"。本轮我自己读了 **附录 E（stream 协议、生命周期与状态机）**，结论是**这条转述不准确**——E.1 直接冲击我 §9.7① 的选择。**这也说明"只取转述"的风险是真实的。**

### 11.1 我的"终态"选择与参考的设计相反 —— 并且我的理由站不住

参考 **E.1 的 context 状态机**明确给了 `CLOSING` 这个可重试状态：

```text
CLOSING
  | cleanup failure: remain CLOSING; dispatch rejected
  | explicit close retry eventually succeeds
  v
CLOSED
```

即：**cleanup 失败就停在 `CLOSING`、拒绝 dispatch、允许显式重试直到成功**。这正是 codex 第七轮给的**选项 1（retry-only）**，而我当时选了**选项 2（终态）**。

**我当时的理由是错的**：我写"承诺重试需要按资源记 cleanup debt，与析构零调用及原则6 冲突"。但参考的 `CLOSING` **并不记 per-resource debt** —— 它只是停在该状态，再次 `close()` 就**重试尚未完成的部分**（主线现有 teardown 本来就是边走边把成员置空，天然可再进入）。所以"需要 debt 账本"这个前提不成立，我的选择缺乏支撑。

→ **改为采用 `CLOSING` + 显式重试**，与参考一致：

- **前置校验失败** → 状态不变，context 可继续用（E.1 写明 "any pre-enqueue validation error: state unchanged"）；
- **进入 destructive teardown 后失败** → 停在 `CLOSING`，**拒绝一切 dispatch**，允许显式 `close()` 重试；
- **析构仍然什么都不做**（§9.7① 的"默认不关"不变，与此正交）；
- `POISONED` 是**独立状态**，由 launch 路径的部分 enqueue 失败进入，不与 `CLOSING` 混用；
- 进入 `CLOSING` 的前提仍是 **"caller first proves external quiescence"** 后调 `begin_close()` —— 与我 §9.7① 的"调用方担保、我们不验证"完全一致。

→ 状态位相应调整：不是"两个 bool + `terminal_`"，而是参考的七态机（`NEW / INITIALIZING / COLLECTING / READY_ENQUEUED / SEALED / POISONED / CLOSING → CLOSED`）。**§10.1 那条补正仍然适用且更重要**：`CLOSING` 里判定"这次重试是否成功"时，同样不能用单个 first-wins 错误槽，否则受控错误会遮住真实 teardown 失败。

另附 E.1 的一条澄清：**`READY_ENQUEUED` 不表示 device 侧 AICPU init/register 已执行完成**，只表示后续 launch 在 stream/event 上正确依赖它；**capture 前的显式 warmup ＋ caller synchronize 才是把异步准备错误暴露出来的用户流程**。

### 11.2 E.9 的部分-enqueue 失败表：比我的"干净拒绝 vs poison"精确得多

我在 §9.7／I.0 只写了一条分界线。参考 **E.9** 给的是**按失败位置逐行**的处置表，应整表吸收为阶段 9（launch 协议）的实现依据：

| 失败位置 | 已入 device queue 的内容 | 处置 |
| -------- | ------------------------ | ---- |
| validate 失败 | 无 | 直接报错，**context 可继续用** |
| 换 stream 时 query previous tail 失败／not-ready | 无 | 直接报错，**context 可继续用**；调用方外部 quiescence 后重试 |
| handshake memset 失败 | 前序 wait 可能已入队 | **poison** |
| start record 失败 | wait/memset 已入队 | poison |
| AICPU launch 失败 | start 已记录，AICore 尚未 launch | poison；**不能自行 sync/reset** |
| hidden wait／AICore launch 失败 | AICPU 可能已运行 | poison，异步错误由外部同步暴露 |
| done record／caller wait 失败 | AICore 可能已运行但 downstream 无完整依赖 | poison，**禁止继续 enqueue** |
| tail record 失败 | 本次 op 可能完成但下次无法安全排序 | poison |

并附两条：**poison 后 `launch`/`prepare` 全部拒绝**；**`close()` 仍要求调用方先完成 graph/stream teardown —— 它不是故障恢复同步点**。
→ 注意只有**前两行**是"可继续用"，其余六行一律 poison。这比"任何 enqueue 之前失败就干净拒绝"更严格：`handshake memset` 已经算"进入序列"了。

### 11.3 附录 E 里已被取代的部分（不要吸收）

E.4／E.5 的伪代码是 **AICPU-first** 的（E.5 标题即"为什么 host 先 enqueue AICPU 仍不等于越过算子边界"）。**最终设计与实现都是 AICore-first**（§8.2 的序列表 ＋ `l1_launch_sequence.h`，且有两条理由：调度环风险与更好的失败闭包）。→ E.4/E.5 属早期快照，**只取其"host enqueue 顺序 ≠ 越过算子边界"的论证，不取其顺序**。

### 11.4 对"吸纳状态"结论的修正

§10.4 我写"附录 A/E/H/I 未逐字读，保留 codex 转述，风险可接受"。**本轮证明这个风险不可接受**——E 里就有一条推翻我决定的内容。
→ 现状：**E 已自读并吸收**。**A／H／I 仍未自读**，且不能再用"codex 说不推翻"当理由。它们的主题分别是完整决策记录（A）、接口 before/after 对照（H）、完整测试矩阵（I）——**I 与阶段 12 的验收清单直接相关，风险最高，应优先自读**。

---

## 12 · 附录 I 自读结果：一条规则修正 ＋ 一条我缺的三分支规则 ＋ 可直接用的验收矩阵

### 12.1 修正：我的"禁止 unregister"过严

9.4① 我写"kernel 模式下 callable 只能 append，**不得 unregister**、不得复用 cid"。附录 I 的 **UT-027** 给的是更准的语义：

> `unregister` | **host handle 失效，但 pinned device state 不释放**

→ **修正**：`unregister` **本身可以允许**，只要它**只失效 host handle、不释放 pinned device state**；真正必须禁止的是 **cid 复用**（复用才会触发 AICPU orch SO 的 `dlclose` + reload，`runtime_c_api.h:462`）。
→ 精确规则：**允许 unregister（仅失效 host 侧句柄）；禁止 cid 复用；pinned device state 一律不随 unregister 释放。** 表一"TRB append-only code entry：不回收；进程结束"因此仍然成立。

### 12.2 我缺的：换 caller stream 的三分支规则（UT-061 / 062 / 062A）

非目标 #4 我只写了"跨 stream 未 quiesce 切换不支持"。附录 I 把它拆成三个可测分支，其中第三支正是 **107024 的合法绕法**：

| 情形 | 规则 |
| ---- | ---- |
| **同一 stream 再次 launch**（UT-061） | **不 wait、也不 query 旧 tail**；依靠 caller FIFO 进入本次 invalidation |
| **换 stream 且旧 tail not-ready**（UT-062） | **在任何新 enqueue 之前直接失败**；context 可重试；**不允许 AICPU overlap** |
| **换 stream 且旧 tail complete**（UT-062A） | host **query 后继续，但不 enqueue 旧 tail 的 wait** —— 于是"标准 `warmup → sync → capture`"流程**不会把 capture 外的 event 导入图中** |

→ 这条把 C19 从"禁止跨 stream"精确成"**换 stream 需要外部静默 ＋ host 侧 query，且绝不 enqueue 旧 tail wait**"。`wait_for_serial_tail = false` 的实现选择由此得到完整解释：不是放弃串行化，而是**串行化靠 host query ＋ caller FIFO，不靠图内 event**。

### 12.3 可直接采用的验收矩阵（挑对我们最有价值的）

**无需硬件即可跑的（mock/fake 驱动）**：

- **UT-048**：binder mock 的 **malloc/copy/free 调用计数全为 0** —— 零分配的 UT 级证明，比真机 allocator trace 便宜得多。
- **UT-063**：steady-state **0 alloc/free、0 stream/event create/destroy、0 sync/capture-query**，且**同 stream 无 event-status query**（最后这条我原来没有）。
- **UT-073**：fake runtime 中 **capture query／model-get／`rtStreamAddToModel` 调用计数全为 0**。
- **UT-011**：更强的结构性判据 —— **L1 state/ABI 里根本没有 `is_capture`／model handle／early-mode 字段**（断言"字段不存在"，不只是"调用不发生"）。
- **UT-070 / 071**：把入口/出口边界写成**调用序的严格序断言**（start record 严格晚于 caller 既有 task/invalidation；hidden wait 严格早于 AICore launch；hidden done 严格晚于 AICore launch；caller wait 严格早于 tail/downstream）。
- **UT-072**：**WithHostArgs 的 stream 参数与 caller stream 逐 bit 相同**，且 hidden stream 只收 wait/AICore/record —— "AICPU 在 caller stream"的精确断言。
- **UT-007**：状态机迁移符合 E.1，且**「CLOSING 失败可重试且拒绝 dispatch」** —— 再次印证 §11.1 的修正。
- **UT-020**：多 program capacity 按**合法最大 sizing 构建**，**不按简单求和／取 max** —— 与表二"不能盲目逐字段 max"一致（第二次独立印证）。
- **UT-049**：**L2 回归** —— `stage_device_args` 仍走原 H2D/D2H/lease 逻辑（owned 路径零回归的具名判据）。
- **UT-065–069**：逐失败点的 poison 判据，与 §11.2 的八行表对应。

**需要硬件的（进阶段 12）**：

- **ST-E-009 / 010**：prepare/warmup/N 次 launch 后**所有 persistent 地址不变**；warmup 后重复调用 **HBM committed 值不持续增长**（正好用表二定的 `committed_device_memory_ctx`）。
- **ST-E-006 / 007**：连续异步、不同地址／不同 scalar，**不同步**地 enqueue N 次后统一同步，**不串包** —— Probe A 的 ST 版。
- **ST-E-013 / 014**：延迟 predecessor / 延迟 AICore tail —— **Probe F 提升为 ST**，判据是"无抢核死锁/早读"与"immediate successor 只在 hidden done 后观察到输出"。
- **ST-E-008**：换 stream 的 fail-closed 与外部同步后切换（对应 12.2 三分支）。
- **ST-G-001–004**：capture/replay、`pre_op → L1 → post_op` 上下游依赖、**input 内容变化靠 replay 前 copy 到固定地址**、scalar 固定语义（**不要求 PyPTO 动态 patch**）。

> 一处待读：**UT-028 引用了"按 D.6 顺序回收"**，而附录 D 我只读了 D.1–D.4。**D.6（回收顺序）还没读**，它与表一的 teardown 顺序直接相关 —— 列入下一轮。

### 12.4 吸纳状态更新

**已自读**：设计文档 §1–§15；附录 **B.9 / C / D.1–D.4 / E / F / G.0 / I / J / K / N**（N.1–3、5、6、9、11、13）；过程记录 10.48–10.62；实现代码关键路径。
**仍未自读**：附录 **A**（完整决策记录）、**H**（接口 before/after）、**D.5–D.6**（含 UT-028 引用的回收顺序）。
→ 按本轮与上轮的经验（E 和 I 各自都改了我一条结论），**不再把"codex 说不推翻"当作不读的理由**。剩下三块里 **D.6 风险最高**（直接关联 teardown 顺序），下一轮读它。

---

## 13 · 附录 D.5／D.6 自读结果：binary 分三层、close 十步顺序，以及**两份参考文档在这里互相冲突**

### 13.1 D.5：binary 要分三层（我表一是平的）

| 层 | 去重依据 | 生命周期 |
| -- | -------- | -------- |
| **Executor binary**（AICPU dispatcher/inner SO ＋ AICore executor ELF） | 每个 kernel context 一份 | 见 13.3 的冲突结论 |
| **Callable orchestration binary** | 按 **orchestration ELF Build-ID** 去重 | **一旦被 graph 使用即 pin 到 context close** |
| **Child/incore binary** | 按 **content identity** 去重，并记录**稳定 GM 地址** | 每个 callable 保存自己的 `(func_id, device_addr)` 快照且**不可重绑**；不同 callable 可用**相同 func_id 数值指向不同 binary** |

v1 明确不做：**LRU、按 graph 数量的 refcount、runtime-completion 驱动的 binary recycle** —— 理由是「**PyPTO 看不到 graph 何时永久不再 replay**」（＝原则6／C30 的同一根因）。

并且给出了我 §12.1 那条修正背后的**确切危害**：

> 简单"unregister callable 就 free child binary"会造成**已经 capture 的 graph 在未来 replay 时读取悬空地址**。

→ 这印证 §12.1 是对的（unregister 只能失效 host 句柄），并把"为什么"补上了。

### 13.2 D.6：close 的十步释放顺序（可直接采用）

前置条件与我 §9.7① 一致：**所有 graph 已销毁、调用方已保证相关 stream quiescent；`close()` 自身不建立这个前置条件，只验证可验证的 host state。**

1. 阻止新的 prepare/launch，取得 context host mutex；
2. 清理 Python callable handles，但**保留强引用到 native close 调用结束**；
3. 销毁/卸载 callable orchestration device descriptors；
4. 释放 child binary GM 与 callable buffers；
5. 释放 shared workspace、TRB arena、Runtime、KernelArgs、regs/FFTS/handshake；
6. 销毁 kernel 模式自有 events；
7. 销毁 hidden AICore stream；
8. ~~卸载 PyPTO 自有 AICPU/AICore binary handles~~ → **见 13.3，此步与当前态冲突，不采用**；
9. 清理 host registry/state；
10. **不调用** `rtDeviceReset`、`aclFinalize`，**不销毁 caller stream**，不改变 torch_npu 的 current device ownership。

末尾还有一条与 J.7 一致的兜底：**若某个 binary handle 当前没有安全的 unload API，就记录为 context/process-lifetime pinned，而不是在 close 里调用未经验证的内部接口。**

### 13.3 两份参考文档在 binary 卸载上直接冲突 —— 以代码为准

- **附录 D.6 第 8 步**（较早的 plan）：close 时**卸载** PyPTO 自有的 AICPU/AICore binary handles。
- **§6.4 当前态所有权表**（较新的完整设计）：AICPU／AICore binary/function handle 是 **process-pinned**，「**进程结束；shutdown 不调用 BinaryUnLoad**」。
- **代码**（决定性）：`load_aicpu_op.cpp` 的 `FinalizeL1Pinned()` 注释写明「graph-visible function handle remain process-pinned; **no BinaryUnLoad API**」；硬不变量 #11 亦然。

→ **以 §6.4 ＋ 代码为准：不卸载。** D.6 第 8 步属被取代的早期设想，**照它实现会把进程 pin 的前提破坏掉**。
→ 表一（§2）无需修改，它本来就写的是 process pin；本节的作用是**显式标记 D.6 第 8 步不可采用**，避免后来者照单执行。
→ 但 D.6 的第 2 步值得单独强调，因为我表一没有：**清理 Python callable handles 时要保留强引用到 native close 返回之后** —— 否则 native 还在用、Python 侧已经把对象放掉了。这与 §9.7① 的"上层只在成功时丢所有权"是同一件事的另一面。

### 13.4 吸纳状态

**已自读**：设计文档 §1–§15；附录 **B.9 / C / D（全部）/ E / F / G.0 / I / J / K / N**（N.1–3、5、6、9、11、13）；过程记录 10.48–10.62；实现代码关键路径。
**仍未自读**：附录 **A**（完整决策记录）、**H**（接口 before/after 对照）。
→ 三轮自读（E、I、D.5–D.6）各改了我一条结论，命中率 3/3。A 与 H 主题上重复度较高（A 是决策记录、H 是接口对照，都应已被正文与我已读的附录覆盖），但按前三轮的经验**不预设它们没有东西**。下一轮读 H（接口对照，与 §9 落点索引最相关），再下一轮读 A。

---

## 14 · 附录 H 自读结果：一处确认、一处接口细节、以及**定案 2 的第三种可能**

### 14.1 H.2 直接确认定案 3，并给出方法名

参考的 per-run vs persistent 对照，与我 §9.6 定案 3 的结论一致，而且把接口形状写出来了：

| # | 当前 L2/L3（per-run） | kernel 模式（persistent） |
| - | --------------------- | ------------------------- |
| prepare | `init_runtime_args(host_runtime, allocator)` ＋ `init_device_kernel_args(allocator)`（各自 alloc ＋ H2D） | **`l1_args.prepare_once(stable_runtime, allocator)`** |
| 每次 launch | `launch_aicore_kernel(run_stream, kernel_args.device_k_args_)` | `launch_aicore_kernel(hidden_stream, **l1_args.device_k_args()**)` |
| 收尾 | 每 run `finalize_device_kernel_args()` ＋ `finalize_runtime_args()` | **`l1_args.finalize_once()`**，在**所有 graph 之后**的显式 context close |

→ 这就是定案 3 说的"独立 owner、复用分配逻辑、不复用生命周期"，三个方法名可直接采用：**`prepare_once` / `device_k_args()` / `finalize_once`**。

### 14.2 H.1 的一处接口细节：kernel launch **不带 `RuntimeHandle`**

```cpp
// 当前（同步）
int simpler_run(ctx, RuntimeHandle runtime, callable_id, args, config);
// 新增，不替换（异步）
int simpler_l1_launch(ctx, callable_id, const ChipStorageTaskArgs *args, void *caller_stream);
```

→ 两点：**"新增，不替换"**再次印证阶段 H 的"新开专用路径"；而且 kernel launch **签名里没有 `RuntimeHandle`**，也没有 wait/poll/run-finalize —— 比我 §5.4 五入口的描述更具体。

### 14.3 H.6 给出定案 2 的第三种可能 —— 而且是参考实际采用的那种

我 §9.6 定案 2（经 §9.4② 修正后）的结论是：**保留 bootstrap/control stream**，因为主线 `ensure_device_initialized` 建的 `stream_aicpu_` 被 AICPU init（`:576`）、`aclrtSynchronizeStreamWithTimeout`（`:583`）、callable 注册（`:857`）共用；并据此把非目标 #8 收窄为"非-launch control 路径允许同步"。

**H.6 走的是另一条路**：

> `pypto_init` 借用 current device，创建**一个 hidden AICore stream** 及 prepare/start/done/tail events，**但它没有 stream 入参**，**不在 init 阶段把 register task 排进某个 caller stream**。`ctx.prepare()`（或普通 eager 的 auto-prepare）**才取当前 stream**，prepare persistent resources 并**异步 enqueue init/register**；**warmup 后由用户同步**；close 只释放自有资源。**init 不创建 private AICPU run stream**，也不预启动 orchestrator/executor 等待未来调用。

→ 也就是说：**根本不要 private AICPU stream**（连 bootstrap 用途也不要）。init/register 改成 **prepare 期在 caller 的 current stream 上异步 enqueue**，错误由 **warmup ＋ 用户同步**暴露（这正好接上 §11.1 记的 E.1 那句"`READY_ENQUEUED` 不代表 device 侧已执行完，capture 前的显式 warmup ＋ caller synchronize 才是暴露异步准备错误的用户流程"）。

**两条路的对比**：

| # | 我的定案 2（保留 bootstrap stream） | H.6（不要 private AICPU stream） |
| - | ----------------------------------- | -------------------------------- |
| 改动量 | 小：`ensure_device_initialized` 按 mode 只 gate execution 侧 | 大：init/register 要**从 init 期移到 prepare 期**并改成异步 |
| 内部 synchronize | 保留（`:583`），靠"限定在非-launch control 路径"来自洽 | **完全没有** PyPTO 内部 synchronize |
| 与单算子边界的一致性 | 需要额外解释"init 期同步不算越界" | 天然一致：所有 device 工作都在 caller stream 上、由调用方同步 |
| 证据 | 无（我推的） | **参考实际采用并上板** |

→ **建议采用 H.6 的形状**，理由是它消掉了唯一一处需要"特例解释"的内部 synchronize，而且有上板证据。**代价要说清**：这不是"按 mode gate 一下 `ensure_device_initialized`"，而是要把 AICPU init／register 的 enqueue 从 init 期迁到 prepare 期并改异步 —— **比定案 2 原本估的工作量大**，且要重新确认 a5 复用 bootstrap 那对 stream 的执行路径（`a5/.../device_runner.cpp:505`）怎么办。
→ 相应地，非目标 #8 **不需要**再收窄成"非-launch control 路径允许同步"——若采用 H.6，kernel 模式**任何路径都没有** PyPTO 内部 synchronize，那条收窄可以撤回。

**这是本轮唯一的未决项**，需要在 Phase 0 之后、动手改 stream 结构之前定：**是按定案 2 保留 bootstrap stream（改动小、需特例解释），还是按 H.6 彻底不要 private AICPU stream（改动大、语义干净、有上板证据）。**

### 14.4 吸纳状态

**已自读**：设计文档 §1–§15；附录 **B.9 / C / D 全部 / E / F / G.0 / H / I / J / K / N**（N.1–3、5、6、9、11、13）；过程记录 10.48–10.62；实现代码关键路径。
**仍未自读**：仅剩附录 **A**（完整决策记录）。
→ 四轮自读（E、I、D.5–D.6、H）**四次都改了或补了结论，命中率 4/4**。下一轮读 A，之后参考侧就算吃干净了。

---

## 15 · 附录 A 自读结果：§14.3 的未决项被裁定，并补上一条我完全没有的 v1 约束

附录 A 是**决策记录**（"哪些事情已经决定、哪些没有决定"），因此它对"我该不该这么定"最有裁定力。

### 15.1 §14.3 的未决项：裁定为 H.6 形状

A.4 第 11 条：

> **L1 launch、prepare 和 close 都不允许 PyPTO 主动做 stream/device synchronize。需要同步的 warmup、测试和 teardown quiescence 由调用方明确完成。**

→ 这是**已确认的决策**，而且**明确覆盖 prepare**。我 §9.6 定案 2 的"保留 bootstrap stream，其 init 期 synchronize 不违反非目标 #8"与它相反。
→ **裁定：采用 §14.3 的 H.6 形状** —— 不保留 private AICPU stream，init/register 迁到 prepare 期在 caller stream 上异步 enqueue，错误由 warmup ＋ 调用方同步暴露。**主线 `device_runner_base.cpp:583` 的 `aclrtSynchronizeStreamWithTimeout` 在 kernel 模式下必须去掉，而不是"限定在非-launch 路径"来解释。**
→ **非目标 #8 那条收窄正式撤回**：kernel 模式的 launch／prepare／close **任何路径都不做** PyPTO 内部 synchronize。
→ 代价照 §14.3 所述：这是重构而非 gate，且要重新确认 a5 复用 bootstrap stream 的执行路径。

### 15.2 我完全没有的一条 v1 约束：**DFX 必须全关**

A.8 第 6 条：

> **v1 关闭所有会增加额外 workspace、collector、回读或同步的 DFX：args dump、PMU、dep-gen、scope stats、L2 swimlane 等。**

→ 我的文档从头到尾**没有这一条**。而它与"launch 期零分配 ＋ 无同步 ＋ 地址稳定"是同一个要求的另一面：任何 DFX 通道只要多申请 workspace、多挂 collector、多做回读或同步，就会破坏 capture 前提。
→ **而且这条与主线近期改动直接相撞**：`dep-gen` 正是 #2057（本轮基线内）刚加强的，`HBG swimlane` 是 #2031。→ **kernel 模式必须能把它们成组关掉，并有测试断言"kernel 模式下这些通道确实未启用"**。已列入阶段 3／6 的实现要求。

### 15.3 第三次确认 §12.1（unregister）

A.6 第 7 条：**"unregister 不能立即释放 graph 可能引用的 binary/device descriptor；v1 可只撤销 host handle，device 对象继续 pin 到 context close"** —— 与 §12.1（UT-027）、§13.1（D.5 的悬空地址危害）一致。这条现在有三处独立出处，可以当定论。

### 15.4 其余值得补进文档的条目

| 出处 | 内容 | 归属 |
| ---- | ---- | ---- |
| A.5.9 | **用户层不应看见 workspace、AICPU stream、AICore stream、event、runtime arena** | 比内存专题 D5 的"stream/event 保持内部"更完整的不透明清单 |
| A.5.6 | **跨进程冲突不在本次解决** | 补入非目标（我原来只有"单 device 单 live context"） |
| A.6.4 | executor binary、child binary、orchestration SO、callable device descriptor **至少存活到所有引用它们的 graph 被销毁** | 表一的生命周期下限，一句话说清 |
| A.6.5 | **child binary 允许在 context 内持续累积**，`close()` 才统一释放 | 与 D.5"不做 LRU/refcount/recycle"配套 |
| A.7.5 | AICore 的关键**不是每次分配一份 device `KernelArgs`，而是消除其中的 per-invocation 内容** | 定案 3／C29 的第三次确认，措辞最准 |
| A.7.7 | child task payload 共享 arena，但**下一次 reset/reuse 必须由 stream/event 顺序保证发生在前一次 AICPU/AICore 都完成之后** | 我只写了"serial-tail 之后"，这条把依据说清了 |
| A.8.1 / A.8.2 | 正常复用只需 host 侧**异步**失效 handshake；**不得为保险每次清零整个 Runtime/arena/workspace**（增加 capture 节点、破坏地址稳定与性能） | 与 F.9／J.5／§10.2 一致，第三处出处 |
| A.9.3 | **关键 teardown 不依赖 `__del__`** | 与 §9.7①「GC/atexit 不得 close」一致 |

**一处口径差异记录**：A.2.4 说 simulator "只需要**稳定返回 unsupported**"，而 codex 第八轮要求 stub **先跑完共同的结构校验与状态机**再按能力分流。两者结论相同（都拒绝），但严格程度不同。**采用 codex 的更严版本**——理由是它让 stub 与真实实现共享同一套校验，避免"sim 上能过、onboard 上才发现参数非法"。A.2.4 是下限，不是上限。

**一处已知被后续取代的记录**：A.2.2 写"首期只实现 `tensormap_and_ringbuffer`，`host_build_graph` 不在首期范围"。这是**计划期**的范围，最终实现两个 runtime 都做了（§C24 已核实：两者 `l1_runtime_supported_impl()` 均返 1、ST 对两者参数化）。→ 读 A.2.2 时不要当成最终状态。

### 15.5 参考侧吸纳完成

**全部自读完毕**：设计文档 §1–§15；实现计划附录 **A / B.9 / C / D（全部）/ E / F / G.0 / H / I / J / K / N**；过程记录 10.48–10.62；以及实现代码的关键头文件与 launch／restore／registry／teardown 路径。

**五轮自读的命中率 5/5** —— E（CLOSING 重试）、I（unregister 语义 ＋ 换 stream 三分支）、D.5–D.6（binary 三层 ＋ 与 §6.4 的冲突）、H（定案 2 的第三种可能）、A（裁定该 fork ＋ DFX 全关）。当初 codex 扫过 A/E/H/I 后报"没有推翻现有结论"，事实是**每一份都有**。这条经验值得记住：**转述可以用来排序优先级，不能用来替代阅读。**
