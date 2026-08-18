# PyPTO L1 接入 ACLGraph 详细实现计划

<!-- markdownlint-disable MD036 MD060 -->

> 状态：第一阶段TRB L1已在A2/A3 device 1完成正式无探针上板：`context.prepare()`、eager warmup、独立caller stream ACLGraph capture、图内PyTorch→L1→PyTorch顺序以及三次连续replay均通过，L2 control亦通过；对应runtime收口提交为`3631ea0d`。A5目前没有同等真实硬件证据，不能泛化为已上板。HBG第二阶段已经实现“每个task/captured node一份tiling-like graph package”：DeviceRunner按本次callable与参数构建immutable `HbgGraphPlan`，生成fresh writable HostArgs blob和placeholder，经独立HBG WithHostArgs run entry交给CANN；AICPU run entry取得prepare-time slot/callable trust roots，exactly-one leader在每次执行/replay前完整恢复共享working SM/runtime arena，peers只在统一restore verdict后进入classify/dispatch。函数地址表是callable-local快照，`callable_id`是context-global身份，两个program可以都从`func_id=0`开始。
>
> runtime `8427ffd7`完成HBG no-reset源码协议：generation内所有有效AICPU participant采用arrive/finalize/snapshot/depart两阶段完成门，generation前错误通过独立64-byte control line释放hidden AICore，已report但physical id/regs mapping无效的core走per-core CANCEL；prepare-time resident control地址覆盖slot registry不可用，Host直传的Runtime override避免AICPU/AICore因坏`KernelArgs::runtime_args`读取不同control，affinity越界在进入barrier前失败。runtime `80615b1e`与PyPTO提交`b8b3dd35`闭合versioned `pypto_orchestration_requirements_v1`生产/消费链：真正生成Host `get_tensor_data/set_tensor_data`的orchestration在borrowed HBG L1构图前fail-closed，只生成device predicate metadata的tensor read保持允许。在这些门禁后，当前工作树已令A2/A3与A5的`l1_runtime_supported_impl()`返回1，并通过`RunConfig(runtime="host_build_graph")`显式选择HBG；默认仍是TRB。
>
> 首轮device 1验证已经证明：同一Host进程先执行TRB后，第一个HBG context的eager、独立stream capture和连续replay可以通过；它同时暴露并修复了两个不能从Host UT推断的真实问题。第一，TRB/HBG AICPU inner SO曾在CANN全局ELF namespace中暴露225个重名C++符号，HBG现以version script只导出5个CANN entry。第二，CANN在ACL binary unload后仍可能保留HBG inner DSO及static registry；当前工作树以context generation在新context的有序init task中reset execution-slot与callable registry，同generation配置re-latch保持幂等。generation在支持范围内只严格保证同一Host进程的顺序context唯一；`CLOCK_MONOTONIC`对跨Host进程顺序复用只是best-effort降重，不是跨进程lease或正式支持面。最新C++无硬件回归为98/98，runtime Python UT串行为1103 passed、8 skipped，顶层compile/JIT/L1相关集合为420 passed，A2/A3与A5的TRB/HBG host、AICPU、AICore共12套onboard产物均构建通过。最后的“同一进程顺序创建第二个HBG context”和双HBG callable capture/replay仍须在device 1恢复可验证状态后复验；完成前不能把HBG第二阶段写成最终上板闭环。本文是指导性设计记录，完整上下文优先，不以篇幅压缩为目标。
>
> 后续完成度审计又把可直接上板的矩阵扩展到runtime scalar与tensor地址异步快照、多输出、多child与内部workspace、两个HBG graph交替replay，以及两个不同HBG context顺序eager/capture/replay。A2/A3、A5在TRB/HBG组合下的lowering和PTOAS完整Host codegen均通过，但这些新增case尚未在device 1执行；large HostArgs边界、runtime args allocator压力、cache多线可见性、memory accounting和N.10.8故障注入仍必须使用专用probe，不能由普通数值ST替代。
>
> runtime提交`3575f60b`已增加独立的`tests/st/l1/host_args_probe`：它不调用PyPTO serializer/parser，而由最小AICPU kernel从真实task-args基址核对三个placeholder、完整payload checksum和首/中/尾字节；Host侧可扫描64 KiB～64 MiB、在launch返回后立即poison/free/reuse scratch、捕获两个不同graph并交替replay、制造其他WithHostArgs任务压力并记录HBM。探针强制显式`--device`且不调用device reset；AICPU/Host交叉编译和一字节错位args基址的无设备自检已通过。由于当前device 1仍处于无可见进程但AICore 100%的残留态，探针尚未执行，N.10的device checkbox保持未勾选。
>
> 首期范围：onboard、`tensormap_and_ringbuffer`（TRB）、`@pl.program`、静态 shape、PyTorch 直接调用验证。
>
> 本文中的接口名是实现建议；编码时可以做小幅命名调整，但不得改变本文确定的所有权、生命周期和流语义。

阅读方式：

- 第 1～15 节是实现主线和验收摘要；
- 附录 A 完整保存讨论结论；
- 附录 B 审计当前 L2/L3 源码调用链，并对照历史 `pto2/pypto` 的 L1/ACLGraph 实现；
- 附录 C～F 展开 task package、workspace、stream、生命周期和方案取舍；
- 附录 G～H 给出逐文件实施步骤和 before/after；
- 附录 I～M 给出完整测试、事实门槛、提交顺序、评审清单和交付物；附录 N 给出第二阶段HBG tiling-like graph payload的准入设计。

摘要和附录发生表述差异时，以附录中更严格、更完整的约束为准；不得以摘要较短为理由省略附录要求。

## 1. 目标

把 PyPTO 从“掌控设备资源并同步完成整次执行”的 L2/L3 执行器，扩展为可被 ACLGraph capture/replay 的 L1 单算子：

- 对外形态与普通 AscendC 自定义算子一致：输入、输出、scalar 和 caller stream 进入一次异步 launch。
- PyPTO 不感知 ACLGraph 的 capture/replay 状态，也不保存 graph handle。
- launch 路径不分配或释放 device memory，不创建或销毁 stream/event，不做 stream/device synchronize。
- AICPU 使用 caller stream；AICore 使用 PyPTO 内部隐藏 stream，两者通过 event 建立依赖。
- 一次 L1 launch 必须是严格的“单算子闭包”：内部 AICore 只能在 caller stream 到达本算子后启动，且 caller stream 离开本算子前必须重新 join 全部内部 task。
- L1 不查询 capture model，不调用 `rtStreamAddToModel`，不为了让 AICPU orchestrator 抢在 caller stream 上此前任务之前运行而引入 early-launch 路径。
- 输入、输出均由调用方提供 device 地址；首期 workspace 仍由 PyPTO 在 prepare 阶段内部申请并持有。
- 首期禁止并发执行同一 L1 context。PyPTO 当前占用全部 AICore，因此并发没有收益，共享 workspace 也不会发生合法调用间踩踏。
- 保持现有 L2 单卡和 L3 单机多卡路径的接口、资源所有权和行为不变。

最终验收形态是：用户先初始化和 warmup，再用 PyTorch 直接调用 PyPTO op 完成 ACLGraph capture/replay。`inductor_pto` 接入不属于本计划。

## 2. 已确认的边界

### 2.1 首期包含

- A2/A3 与 A5 共用一套 L1 host 设计；架构差异只留在已有 arch-specific `KernelArgs` 和构建产物中。
- 仅支持 onboard TRB；simulator 和 `host_build_graph` 明确返回 unsupported。
- 仅支持 `@pl.program` 编译产物。
- 输入输出个数、dtype、shape和参数布局由compiled callable固定；stride metadata在第一次成功enqueue后绑定。
- scalar 值及 tensor device 地址作为每次调用参数；capture 后它们是否更新由 ACLGraph 的参数语义负责，PyPTO 不判断。
- 关闭 args dump、PMU、dep-gen、L2 swimlane、scope stats 等可能引入额外资源或回读的 DFX 功能。
- Python convenience wrapper 只做 forward，不做 autograd。
- Eager 可以自动完成普通注册；ACLGraph capture 前必须显式 `prepare/warmup` 并由用户在外部完成同步。

### 2.2 首期不包含

- 动态输出 shape、运行时重新 tiling、capture 后改变参数布局。
- 外部 workspace 入参和 workspace size query 公共 API。
- 同一 context 的并发 graph replay、多个 caller stream 上的重叠执行。
- child kernel binary 的回收、复用或 `aclrtRegisterBin` 迁移。
- L1 的 simulator 语义仿真。
- `inductor_pto`、backward/autograd、多进程或多卡 L1 编排。
- PyPTO 内部主动等待异步错误、主动 stream sync、依赖某个固定的 kernel-launch 数量上限。
- 单算子之外的 orchestration 提前展开、跨算子调度或用隐藏 AICPU stream 越过 caller-stream 边界的性能优化。这类能力属于后续 `host_build_graph` 方案，不属于 L1 单算子。

“首期静态 shape”只约束一次 prepared operator 的外部调用契约。即使 PyPTO 内部由支持动态 shape 的程序编译得到，只要本次 capture 的 task、参数布局和 tensor metadata 固定，L1 runtime 不需要感知“动态 shape”这个概念。

## 3. L1 与现有 L2/L3 的根本差异

| 项目 | L2/L3 当前行为 | L1 目标行为 |
| --- | --- | --- |
| device 生命周期 | PyPTO attach、初始化并在 finalize 中 reset | 借用 torch_npu/调用方已建立的 device context，绝不 reset |
| tensor 内存 | runtime maker staging、H2D/D2H、内部 output 分配 | 直接使用调用方传入的 device 地址 |
| workspace | 每次 run 可能准备、扩容或回收 | prepare 时一次分配，context 生命周期内固定 |
| AICPU stream | PyPTO 创建和持有 | 使用 caller stream |
| AICore stream | 当前 run stream 或内部 stream | 每个 device context 一个隐藏持久 stream |
| 完成语义 | `run/wait/finalize` 可同步等待 | launch 仅 enqueue，caller stream 表示依赖完成 |
| task args | host Runtime 构建后同步复制到 device | AICPU 参数由 CANN 做每次 launch 快照；AICore只引用持久状态 |
| 并发 | L3 pipeline slot 可并发准备/执行 | v1 明确单执行序列 |
| capture | 不作为核心约束 | launch 路径的每个 API 都必须可 capture/replay |
| 调度边界 | PyPTO 掌控 whole-run，可内部安排启动/收尾 | caller stream 上严格 fork/join 的单算子，不得让 device task 越过入口或出口 |
| 图接入 | 无普通 op 形态 | 只依赖 caller/hidden stream 的 event 依赖被 ACLGraph 捕获，不查询或修改 capture model |

L1 不能复用“在现有 L2 run 上删掉一个 synchronize”的做法。内存 staging、stream ownership、device reset、per-run `KernelArgs`、注册预热和错误清理都必须有独立的 L1 分支。

## 4. 当前代码中的关键阻碍

### 4.1 当前 C ABI 是拥有型执行模型

当前入口把 caller-owned `RuntimeHandle` 贯穿 prepare、launch、poll、wait、finalize：

```cpp
int simpler_run(DeviceContextHandle ctx, RuntimeHandle runtime, int32_t callable_id,
                const void *args, const CallConfig *config);
int simpler_prepare_run(...);
int simpler_launch_run(...);
int simpler_wait_run(...);
int simpler_finalize_run(...);
```

这些接口允许 host 在结束时验证、D2H 和释放资源，不适合一个异步返回的普通算子。

### 4.2 当前 TRB binder 会替换 tensor 地址

`runtime/src/a2a3/runtime/tensormap_and_ringbuffer/host/runtime_maker.cpp` 的 `stage_device_args()` 会：

1. 为 tensor 分配或切分内部 device buffer；
2. 把输入从 host copy 到 device；
3. 把 `ChipStorageTaskArgs` 内地址替换为 staging 地址；
4. 在 validate/finalize 时 copy back 并释放。

L1 必须增加 direct-device binder，原地址只做校验和封装，不发生 staging/copy/free。

### 4.3 当前 AICore 参数是 per-run device allocation

`KernelArgsHelper` 当前每次构造 device `Runtime` 和 device `KernelArgs`：

```cpp
allocator.alloc(runtime_device_copy_size(runtime));
rtMemcpy(runtime_dev, ..., &runtime, ..., RT_MEMCPY_HOST_TO_DEVICE);
allocator.alloc(sizeof(KernelArgs));
rtMemcpy(device_k_args, ..., &args, ..., RT_MEMCPY_HOST_TO_DEVICE);
```

随后 AICore launch 只传这个 device 指针：

```cpp
struct Args { KernelArgs *k_args; };
rtKernelLaunchWithHandleV2(..., &rt_args, ..., stream, ...);
```

L1 不能在 launch 中重复 alloc/copy/free，但这个“一层指针”的 AICore ABI 本身可以保留。

### 4.4 当前初始化和销毁拥有设备

`ensure_device_initialized()` 创建 AICPU/AICore stream，并在初始化、注册等路径调用 `aclrtSynchronizeStreamWithTimeout`。架构侧 finalize 还会进入 `rtDeviceReset`。L1 必须使用独立 mode，避免走这些 L2/L3 资源所有权分支。

### 4.5 kernel binary 已有两种所有权

- executor AICore ELF 由 `rtRegisterAllKernel` 注册，handle 在 `DeviceRunnerBase` 中缓存；当前没有对应的公开 unregister。
- child/incore kernel binary 由 PyPTO 上传到 ChipCallable 相关 GM，没有走 `aclrtRegisterBin`。历史L2/L3使用runtime `func_id_to_addr_`；L1每个callable保存独立映射快照并在本次invocation重放。

首期二者都按 context 生命周期 pin 住。child binary 允许持续累积，不在本次引入回收算法。

## 5. 核心设计：把状态分成“持久状态”和“调用快照”

### 5.1 context 生命周期持久状态

新增 `L1ExecutionState`，由 onboard `DeviceRunnerBase` 持有：

```cpp
struct L1ExecutionState {
    L1Phase phase;                 // new/initializing/collecting/ready/sealed/poisoned/closing/closed
    int32_t device_id;
    rtStream_t aicore_stream;      // hidden, persistent
    rtEvent_t start_event;
    rtEvent_t aicore_done_event;
    rtEvent_t serial_tail_event;
    Runtime *runtime_dev;          // persistent device runtime descriptor
    KernelArgs kernel_args_host;   // immutable after prepare
    KernelArgs *kernel_args_dev;   // persistent AICore-visible copy
    L1Workspace workspace;         // persistent shared arenas/buffers
    L1CallableTable callables;     // append-only while context is live
    std::mutex enqueue_mutex;      // serializes host-side enqueue sequences
};
```

以下内容必须在 capture 前完成并保持到所有 graph 销毁之后：

- executor AICPU/AICore binary handle、child kernel binary GM 地址；
- orchestration SO、callable descriptor、callable ID 到 binary/entry/metadata 的映射；
- hidden stream/events，以及 `Runtime`、`KernelArgs`、regs/FFTS、handshake、TRB arena、GM SM、workspace。

`KernelArgs` 在 prepare 结束后不得含 per-invocation 字段。AICore 每次 launch 只收到同一个 `kernel_args_dev`，从而不需要 PyPTO 自己管理 device task-args 内存池。

### 5.2 每次调用的 AICPU 参数快照

建议新增固定布局、带 ABI version 的 host args：

```cpp
struct L1AicpuInvocationArgs {
    uint32_t abi_version;
    uint32_t total_size;
    KernelArgs kernel_args;             // fields point to persistent device state
    const L1CallableDeviceDesc *callable_desc;
    int32_t callable_id;
    uint32_t invocation_flags;
    ChipStorageTaskArgs orch_args;       // caller tensor addresses + scalars
};
```

它作为普通 host 参数传给：

```cpp
aclrtLaunchKernelWithHostArgs(func_handle, aicpu_thread_num, caller_stream, nullptr,
                              &invocation, sizeof(invocation), nullptr, 0);
```

这里利用 CANN runtime 已有的 task-args 管理能力：API 在 enqueue 时接收完整 host args，runtime 内部在 task 真正完成后才回收对应参数存储；实现依据可核对 `stars_arg_manager.cc` 的参数池和 `stream_david.cc` 的 task-complete recycle。PyPTO 的栈上 `invocation` 在 API 返回后无需继续存活，也无需复制到自建的 device task pool。

这正好解决连续异步 host launch 的参数覆盖问题：每次 AICPU 调用有独立的 runtime-owned 快照；共享 workspace 仍只有一份，并由 stream/event 顺序保护。

### 5.3 AICore 不携带每次调用的 args

AICore 仍走现有内部接口：

```cpp
struct AicoreLaunchArgs { KernelArgs *k_args; };
rtKernelLaunchWithHandleV2(aicore_bin_handle, 0, block_dim, &rt_args, nullptr, hidden_stream, &cfg);
```

区别仅是 `k_args` 指向 context-lifetime 的固定 device object，而不是每次 run 新分配的 `device_k_args_`。

`orch_args`、`callable_id` 等动态内容由 AICPU entry 直接从 `L1AicpuInvocationArgs` 读取。不要再把它们先写进共享 `DeviceRuntimeLaunchDesc`，否则会重新引入调用间覆盖和额外 H2D 生命周期问题。AICPU 生成的 AICore child task payload 继续写入 TRB arena；下一次调用必须在前一次 AICPU/AICore 都完成后才重置和复用 arena。

### 5.4 DeviceRuntimeLaunchDesc 的 L1 约束

现有 `DeviceRuntimeLaunchDesc` 混合了静态字段和 `orch_args_storage_`、`active_callable_id_` 等 per-run 字段。首期不破坏 L2 ABI，采用以下分支：

- L2/L3 保持现有字段和 copy 流程。
- L1 prepare 一次性构造只含稳定地址、core 数、func 映射和 arena 地址的 device descriptor。
- L1 AICPU executor 从 invocation args 获取 callable 和 orch args。
- 用 static assertion 保证 L1 依赖的共享字段 offset/size 在 A2/A3、A5 对应构建中一致。

如果直接复用 `Runtime` 使静态/动态边界过于含糊，则新增 `L1DeviceContext`，让 `KernelArgs.runtime_args` 在 L1 mode 下指向它；不要为了复用一个 struct 而在 launch 中更新整块 Runtime。

## 6. stream 与事件协议

本节的最高优先级不变量是“单算子闭包”，而不是尽可能早地启动 orchestrator。对 caller stream 上一次可见的 L1 调用，必须同时满足：

1. **入口边界：** hidden AICore stream 上的本次 task 不得早于 caller stream 上本算子之前的任何 task；
2. **内部并行：** caller stream 上的 AICPU orchestrator 和 hidden stream 上的 AICore executor 可以在本算子边界内并行，由现有 handshake/ring 协议协作；
3. **出口边界：** caller stream 上的 downstream task 不得早于 AICPU task 和 hidden AICore task 中任何一方完成；
4. **图透明：** 上述顺序在 eager 和 capture 中完全一致，不存在 capture-only early launch mode。

`start_event` 和 `aicore_done_event` 分别是这个闭包的 fork 和 join，不是可选性能优化。

### 6.1 单次 launch 的设备顺序

```text
caller stream                                hidden AICore stream
-------------                                --------------------
... predecessor tasks
| L1 operator entry
| [host only on stream switch: query previous serial_tail; never enqueue a wait]
| aclrtMemsetAsync(handshake only)
| record(start) ---------------------------> wait(start)
| launch AICPU with host args                 launch persistent AICore executor
| wait(aicore_done) <----------------------- record(aicore_done)
| record(serial_tail)
| L1 operator exit
... downstream PyTorch ops
```

Host enqueue 顺序必须固定为：

1. 用 context mutex 防止两个 host 线程交叉下发同一组 event 操作；
2. 若caller raw stream与上一调用不同，host用 `aclrtQueryEventStatus` 非阻塞确认上一 `serial_tail` 已完成；未完成直接报错且不enqueue，完成后也不把旧tail wait加入新stream；同一raw stream依赖FIFO；
3. `aclrtMemsetAsync` 只清理 handshake/invalidation 区域；
4. caller stream record `start_event`；
5. caller stream enqueue AICPU `aclrtLaunchKernelWithHostArgs`；
6. hidden stream wait `start_event`；
7. hidden stream enqueue AICore `rtKernelLaunchWithHandleV2`；
8. hidden stream record `aicore_done_event`；
9. caller stream wait `aicore_done_event`；
10. caller stream record `serial_tail_event` 并立即从 host API 返回。

caller stream 自身保证 AICPU 完成后才越过步骤 9；`aicore_done_event` 保证 AICore 也完成。因此调用后的 PyTorch op 同时依赖两侧完成。

`start_event` 不能省略。PyPTO AICore executor 会占用全部 AICore；如果 hidden AICore 可在 caller stream 的 predecessor AICore task 之前启动，它可能先占满核并等待尚未轮到的 caller-stream AICPU task，而 predecessor 又因无核可用而无法完成，形成资源饥饿/死锁环。因此 fork 必须位于 predecessor 之后。

`aicore_done_event` 同样不能用“AICPU orchestrator 已返回”替代。AICore executor 可能仍在执行最后的 child task、shutdown 或 cache/visibility 收尾；caller stream 必须等待 hidden stream 的真正完成点后才能对外暴露算子完成依赖。

`serial_tail_event` 是host在**普通eager换流前**使用的完成证明，不再是跨stream自动插入的device依赖。原因是CANN会拒绝capture stream等待一个在capture外record且仍带record状态的event；即使用户已同步，event synchronize/stream synchronize也不会清掉这个capture-isolation状态。标准 `warmup -> external synchronize -> capture stream` 流程只能在host侧确认旧eager tail已经完成，然后不把旧event依赖带入capture。

这个query不是capture状态查询，也不感知graph/model；它只查询public event对应的先前eager task是否完成。capture会创建自己的event generation/clone，host以后查询原event不能证明graph replay已经完成。因此ACLGraph replay的并发仍不作为v1支持契约；graph→eager、两个graph交替或并发replay，以及capture后切换stream，都必须由调用方先保证外部quiescence。同一context的多个graph不得重叠replay。

### 6.2 为什么只 reset handshake

TRB 的 scheduler/arena 状态已在 AICPU executor 内通过 `init_per_ring()`、`runtime_reset_for_reuse()` 等逻辑重置。host 侧只需要消除 AICore 启动握手的旧值，避免下一次 AICPU 把旧 `aicore_done` 当成本次上报。

因此不得在每次 launch 清零整个 Runtime、arena 或 workspace。若 onboard 测试证明 handshake 中还有未覆盖字段，扩大的是“明确列出的 invalidation region”，不是恢复全量 reset。

### 6.3 capture 可行性是 Phase 0 硬门槛

必须在正式改造前用最小 onboard probe 验证以下组合可被 ACLGraph capture/replay：

- caller stream 上的 `aclrtMemsetAsync`；
- external/hidden stream 的 event record/wait；
- AICPU `aclrtLaunchKernelWithHostArgs`；
- hidden stream 的 `rtKernelLaunchWithHandleV2`；
- event 在连续 replay 中复用且不会错配 generation。

这个 probe 必须从 caller capture stream 出发，仅通过 `record(start) -> wait(start)` 和 `record(done) -> wait(done)` 的依赖关系让 ACLGraph 捕获 hidden AICore 分支。L1 路径和 probe 都不允许：

- 查询 caller stream 是否正在 capture；
- 获取或保存 capture model/graph handle；
- 调用 `rtStreamAddToModel` 或等价内部 API 主动把 hidden stream 挂到 model；
- 因 capture 而略过入口依赖，或在 caller stream 到达算子前提前发射 AICPU orchestrator。

任一原语失败都先停在 Phase 0，记录失败 API 和 runtime error。即使只有“hidden stream 无法通过 event 依赖被自然捕获”这一项失败，也不能以 `rtStreamAddToModel`、stream sync、capture-aware 分支或 early launch 绕过。这意味着当前 runtime 事实不满足本 L1 架构，应先停止主线并重新评估可捕获的单算子封装。

## 7. 建议的 native API

为避免改动现有 L2/L3 ABI，新增 L1 专用入口；共同实现可以下沉到内部 helper：

```cpp
int simpler_l1_supported(DeviceContextHandle ctx);

int simpler_l1_init(
    DeviceContextHandle ctx,
    int device_id,
    const uint8_t *aicpu_binary,
    size_t aicpu_size,
    const uint8_t *aicore_binary,
    size_t aicore_size,
    const uint8_t *dispatcher_binary,
    size_t dispatcher_size,
    const CallConfig *config);

int simpler_l1_prepare_callable(
    DeviceContextHandle ctx,
    int32_t callable_id,
    const void *callable,
    size_t callable_size,
    void *caller_stream);

int simpler_l1_launch(
    DeviceContextHandle ctx,
    int32_t callable_id,
    const ChipStorageTaskArgs *args,
    void *caller_stream);
```

约束：

- `simpler_l1_init` 读取当前 device id 并校验，不取得 device reset 所有权。
- `prepare_callable`可做allocation、binary upload、AICPU callable load和arena构建，但必须在capture外调用；异步准备使用caller stream，不在内部sync。`callable_size`是canonical `ChipCallable` blob的精确长度，native必须在任何hash/upload前校验header、name、binary和全部child offset/length恰好落在该范围内，拒绝truncation与trailing bytes。
- `launch` 只接收预注册 callable、固定布局 args 和 stream；不接收 `RuntimeHandle`，不提供 poll/wait/finalize-run。
- `finalize_device` 根据 mode 释放 PyPTO 自己持有的 L1 资源，但 L1 分支绝不 `rtDeviceReset/aclFinalize`。
- 所有 runtime variant 导出同名 symbol；sim/HBG 的 `simpler_l1_supported()` 返回 0，其余 L1 API 返回清晰 unsupported error，避免 `dlsym` 差异。
- `CallConfig` 在 prepare 后固定。launch 前发现 callable 未 prepare、DFX 开启、shape/layout 不匹配、资源需求增长时直接报错。

不新增 `reset()` API。若一组 event/kernel 已部分 enqueue 后 host API 失败，将 context 标记为 poisoned；调用方先按其 runtime 规则排空/销毁相关 graph，再 close 并重建 context。

## 8. Python 与 PyTorch API

### 8.1 低层 Python binding

nanobind 层只暴露原始能力，不依赖 torch_npu：

```python
worker.init_l1(device_id, runtime_binaries, call_config)
worker.l1_prepare_callable(callable_id, callable, raw_stream_address)
worker.l1_launch(callable_id, chip_storage_args, raw_stream_address)
prepare_call = worker.l1_make_prepare_queue_call(callable_id, callable)
launch_call = worker.l1_make_launch_queue_call(callable_id, chip_storage_args)
```

direct入口的raw stream是整数地址（`uintptr_t`语义），不是capsule；命名capsule只封装taskQueue deferred prepare/launch descriptor。native入口立即校验null和device。binding在native enqueue时释放GIL，但不等待设备完成。

### 8.2 独立 PyTorch convenience wrapper

新增尽量独立的 torch_npu adapter，职责仅包括：

- 从 `c10_npu::getCurrentNPUStream().stream(false)` 获取真实 current stream，并遵循 torch_npu taskQueue 调用方式；`false`禁止这个入口因取stream而隐式排空taskQueue；
- 校验所有 tensor 位于同一 NPU device，将Python tensor立即转换为C++ `at::Tensor` handle，并在callback入队前对普通torch_npu caching-allocator storage调用 `recordStream`；
- 保持deferred descriptor和C++ Tensor handle到taskQueue callback完成，再调用低层 L1 launch。adapter不捕获Python对象，也不代替用户持有context、graph-bound tensors及external/custom storage owner到graph真正销毁和device完成。

taskQueue 适配不得进入 simpler runtime core。core 永远只看一个外部 `aclrtStream`。

建议用户 API：

```python
ctx = pypto.l1.pypto_init(
    device=device_id,  # mandatory; must already equal torch_npu current device
    programs=[compiled_program],
    config=l1_config,
)
op = ctx.operator(compiled_program)

# Canonical graph-safe path: caller owns output storage.
op(x, out=y)
```

v1对纯Out参数始终要求预分配 `out=`，不实现 `y = op(x)` 的自动output allocation。若未来增加eager convenience，必须作为独立支持面；ACLGraph核心路径仍使用预分配 `out=`，把torch allocator capture行为与PyPTO runtime正确性分开验证。

### 8.3 prepare、warmup 和销毁

```python
ctx = pypto.l1.pypto_init(programs=[compiled], device=device_id)
op = ctx.operator(compiled)
op.prepare()          # outside capture; idempotent
op(x, out=y)          # eager warmup
torch_npu.npu.synchronize()  # explicitly owned by user/test

graph = torch_npu.npu.NPUGraph()
with torch_npu.npu.graph(graph, stream=capture_stream):
    op(x, out=y)
graph.replay()

torch_npu.npu.synchronize(device_id)
graph.reset()         # all possible replays die first
ctx.close()           # no implicit device/stream synchronization
```

禁止依赖 Python `__del__` 做关键 teardown。`close()` 在 context 仍可能被 graph 使用时无法可靠检测，因此文档契约要求用户先销毁 graph 并确保相关工作完成。

## 9. callable、kernel binary 和 workspace 生命周期

### 9.1 callable ID 和 binary pinning

- context 内 callable table 采用 append-only 语义。
- 已注册 ID 不允许指向另一份 callable；重复注册同一 identity 幂等，冲突直接报错。
- `callable_id` 不允许改指另一个identity；同一callable内的 `func_id -> child binary address` 快照不可变。不同独立program/callable可以都使用 `func_id=0` 并映射到不同binary。
- unregister 在 L1 v1 中只移除 host 可见 handle，不能释放 graph 可能引用的 device binary/descriptor。
- 所有 executor/child binary 与 device callable descriptor 在 `ctx.close()` 才统一回收。
- 不依赖 runtime 内部“约 2048 次 launch”等规格，不把任何观察值写成 PyPTO 上限。

这个策略会增加长生命周期 context 的 HBM 使用，但它把 graph 引用悬空风险降到最低，符合首期正确性优先目标。

### 9.2 workspace

- 根据 prepared programs 的最大需求在 init/prepare 阶段建立一份共享 workspace。
- 新 callable 若需要更大 workspace，可在 capture 外显式 prepare 时增长；已有地址如果会被 graph 引用，则不得搬迁，只能追加新 backing storage 或直接拒绝。
- launch 路径发现资源不足一律报错，不能隐式 grow。
- v1 不暴露 workspace 指针/size 给用户。
- 同raw stream由FIFO保证调用不重叠；换stream只在上一 `serial_tail` 已完成时放行；graph replay并发由业务契约禁止。因此v1可以复用同一份workspace。host mutex本身不是device完成证明，不能单独作为workspace复用依据。

## 10. AICPU/AICore executor 改造

### 10.1 新 AICPU entry

新增独立 symbol，例如 `simpler_aicpu_l1_run`：

1. 校验 `abi_version/total_size/callable_id`；
2. 从 invocation 获得 persistent Runtime/arena 指针和本次 `orch_args`；
3. 由 leader 完成本次 runtime reset 和 callable 发布；
4. 在 scheduler/orchestrator barrier 前发布完成；
5. 复用现有 TRB `AicpuExecutor` 的 init/run/deinit 主流程；
6. 不保存 invocation host pointer 到 task 生命周期之外的全局状态。

这个 entry 只能作为 caller stream 上本次 L1 operator 的一个普通 AICPU task 被 enqueue。它不允许在 prepare 时预先启动，不允许在 private AICPU stream 上长驻等待未来调用，也不允许根据 capture mode 抢在 caller stream predecessor 之前执行。AICPU/AICore 的并行只发生在第 6 节定义的 entry/exit event 之间。

L2/L3 的 `simpler_aicpu_exec` 保持原参数 ABI。共享逻辑通过内部函数复用，不能让 L1 条件分支散布在 scheduler hot path。

### 10.2 AICore entry

首选不修改现有 kernel entry ABI。它继续读取 persistent `KernelArgs` 和 Runtime 中稳定地址，等待 handshake/task ring。只有当 Phase 0/原型证明 AICore 在 AICPU 发布前读取了必须动态变化的字段时，才增加一个 invocation epoch/ready word；不得退回 per-run device `KernelArgs` allocation。

### 10.3 错误模型

- 参数、状态、resource、API enqueue 错误同步返回给 Python。
- 第一个 kernel 已 enqueue 后的后续 enqueue 失败会 poison context，防止错误 event 序列被继续复用。
- kernel 真正执行时产生的异步错误由 torch_npu/ACL runtime 在后续同步点报告；PyPTO launch 不主动查询。
- AICPU/AICore 内部错误仍使用现有 device error 状态；v1 不新增 host polling 节点。

## 11. 文件级实施清单

| 文件 | 计划改动 |
| --- | --- |
| `runtime/src/common/worker/pto_runtime_c_api.h` | 声明 L1 init/prepare/launch/support ABI 和生命周期契约 |
| `runtime/src/common/platform/onboard/host/c_api_shared.cpp` | 实现 mode 分流、异常边界和 C ABI 包装 |
| `runtime/src/common/platform/onboard/host/l1_execution_state.{h,cpp}` | 新增 persistent state、event 协议、poison 状态和 append-only registry |
| `runtime/src/common/platform/onboard/host/device_runner_base.{h,cpp}` | 接入 L1 state；复用 binary loader；增加 WithHostArgs launch；L1 finalize 不 reset device |
| `runtime/src/common/platform/onboard/host/device_runner_helpers.{h,cpp}` | 增加 prepare-once 的 persistent Runtime/KernelArgs helper，保留现有 per-run helper |
| `runtime/src/common/aicpu_loader/host/load_aicpu_op.{h,cpp}` | 用已解析 `rtFuncHandle` 封装 `aclrtLaunchKernelWithHostArgs` |
| `runtime/src/a2a3/platform/onboard/host/device_runner.{h,cpp}` | L1 device attach/finalize 分支，禁止 reset 和内部 sync |
| `runtime/src/a5/platform/onboard/host/device_runner.{h,cpp}` | 与 A2/A3 对齐的 L1 ownership 分支 |
| `runtime/src/{a2a3,a5}/runtime/tensormap_and_ringbuffer/runtime/runtime.h` | 定义/校验 L1 稳定 device state 与 invocation ABI |
| `runtime/src/{a2a3,a5}/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp` | 新 L1 entry，直接消费 invocation callable/args，复用 reset/scheduler |
| `runtime/src/{a2a3,a5}/runtime/tensormap_and_ringbuffer/host/runtime_maker.cpp` | 新 direct-device binder；现有 staging binder 不变 |
| `runtime/src/{a2a3,a5}/platform/onboard/aicore/kernel.cpp` | 原则上仅增加 ABI/static 校验；必要时实现 ready epoch |
| `runtime/src/common/worker/chip_worker.{h,cpp}` | 动态加载 L1 symbols，增加 L1 lifecycle/mode 方法 |
| `runtime/python/bindings/task_interface.cpp` | 暴露 raw-stream L1 binding |
| `runtime/python/simpler/task_interface.py` | 低层 context/handle 生命周期、幂等 prepare、显式 close |
| `python/pypto/ir/compiled_program.py` | 导出 L1 所需静态 signature/output/workspace/callable metadata |
| `python/pypto/runtime/l1.py`（新增） | `pypto_init`、`L1Context`、`L1Operator` 高层 API |
| `python/pypto/runtime/task_interface.py` | 连接 compiled program 与 simpler L1 binding |
| 独立 torch_npu adapter（新增，位置按构建系统确定） | current stream、taskQueue、tensor/scalar 打包和 forward op |
| `runtime/tests/ut/`、`tests/ut/runtime/` | ABI、state machine、无分配/无同步、binder 单测 |
| `tests/st/runtime/l1/`（新增） | onboard eager、ACLGraph capture/replay、negative/stress 测试 |

新增源文件后同步更新对应 CMake source list；若 Python public surface 有手写 type stub，则同步更新 stub。不要把 torch/torch_npu link dependency加入 `libhost_runtime.so`。

## 12. 分阶段实施顺序

阶段严格串行推进；前一阶段的退出条件是后一阶段的进入条件：

| 阶段 | 实施内容 | 退出条件 |
| --- | --- | --- |
| 0：onboard probe | 最小 C++/Python probe 验证 caller-stream AICPU + event-forked hidden AICore、async memset、WithHostArgs、AICore handle launch；覆盖连续 replay、warmup/capture 不同 stream；A2/A3、A5 共用源码 | 最小图可在不查询 capture model、不调用 `rtStreamAddToModel`、不 early-launch 的前提下重复 replay，结果正确且无 sync/event 错代；否则停止正式改造 |
| 1：ABI/ownership | 增加 L1 ABI、symbol loader、mode state 和 unsupported stubs；attach 当前 device，建立 hidden stream/events；fake-runtime UT 锁定禁止 reset/sync | 空 context 可 init/close；sim/HBG 稳定报 unsupported；L2/L3 回归不变 |
| 2：persistent state | 实现 `L1ExecutionState`；一次性分配 Runtime、KernelArgs、TRB arena、handshake/workspace；executor 注册一次，child/callable append-only | prepare 幂等；graph 可见地址稳定；launch allocator 计数不变；冲突注册正确报错 |
| 3：AICPU snapshot | 定义 versioned invocation；增加 `aclrtLaunchKernelWithHostArgs`；实现 `simpler_aicpu_l1_run`；覆盖 host 栈参数立即改写 | 连续异步调用不同 tensor/scalar 不串包 |
| 4：binder/launch | direct-device binder；实现第 6 节 event 顺序和 poison；只 invalidation handshake；拒绝 DFX | eager 单 kernel、多 child kernel、workspace 均正确，host launch 不等待设备 |
| 5：PyTorch | raw-stream binding；`pypto_init`/operator/prepare/warmup/close；独立 taskQueue/current-stream adapter；先 `out=` 后 eager 自动分配 | 不经 `inductor_pto` 即可用 PyTorch tensor 驱动 eager L1 op |
| 6：ACLGraph/回归 | capture 单 op；图内前后普通 NPU op；覆盖 scalar、多 output、多 child/workspace；完整 L2/L3 回归 | 达到第 15 节全部完成标准 |

## 13. 测试计划

### 13.1 Host/UT

- `L1ExecutionState` 合法/非法状态迁移。
- init、prepare、launch、close 的 ACL API 调用序列精确匹配预期。
- launch trace 中没有 capture query、capture model handle、`rtStreamAddToModel` 或 private-AICPU-stream launch。
- launch 期间 device alloc/free、stream/event create/destroy、sync 调用计数均为 0。
- direct binder 保持 tensor device 地址、dtype、shape、stride、scalar bit pattern 不变。
- callable ID 重复同 identity 幂等，不同 identity 冲突。
- workspace 不足、动态 metadata、DFX 开启、未 prepare、null stream、device mismatch 直接报错。
- 任一步骤部分 enqueue 失败后 context 进入 poisoned，后续 launch 被拒绝。
- L2/L3 原接口仍走原 helper，不使用 L1 persistent state。

### 13.2 Onboard eager

- 单输入单输出 elementwise program。
- 多输入、多输出和 scalar 参数。
- 一个 `@pl.program` 内多个 child kernel。
- 使用 TRB workspace/ringbuffer 的程序。
- 连续异步调用使用不同 input/output 地址，最后由测试统一 synchronize，结果不串包。
- warmup stream 和后续调用 stream 不同但不重叠，验证 `serial_tail`。
- A2/A3 与 A5 至少各跑一套 smoke；若 CI 设备不足，代码评审不得假设只在单架构成立。

### 13.3 ACLGraph

- 显式 warmup + synchronize 后 capture/replay。
- capture 图中 `pre_op -> PyPTO L1 -> post_op`，验证上下游 stream 顺序。
- 用可控延迟的 predecessor 验证 hidden AICore 不能提前占核；用可控延迟的 AICore 收尾验证 post-op 不能越过 done join。
- 多次 replay 使用固定 graph tensor 地址但改变输入内容。
- 两个顺序执行的 PyPTO L1 节点共享 context/workspace。
- capture 失败路径不泄漏 graph 引用资源。
- replay 次数由测试配置控制并做压力覆盖，不把某个 runtime 内部 launch 上限编码成产品常量。
- 明确负测并发 replay：v1 不承诺正确并发，wrapper/runtime 在能够检测时直接拒绝，文档也保留限制。

### 13.4 回归命令

实现后按仓库环境执行，具体 ST selector 以新增测试 marker 为准：

```bash
cd /mnt/workspace/inductor/pto/gpt_pypto
source .claude/skills/testing/load-env.sh
cmake --build build --parallel "$PYPTO_BUILD_JOBS"
python -m pytest runtime/tests/ut -n "$PYPTO_TEST_JOBS"
python -m pytest tests/ut/runtime -n "$PYPTO_TEST_JOBS"
# 仅在通过合规调度确认device 1空闲后执行；禁止fallback到device 0。
python -m pytest tests/st/runtime/l1 -v --platform=a2a3 --device=1
```

还要运行受改动模块现有的 compiled-program、task-interface 和 onboard TRB 回归；不能只跑新增 L1 case。

## 14. 主要风险与回退原则

| 风险 | 最早验证点 | 处理原则 |
| --- | --- | --- |
| hidden stream 无法仅通过 event fork/join 被 ACLGraph capture | Phase 0 | 停止正式改造，记录 runtime 能力缺口；不用 `rtStreamAddToModel`、capture query、early launch 或 sync 绕过 |
| `rtKernelLaunchWithHandleV2` 不可捕获 | Phase 0 | 评估等价 public AICore launch API作为后续独立迁移，不影响 AICPU WithHostArgs 结论 |
| event 重复 record/wait generation 错配 | Phase 0 | 调整 prepare-time event 拓扑；不依赖固定 launch 次数 |
| AICore 过早读取动态 Runtime 字段 | Phase 2/4 | 动态字段移入 AICPU invocation；必要时加 device ready epoch |
| AICPU callable load 在 capture 内触发 | Phase 3 | prepare/warmup 强制完成 load；capture 内发现未 ready 直接报错 |
| hidden AICore 早于 caller predecessor 启动 | Phase 0/4 | `start_event` 是必选入口 gate；trace + 延迟 predecessor 测试验证；不提供 early mode |
| caller 在 AICore 收尾前越过算子 | Phase 0/4 | `aicore_done_event` 是必选出口 join；不用 AICPU return/ack 推断整个算子完成 |
| 新 callable 使稳定 workspace 地址失效 | Phase 2 | 追加 backing storage 或拒绝；绝不搬迁 graph 已引用地址 |
| context 提前 close 导致 graph 悬空 | Python API | 调用方显式持有context和graph-bound tensors，并按external quiescence -> graph reset/destroy -> close顺序处理；不由 `__del__` 自动回收 |
| L1 finalize 误 reset torch_npu device | Phase 1 UT/ST | mode-specific finalize，mock 断言 reset/aclFinalize 从未调用 |
| 异步错误导致 AICore/AICPU 等待不退出 | Onboard negative | 沿用 runtime timeout/error 机制；v1 不在 host launch 引入 wait |
| L1 分支影响 L2/L3 hot path | 每阶段回归 | 新入口、新 state；共享 helper 只抽取纯公共逻辑 |

## 15. 完成标准

以下条件全部满足才算 L1 首期完成：

1. `@pl.program` 可通过 PyTorch 直接调用，不依赖 `inductor_pto`。
2. eager 与 ACLGraph replay 的数值结果在单 kernel、多 child kernel、workspace case 中正确。
3. native launch API 明确接收 caller stream；AICPU 在该 stream，AICore stream 对用户不可见。
4. caller predecessor 通过 `start_event` 先于 hidden AICore，caller downstream 通过 `aicore_done_event` 后于 AICPU/AICore 全部完成；不存在越过单算子入口或出口的 device task。
5. L1 launch 在 eager/capture 中走完全相同的路径，不查询 capture 状态，不持有 model/graph handle，不调用 `rtStreamAddToModel`，不存在 early-launch mode。
6. launch 路径没有 device allocation/free、stream/event create/destroy、H2D/D2H staging、stream/device sync。
7. 每次 AICPU 调用参数由 `aclrtLaunchKernelWithHostArgs` 形成独立快照；AICore 只引用 context-lifetime persistent device state。
8. workspace和child binary地址在graph生命周期内稳定；callable ID不重绑，同callable内func ID/address snapshot不变，不同callable可重复使用func ID数值。
9. L1 close 不执行 `rtDeviceReset`、`aclFinalize`，不破坏 torch_npu device context。
10. capture 前未 prepare、资源不足、unsupported runtime、DFX 开启等情况都在 host 侧清晰报错。
11. 不依赖任何固定 kernel-launch 上限，也不向用户暴露 ACLGraph capture/replay 细节。
12. 现有 L2/L3 API、测试和资源掌控行为保持不变。

达到这些条件后，再单独设计动态 shape/参数更新、外部 workspace、binary 回收和并发执行。需要跨算子提前展开 orchestration 的性能优化应作为后续 `host_build_graph` 能力单独设计；上述后续能力都不能反向破坏本计划建立的 L1 单算子、task-package 与 persistent-state 边界。

---

## 附录 A：完整决策记录

本附录把前面多轮讨论中已经确认的结论完整保留下来。它与正文有意重复：正文描述“准备怎么实现”，本附录描述“哪些事情已经决定、哪些事情没有决定”，避免实现过程中重新打开已关闭的设计问题。

### A.1 层级定义和最终目标

| 主题 | 已确认结论 | 对实现的直接约束 |
| --- | --- | --- |
| L3 | 单机多卡，由 hierarchical `Worker(level=3)` 管理多张卡、子进程、mailbox、pipeline lease 和通信资源 | L3 继续使用现有 leaf `ChipWorker`；本次不改其上层调度协议 |
| L2 | 单卡、全资源掌控；PyPTO/Simpler 分配 tensor/runtime/workspace，创建 stream，等待并回收整次 run | 现有 `simpler_run` 和 native-run prepare/launch/wait/finalize 语义不变 |
| L1 | 单算子、不掌控整张设备；行为应像一个普通 AscendC 自定义算子 | 必须借用外部 stream、异步返回、只使用调用方 tensor 地址、不得 reset 设备；所有内部 task 严格封闭在该 op 的 caller-stream entry/exit 之内 |
| 最终目标 | PyPTO 以 L1 方式进入 ACLGraph | PyPTO 不感知 graph capture/replay，只提供可被 capture 的普通 launch 形态 |
| 本轮验证 | 由 PyTorch 直接调用 PyPTO 完成 eager 和 ACLGraph 验证 | 不等待 `inductor_pto` 接入，也不在本计划修改 `inductor_pto` |

### A.2 平台、runtime 和入口范围

1. A2/A3 与 A5 的 ACLGraph 相关 runtime API 没有预期中的本质差异，L1 host 设计应共用；不能先写一套只对某一架构成立的私有协议。
2. 首期只实现 `tensormap_and_ringbuffer`。这是因为本次关键问题是 device orchestration、AICPU/AICore 双 stream 和 TRB device state；`host_build_graph` 不在首期范围。
3. `host_build_graph` 是后续跨算子调度和“orchestration 提前展开”优化的正确架构归属：它以完整 graph 而不是单个 L1 op 为调度单元，可在显式图依赖下合法安排更早的工作。本计划不实现该优化，也不在 TRB L1 中模拟它。
4. simulator 对 stream capture 改造没有有效验证价值。语义正确性必须 onboard 验证；sim 只需要稳定返回 unsupported，不能用 sim 通过代替 onboard gate。
5. 首期只接 `@pl.program`，不同时扩展其他入口、distributed HOST program 或更高层级调用。
6. A2/A3、A5 都必须导出相同的 L1 C symbols；未支持的 runtime variant 用明确 stub 返回，不通过缺少 symbol 表达能力差异。
7. device id 使用 ACL/runtime 的当前 device 查询能力获得并与 Python 请求值校验；不能假定永远是 device 0，也不能由 L1 finalize reset 当前 device。

### A.3 shape、tensor 和 scalar 契约

1. v1 采用静态 shape：prepare 后 tensor 数量、shape、dtype、stride、参数方向和 output spec 固定。
2. “静态 shape”是 L1 operator 实例的调用契约，不是限制 PyPTO 编译器只能生成静态能力。即使编译产物内部具有动态 shape 能力，本次 capture 的 task 和参数布局固定时，L1 runtime 不需要感知它。
3. 这与普通 AscendC op 的 tiling/capture 关系一致：capture 的是已经完成前置准备后形成的 launch；replay 时是否使用同一组动态值，是上层图和参数更新机制的问题。
4. 低层 native API 强制要求 input/output tensor 全部由调用方提供，地址必须是当前 device 上的 device pointer。
5. PyTorch convenience wrapper 可以根据静态 output metadata 分配 output，但 ACLGraph 首轮验证使用显式预分配 `out=`，把 allocator capture 行为从 PyPTO runtime 验证中隔离。
6. scalar 进入每次 invocation 快照。scalar 在 capture 时是否稳定、replay 是否更新是 ACLGraph 使用者的责任；PyPTO 不检测或缓存 capture 状态。
7. v1 不允许 launch 时改变 shape/layout 后偷偷重建 Runtime；不匹配直接报错。

### A.4 stream 和同步契约

1. native L1 API 原则上必须显式接收 caller stream。
2. PyTorch convenience wrapper 从 torch_npu current stream 获取底层 stream，参考用户给出的 torchair custom-op 用法；必须调用 `c10_npu::getCurrentNPUStream().stream(false)`，不能使用会处理/排空taskQueue的默认 `.stream()` 重载。
3. taskQueue 适配完全留在接近独立的 PyTorch wrapper，不把 torch/torch_npu 依赖传入 simpler core。
4. AICPU 主执行 task 使用 caller stream，这是让 PyPTO 在外部看起来像普通 op 的关键。
5. AICore 使用 PyPTO 内部持久 hidden stream；AICPU/AICore 双 stream 不出现在用户 API 中。
6. caller stream 在本 op 的 handshake invalidation 之后 record `start_event`，hidden AICore stream 必须 wait 它才能启动；这保证所有 caller predecessor 先于 PyPTO 占用全核。
7. hidden AICore stream 在 executor 真正返回后 record `aicore_done_event`，caller stream 必须 wait 它才能越过 op 出口；AICPU 返回或 shutdown ack 都不能代替这个 join。
8. 对外可见的是一个严格闭合的 caller-stream op；内部双 stream 并行只能发生在 `start_event` 和 `aicore_done_event` 之间。
9. ACLGraph 必须仅通过 event fork/join 捕获 hidden AICore 分支。PyPTO 不调用 capture query，不获取 capture model，不调用 `rtStreamAddToModel`，不分支处理 capture/replay，不持有 graph handle。
10. capture 和 eager 使用同一条 launch 路径；不存在 capture-only 跳过 pre-dependency 的 early mode，也不允许 private AICPU stream 上的 orchestrator 抢跑。
11. L1 launch、prepare 和 close 都不允许 PyPTO 主动做 stream/device synchronize。需要同步的 warmup、测试和 teardown quiescence 由调用方明确完成。
12. 普通 eager 模式可便利地自动注册；ACLGraph capture 流程必须先显式 prepare/warmup，随后由测试或用户执行外部 synchronize。

### A.5 内存、workspace 和并发契约

1. L1 不做 input/output tensor 的 device allocation、H2D staging、D2H copy-back 或 per-run free。
2. 当前 workspace 暂时保留在 PyPTO 内部，不在 v1 增加外部 workspace 参数。
3. 但 workspace 仍必须参与 L1 改造：它要从“可能在 run 中申请/扩容”改成“prepare 前申请、地址固定、launch 只复用”。这部分不能因为暂时不暴露外部 API 而完全不处理。
4. PyPTO 当前一次执行占用全部 AICore，合法执行之间没有可用的 AICore 并发。因此共享一份 workspace 在 v1 是合理的。
5. v1 明确禁止同一 device L1 context 的并发执行和并发 graph replay。普通 host launch 通过 context mutex 和 event tail 排序；graph replay 端无法完全由 PyPTO host 检测，仍保留调用契约。
6. 建议一个进程内每个 device 只允许一个 live L1 context；多个 callable 放入同一个 context。跨进程冲突不在本次解决。
7. `pypto_init(programs=[...])` 是首选整体准备入口：预先汇总程序资源需求，建立共享 persistent state，减少第一次 op call 的隐式行为。
8. launch 发现 workspace、ring、callable slot 或其他资源不足时直接报错，不在 capture 中 grow。
9. 用户层不需要看见 workspace、AICPU stream、AICore stream、event 或 runtime arena，避免把实现负担转移给用户。

### A.6 kernel binary 和 callable 生命周期

1. 当前 executor AICore binary 由 `rtRegisterAllKernel` 注册并缓存 handle，没有通过 public `aclrtRegisterBin` 完成完整公共 API 生命周期。
2. 当前 incore/child kernel binary 是 PyPTO 自己上传和管理的 GM 地址。历史L2/L3通过context runtime映射分发；L1对每个callable保存独立 `func_id -> addr` 快照，每次invocation只重放当前callable的映射。它们都没有注册成普通runtime kernel。
3. v1 不要求立刻迁移 binary 注册 API，也不要求实现 device binary 内存复用。
4. executor binary、child binary、orchestration SO、callable device descriptor 都至少存活到所有引用它们的 graph 被销毁。
5. 为先保证正确性，child binary 允许在 context 内持续累积；`ctx.close()` 才统一释放 PyPTO 能释放的部分。
6. callable ID在context生命周期内不重绑定；同一callable内的func ID/address snapshot不变。不同callable允许重复使用数值相同的func ID。相同identity的重复prepare可以幂等，不同identity冲突直接失败。
7. unregister 不能立即释放 graph 可能引用的 binary/device descriptor；v1 可只撤销 host handle，device 对象继续 pin 到 context close。

### A.7 task package 和参数生命周期

1. 问题的重点不是 AICore 能否并发，而是连续异步 host launch 时，tensor 地址、scalar、KernelArgs 和 runtime 描述何时可以覆盖或回收。
2. “kernel launch 后都是 device 地址”并不自动解决生命周期：runtime 仍需要保存 launch task 描述和 host args，直到 device 真正消费该 task。
3. CANN runtime 能看见 task 的真实完成点，因此其内部参数池适合管理普通 kernel-launch host args；PyPTO 不应重复实现一套不知道 completion 的 host/device task pool。
4. AICPU task 已经可以按普通 runtime kernel launch 处理，使用 `aclrtLaunchKernelWithHostArgs` 形成每次独立参数快照。
5. AICore 的关键不是给每次调用分配一份 device `KernelArgs`，而是消除 `KernelArgs` 中的 per-invocation 内容：AICore launch 永远只传一个 context-lifetime persistent `KernelArgs *`。
6. tensor 地址、scalar、callable ID 等动态参数只进入 AICPU invocation；AICPU 据此产生本次 child task payload，AICore 从 TRB/ringbuffer 消费。
7. child task payload 使用共享 arena，但下一次 reset/reuse 必须由 stream/event 顺序保证发生在前一次 AICPU/AICore 都完成之后。
8. 如果将来支持并发，才需要多 invocation slot、generation 和基于真实 completion 的回收；不能把一个猜测的 launch 数量上限当作 slot 数量。

### A.8 reset、错误、DFX 和可见性

1. 不需要新增用户可见 `reset()`。正常复用只需 host 侧异步失效 handshake；TRB scheduler/arena 已在 AICPU 内部 reset。
2. 不能为了保险每次清零整个 Runtime/arena/workspace，这既增加 capture 节点，也可能破坏地址稳定和性能。
3. host 参数、状态、资源和 enqueue 错误同步返回；设备执行期错误由 ACL/torch_npu 在调用方后续同步点报告。
4. 若一组双 stream 操作已经部分 enqueue，再发生 host API 失败，context 进入 poisoned。v1 不尝试在未知异步状态下“就地恢复”。
5. L1 不 reset device；即使出现 poisoned context，也由调用方排空/销毁 graph 后 close 并重建 context，不能破坏同设备上的 torch_npu 所有权。
6. v1 关闭所有会增加额外 workspace、collector、回读或同步的 DFX：args dump、PMU、dep-gen、scope stats、L2 swimlane 等。
7. 不依赖“约 2048 次 kernel launch”之类内部规格。这个值属于 runtime 内部实现，可能变化，也未必能从公开 API 查到。

### A.9 Python API 和验证范围

1. simpler/native 提供 raw stream 的低层 L1 API；PyPTO 提供 `pypto_init`、operator、prepare/warmup/close。
2. wrapper 只做 forward，首期不注册 autograd/backward。
3. context强持有全部native state，operator强持有context；但ACLGraph不会自动为PyPTO保持这些Python owner，调用方必须显式持有context和graph-bound tensors。关键teardown不依赖 `__del__`。
4. 用户必须在 graph 销毁且相关 stream 工作完成后再 `ctx.close()`。PyPTO close 本身不偷偷同步。
5. 测试首先覆盖 eager、单 kernel、多 child kernel、workspace，再做 ACLGraph capture/replay。
6. 不使用 simulator 证明 stream 语义，不等待 inductor 接入，不把 dynamic shape、并发、外部 workspace 一并塞入首期。

## 附录 B：当前调用逻辑的源码审计

本附录先记录设计所依据的当前 L2/L3 代码路径，再对照工作区中历史 `pto2/pypto` 的 L1/ACLGraph 路径。行号是本计划编写时的快照，后续提交可能漂移；实现者应优先按 symbol 搜索。

### B.1 当前 L2 Python 调用链

普通单卡调用大致经过：

```text
CompiledProgram.__call__
  -> _invoke_compiled
    -> execute_compiled
      -> compile_and_assemble
      -> _coerced_to_orch_args
      -> execute_on_device(level=2)
        -> reuse active pypto.runtime.worker.ChipWorker, or create simpler.Worker(level=2)
        -> worker.init(prewarm_config)
        -> worker.register(chip_callable)
        -> worker.run(callable_handle, orch_args, CallConfig)
        -> worker.close()
```

对应源码锚点：

- `python/pypto/ir/compiled_program.py::_invoke_compiled`：完成参数 coercion、选择 platform、调用 `execute_compiled`。
- `python/pypto/runtime/runner.py::execute_compiled`：assemble callable、构造 orch args、设置 DFX，进入 `execute_on_device`。
- `python/pypto/runtime/device_runner.py::execute_on_device`：复用 active `ChipWorker`，或走一次性 init/register/run/close。
- `python/pypto/runtime/worker.py::ChipWorker._run_chip`：缓存 callable handle，调用 simpler worker。
- `runtime/python/simpler/task_interface.py::ChipWorker.run`：把公开 handle 解析成私有 callable slot，调用 nanobind `_ChipWorker.run`。

当前 L2 的“torch.Tensor”并不等价于 L1 的 NPU tensor：`CompiledProgram` 现有入口主要允许 host `torch.Tensor` 或显式 `DeviceTensor`。普通 host tensor 会在 runtime maker 中 staging，只有标记为 `child_memory` 的 `DeviceTensor` 才透传 device 地址。L1 PyTorch wrapper 必须明确接受 torch_npu tensor 并直接传 `data_ptr()`，不能误走现有 host tensor 语义。

### B.2 当前 L2 native 调用链

`ChipWorker::run_on_slot` 最终调用 C ABI `simpler_run`：

```text
ChipWorker::run_on_slot
  -> select_pipeline_slot_ctx / select_arena_bank_ctx
  -> simpler_run
       -> simpler_prepare_run
       -> simpler_launch_run
       -> simpler_wait_run
       -> simpler_finalize_run
```

关键行为如下：

1. `simpler_prepare_run` 在 caller-owned opaque `RuntimeHandle` 上 placement-new `OnboardNativeRunState`。
2. 它 reserve native run、provision run resources、确定 launch shape，并调用 `bind_callable_to_runtime`。
3. TRB `runtime_maker.cpp::stage_device_args` 为普通 tensor staging device buffer，执行 H2D，并记录结束时的 D2H/free lease。
4. `simpler_launch_run` 创建 host executor thread；thread attach device 后调用 arch-specific `DeviceRunner::run`。
5. `DeviceRunner::run` 初始化 per-run regs、Runtime、device `KernelArgs`、diagnostic resources 和 run streams。
6. 当前实现先 enqueue AICore，再 enqueue AICPU；两边执行完后 `sync_run_streams()`。
7. `simpler_wait_run` join executor；`simpler_finalize_run` validate/copy-back/free，并释放 run claim/resource。
8. 同步包装 `simpler_run` 强制执行 prepare → launch → wait → finalize 的完整闭环。

这条路径适合 L2/L3 leaf：它拥有 run 的开始和结束，能够等待 completion 后安全 copy-back/free。它不适合 L1，因为 L1 的完成点由 caller stream/ACLGraph 持有，PyPTO host 在 launch 返回时看不到 graph replay 的完成。

### B.3 当前 L3 调用链及其与 L2 的关系

L3 入口不是把单卡 `CompiledProgram` 简单传 `level=3`，而是独立的 distributed path：

```text
DistributedCompiledProgram.__call__ / prepare
  -> execute_distributed / DistributedWorker
    -> simpler.Worker(level=3)
      -> root hierarchical scheduler
      -> shared-memory control/task frames
      -> one forked chip process per device
        -> local simpler ChipWorker
          -> _prepare_native_run_from_blob
          -> _launch_native_run
          -> _poll_native_run
          -> _finalize_native_run
```

L3 的重要区别：

- root/host orchestration 管理多卡和 SubWorker；每张卡的 leaf 仍是 L2 `ChipWorker`。
- host tensor 必须在 fork 前变成 shared memory，或使用 worker-resident `DeviceTensor`。
- task frame 携带 run_id、slot、generation、dispatch_id、callable digest、config 和 serialized task args。
- pipeline lease 允许 active run 期间准备 successor；但注释和实现都明确 device execution 仍是一条 whole-run FIFO，整张卡一次只执行一个 run。
- leaf chip process 用 native prepare/launch/poll/finalize 分离接口，使 hierarchical scheduler 能异步观察完成并管理 mailbox 生命周期。

因此 L3 的 native-run 拆分虽然有“异步”外观，仍不是 L1：它有自己的 progress loop，最终会 poll/finalize，能在完成后释放每次 run 的 Runtime 和 staging。ACLGraph replay 不会再次进入这个 host progress loop，所以不能直接拿 L3 token 机制充当 L1 operator。

### B.4 当前 callable 注册路径

当前 `simpler_register_callable`：

1. 解析 `ChipCallable`，上传 child kernel/ChipCallable buffer；
2. host-build-graph 记录 host orchestration function；TRB 记录 device orchestration SO、entry/config name 和 child kernel 地址；
3. TRB 通过 `launch_device_register(callable_id)` 让 AICPU 侧加载 callable；
4. 当前 `launch_device_register` 使用内部 AICPU stream 并调用 `aclrtSynchronizeStreamWithTimeout` 等待完成；
5. callable state 保存 kernel address mapping，run 时重放到 `Runtime.func_id_to_addr_` 并写 `active_callable_id_`。

L1 需要保留注册结果，但不能原样复用第 4 步。L1 prepare 应把 device register/init enqueue 到 caller stream，并通过 prepare event 与后续 launch 排序；capture 前由用户执行 warmup/synchronize，把异步注册错误暴露出来。

### B.5 当前 AICPU launch

`runtime/src/common/aicpu_loader/host/load_aicpu_op.cpp` 当前流程：

- `rtsBinaryLoadFromFile` 加载 dispatcher/inner AICPU binary；
- `rtsFuncGetByName` 为每个 symbol 得到 `rtFuncHandle`；
- `AicpuKernelLaunch` 构造 `rtCpuKernelArgs_t`；
- 通过 `rtsLaunchCpuKernel` enqueue 到指定 stream。

L1 不需要重做 binary load/resolve，只需要在相同 `rtFuncHandle` 上增加 `aclrtLaunchKernelWithHostArgs` 封装，并为 L1 entry 使用新的 invocation ABI。现有 L2 `rtsLaunchCpuKernel` 路径保持不变。

### B.6 当前 AICore launch 和参数构造

`KernelArgsHelper` 当前有两层 per-run device copy：

```text
host Runtime.dev
  --rtMemcpy--> device Runtime descriptor

host KernelArgs { runtime_args=device Runtime, regs, FFTS, DFX... }
  --rtMemcpy--> device KernelArgs

rtKernelLaunchWithHandleV2 args
  = one pointer to device KernelArgs
```

`DeviceRunnerBase::launch_aicore_kernel` 首次调用时用 `rtRegisterAllKernel` 注册 executor ELF，随后缓存 `aicore_bin_handle_`。每次调用的 launch args 只有 `KernelArgs *`。

这个现状给出一个重要结论：AICore kernel entry ABI 本来就是“读取一个 device context 指针”。L1 不需要为 AICore 发明可变 host args，只要把该指针指向 prepare-once、context-lifetime 的 immutable/stable state。

### B.7 当前 TRB reset 和 handshake

当前 TRB 已经在 AICPU executor 内部重置绝大多数可复用状态：

- `init_per_ring()` 重建 ring flow-control/header；
- `runtime_reset_for_reuse()` 重置 arena runtime data；
- mailbox 通过 `tail := head` 丢弃错误中止后未消费的旧消息；
- scheduler/context 在每次 boot 重新建立本次调度状态。

host 当前还在 AICore launch 前清 `Handshake::aicore_done`，原因是 workers 区域位于 pooled arena，会跨 run 保留。AICore 随后覆盖 `physical_core_id/core_type`，所以正常路径只需 invalidation `aicore_done`，不需要把整个 handshake/Runtime 清零。

### B.8 源码锚点索引

| 关注点 | 当前源码位置 |
| --- | --- |
| 高层单卡入口 | `python/pypto/ir/compiled_program.py:912` 附近 |
| L2 execute/assemble | `python/pypto/runtime/runner.py::execute_compiled` |
| L2 worker lifecycle | `python/pypto/runtime/device_runner.py::execute_on_device` |
| L3 高层入口 | `python/pypto/ir/distributed_compiled_program.py:318`、`:399` 附近 |
| L3 chip task progress | `runtime/python/simpler/worker.py` 中 `_prepare_native_run_from_blob`、`_launch_native_run`、poll/finalize loop |
| C ABI | `runtime/src/common/worker/pto_runtime_c_api.h:246` 起 |
| C ABI 实现 | `runtime/src/common/platform/onboard/host/c_api_shared.cpp:640`、`:739`、`:842`、`:852`、`:932` 附近 |
| AICPU load/launch | `runtime/src/common/aicpu_loader/host/load_aicpu_op.cpp:351`、`:382` 附近 |
| AICore launch | `runtime/src/common/platform/onboard/host/device_runner_base.cpp:1229` 附近 |
| per-run KernelArgs | `runtime/src/common/platform/onboard/host/device_runner_helpers.cpp:25`、`:60` 附近 |
| A2/A3 AICore entry | `runtime/src/a2a3/platform/onboard/aicore/kernel.cpp:90` 附近 |
| A5 AICore entry | `runtime/src/a5/platform/onboard/aicore/kernel.cpp:104` 附近 |
| KernelArgs layout | `runtime/src/{a2a3,a5}/platform/include/common/kernel_args.h` |
| Runtime device descriptor | `runtime/src/{a2a3,a5}/runtime/tensormap_and_ringbuffer/runtime/runtime.h:158` 附近 |
| L2 tensor staging | `runtime/src/{a2a3,a5}/runtime/tensormap_and_ringbuffer/host/runtime_maker.cpp:550` 附近 |
| TRB internal reset | `runtime/src/{a2a3,a5}/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp:604` 附近 |
| stale handshake consumer | `runtime/src/{a2a3,a5}/runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler_cold_path.cpp:729` 附近 |
| CANN WithHostArgs 声明 | `torch_npu/third_party/acl_src/runtime/include/external/acl/acl_rt.h:5060` |
| CANN launch routing | `torch_npu/third_party/acl_src/runtime/src/runtime/api/api_c_kernel.cc:42` 附近 |
| CANN task arg pool | `torch_npu/third_party/acl_src/runtime/src/runtime/core/src/kernel/arg_loader/stars_arg_manager.cc:89` 附近 |
| CANN completion recycle | `torch_npu/third_party/acl_src/runtime/src/runtime/core/src/stream/stream_david.cc:1215` 附近 |

### B.9 历史 `pto2/pypto` 的 L1/ACLGraph 实现：可复用机制与明确拒绝的调度语义

工作区中的历史实现位于 `../../pto2/pypto`。它已经打通了“torch current stream -> native kernel launch -> ACLGraph capture/replay”的基本形态，因此是有价值的机制参考；但它的 AICPU/AICore 拆分方式、capture 感知和 early-launch 策略与本计划确立的单算子边界相冲突，不能整体照搬。

#### B.9.1 历史实现的实际 stream 拓扑

1. `../../pto2/pypto/python/pypto/frontend/parser/entry.py:646` 把 torch 当前 stream 传给 `LaunchKernelTorch`。
2. `../../pto2/pypto/python/src/bindings/runtime.cpp:625-650` 把该 stream 保存为 `aicoreStream`，并主动调用 `DeviceLauncher::GetCaptureInfo(aicoreStream, rtModel)` 查询 capture 状态和 model handle。
3. AICore executor 通过 `RuntimeKernelLaunchWithHandleV2(..., aicoreStream, ...)` 发射到 torch caller stream；AICPU control/scheduler/orchestrator 则使用 PyPTO 自己的 ctrl/sched streams。这与本计划的“AICPU 在 caller stream，AICore 在 hidden stream”正好相反。
4. `../../pto2/pypto/framework/src/machine/runtime/launcher/device_launcher.cpp:341-346` 的 `AddAicpuStream` 在 capture 时调用 `RuntimeStreamAddToModel`，把 ctrl/sched private streams 主动挂到 caller stream 对应的 capture model。同文件 `SetCaptureStream` 也有相同形态的 `RuntimeStreamAddToModel(aicpuStream, rtModel)`。
5. `DeviceLauncher::LaunchSyncTask` 在 `launchEarlyMode == 0 && isCaptureMode` 时直接返回，跳过 `RunPreSync`。而 `RunPreSync` 原本会在 caller/AICore stream record event，让 private AICPU streams wait，从而把 orchestrator 排在 caller predecessor 之后。
6. `DeviceLauncher::LaunchKernel` 的 host enqueue 顺序是先 `LaunchAicpuKernel`，再 `LaunchAicoreKernel`。在 capture early mode 中，private AICPU streams 已被加入 model，又没有 caller-stream entry event，所以 AICPU orchestrator 可在 caller stream 真正走到这个 PyPTO kernel 之前就提前展开。

历史 capture 拓扑可概括为：

```text
caller/torch stream                       private AICPU ctrl/sched streams
-------------------                       ---------------------------------
... predecessor tasks                     [added to capture model]
launch AICore executor                     launch AICPU orchestrator early
... successor tasks                        expand/schedule child work
```

关键不是 host C++ 中哪个 launch 调用写在前面，而是 capture model 内 private AICPU stream 与 caller predecessor 之间没有单算子 entry dependency。`rtStreamAddToModel` 使 private stream 的 task 进入图，跳过 `RunPreSync` 则刻意移除了它与 caller stream 之前任务的边界。二者结合才形成这里反对的 early-orchestrator 语义。

#### B.9.2 为什么这个性能优化不属于 L1

这种方式的目标是让 AICPU orchestrator 提前生成/提交 child work，从而隐藏 orchestration 延迟。它可以带来性能收益，但对 L1 来说有以下不可接受的代价：

1. **越过算子入口。** caller stream 上的“本 op 之前”不再意味着 PyPTO 的 device 行为尚未开始。一个外观上普通的 custom op 在 caller 到达它之前已产生内部工作，破坏了组合性。
2. **eager/capture 语义分裂。** 历史代码通过 `IsCaptureMode()` 和 `launchEarlyMode` 选择是否保留 pre-sync；同一个 op 在 eager 和 graph 中不是同一条调度路径。
3. **依赖 capture 内部对象。** PyPTO 必须查询 capture info、拿到 model handle 并主动改变 model 的 stream 集合，不再是一个图透明的普通 launch API。
4. **隐藏资源调度越权。** PyPTO 目前占用所有 AICore。一旦允许 hidden work 在 caller predecessor 之前启动，除了语义越界，还会带来抢核、饥饿甚至形成循环等待的风险。
5. **无法作为单算子局部优化证明。** “这个 op 提前跑一点”是否安全，需要看完整图的资源和依赖；单个 L1 API 不拥有这个视野。

因此，本计划对 L1 做出硬性结论：

- 不提供 `launch_early_mode` 或任何等价开关；
- 不把 AICPU orchestrator 放到 private stream；
- 不使用 capture query + `rtStreamAddToModel` 形成 capture-only 调度拓扑；
- 不在 prepare 时预启动一个等待未来 invocation 的 AICPU/AICore kernel；
- 不以“性能更好”为理由破坏 entry/exit 之间的严格单算子闭包。

这个结论只限定 PyPTO L1 的架构边界，不对 `rtStreamAddToModel` 在其他 runtime/图执行器中的通用合法性作判断。但在本 L1 的实现、probe、测试和回退方案中，该 API 都是明确禁止项。

#### B.9.3 历史实现中值得复用的部分

| 历史机制 | 源码事实 | 本计划的取舍 |
| --- | --- | --- |
| torch current stream 入参 | Python 将 `_current_stream()` 传到 native | 复用概念；由新 PyTorch adapter 取 `c10_npu::getCurrentNPUStream().stream(false)`，对 simpler core 只暴露 raw stream |
| AICPU WithHostArgs | `load_aicpu_op.cpp:89-101` 调用 runtime WithHostArgs API | 复用机制；用它建立每次 invocation 的 runtime-owned 参数快照 |
| AICore handle launch | `device_launcher.cpp:452-462` 调用 `RuntimeKernelLaunchWithHandleV2` | 作为 Phase 0 优先验证路径；但参数改为 context-lifetime persistent `KernelArgs *` |
| executor binary 注册 | `kernel_binary.cpp:70-102` 注册/卸载整个 kernel binary | 参考 runtime-owned handle 思路；不假定它等价于当前 TRB child/incore binary GM 管理 |
| host args 固定布局 | `KernelBinary::InitLaunchArgs` 建立 args/host-input 描述 | 复用 ABI 固定的原则，重新定义 versioned `L1AicpuInvocationArgs` |
| capture 前 current-stream plumbing | binding 保存 caller stream 并用它 launch AICore | 复用“外部传 stream”这一 API 方向，但改为 AICPU 在 caller、AICore 在 hidden |
| capture query/model attach | `GetCaptureInfo` + `AddAicpuStream` | 明确不复用 |
| capture-only early launch | capture 时跳过 `RunPreSync` | 明确不复用 |
| eager 内部 sync | `LaunchAicoreKernel` 在非 capture 等条件下 `DeviceSynchronize` | 明确不复用；新 L1 eager/capture 都异步 |

历史实现还在 `../../pto2/pypto/python/src/bindings/runtime.cpp:697-702` 通过外部 Python/module allocator 提供 workspace，并在 `../../pto2/pypto/examples/03_advanced/aclgraph/aclgraph.py:126-145` 的示例中直接 capture 首次 model 调用。这两点都不改变本计划已经确定的选择：

- workspace 首期继续由当前 PyPTO 内部 prepare-time 分配和 pin，不新增外部 workspace API；
- ACLGraph capture 前必须显式 prepare/warmup，不在 capture 中触发编译、binary load、allocation 或 lazy register。

#### B.9.4 历史拓扑与目标拓扑的精确差异

| 项目 | 历史 `pto2/pypto` | 本计划的 TRB L1 |
| --- | --- | --- |
| caller stream 上的主 task | AICore executor | AICPU orchestrator |
| private stream | AICPU ctrl/scheduler streams | 只有 hidden AICore stream |
| hidden branch 入口 | capture 时可跳过 caller pre-sync | 必须 wait caller record 的 `start_event` |
| hidden branch 加入 graph | capture query + `rtStreamAddToModel` | ACLGraph 仅通过 caller/hidden event fork 自然捕获；失败即 Phase 0 不通过 |
| 算子出口 | 依历史 capture/sync 分支而异 | caller 必须 wait hidden AICore `done_event` |
| eager/capture 路径 | 感知 capture，可切换 early/sync | 完全相同，不感知 graph |
| 优化边界 | 单 op 可让 orchestrator 越过 entry 抢跑 | 单 op 严格闭合，不做跨边界优化 |

目标拓扑是：

```text
caller/torch stream                         hidden AICore stream
-------------------                         --------------------
... predecessor tasks
[L1 entry]
invalidate -> record(start) --------------> wait(start)
AICPU orchestrator                           AICore executor
wait(done) <------------------------------- record(done)
[L1 exit]
... successor tasks
```

这里 AICore 仍可以在 AICPU orchestrator 完成之前启动，因为 TRB 需要两者在算子内通过 handshake/ring 协作；但两者都不能早于 caller 的 `start_event`，且 caller 不能早于 AICore `done_event` 离开该 op。“内部并行”与“越过单算子边界提前执行”必须在评审中被当作两件完全不同的事。

#### B.9.5 后续性能优化归属 `host_build_graph`

当前仓库中 `runtime/src/a2a3/runtime/host_build_graph/host/runtime_maker.cpp` 已经体现了 host-orchestration-first 的基本模型：host 先执行 orchestration，构造完整 task graph image，做地址 relocation 并上传，device 侧再以 scheduler-only 形态执行。这个 runtime 拥有完整 graph 任务和依赖视野，所以将来需要通过“提前展开 orchestration”来隐藏 host/AICPU 延迟时，应在 `host_build_graph` 的图级语义下单独设计：

1. 以 graph 而不是 L1 op 为调度和资源单元；
2. 把跨 op 的先后关系显式编码在 host-built graph 中；
3. 在完整资源视野下决定哪些 orchestration 可提前，而不是由一个局部 op 私自抢跑；
4. 独立定义它与 ACLGraph 的 capture/instantiate/replay 分层，不复活 L1 中的 capture query/`rtStreamAddToModel` 分支；
5. 保留本文定义的 L1 单算子语义；图级优化不能暗中改变单独调用 L1 API 时的顺序保证。

本计划仍然对 `host_build_graph` 的 L1 C API 返回 unsupported。本节只确定后续架构边界，不把 host-build-graph 改造偷渡进当前 TRB L1 的实现范围。

第二阶段 HBG L1 + ACLGraph 的完整对象模型、WithHostArgs inline payload候选、per-replay restore、capacity/lifetime协议和device 1 P0矩阵见**附录N**。附录N是后续阶段准入设计，不改变本计划前文对当前TRB L1的API、完成定义和禁止项。

## 附录 C：kernel task package 与 device state 的完整推导

这是 L1 改造最关键的部分。实现评审时应先审这一节，再审具体 API 名字。

### C.1 必须区分六类对象

| 对象 | 内容 | 谁分配/持有 | 最短安全生命周期 |
| --- | --- | --- | --- |
| Python/PyTorch call args | tensor object、scalar object、output references | 调用方/wrapper | native enqueue 完成；tensor storage 需延续到设备使用完 |
| `L1AicpuInvocationArgs` host image | device 地址值、scalar bits、callable desc、persistent context pointers | PyPTO host 栈或临时对象 | 只需活到 `aclrtLaunchKernelWithHostArgs` 返回 |
| CANN task args snapshot | runtime 从 host args 建立的 task 参数存储 | CANN runtime 内部 pool | AICPU task 真正完成并被 runtime 回收 |
| persistent `KernelArgs`/Runtime | AICore 所需稳定地址、regs、arena、handshake 等 | PyPTO L1 context | 所有相关 graph 销毁且 device quiescent 后 |
| TRB child task payload | AICPU 生成、AICore 消费的每次 invocation task/args | PyPTO shared arena/ring | 本次 AICPU/AICore 都结束；下一次 reset 前 |
| executor/child binary | AICore executor ELF 和 incore function binary | runtime handle + PyPTO GM | 所有可能 launch/dispatch 它们的 graph 销毁后 |

这六类对象不能统称为“KernelArgs”。早期方案容易混淆的正是第 2、3、4、5 类。

### C.2 AICPU invocation 的时间线

```text
T0 host: 读取 tensor.data_ptr、shape metadata、scalar bits
T1 host: 在栈上构造 L1AicpuInvocationArgs
T2 host: aclrtLaunchKernelWithHostArgs(..., &invocation, sizeof(invocation), ...)
T3 runtime: 为本次 task 建立独立参数快照并入队
T4 host: API 返回；栈上 invocation 可以销毁或被下一次调用覆盖
T5 device AICPU: 从本次 runtime-owned task args 读取 callable/torch tensor 地址/scalar
T6 AICPU: 在共享 TRB arena 中构造 child task payload
T7 AICore: 从 persistent Runtime/ring 取得并消费 child task
T8 hidden AICore record done，caller stream 在自身 AICPU task 之后 wait done；单算子 exit 边界闭合
T9 CANN runtime: 确认 task completion 后回收自己的 task args slot
T10 下一次 PyPTO invocation: 在 serial-tail 依赖后 reset/reuse shared arena
```

PyPTO 不需要知道 T9 的内部实现细节，也不需要自己查询哪一个 task slot 已完成。它依赖的是 `aclrtLaunchKernelWithHostArgs` 对 host args 的异步 launch 契约；本地 CANN 源码中的参数池和 completion recycle 用来验证该选择与当前实现一致，而不是让 PyPTO 调用内部 pool API。

### C.3 为什么 AICore 不需要 per-call task args pool

AICore executor 每次真正需要的入口参数只有一个 `KernelArgs *`。把其中所有可变字段移走以后：

```cpp
struct L1PersistentKernelArgs {
    L1DeviceContext *runtime_args;  // stable device address
    uint64_t regs;                  // stable table
    uint64_t ffts_base_addr;        // stable resource
    uint64_t workspace_base;        // stable shared workspace
    // v1 DFX fields remain zero
};
```

这个 object 在 prepare 时做一次 H2D，后续所有 AICore launch 都只把相同 device pointer 交给 `rtKernelLaunchWithHandleV2`。launch task 自己携带的只是一个 8-byte pointer value，而被指向的 object 在 graph 生命周期内不释放，因此不存在“下一次 host 调用覆盖本次 AICore KernelArgs”的问题。

动态的 tensor/scalar 不应重新塞回 persistent Runtime。AICore 真正使用它们时，它们已经被 AICPU 编码进对应 child task payload；TRB ring 的 producer/consumer 协议负责本次执行内部的可见性。

### C.4 哪些 Runtime 字段必须重新分类

当前 `DeviceRuntimeLaunchDesc` 同时包含：

- 稳定字段：workers/handshake 基址、worker/core 数、GM SM、prebuilt arena、function binary address mapping；
- 配置字段：AICPU thread/affinity、ring sizing；
- per-call 字段：`orch_args_storage_`、`active_callable_id_`；
- 会被 executor reset 的运行时字段。

L1 需要逐字段审计，而不是直接把当前 struct 标记为 persistent：

1. prepare 固定的字段写入 L1 persistent device context；
2. AICPU invocation 携带 `orch_args`、callable id/descriptor 和允许的 scalar；
3. AICPU 内部 reset 的字段继续留在 runtime arena；
4. 任何仍会在 host launch 前变化的字段都不能由 AICore 无保护地读取；要么移入 invocation，要么加入 device-side publish/ready 协议；
5. L2/L3 的现有 `DeviceRuntimeLaunchDesc` ABI 不改，必要时为 L1 新建 `L1DeviceContext`，不要强求一个 struct 同时服务两套生命周期。

### C.5 AICPU 与 AICore 启动竞态

`start_event` 先保证一个更外层的顺序：handshake invalidation 和所有 caller-stream predecessor 都已到达 fork 点，hidden AICore 才有资格启动。这个 gate 是单算子入口和全核资源安全的硬约束，不能为了提前 orchestration 而略过。

在 `start_event` 之后，event 并不保证 AICPU 已发布动态字段才启动 AICore。这是单算子内部的正常启动竞态，与“在 caller 到达算子前提前启动”不是同一件事。正确设计应尽量让 AICore 启动阶段只读 persistent 字段：

- AICore 可以先启动并上报 handshake；
- AICPU 从自己的 invocation snapshot 获得本次 callable/args；
- AICPU 完成 runtime reset 和 task publish 后，通过现有 handshake/ring protocol 开窗；
- AICore 不在开窗前读取本次 child task payload。

只有源码审计或 onboard probe 证明 AICore entry 在开窗前必须读某个动态字段时，才增加 `invocation_generation/ready`：AICPU leader release-store ready，AICore acquire-load 后继续。不能用 host 同步 memcpy 或 per-run allocation掩盖竞态。

同理，AICPU 的 deinit/shutdown ack 只能证明 orchestrator 侧进展，不代表 hidden AICore stream 的 kernel task 已在 runtime 层完整返回。只有 hidden stream 在 AICore launch 之后 record 的 `aicore_done_event` 才能关闭 op 出口边界。

### C.6 将来并发时才需要解决的事情

若未来 PyPTO 不再占用全部 AICore，且产品要求并发 L1 invocation，需要重新设计：

- N 份 workspace/runtime arena/handshake；
- N 份 invocation generation/ownership；
- graph node 到 slot 的稳定绑定；
- 基于 event/task completion 的 slot recycle；
- callable binary/descriptor 的并发读和 unregister；
- 跨 graph/stream 的公平性和错误隔离。

这些不能通过“创建 2048 个 slot”解决。runtime 某个内部最大 launch 数既不是 completion protocol，也不是稳定 API，因此 v1 不预埋与该数字绑定的实现。

## 附录 D：workspace、binary 与全部资源所有权

### D.1 总体所有权表

| 资源 | 创建时机 | 所有者 | launch 是否可修改 | 销毁时机 |
| --- | --- | --- | --- | --- |
| torch_npu device/context | PyTorch 初始化 | torch_npu/调用方 | 否 | 不由 PyPTO 销毁 |
| input tensor storage | 用户/PyTorch | 调用方 | 只读或按 signature | default NPU caching allocator由adapter `recordStream`保护本次device use；调用方仍持有graph-bound tensor到graph销毁 |
| output tensor storage | 用户预分配 | 调用方/PyTorch | kernel 写 | 同上；v1不自动分配纯Out storage |
| external/from-blob/custom storage | 调用方/外部allocator | 外部owner | 按signature | `recordStream`可能无法接管；owner必须持有到graph销毁且最后真实device use完成 |
| raw caller stream | PyTorch current stream | torch_npu | PyPTO只 enqueue | 不由 PyPTO 销毁 |
| hidden AICore stream | `pypto_init` | L1 context | 只 enqueue | graph 销毁、外部 quiescence 后 `ctx.close()` |
| start/done/tail events | `pypto_init` | L1 context | record/wait | 与 hidden stream 同期 |
| AICPU binary handle | `pypto_init` | L1 context/`LoadAicpuOp` | 否 | `ctx.close()`；不 reset device |
| AICore executor handle | prepare/首次显式 binary prepare | L1 context | 否 | 若 API 可卸载则 close；否则遵守当前 runtime 生命周期 |
| child kernel binary GM | callable prepare | L1 context append-only pool | 否 | `ctx.close()` |
| orchestration SO/device descriptor | callable prepare | L1 context | 否 | `ctx.close()` |
| callable ID / callable-local func ID maps | callable prepare | L1 context | callable ID只追加；同callable内mapping不重绑 | `ctx.close()` |
| regs/FFTS tables | context prepare | L1 context | 否 | `ctx.close()` |
| persistent Runtime/KernelArgs | context prepare | L1 context | 稳定字段不改 | `ctx.close()` |
| handshake invalidation region | context prepare | L1 context | 每次 `aclrtMemsetAsync` | `ctx.close()` |
| TRB prebuilt arena/GM SM | `pypto_init`/prepare | L1 context | device 内部 reset/reuse | `ctx.close()` |
| L1 workspace | `pypto_init`/prepare | L1 context | kernel 内容可变，地址不变 | `ctx.close()` |
| AICPU invocation host image | 每次 Python/native launch | host 调用栈 | API 返回后可覆盖 | API 返回 |
| CANN task args snapshot | 每次 WithHostArgs launch | CANN runtime | 不由 PyPTO访问 | runtime 确认 task completion 后 |
| child task payload | 每次 AICPU execution | TRB arena | 本次 producer/consumer 使用 | serial-tail 后下一次 reset/reuse |
| ACLGraph handle/graph pool | capture | 调用方/torch_npu | replay 使用 | 必须早于 `ctx.close()` 销毁；不假定graph会强持有PyPTO context |

### D.2 workspace 是否应在本次一起改造

结论是“内部 workspace API 不扩展，但 workspace 生命周期必须一起改造”。原因分三层：

1. **不需要对外暴露。** v1 无合法并发执行，PyPTO 占用全部 AICore，共享内部 workspace 不会在正确使用下踩踏；外部 workspace 只会增加用户负担。
2. **不能保持现状不动。** ACLGraph capture 后要求 launch 路径无 allocation，且 graph 中使用的 device 地址必须稳定；若 workspace 仍在每次 run 中申请、扩容或移动，L1 根本不成立。
3. **需要 prepare-time 容量规划。** `pypto_init(programs=[...])` 应收集全部已知 program 对 GM heap、GM SM、TRB arena、DMA workspace 和其他 scratch 的需求，按可共享资源的最大值而不是总和进行一次 provision。

因此本次 workspace 改造的准确边界是：

- 改资源的申请时机、地址稳定性、容量检查和 close 回收；
- 不新增用户传入 workspace pointer；
- 不新增公共 workspace-size query；
- 不实现多 invocation workspace slot；
- 不允许 launch-time fallback allocation。

### D.3 建议的容量冻结规则

`L1ExecutionState` 维护资源阶段：

```text
COLLECTING
  - pypto_init 收集 programs
  - callable prepare 可增加需求
  - 尚未发生任何 L1 op launch

SEALED
  - 第一次完整warmup/launch序列成功enqueue后进入
  - persistent addresses 和 capacity 固定
  - 新callable一律不冁许prepare；全部program必须由context-wide prepare先收集

POISONED
  - 不再允许dispatch，只允许外部quiescence后close

CLOSING
  - 第一项destructive teardown前进入；释放失败仍保持Closing
  - 不允许prepare/launch，只允许显式retry close

CLOSED
  - 所有资源已释放，close幂等
```

不需要把 `seal()` 暴露给普通用户。`pypto_init(programs=[...])` 声明全部program，context-wide `prepare()` 完成收集/异步注册，第一次完整launch enqueue成功后由内部seal。若将来要支持prepare后追加大program，应采用append-only backing storage并保证旧地址不动；v1的行为是sealed后新callable直接报错。

### D.4 workspace 需求计算

实现时至少要区分：

- `gm_heap`: child task 或 runtime allocator 使用的共享堆；
- `gm_sm`: PTO2 shared memory/ring region；
- prebuilt runtime arena image；
- handshake/workers region；
- regs/FFTS address tables；
- dispatcher/AICPU init 所需持久 buffer；
- optional SDMA workspace（只有首期明确允许的功能才能 provision）；
- program 静态 workspace/scratch metadata。

对串行 callable，原则上取各 program 相同资源种类的最大需求；但不能盲目把所有字段做 `max()`：某些 resource descriptor 内含相对地址和布局，必须重新构建一个能够容纳最大 sizing 的合法 arena image。沿用现有 prebuilt-arena builder 的 sizing/cache key，比把现有任意 program 的 image 扩大更安全。

### D.5 binary pinning 的精确策略

建议将 binary 状态分成三层：

1. **Executor binary**：AICPU dispatcher/inner SO 和 AICore executor ELF，每个 L1 context 一份。
2. **Callable orchestration binary**：按 orchestration ELF Build-ID 去重，但一旦被 graph 使用便 pin 到 context close。
3. **Child/incore binary**：按content identity去重并记录稳定GM地址；每个callable保存自己的 `(func_id, device_addr)` 快照且不可重绑。不同callable可使用同一func ID数值指向不同binary。

v1 不做 LRU、引用 graph 数量的 refcount 或 runtime completion 驱动的 binary recycle，因为 PyPTO 看不到 graph 何时永久不再 replay。简单“unregister callable 就 free child binary”会造成已经 capture 的 graph 在未来 replay 时读取悬空地址。

### D.6 L1 close 的释放顺序

调用前置条件：所有 graph 已销毁，调用方已经保证相关 stream quiescent。`close()` 自身不建立这个前置条件，只验证可验证的 host state。

建议释放顺序：

1. 阻止新的 prepare/launch，取得 context host mutex；
2. 清理 Python callable handles，但保留到 native close 调用结束的强引用；
3. 销毁/卸载 callable orchestration device descriptors；
4. 释放 child binary GM 和 callable buffers；
5. 释放 shared workspace、TRB arena、Runtime、KernelArgs、regs/FFTS/handshake；
6. 销毁 L1 自有 events；
7. 销毁 hidden AICore stream；
8. 卸载 PyPTO 自有 AICPU/AICore binary handles；
9. 清理 host registry/state；
10. **不调用** `rtDeviceReset`、`aclFinalize`，不销毁 caller stream，不改变 torch_npu current device ownership。

若当前 CANN API 对某个 binary handle 没有安全 unload，记录为 context/process-lifetime pinned，而不是在 close 中调用未经验证的内部接口。

## 附录 E：stream 协议、生命周期和状态机

### E.1 context 状态机

```text
NEW
  | simpler_l1_init / pypto_init
  v
INITIALIZING
  | host resources created; async device init may be enqueued
  v
COLLECTING
  | callable prepare + resource planning
  v
READY_ENQUEUED
  | prepare work is ordered on caller stream; warmup must precede capture
  v
SEALED
  | first L1 launch; resource addresses frozen
  | repeated async launch/capture/replay
  +------------------------------+
  | any pre-enqueue validation   | partial enqueue/API failure
  | error: state unchanged       v
  |                           POISONED
  |                              |
  +------------------------------+
                                 | caller first proves external quiescence
COLLECTING / READY_ENQUEUED / SEALED / POISONED
                                 | begin_close(), before destructive teardown
                                 v
                              CLOSING
                                 | cleanup failure: remain CLOSING; dispatch rejected
                                 | explicit close retry eventually succeeds
                                 v
                               CLOSED
```

“READY_ENQUEUED”不表示 device AICPU init/register 已经执行完成，只表示后续 launch 在 stream/event 上正确依赖它。ACLGraph capture 前的显式 warmup + caller synchronize 才是把异步准备错误暴露出来的用户流程。

### E.2 callable 状态机

```text
ABSENT
  -> REGISTERING_HOST
  -> DEVICE_PREPARE_ENQUEUED
  -> READY_FOR_ORDERED_LAUNCH
  -> PINNED_BY_CONTEXT
  -> released only at context close
```

同一 identity 的 prepare 重入返回现有 handle；同一 callable ID 的不同 identity 在 `REGISTERING_HOST` 之前就拒绝。任何已进入 `PINNED_BY_CONTEXT` 的 callable 都不能通过普通 unregister 释放 device state。

### E.3 prepare 的异步顺序

当前 L2 registration 会启动 AICPU register task 并内部 sync。L1 建议改为：

```text
caller stream
  [optional L1 device init WithHostArgs]
  [callable register/load WithHostArgs]
  record(callable_prepare_done_event)

first launch on same or another stream
  wait(callable_prepare_done_event)
  continue normal L1 launch sequence
```

这样 `prepare_callable()` 的同步返回只报告 host validate/enqueue 是否成功；device dlopen/config 错误会在用户 warmup 后的外部 synchronize 报告。不能为了得到一个同步 prepare 返回值重新引入内部 stream sync。

当前实现把所有 `pypto_init(programs=[...])` 准备汇总到context-level prepare tail event。该event只由第一个成功的完整warmup/launch enqueue消费；后续capture launch绝不再wait它，避免capture外record的event触发capture-isolation。

### E.4 单次 launch 的完整 host enqueue 伪代码

```cpp
int DeviceRunnerBase::l1_launch(int32_t cid,
                                const ChipStorageTaskArgs &args,
                                aclrtStream caller) {
    std::lock_guard lock(l1_.enqueue_mutex);
    validate_l1_launch(cid, args, caller);       // must not enqueue on failure
    // No capture query, graph/model handle, StreamAddToModel, or early mode.

    L1AicpuInvocationArgs inv = make_invocation(cid, args);

    if (l1_.has_prepare_tail && !l1_.prepare_tail_consumed)
        ACL_TRY(aclrtStreamWaitEvent(caller, l1_.prepare_tail));
    if (l1_.has_serial_tail && l1_.last_caller_stream != caller) {
        aclrtEventRecordedStatus status{};
        ACL_TRY(aclrtQueryEventStatus(l1_.serial_tail, &status));
        if (status != ACL_EVENT_RECORDED_STATUS_COMPLETE)
            return PTO_RUNTIME_ERR_BUSY;  // no task enqueued; caller must externally quiesce and retry
    }

    ACL_TRY(aclrtMemsetAsync(l1_.handshake_invalidate_addr,
                             l1_.handshake_invalidate_size, 0,
                             l1_.handshake_invalidate_size, caller));
    ACL_TRY(aclrtRecordEvent(l1_.start_event, caller));

    ACL_TRY(aclrtLaunchKernelWithHostArgs(l1_.aicpu_run_func,
                                          l1_.aicpu_launch_count,
                                          caller, nullptr,
                                          &inv, sizeof(inv), nullptr, 0));

    ACL_TRY(aclrtStreamWaitEvent(l1_.aicore_stream, l1_.start_event));
    ACL_TRY(launch_aicore_kernel(l1_.aicore_stream,
                                 l1_.persistent_kernel_args_dev));
    ACL_TRY(aclrtRecordEvent(l1_.aicore_done_event,
                             l1_.aicore_stream));

    ACL_TRY(aclrtStreamWaitEvent(caller, l1_.aicore_done_event));
    ACL_TRY(aclrtRecordEvent(l1_.serial_tail, caller));
    l1_.prepare_tail_consumed = true;  // only after the complete sequence was enqueued
    l1_.has_serial_tail = true;
    return 0;
}
```

这是语义伪代码，不要求最终 API 函数名完全一致。真正实现必须对每个 `ACL_TRY` 标记“此前是否已经 enqueue”，以决定普通返回错误还是 poison context。伪代码中故意没有 capture query 和 `rtStreamAddToModel`：这不是省略的工程细节，而是 L1 ABI 的硬约束。

### E.5 为什么 host 先 enqueue AICPU 仍不等于越过算子边界

host API 调用只负责 enqueue，不会等待 device 执行。caller stream 上先 record start、再 enqueue AICPU；host 随后立刻在 hidden stream enqueue wait/start/AICore。device 侧：

- hidden AICore 只能在 handshake invalidation 和 start record 完成后开始；
- AICPU 位于同一个 start record 之后；
- 两者可以并行启动，符合当前 TRB handshake 需求；
- caller stream 在自己的 AICPU task 后再等待 AICore done，因此 downstream 同时等待两边。

不能把 start event record 放到 AICPU 完成之后，否则 hidden AICore 无法与 AICPU 并行，AICPU scheduler 等待 AICore handshake 时会死锁。

也不能把 hidden AICore 的 wait/start 删掉或在 capture 时略过。“host 先调用某个 enqueue API”和“device task 可在 caller predecessor 之前执行”是两件事；真正的 device 语义由 stream/event 依赖决定。本协议允许 AICPU/AICore 在 start 后竞速，但绝不允许任何一方在 start 前抢跑。

### E.6 多 caller stream 的fail-closed规则

仅用hidden AICore stream的FIFO不够：两个caller stream上的AICPU task仍可能重叠并同时reset/use shared arena。但不能通过“下一stream等待上一 `serial_tail`”自动串行，因为capture stream等待capture外record的event会触发runtime capture-isolation错误。

当前规则是：

1. 上一次caller stream在AICPU task之后wait AICore done，并在算子出口record serial tail；因此tail complete代表该次AICPU与AICore均已完成；
2. 下一调用仍在同一raw stream时依赖stream FIFO，不额外wait/query；
3. 下一调用切换raw stream时，在任何新task enqueue前非阻塞query上一tail；not-ready直接返回busy/invalid-state，context仍可使用，调用方外部同步后重试；
4. query为complete时允许切stream，但不enqueue旧tail wait，避免把capture外event导入图；
5. host mutex只防止host enqueue序列交叉，不代表device task已经完成。

这使普通eager跨stream在调用方已经排空前序工作时可用，并对未排空情况fail-closed。它不能检测capture clone或未来graph replay：graph replay不重新进入PyPTO host代码，且host查询原public event只看到先前eager generation。因此两个graph并发、graph→eager或任意capture后的换流仍由v1外部quiescence契约约束。

### E.7 ACLGraph capture/replay 时 PyPTO 实际发生什么

**capture 前：** Python 调用 `pypto_init/prepare/warmup`，runtime 分配资源并加载 binary；用户外部 synchronize。

**capture 中：** Python wrapper 调用一次 native `l1_launch`。ACLGraph 从 caller stream 的 capture 出发，依靠 start/done event 依赖自然发现并记录 hidden AICore 分支：caller 上的 memset/record/AICPU/wait/tail，hidden 上的 wait/AICore/record。PyPTO 不查询 capture 状态，不拿 model handle，不主动 add stream to model。

**replay 中：** 不再进入 Python/PyPTO host launch。ACLGraph 直接重放已经 capture 的 runtime tasks；因此所有 task 中引用的 context、events、streams、Runtime、KernelArgs、workspace、binary 和 tensor storage 必须仍有效。

**graph 销毁后：** 只有调用方知道不会再 replay。PyPTO 无法从一次 unregister 推导 graph 已死亡，所以资源默认 pin 到显式 context close。

如果 ACLGraph 不能通过这个完整 fork/join 捕获 hidden 分支，则 Phase 0 失败。不允许把上面的 capture 描述改成“PyPTO 查询 model 并强制加入 private stream”。

### E.8 并发限制的可执行定义

v1 的“不并发”包含：

- 同一 context 不允许两个 host 线程同时进入 enqueue；mutex 强制。
- 普通eager同stream依赖FIFO；不同stream只有在上一tail已完成时才允许继续，否则fail-closed，不自动插入跨 stream event wait。
- 一个 process/device 只允许一个 L1 context，wrapper/native registry 尽量强制。
- 同一 context 的两个 ACLGraph 不得并发 replay。
- L1 与同设备 L2/L3 全资源 run 不得重叠。
- 跨进程是否有人占用整张设备不由本次 API 检测。

ACLGraph replay 不经过 PyPTO host，所以“并发 replay”无法单靠 host mutex 检测。captured serial-tail event 可能实际把它们串行化，但在 Phase 0/正式测试证明多 graph 共享 event 的语义之前，不能把它升级成支持承诺。

### E.9 部分 enqueue 失败表

| 失败位置 | 已进入 device queue 的内容 | context 处理 |
| --- | --- | --- |
| validate 失败 | 无 | 直接报错，context 可继续用 |
| 换stream时query previous tail失败/not-ready | 无 | 直接报错，context可继续用；调用方外部quiescence后重试 |
| handshake memset 失败 | 前序 wait 可能已入队 | poison，避免下一调用跨过不完整序列 |
| start record 失败 | wait/memset 已入队 | poison |
| AICPU launch 失败 | start 已记录，AICore 尚未 launch | poison；不能自行 sync/reset |
| hidden wait/AICore launch 失败 | AICPU 可能已运行 | poison，异步错误由外部同步暴露 |
| done record/caller wait 失败 | AICore 可能已运行但 downstream 无完整依赖 | poison，禁止继续 enqueue |
| tail record 失败 | 本次 op 可能完成但下次无法安全排序 | poison |

poison 后 `l1_launch/prepare` 全部拒绝。`close()` 仍要求调用方先完成 graph/stream teardown；它不是故障恢复同步点。

## 附录 F：考虑过但明确不采用的方案

### F.1 直接复用 `simpler_run`，只删除 `sync_run_streams`

不采用。`simpler_run` 仍会构造 `OnboardNativeRunState`、provision per-run resources、staging tensor、创建 executor thread、分配/copy device Runtime/KernelArgs，并依赖 finalize 做 copy-back/free。删掉最后一个 sync 会让这些资源在 device 尚未使用完时提前回收。

### F.2 把 `simpler_prepare_run/launch_run` token 暴露给 ACLGraph

不采用。native-run token 需要 host poll/wait/finalize 才完成生命周期，而 replay 不回到 PyPTO host。它适合 L3 progress loop，不适合普通 graph operator。

### F.3 对外暴露 AICPU stream 和 AICore stream

不采用。L1 应表现为一个 caller-stream op；双 stream 是 PyPTO executor 的内部实现。暴露会让用户承担 event 拓扑和错误清理，违背目标。

### F.4 AICPU 继续使用 PyPTO 内部 stream

明确不采用，不只是“首选顺序”问题。AICPU 是本次 orchestration 主 task，放 caller stream 才能自然继承 torch_npu taskQueue、capture 和上下游依赖；hidden stream 仅承载必须在单算子内并行的 AICore executor。

特别禁止历史 `pto2/pypto` 的组合：把 private AICPU ctrl/scheduler stream 加入 capture model，并在 capture 时跳过 caller-stream pre-dependency，以便 orchestrator 早于 caller stream 上其他 task 启动。它虽可隐藏 orchestration 延迟，但会让 device 行为越过单算子 entry，同时使 eager/capture 语义分裂。后续需要此类性能收益时，只能在 `host_build_graph` 的图级调度下重新设计。

### F.5 每次 launch 把整个 Runtime async memcpy 到同一 device buffer

不采用。连续 host launch 会遇到 host source buffer 生命周期、同一 device destination 覆盖和 graph replay 参数更新问题；也会把不必要的大块 H2D 节点放入图。动态参数应由 WithHostArgs 快照交给 AICPU，persistent Runtime 不应每次重写。

### F.6 PyPTO 自建一个固定大小 device task-args pool

不采用。AICPU host args 已由 CANN runtime 的 completion-aware pool 管理；AICore 在新设计下不需要 per-call device args。自建 pool 既看不到真实 task completion，又容易错误依赖内部 launch 数上限。

### F.7 立即把 AICore binary 全迁到 public `aclrtRegisterBin`

不作为 v1 前置条件。长期看统一 public API 值得单独评估，但这会扩大 binary loader/ABI 变更面。首期保留 `rtRegisterAllKernel + rtKernelLaunchWithHandleV2`，把 capture 可行性列为 Phase 0 gate。

### F.8 AICPU 继续用 `rtsLaunchCpuKernel`

L2/L3 保留；L1 不采用。L1 要利用 `aclrtLaunchKernelWithHostArgs` 的普通 launch 参数快照和 ACLGraph 形态，减少 PyPTO 自己管理参数 lifetime 的责任。

### F.9 每次 launch 全量 reset workspace/Runtime

不采用。TRB 内部已有 reset，host 只需 invalidation stale handshake。全量 memset 增加图节点和带宽，并可能清除本应持久的 descriptor。

### F.10 v1 同时暴露外部 workspace

不采用。内部共享 workspace 在无并发前提下足够；当前最重要的是 prepare-time pin 和 launch-time no-allocation。外部 workspace size/query/caller ownership 留给后续独立设计。

### F.11 为兼容 dynamic shape 在每次 launch 重新 prepare

不采用。v1 operator metadata 固定。编译产物内部支持 dynamic 并不意味着 L1 runtime 要在 graph replay 时重新 tiling/prepare；真正的动态参数更新后续单独设计。

### F.12 在 PyPTO 内检测 ACLGraph capture/replay

不采用。capture-aware 分支会让 eager 与 graph 形成两套行为，也无法在 replay 时执行 host 逻辑。所有 launch API 本身必须可 capture，PyPTO 始终走同一路径。

### F.13 在 launch/close 中做一次“保险同步”

不采用。任何内部 sync 都破坏普通异步 op 形态，可能阻断 ACLGraph capture，还会让 PyPTO越权等待调用方 stream。warmup 和 close 前 quiescence由调用方显式负责。

### F.14 用 simulator 先证明 stream 改造正确

不采用作为语义证据。simulator 不能复现真实 ACLGraph、RTS stream/event、AICPU/AICore 并行和 task arg recycle；它最多用于 host state machine UT，onboard probe 是硬门槛。

### F.15 查询 capture model 并用 `rtStreamAddToModel` 强制挂载 hidden stream

不采用，也不作为 event-based capture 失败后的回退方案。这会要求 PyPTO 感知 capture、持有 runtime model handle 并主动修改图的 stream 集合；更重要的是，它容易与“跳过 caller entry dependency”绑定，重新引入历史 early-orchestrator 语义。

新 L1 只允许 ACLGraph 沿 caller record -> hidden wait -> hidden record -> caller wait 的完整 event 闭环捕获 hidden AICore branch。如果目标 CANN 版本不支持这个模式，结论是当前 L1 架构的 Phase 0 门槛未通过，而不是授权实现者降级单算子边界。

## 附录 G：可直接执行的详细实施步骤

正文第 12 节给出了摘要顺序；本附录保留每一步的前置条件、具体改动、验证和失败处理。阶段不得跨越 Phase 0 硬门槛并行推进。

### G.0 Phase 0：最小 onboard capture probe

**目标：** 在不改完整 PyPTO runtime 前，证明所依赖的 CANN/ACLGraph 原语组合可用。

**建议新增：**

- `tests/st/runtime/l1/probe_l1_multistream_capture.py`
- `runtime/tests/st/l1/` 下最小 native helper（若 Python 无法直接发起所有 RT API）

**Probe A：WithHostArgs 参数快照**

1. 复用现有 `LoadAicpuOp` 得到一个测试 AICPU `rtFuncHandle`。
2. 在 host 栈构造包含一个 device pointer 和几个 scalar 的固定 struct。
3. 调用 `aclrtLaunchKernelWithHostArgs` 后立刻覆盖 host struct。
4. 不在两次 launch 之间同步，连续 enqueue 多份不同参数。
5. 最后由测试外部 synchronize，验证每份 task 消费自己的快照。

**Probe B：双 stream capture**

1. 使用 torch_npu current/capture stream 作为 caller stream。
2. prepare 前创建一个 hidden stream 和 start/done events。
3. caller stream 上 enqueue AICPU test task，hidden stream 上 enqueue AICore test task。
4. capture caller record → hidden wait/launch/record → caller wait，仅依赖 event fork/join 让 ACLGraph 捕获 hidden branch。
5. probe 的 runtime shim 将 capture query、model-handle 获取和 `rtStreamAddToModel` 设为禁止 API；任一调用立即使 probe 失败。
6. 图前后各放一个可观察的普通 NPU op，证明上下游依赖。
7. replay 多次并改变 graph input buffer 内容。

**Probe C：mixed launch API**

1. AICPU 使用 public `aclrtLaunchKernelWithHostArgs`。
2. AICore 使用当前 `rtRegisterAllKernel + rtKernelLaunchWithHandleV2`。
3. 两个 task 在同一 graph 中 capture/replay。
4. 记录 capture、instantiate、replay、external synchronize 的准确 error code。

**Probe D：event generation/reuse**

1. 同一 start/done/tail event 在同一个 capture 中使用一次。
2. 同一 graph 连续 replay。
3. 两个 graph 顺序 replay并共享同一 context events。
4. warmup stream 与 capture stream 不同。
5. 验证没有等待旧 generation、提前放行或死锁。

**Probe E：handshake invalidation**

1. 只清 `aicore_done`/明确 invalidation region。
2. 连续运行同一个最小 TRB executor。
3. 验证旧 physical core report 不被误认，本次 worker 数和 core type 正确。

**Probe F：单算子 entry/exit 边界**

1. 在 caller stream 上放置一个可控延迟的 predecessor AICore task，完成时写入 entry marker。
2. hidden AICore test task 在启动时读 entry marker；若 marker 未就绪则报错，并通过 runtime trace/时序确认 hidden task 没有提前占核。
3. 让 hidden AICore task 在末尾延迟写入 exit marker，caller stream 上的 immediate successor 必须观察到 marker，证明 done join 不可越过。
4. 同一套检查分别在 eager 和 ACLGraph replay 下执行，两者的 runtime task/event 序列除 graph 自身节点容器外不得分叉。
5. 记录 AICPU 只在 caller stream，hidden stream 上只有 AICore wait/launch/record，不存在 private AICPU task。

**Phase 0 通过条件：**

- 上述 probe 均在至少一个 A2/A3 和一个 A5 环境给出正确结果；
- launch/capture path 内无 PyPTO stream sync；
- launch/capture path 内无 capture query、capture/model handle、`rtStreamAddToModel` 和 early-launch mode；
- caller predecessor -> start -> hidden AICore -> done -> caller successor 的边界在 eager/capture 中都成立；
- 参数不会串包；
- event replay 不错代；
- mixed API 可被图接受。

**Phase 0 失败处理：** 停止实现主线，记录失败原语。若只有 `rtKernelLaunchWithHandleV2` 不可 capture，再单独评估 AICore public binary/launch 迁移；不要先写一半 L1 state 再用同步规避。若 event fork/join 不能自然捕获 hidden stream，同样停止，不允许用 `rtStreamAddToModel` 恢复历史 capture-aware/early-launch 路径。

### G.1 Phase 1：增加独立 L1 ABI 和 execution mode

**修改文件：**

- `runtime/src/common/worker/pto_runtime_c_api.h`
- `runtime/src/common/platform/onboard/host/c_api_shared.cpp`
- sim/HBG 对应 C API 实现或 shared weak/strong stubs
- `runtime/src/common/worker/chip_worker.{h,cpp}`

**具体改动：**

1. 新增 `simpler_l1_supported`、`simpler_l1_init`、`simpler_l1_prepare_callable`、`simpler_l1_launch` symbols。
2. 在 `DeviceRunnerBase` 增加不可逆 execution mode：`Uninitialized -> L2Owned` 或 `Uninitialized -> L1Borrowed`。同一 context 不允许两种 mode 混用。
3. `simpler_init` 继续选择 `L2Owned`，所有旧调用不变。
4. `simpler_l1_init` 选择 `L1Borrowed`，检查当前 ACL device/context 与请求 device 一致。
5. L1 不执行 `aclInit` ownership、`rtDeviceReset` 或 `aclFinalize`；只创建明确由 L1 context 拥有的资源。
6. `finalize_device` 根据 mode 分流。L2/L3 继续现有 force-reset/recovery；L1 只走 borrowed teardown。
7. `ChipWorker` 动态加载所有 L1 symbols，但只有显式 L1 init 才使用；symbol 在 unsupported variant 也存在。
8. `ChipWorker::~ChipWorker()` 当前自动 `finalize()`。L1 mode 不能在未知 graph 生命周期下盲目 free；高层 wrapper 必须强制显式 close。C++ destructor 在未显式 close 时应至少记录 fatal-level lifecycle error，必要时选择保守泄漏 L1 pinned resources而不是制造 use-after-free。
9. L1 state/API 中不引入 `is_capture`、`capture_model`、`launch_early_mode` 或等价字段；这些字段的出现本身就应触发架构评审。

**单测：**

- L1 init 后调用 L2 run 拒绝；L2 init 后调用 L1 API 拒绝。
- mock 记录 L1 init/close 从未调用 reset/finalize ownership API。
- mock 记录 L1 init/prepare/launch 从未调用 capture query、model API 和 `rtStreamAddToModel`。
- sim/HBG support query 为 false，其他入口返回稳定错误码。
- 重复 init、null ctx、device mismatch、close twice 的状态行为明确。

### G.2 Phase 2：实现 `L1ExecutionState`

**新增文件：**

- `runtime/src/common/platform/onboard/host/l1_execution_state.h`
- `runtime/src/common/platform/onboard/host/l1_execution_state.cpp`

**建议成员：**

```cpp
class L1ExecutionState {
public:
    L1Phase phase;  // New/Initializing/Collecting/ReadyEnqueued/Sealed/Poisoned/Closing/Closed
    int device_id;
    rtStream_t aicore_stream;
    rtEvent_t prepare_tail;
    rtEvent_t start_event;
    rtEvent_t aicore_done_event;
    rtEvent_t serial_tail;
    bool has_prepare_tail;
    bool prepare_tail_consumed;
    bool has_serial_tail;
    rtStream_t last_caller_stream;
    L1PersistentDeviceState device;
    L1WorkspaceState workspace;
    std::array<L1CallableState, MAX_REGISTERED_CALLABLE_IDS> callables;
    std::mutex enqueue_mutex;
    bool any_launch_enqueued;
    bool poisoned;
};
```

**具体改动：**

1. init 创建 hidden AICore stream 和全部 events；launch 只复用。
2. 将 allocation accounting 与现有 `MemoryAllocator` 对接，但 L1 close 路径不能触发 device reset。
3. 建立 process-local per-device live context registry，拒绝第二个 L1 context。
4. 建立 poisoned sticky state，保存第一个 partial-enqueue error code/step，后续错误信息引用原始失败点。
5. 所有资源初始化使用 RAII rollback；只有成功发布到 state 后才进入下一 phase。
6. state 的 device pointer 字段在 seal 后通过 debug assertion 禁止修改。

**单测：** fault injection 覆盖每一个 stream/event/allocation 创建步骤，验证 rollback 只释放已拥有资源且不触碰 caller device/context。

### G.3 Phase 3：prepare-time persistent resources

**修改文件：**

- `runtime/src/common/platform/onboard/host/device_runner_helpers.{h,cpp}`
- `runtime/src/common/platform/onboard/host/device_runner_base.{h,cpp}`
- `runtime/src/{a2a3,a5}/platform/onboard/host/device_runner.{h,cpp}`
- `runtime/src/{a2a3,a5}/runtime/tensormap_and_ringbuffer/host/runtime_maker.cpp`

**具体改动：**

1. 保留现有 `KernelArgsHelper` 给 L2/L3；新增 `L1PersistentArgsHelper`，不要让一个 helper 同时承担 per-run 和 context-lifetime 两套状态。
2. 根据 `programs=[...]`/callable metadata 计算最大 block/core、ring sizing、GM SM、arena 和 workspace 需求。
3. prepare 一次性获取 regs/FFTS 表、构建 prebuilt arena、分配 Runtime device context 和 device `KernelArgs`。
4. 初始化 v1 DFX 字段为 0，并在 native config validate 时拒绝打开。
5. executor AICore ELF 在 prepare 明确注册，不能把“首次 `rtKernelLaunchWithHandleV2` 懒注册”留进 capture。
6. 所有 host-to-device 初始化 copy 在 capture 前完成；launch 中只允许 handshake async memset 和 kernel/event enqueue。
7. 记录每种 allocation 的地址和 size，UT/ST 可比较 warmup/capture/replay 前后地址是否稳定。

**重要检查：** 当前 `launch_aicore_kernel` 内包含 lazy `rtRegisterAllKernel`。L1 必须拆出 `ensure_l1_aicore_registered()` 并在 prepare 调用；L2 保持 lazy 行为或也复用 helper，但不得改变其时序/性能契约。

### G.4 Phase 4：callable 注册与 append-only binary state

**修改文件：**

- `runtime/src/common/platform/onboard/host/c_api_shared.cpp::simpler_register_callable`
- `runtime/src/common/platform/onboard/host/device_runner_base.{h,cpp}`
- `runtime/src/{a2a3,a5}/runtime/tensormap_and_ringbuffer/host/runtime_maker.cpp`

**具体改动：**

1. 把当前注册拆成可共享的 host parse/upload 部分，以及 L2 sync device-register、L1 async device-register 两个尾部。
2. L1 `record_device_orch_callable` 写 append-only table；保存 content identity、orchestration SO device address、entry/config 和 child `(func_id, addr)`。
3. 在写device state前检查callable ID/content冲突及同callable内func ID重复/越界，避免失败后部分注册。不对不同callable中数值相同的func ID做全局冲突判定。
4. L1 device register task使用传入 caller stream，通过 WithHostArgs 或已验证的 capture-compatible普通 launch enqueue。
5. record context-level `prepare_tail`；只有第一个完整成功enqueue的warmup/launch wait并消费该event，后续capture launch不再wait。
6. resource capacity 已 sealed 时，新 callable 若超出现有 arena/workspace/callable capacity直接失败；符合容量且不重绑定的 binary 可追加。
7. L1 unregister 不释放 device binary；只把 Python handle 标为不可再提交。

**验证：** 两个program共用相同child binary时地址去重；两个program都使用 `func_id=0` 但指向不同binary时均正确分发；同callable内重复func ID失败；注册后多次capture不增加HBM。

### G.5 Phase 5：新增 L1 AICPU entry 和 invocation ABI

**修改文件：**

- `runtime/src/common/aicpu_loader/host/load_aicpu_op.{h,cpp}`
- `runtime/src/{a2a3,a5}/platform/onboard/aicpu/kernel.cpp`
- `runtime/src/{a2a3,a5}/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp`
- `runtime/src/{a2a3,a5}/runtime/tensormap_and_ringbuffer/host/runtime_maker.cpp` 中 extra symbol 列表
- 相关 AICPU build/export/JSON 配置

**具体改动：**

1. 在 `KernelNames` 增加 `L1RunName = "simpler_aicpu_l1_exec"`。
2. 让 TRB build 报告/导出该 symbol；HBG 不实现语义但 host C API 已返回 unsupported。
3. `LoadAicpuOp` 增加按 name 取得已解析 handle并调用 `aclrtLaunchKernelWithHostArgs` 的方法；现有 `LaunchBuiltInOp/rtsLaunchCpuKernel` 不改。
4. 定义 `L1AicpuInvocationArgs` 的 ABI version、size、alignment 和 compile-time static assertions；A2/A3、A5 host/device 编译都包含同一 wire header。
5. platform `kernel.cpp` 的新 entry 只负责过滤/线程 index、参数校验并进入 TRB `aicpu_execute_l1`。
6. `aicpu_execute_l1` 直接从 invocation 获得 callable descriptor 和 `ChipStorageTaskArgs`，不读取 `Runtime.orch_args_storage_`/`active_callable_id_`。
7. 抽取 L2/L1 共用的 scheduler init/run/deinit，不复制一份 scheduler 实现。
8. L1 entry 不写 invocation host pointer 到静态全局；只在本 AICPU task 生命周期内使用 runtime-owned args image。

**ABI 测试：** host/device `sizeof/offsetof/alignment`；坏 version/size/cid；最大 tensor/scalar count；scalar bit-exact；多个连续异步 invocation不串包。

### G.6 Phase 6：AICore persistent ABI 审计

**修改文件：**

- `runtime/src/{a2a3,a5}/platform/include/common/kernel_args.h`
- `runtime/src/{a2a3,a5}/platform/onboard/aicore/kernel.cpp`
- TRB scheduler/executor 中实际读取 Runtime 的位置

**具体改动：**

1. 列出 AICore entry 到第一次等待/开窗前读取的所有 `KernelArgs`/Runtime 字段。
2. 将它们分类为 prepare-stable、AICPU-published、纯 device-reset 三类。
3. 保证 prepare-stable 字段在 capture 前写好且不再修改。
4. 把 per-call 字段从 AICore早期读取路径移走。
5. 若仍存在必要动态字段，新增单独 cache-line-aligned `L1InvocationEpoch`，由 AICPU release publish、AICore acquire wait；不要改变 tensor/scalar 快照方案。
6. AICore kernel entry function name/launch ABI原则上保持不变，降低 ptoas/binary 影响。

**验证：** 在 AICPU/AICore 启动次序上做扰动测试，分别让 AICore先到和 AICPU先到；结果和 handshake 均正确。

### G.7 Phase 7：direct-device binder

**修改文件：**

- `runtime/src/{a2a3,a5}/runtime/tensormap_and_ringbuffer/host/runtime_maker.cpp`
- `runtime/src/common/task_interface/` 中需要的 validation helper

**建议接口：**

```cpp
int bind_l1_device_args(const L1PreparedSignature &signature,
                        const ChipStorageTaskArgs &input,
                        ChipStorageTaskArgs *validated_snapshot);
```

**具体规则：**

- tensor count/scalar count 必须匹配 prepared signature；
- device address 对非零 size 不得为 0；
- dtype、rank、shape、stride/contiguity 契约匹配；
- input/output direction 只用于校验，不触发 copy；
- `child_memory` 不再是 L1 透传的特殊后门，所有 tensor 都按 caller device storage 处理；
- zero-size tensor 的地址规则明确；
- scalar 按 dtype 转成稳定 64-bit bit pattern，不做数值重新解释；
- 不调用 `device_malloc/copy_to_device/copy_from_device/free`；
- 不建立 L2 `tensor_leases_`。

L2 `stage_device_args()` 保持原样，不能在其中添加一个容易漏判的 `if (l1)`；用独立 binder 让 code review 可以证明 no-staging。

### G.8 Phase 8：实现固定 stream/event launch 序列

**修改文件：**

- `runtime/src/common/platform/onboard/host/l1_execution_state.cpp`
- `runtime/src/common/platform/onboard/host/device_runner_base.cpp`
- `runtime/src/{a2a3,a5}/platform/onboard/host/device_runner.cpp` 的 arch launch-shape helper
- `runtime/src/common/platform/onboard/host/c_api_shared.cpp::simpler_l1_launch`

**具体改动：**

1. launch 开始时执行全部不产生 task 的 validation；失败时 queue 必须未改变。
2. 在 context mutex 内构造 invocation snapshot并执行 E.4 固定 enqueue 顺序。
3. 使用 `aclrtMemsetAsync` 清明确的 handshake invalidation region。
4. AICPU WithHostArgs 使用 caller stream；AICore handle launch 使用 hidden stream。
5. `start_event` 必须在 caller 的 predecessor 和 handshake invalidation 之后 record，hidden stream 上的 AICore launch 必须位于对应 wait 之后。
6. `aicore_done_event` 必须在 hidden AICore launch 后 record，caller 必须在 downstream/tail 前 wait；不以 AICPU ack 代替。
7. 维护 `has_prepare_tail/prepare_tail_consumed/has_serial_tail/last_caller_stream`；只在第一个完整序列enqueue成功后标记prepare tail consumed。同stream依赖FIFO，换stream仅在pre-enqueue阶段query已完成的tail，禁止enqueue旧tail wait。
8. 实现中不得查询 capture 状态、保存 capture/model handle、调用 `rtStreamAddToModel` 或分支出 early-launch 序列。
9. 对每个 enqueue step 编号，部分失败时记录 step、原 error code、callable id并 poison。
10. API在最后一个record成功后立即返回，不调用host wait/poll/sync。唯一允许的host query是换raw stream时、任何新task enqueue前的非阻塞 `aclrtQueryEventStatus(serial_tail)`；它不是capture query。stream-wait-event是enqueue一个device依赖节点，不是host blocking wait。
11. launch 前后读取 allocator/stream-create/capture-model API counters 的 debug test hook，证明 steady-state 中 allocation/create/model-mutation 都为零变化。

### G.9 Phase 9：low-level Python binding

**修改文件：**

- `runtime/src/common/worker/chip_worker.{h,cpp}`
- `runtime/python/bindings/task_interface.cpp`
- `runtime/python/simpler/task_interface.py`

**具体改动：**

1. `_ChipWorker` 增加 `l1_init/l1_prepare_callable/l1_launch/l1_close` 或等价独立 `_L1Context` native class。
2. raw stream 通过 `uintptr_t` 进入 binding，禁止把 Python stream object 传到 simpler core。
3. native enqueue 期间释放 GIL；返回后不保存 Python args object地址。
4. Python wrapper 建立 opaque callable handle 与 native context owner id 检查。
5. 显式 `close()` 幂等；未 close 的析构路径给出强警告并选择安全策略。
6. L1 和现有 `ChipWorker` registry 分离，避免现有 L2 `unregister` 立即释放语义泄漏到 L1。

### G.10 Phase 10：PyPTO L1 API 与 torch_npu adapter

**建议新增/修改：**

- `python/pypto/runtime/l1.py`
- `python/pypto/runtime/task_interface.py`
- `python/pypto/ir/compiled_program.py`
- 独立 torch extension/adapter 目录及其 CMake/setup 配置

**`CompiledProgram` 增加只读 metadata：**

- static parameter signature；
- output indices/spec；
- callable identity和 assembled `ChipCallable`；
- runtime name/platform/backend；
- workspace/ring/resource requirements；
- 是否包含 v1 unsupported 功能。

**建议高层 API：**

```python
ctx = pypto.l1.pypto_init(
    programs=[compiled_a, compiled_b],
    device=device_id,  # mandatory; caller already made this torch_npu device current
    config=L1Config(),
)
op_a = ctx.operator(compiled_a)

op_a.prepare()                  # idempotent; outside capture
op_a(x, scale, out=y)           # warmup
torch_npu.npu.synchronize()     # caller-owned

graph = torch_npu.npu.NPUGraph()
with torch_npu.npu.graph(graph, stream=capture_stream):
    op_a(x, scale, out=y)
```

**torch adapter 职责：**

1. 导出nanobind `enqueue(queue_call, tensors, expected_device, op_name)`，不注册TORCH_LIBRARY/custom op schema；
2. 校验queue-call ABI、torch/torch_npu build/runtime version和NPU device；dtype/shape/output等签名检查由Python L1 wrapper和native双层负责；
3. 获取 `c10_npu::getCurrentNPUStream().stream(false)`；
4. 用 `OpCommand::RunOpApiV2` 把纯C++ deferred callback插入taskQueue；
5. 持有descriptor/Tensor到callback，并对default allocator storage做 `recordStream`；
6. callback调用raw-stream L1 API；Python wrapper返回调用方显式传入的 `out=`；
7. 不接管workspace、events、graph或external/custom storage owner。

首个ACLGraph ST使用显式 `out=`。`op(x)` 自动分配不在v1实现/支持面中；若未来需要，作为eager-only convenience独立设计。

### G.11 Phase 11：close、文档和旧路径回归

1. 实现 D.6 borrowed-resource teardown。
2. v1不实现context manager/`__exit__`：语法作用域结束不能证明ACLGraph不再replay或device已quiescent。只提供显式 `close()`，文档给出graph reset/destroy、外部同步、close的正确顺序。
3. L1 未显式 close 的 process teardown不应 crash；可以报告 pinned-resource leak。
4. 跑完整 L2 direct worker、one-shot L2、L3 persistent/one-shot、A2/A3/A5 TRB 回归。
5. 比较 L2/L3 的 API symbol、stream count、allocation和 timing，确认 L1 分支没有改变原有语义。
6. 把“跨算子提前 orchestration”记入 `host_build_graph` 后续 backlog，明确不通过 L1 early-mode 或 `rtStreamAddToModel` 实现。

## 附录 H：接口细节与 before/after 对照

### H.1 同步 L2 API 与异步 L1 API

**当前：**

```cpp
int simpler_run(DeviceContextHandle ctx, RuntimeHandle runtime,
                int32_t callable_id, const void *args,
                const CallConfig *config);
// internally prepare -> launch -> wait -> finalize
```

**新增，不替换：**

```cpp
int simpler_l1_launch(DeviceContextHandle ctx,
                      int32_t callable_id,
                      const ChipStorageTaskArgs *args,
                      void *caller_stream);
// validate + enqueue only; no RuntimeHandle, wait, poll or run-finalize
```

### H.2 per-run KernelArgs 与 persistent KernelArgs

**当前 L2/L3：**

```cpp
kernel_args.init_runtime_args(host_runtime, allocator);  // alloc + H2D
kernel_args.init_device_kernel_args(allocator);          // alloc + H2D
launch_aicore_kernel(run_stream, kernel_args.device_k_args_);
kernel_args.finalize_device_kernel_args();
kernel_args.finalize_runtime_args();
```

**L1：**

```cpp
// prepare, before capture
l1_args.prepare_once(stable_runtime, allocator);

// every launch
launch_aicore_kernel(hidden_stream, l1_args.device_k_args());

// explicit context close, after all graphs
l1_args.finalize_once();
```

### H.3 tensor staging 与 direct binding

**当前 L2 普通 host tensor：**

```text
host tensor address -> device_malloc -> H2D -> replace orch arg
                                      -> execute
host tensor address <- D2H <- device buffer <- validate/finalize/free
```

**L1 NPU tensor：**

```text
torch_npu tensor data_ptr -> validate metadata -> copy pointer value into AICPU invocation snapshot
                                             -> no allocation/copy-back/free
```

### H.4 AICPU launch

**当前 L2/L3：**

```cpp
rtsLaunchCpuKernel(func_handle, aicpu_num, internal_stream,
                   &launch_cfg, &cpu_args);
```

**L1：**

```cpp
aclrtLaunchKernelWithHostArgs(func_handle, aicpu_num, caller_stream,
                              nullptr, &invocation, sizeof(invocation),
                              nullptr, 0);
```

这个 launch 与 caller 之前/之后 task 的顺序直接由 caller stream FIFO 表达。它不是先 launch 到 private stream 再依赖 capture model 将其“挂入”本算子。

### H.5 current stream 隔离层

```text
PyTorch/torch_npu adapter
  c10_npu::getCurrentNPUStream().stream(false)
  tensor.data_ptr()
  taskQueue custom-op boundary
          |
          v raw stream + POD task args
Simpler L1 C ABI
  no torch dependency
```

### H.6 初始化入口

**当前一次性 L2：** init 创建自有 AICPU/AICore streams，register 同步 AICPU load，run sync，close reset/teardown device。

**L1：** `pypto_init` 借用current device，创建一个hidden AICore stream及prepare/start/done/tail events，但它没有stream入参，不在init阶段将register task排入某个caller stream。`ctx.prepare()`（或普通eager的auto-prepare）才取当前stream，prepare persistent resources并异步enqueue init/register；warmup后由用户同步；close只释放自有资源。init不创建private AICPU run stream，也不预启动orchestrator/executor等待未来调用。

## 附录 I：完整测试矩阵

测试编号用于实施和评审跟踪，不要求最终 pytest 名字完全相同，但每个行为必须有对应覆盖。

### I.1 Host state/ABI 单测

| 编号 | 场景 | 关键断言 |
| --- | --- | --- |
| UT-001 | `simpler_l1_supported` | TRB onboard true，sim/HBG false，所有 variant symbol 可解析 |
| UT-002 | L1 init 正常 | mode 为 borrowed；创建一组 hidden stream/events；不调用 reset/aclFinalize |
| UT-003 | device mismatch | 在任何 allocation/enqueue 前失败 |
| UT-004 | L1/L2 mode 混用 | 双向都拒绝；旧 L2 context 行为不变 |
| UT-005 | init fault injection | 每个资源创建点失败均只回收已拥有对象 |
| UT-006 | close twice | 第一次释放，第二次幂等，不触碰 caller stream/device |
| UT-007 | context state | NEW/INITIALIZING/COLLECTING/READY_ENQUEUED/SEALED/POISONED/CLOSING/CLOSED 迁移符合 E.1；Closing失败可重试且拒绝dispatch |
| UT-008 | invocation ABI | A2/A3、A5 host/device `sizeof/alignof/offsetof` 相同 |
| UT-009 | invocation version | bad version、bad size、null descriptor被 AICPU entry 拒绝 |
| UT-010 | max args | tensor/scalar capacity边界准确，越界在 host validate 失败 |
| UT-011 | graph-transparent state | L1 state/ABI 无 `is_capture`、model handle、early-mode 字段；API trace 无 capture query/`rtStreamAddToModel` |

### I.2 资源与 callable 单测

| 编号 | 场景 | 关键断言 |
| --- | --- | --- |
| UT-020 | 多 program capacity | shared resource按合法最大 sizing构建，不按错误简单求和/取 max |
| UT-021 | seal 前增长 | prepare 可增长但最终 graph-visible 地址只在 seal 后发布 |
| UT-022 | seal 后资源不足 | 直接报错；allocator/address/state均不改变 |
| UT-023 | 相同 callable identity | prepare 幂等，binary/device descriptor地址相同 |
| UT-024 | callable ID 冲突 | 不同 identity 重用 ID 在任何 device 修改前失败 |
| UT-025 | callable-local func ID | 同callable内重复/invalid func ID被拒绝；不同callable的同数值func ID可以指向不同binary |
| UT-026 | child binary 去重 | 相同 content identity 重用地址；没有重复 HBM增长 |
| UT-027 | unregister | host handle失效，但 pinned device state不释放 |
| UT-028 | close | callable/binary/workspace按 D.6 顺序回收 |
| UT-029 | AICore lazy register | L1 prepare后 launch不再调用 `rtRegisterAllKernel` |

### I.3 direct binder 单测

| 编号 | 场景 | 关键断言 |
| --- | --- | --- |
| UT-040 | 正常 input/output | output/input device地址逐 bit 保持，不 staging |
| UT-041 | scalar types | int/uint/float/bool/ctypes 按声明 dtype bit-exact |
| UT-042 | tensor count mismatch | launch 前失败 |
| UT-043 | scalar count mismatch | launch 前失败 |
| UT-044 | shape/dtype/stride mismatch | v1 静态契约报清晰错误 |
| UT-045 | null nonempty tensor | launch 前失败 |
| UT-046 | zero-size tensor | 按明确规则接受或拒绝，A2/A3/A5 一致 |
| UT-047 | wrong device | wrapper/native 均拒绝 |
| UT-048 | no allocator use | binder mock 的 malloc/copy/free 调用计数全为 0 |
| UT-049 | L2 regression | `stage_device_args` 仍进行原 H2D/D2H/lease 逻辑 |

### I.4 launch 序列单测

用 fake ACL/RT function table 记录每次调用：

| 编号 | 场景 | 预期序列/行为 |
| --- | --- | --- |
| UT-060 | 首次 launch | prepare-tail wait（若有）→ memset → start record → AICPU → hidden wait/AICore/done → caller wait/tail |
| UT-061 | 第二次同 stream | 不wait/query旧tail；依靠caller FIFO进入本次invalidation |
| UT-062 | 第二次不同 stream且旧tail not-ready | 在任何新enqueue前直接失败，context可重试，不允许AICPU overlap |
| UT-062A | 第二次不同 stream且旧tail complete | host query后继续，但不enqueue旧tail wait；标准warmup→sync→capture不会导入capture外event |
| UT-063 | steady-state | 0 alloc/free、0 stream/event create/destroy、0 sync/capture-query；同stream无event-status query |
| UT-064 | pre-enqueue validation error | 调用记录为空，context 仍可使用 |
| UT-065 | memset 失败 | context poisoned，记录准确 step/error |
| UT-066 | AICPU launch 失败 | hidden launch不得继续或按确定状态 poison；后续 launch拒绝 |
| UT-067 | AICore launch 失败 | context poison；不内部 sync/reset |
| UT-068 | done/tail 失败 | context poison；后续 launch拒绝 |
| UT-069 | poisoned close | 只在 caller满足外部 quiescence契约后释放，不做恢复同步 |
| UT-070 | predecessor 入口 gate | start record 严格位于 caller 已有 task/invalidation 之后，hidden wait 严格位于 AICore launch 前 |
| UT-071 | downstream 出口 join | hidden done 严格位于 AICore launch 后，caller wait 严格位于 tail/downstream 前 |
| UT-072 | no private AICPU stream | WithHostArgs 的 stream 参数与 mandatory caller stream 逐 bit 相同，hidden stream 只接收 wait/AICore/record |
| UT-073 | forbidden capture APIs | fake runtime 中 capture query、model-get、`rtStreamAddToModel` 调用计数全为 0 |

### I.5 Onboard eager ST

| 编号 | 程序 | 覆盖点 |
| --- | --- | --- |
| ST-E-001 | 单 input/output add | 最小 `@pl.program`、explicit `out=`、数值正确 |
| ST-E-002 | 多 input + scalar | WithHostArgs scalar/tensor snapshot |
| ST-E-003 | 多 output | 参数方向和多个 output地址 |
| ST-E-004 | 多 child kernel | AICPU 生成多个 task，child binary mapping稳定 |
| ST-E-005 | TRB workspace | 内部共享 workspace、arena reset/reuse |
| ST-E-006 | 连续异步不同地址 | 不同步 enqueue N 次，最后统一同步，不串包 |
| ST-E-007 | 连续异步不同 scalar | 每次结果对应本次 scalar，不被下一 host call覆盖 |
| ST-E-008 | caller stream 切换 | 前序未完成时fail-closed；外部同步后query tail complete并换流，不导入旧event wait |
| ST-E-009 | address stability | prepare/warmup/N 次 launch 后所有 persistent地址不变 |
| ST-E-010 | memory stability | warmup后重复调用 HBM committed值不持续增长 |
| ST-E-011 | A2/A3 | 核心 eager suite在 910B 系列 onboard 通过 |
| ST-E-012 | A5 | 同一 API/测试在 A5 onboard 通过 |
| ST-E-013 | 延迟 predecessor | caller 上前置 AICore task 完成后 hidden AICore 才启动，无抢核死锁/早读 |
| ST-E-014 | 延迟 AICore tail | immediate caller successor 只在 hidden done 后观察到最终输出 |

连续异步测试必须让 host 参数容器尽快离开作用域或被覆盖，专门复现“下一次调用提前覆盖”的风险，而不是每次都保留 Python list 直到同步。

### I.6 ACLGraph ST

| 编号 | 场景 | 关键断言 |
| --- | --- | --- |
| ST-G-001 | 单 L1 op capture/replay | warmup后 capture 成功，N 次 replay结果正确 |
| ST-G-002 | `pre_op -> L1 -> post_op` | caller stream上下游依赖完整 |
| ST-G-003 | input 内容变化 | graph tensor地址固定，replay前 copy新内容，结果更新 |
| ST-G-004 | scalar capture | 固定 scalar 与图语义一致；不要求 PyPTO动态 patch |
| ST-G-005 | 多 child kernel graph | hidden AICore stream/task sequence完整 replay |
| ST-G-006 | workspace graph | 多次 replay 后 shared arena正确 reset，不读旧 task |
| ST-G-007 | 一个图两个 L1 op | 同 context、同 workspace按 captured tail顺序复用 |
| ST-G-008 | 两个图顺序 replay | 共享 context events在顺序 replay下 generation正确 |
| ST-G-009 | warmup/capture 不同 stream | warmup外部同步后tail query通过；capture不包含对capture外serial tail的wait |
| ST-G-010 | replay stress | 次数由环境/CI预算配置，不编码内部 launch上限 |
| ST-G-011 | graph lifetime | 调用方显式持有context和graph-bound tensors；external quiescence + `graph.reset()` 之前不close/free native state |
| ST-G-012 | close order | graph销毁+外部同步后 close不破坏 torch_npu后续 op |
| ST-G-013 | capture 前未 prepare | 清晰失败，不在 capture内隐式 allocation/load |
| ST-G-014 | unsupported DFX | capture前 host报错，不出现 collector task |
| ST-G-015 | capture entry boundary | 延迟 predecessor + entry marker 证明 hidden AICore 不早于 caller `start_event` |
| ST-G-016 | capture exit boundary | 延迟 AICore tail + exit marker 证明 post-op 不越过 caller `done_event` wait |
| ST-G-017 | event-only hidden capture | trace 显示 hidden branch 由 event fork/join 纳入图，无 capture query、model handle、`rtStreamAddToModel` |
| ST-G-018 | eager/capture 拓扑对照 | 除 graph 容器本身外，PyPTO 的 memset/event/AICPU/AICore 序列一致，无 capture-only early mode |

### I.7 并发和负面 ST

1. 两个 host 线程同时调用同一 op：host enqueue 序列不交叉，或第二个明确拒绝；不能出现部分 event交叉。
2. 同 process/device 创建第二个 L1 context：直接失败。
3. 同 context 两个 graph 并发 replay：作为 unsupported 测试记录当前 runtime 行为；若 wrapper能检测则拒绝，不能据偶然成功宣布支持。
4. L1 与 L2 worker 同设备重叠：文档负面契约，测试环境可在可控条件下验证拒绝/失败信息，不要求 runtime跨组件仲裁。
5. prepare 后改变 output shape/dtype：host拒绝。
6. context poisoned 后任何新 prepare/launch：拒绝并报告首个 poison原因。

### I.8 L2/L3 回归

至少运行：

- `runtime/tests/ut/py/test_task_interface.py`；
- `tests/ut/ir/test_compiled_program.py`；
- `tests/st/runtime/framework_and_models/test_compiled_program.py`；
- TRB A2/A3、A5 现有 run/register/prewarm/DFX tests；
- simpler L2 direct `ChipWorker.run`；
- L3 one-shot `execute_distributed`；
- L3 persistent `DistributedWorker`、pipeline prepare/launch/poll/finalize；
- callable register/unregister和 binary count tests；
- device failure/recovery tests，确认 L2 force-reset 行为未被 L1 borrowed mode削弱。

### I.9 测试中的禁止项断言

ACL/RT shim 或 trace 必须能断言 capture launch interval 内未出现：

- `rtMalloc/aclrtMalloc/device_malloc`；
- `rtFree/aclrtFree/device_free`；
- `rtStreamCreate/Destroy`、event create/destroy；
- `rtMemcpy` whole Runtime、tensor H2D/D2H；
- `aclrtSynchronizeStreamWithTimeout`；
- `aclrtSynchronizeDevice`；
- stream-capture status/model query（包括 `GetStreamCaptureInfo` 或等价 API）；
- `rtStreamAddToModel`/`RuntimeStreamAddToModel` 或任何等价的 capture-model stream attach；
- private AICPU run stream 上的 orchestrator launch；
- `launch_early_mode` 分支或为 capture 删除 caller-entry dependency 的任何节点；
- host poll/wait/finalize-run；
- `rtRegisterAllKernel` 或 AICPU binary load；
- callable register/dlopen；
- DFX collector init/teardown。

允许出现的 launch-time runtime 节点只有经过 Phase 0 确认的 event wait/record、handshake async memset、caller-stream AICPU WithHostArgs 和 hidden-stream AICore executor launch。trace 还必须证明 event 形成 caller predecessor -> start -> hidden AICore -> done -> caller successor 的完整闭环。

## 附录 J：仍需 onboard 事实确认的硬门槛

这些不是需要用户重新选择的设计问题，而是实现前必须由实验回答的 runtime 事实。

### J.1 `rtKernelLaunchWithHandleV2` capture

确认该内部 AICore launch 在目标 CANN 版本的 ACLGraph capture stream 中是否形成可 replay node。若失败，记录具体 API/error和图状态，再评估 public AICore binary/func handle launch；不能假设 A2/A3成功就自动代表 A5。

### J.2 `aclrtLaunchKernelWithHostArgs` 对 AICPU func handle 的行为

本地 CANN API 接收 `aclrtFuncHandle`，而当前 loader 保存 `rtFuncHandle`。需要在实际头文件/type ABI和 onboard调用中确认二者兼容、AICPU block count/thread语义正确、参数由 runtime复制而非只保存 host pointer。

### J.3 多 stream event capture 与复用

确认：

- hidden stream 能否仅因为 wait caller-captured `start_event` 而进入同一 graph，并通过 record `done_event` + caller wait 完成 join；
- record/wait API组合的支持范围；
- event重复 record在 graph replay中的 generation语义；
- prepare/warmup stream与 capture stream不同的行为；
- graph销毁前 event对象必须保持的生命周期。

这里的“加入同一 graph”只指 ACLGraph 通过 event dependency 自然捕获 hidden branch，不允许 probe 调用 capture query 或 `rtStreamAddToModel`。如果只有主动 add-to-model 才能工作，J.3/Phase 0 结论就是“不通过”，不能把该 API 写入 L1 正式路径。

### J.4 AICore entry 的动态字段读取

通过代码审计和延迟扰动确认 AICore启动早期是否读取 `active_callable_id_`、`orch_args_storage_` 或其他本次字段。如果读，按 C.5 改成 invocation/ready protocol；不做 host同步。

### J.5 handshake invalidation 最小区域

验证只清 `aicore_done` 是否对 A2/A3和 A5均足够；如果新 L1 persistent布局让其他 generation/status跨调用保留，列出每个字段及原因，形成明确 region。禁止以“全量清零能跑”结束分析。

### J.6 torch_npu taskQueue 接入形态

确认 wrapper 应使用的注册宏、current stream获取位置和 taskQueue callback边界。用户给出的 torchair `add_custom.asc` 是参考；本仓库本地 torch_npu/torchair实现用于确认版本一致性。这个适配只能影响 wrapper，不得改变 native L1 ABI。

### J.7 binary unload/close

确认目标 runtime 对 `rtRegisterAllKernel` handle 是否有安全且当前可用的 unregister/unload；若没有，文档和实现都明确 executor binary是 context/process lifetime pinned。不能调用未公开或未经验证的卸载 API。

### J.8 graph replay 并发与 shared event

v1 契约仍禁止并发 replay，但需观察两个图共享 context events时 runtime是否天然串行、报错或发生未定义行为。结果用于将来并发设计，不影响 v1 单序列验收。

## 附录 K：建议提交顺序

为了让每个提交可独立审查和回退，建议按以下顺序：

1. **Probe only**：加入 Phase 0 onboard probe和结论记录，不改正式 API；probe 必须证明 event-only hidden-stream capture 和 entry/exit 闭包，trace 必须证明无 capture query/`rtStreamAddToModel`/early mode。
2. **ABI/mode skeleton**：L1 symbols、unsupported stubs、borrowed mode、no-reset UT。
3. **Persistent state**：`L1ExecutionState`、hidden stream/events、prepare-once allocator和 fault-injection UT。
4. **Callable/resource prepare**：capacity freeze、binary pinning、async device register。
5. **AICPU invocation**：WithHostArgs loader、新 L1 entry、ABI测试。
6. **AICore persistent args**：字段审计、prepare-time register、必要 ready epoch。
7. **Direct binder/launch**：no-staging binder、固定 event序列、poison状态。
8. **Low-level Python**：raw stream binding、explicit close、handle ownership。
9. **PyTorch adapter**：current stream、taskQueue、explicit `out=` forward。
10. **Eager ST**：A2/A3、A5 basic/multi-kernel/workspace/async args。
11. **ACLGraph ST**：capture/replay、pre/post op、entry/exit 延迟 marker、event-only hidden capture、lifetime/stress。
12. **Convenience/output allocation（未来可选 backlog）**：不属于v1完成标准。只有在核心 graph路径通过后才能单独设计和加入；v1继续强制调用方预分配并显式传入 `out=`。
13. **Regression/docs**：完整 L2/L3 suite、API reference和限制说明。

不要把 Probe、native ABI、PyTorch wrapper和 ACLGraph tests压在一个无法定位问题的大提交里。

## 附录 L：实现和评审检查清单

### L.1 Native API

- [ ] L1 symbol在所有 variant可解析，unsupported返回明确。
- [ ] L1/L2 mode互斥，旧 ABI没有签名变化。
- [ ] caller stream是 mandatory native入参。
- [ ] L1 launch没有 `RuntimeHandle`、poll、wait、finalize-run概念。
- [ ] 所有 pre-enqueue validation在第一个 runtime task之前完成。
- [ ] partial enqueue后 sticky poison。

### L.2 Resource ownership

- [ ] device/context/caller stream从不由 L1销毁。
- [ ] hidden stream/events只在 init创建、close销毁。
- [ ] workspace/arena/KernelArgs/Runtime在 capture前固定地址。
- [ ] launch中没有 allocation/free或 lazy binary load/register。
- [ ] callable ID不重绑定；每个callable内func ID/address snapshot不变，不同callable可重复使用func ID数值。
- [ ] graph可能引用的 binary/device descriptor pin到 context close。
- [ ] L1 close不调用 reset/aclFinalize，不依赖隐式 sync。
- [ ] 第一项destructive teardown前进入粘性 `CLOSING`；任何释放失败后prepare/launch均fail-closed，但close可显式重试且仍保留device claim/DSO/handle ownership。

### L.3 Task args

- [ ] AICPU每次使用独立 WithHostArgs invocation snapshot。
- [ ] stack invocation在 API返回后不被 PyPTO保存。
- [ ] AICore只接收 persistent device `KernelArgs *`。
- [ ] per-call tensor/scalar/callable不写进可被下一 launch覆盖的共享 host/device args。
- [ ] child task arena只在 serial-tail之后 reset/reuse。
- [ ] 不存在与固定 launch数量绑定的自建 pool。

### L.4 Stream/capture

- [ ] AICPU在 caller stream，AICore在 hidden stream。
- [ ] 不存在 private AICPU run stream，不预启动 AICPU/AICore kernel 等待未来 invocation。
- [ ] start 依赖位于 caller predecessor 与 handshake invalidation 之后，并在 AICPU/AICore并行前建立，不产生抢核死锁。
- [ ] hidden AICore 只能在 wait start 之后启动，不能越过单算子 entry。
- [ ] downstream caller stream同时等待 AICPU自身顺序和 AICore done，不能越过单算子 exit。
- [ ] 同caller stream依赖FIFO；host调用换stream时在enqueue前非阻塞query上一operator tail，not-ready直接fail-closed，complete也不将capture外event wait导入图。
- [ ] capture时执行一次PyPTO host launch以录制runtime task；replay不再进入Python/PyPTO host callback。
- [ ] eager/capture 使用同一 launch 拓扑，没有 capture-only early mode。
- [ ] PyPTO不查询 capture状态，不获取/保存 model/graph handle，不调用 `rtStreamAddToModel`。
- [ ] hidden AICore branch 只通过 event fork/join 被 graph 捕获；不支持就停在 Phase 0，不降级边界。
- [ ] 调用方在所有graph销毁前持有context和graph-bound tensors；adapter只承担deferred callback lease与default allocator `recordStream`，不伪装成graph lifetime owner。

### L.5 Python/PyTorch

- [ ] simpler core不 link torch/torch_npu。
- [ ] taskQueue/current stream只在 adapter。
- [ ] Python L1 wrapper强制显式 `out=`；native低层对完整tensor descriptor/direction/pointer/device做独立强校验。
- [ ] graph核心测试使用预分配 `out=`。
- [ ] wrapper forward-only，`requires_grad=True` 明确拒绝，不声称autograd/alias schema。
- [ ] default torch_npu caching-allocator storage经 `recordStream` 保护；external/from-blob/custom allocator storage明确要求外部owner持有到最后真实stream/device use完成，capture时还覆盖graph销毁后的外部quiescence。
- [ ] explicit prepare/warmup/close文档和错误信息完整。
- [ ] `__del__` 不在未知 graph生命周期下释放关键资源。

### L.6 Test/compatibility

- [ ] Phase 0硬门槛在 A2/A3、A5都有结果。
- [ ] launch禁止项有可自动检查的 trace/counter。
- [ ] trace/counter 明确检查 capture query、model attach、private AICPU launch 和 early-mode 均为零。
- [ ] eager/ACLGraph 都有延迟 predecessor 和延迟 AICore tail 的 entry/exit 边界测试。
- [ ] 连续异步 args不串包。
- [ ] eager、multi-kernel、workspace、graph pre/post op均覆盖。
- [ ] replay stress不编码 runtime内部上限。
- [ ] L2 one-shot/reuse和 L3 persistent/pipeline回归通过。
- [ ] poisoned/close/device ownership负面路径通过。

## 附录 M：最终交付物

首期实现完成时应同时交付：

1. Phase 0 probe源码及 A2/A3、A5 结果记录，其中必须包含 event-only hidden-stream capture、entry/exit marker 和无 `rtStreamAddToModel` trace；
2. L1 native C ABI、borrowed execution mode和 persistent state；
3. TRB AICPU WithHostArgs entry与 AICore persistent args；
4. direct-device tensor binder和固定 stream/event launch；
5. `pypto_init`/operator/prepare/warmup/close Python API；
6. 独立 torch_npu current-stream/taskQueue adapter；
7. eager/ACLGraph onboard tests及 launch禁止项检查，包括无 capture query/model attach/private-AICPU early launch；
8. L2/L3 全量回归结果；
9. 用户使用说明：静态 shape、显式 warmup、无并发、graph先销毁再close、内部 workspace；
10. 后续设计 backlog：dynamic shape、外部 workspace、binary recycle、并发 invocation、public AICore binary API迁移；
11. 独立的 `host_build_graph` 性能优化 backlog：以完整图为调度边界提前展开 orchestration，不在 L1 中复活 early mode 或 `rtStreamAddToModel`。

任何只实现“传入 stream并删除 sync”、但仍在 launch 中 staging/alloc/free、仍让 AICore使用 per-run device args、或仍可能在 graph存活时释放 binary/workspace 的版本，都不满足本计划定义的 L1。同样，任何通过 capture query + `rtStreamAddToModel`、private AICPU stream 或 capture-only early mode 让 orchestrator 越过单算子入口的版本，即使数值正确或性能更好，也不满足本计划定义的 L1。

## 附录 N：第二阶段 `host_build_graph` L1 + ACLGraph 实现准入设计

### N.1 本附录的状态、范围与不变量

本附录定义的是**当前TRB L1完成之后**的第二阶段。到runtime `74d0ff65`为止，HBG native源码主链已经实现到per-task package、独立run entry和per-execution leader restore；但没有device 1证据，也没有开放capability/Python API。因此这里的“未完成”是指产品支持和设备验证未完成，不再是指device bridge/restore源码尚不存在。在本附录的P0门槛通过前：

- `host_build_graph` 的L1 capability继续返回unsupported；
- 不将HBG variable-length payload偷渡进当前TRB `L1AicpuInvocationArgs`；
- 不修改本计划已确定的TRB Python/C++高层API、workspace内部管理、静态shape、显式warmup和close契约；
- 不让HBG单算子通过capture query、`rtStreamAddToModel`、private AICPU stream或capture-only early launch越过单算子边界；
- 不依赖“大约2048次launch”、256 MiB args或任何CANN内部常量作为公开设计规格；
- 不因“PyPTO占用全部AICore”就把尚未完成的device task、captured graph package或external tensor storage当成可回收。

对外语义仍是一个普通AscendC单算子：调用方只传tensor/scalar/output和caller stream，不感知AICPU/AICore双stream、graph package、restore manifest或ACLGraph capture/model handle。host-build-graph提前展开orchestration的性能收益只能在HBG完整图语义内实现，不回流到旧PyPTO跨op抢跑方案。

### N.2 当前HBG源码对第二阶段的硬约束

#### N.2.1 当前“graph image”至少跨越四类device存储

| 区域 | 当前内容 | 第二阶段归类 |
| --- | --- | --- |
| GM SM | task descriptor、payload、slot state、completion flags、flow-control header | pristine source和mutable working copy都需覆盖 |
| runtime arena | `PTO2Runtime`、orchestrator/scheduler objects、ready queues、mailbox、layout storage，以及task-owned完整函数地址表与host task数 | pristine source和mutable working copy都需覆盖 |
| GM heap | HBG intermediate/output/workspace地址和必要的host-built initializer目标 | v1 context-wide mutable slot；只将有语义的initializer spans放入restore manifest |
| outer Runtime/KernelArgs/handshake | prebuilt arena base、GM base、worker handshake、device KernelArgs，以及仅供host build暂存的legacy function table | context-wide mutable/persistent execution state；scheduler不得把其中可被后续build覆盖的function table/task count当作invocation source |

源码证据：

- `runtime/src/a2a3/runtime/host_build_graph/host/runtime_maker.cpp` 的 `build_host_orchestration_image`：本地host SM、最终device base relocation和owning SM bytes；该函数在提交 `11b7a4b1` 后不再自行H2D；
- 同文件 `bind_callable_to_runtime_impl`：GM heap/SM/runtime arena三个region、本地host arena，并在SM与arena两份host image完成后进入显式同步H2D区域；
- `runtime/src/common/platform/onboard/host/device_runner_helpers.cpp:25-75`：outer `Runtime`和device `KernelArgs`分别分配/复制；
- `runtime/src/a2a3/runtime/host_build_graph/runtime/shared/runtime.cpp`：HBG仍将整个outer `Runtime` 作为context/device control image复制；
- runtime提交 `6228b481`：A2/A3与A5的pristine `PTO2Runtime`新增task-owned prebuilt invocation state，host build把完整1024项函数地址表和实际host task数深拷贝进runtime-arena image；AICPU boot先校验该state，scheduler只绑定恢复后的表，不再读取outer `Runtime::func_id_to_addr_`和 `Runtime::host_total_tasks`作为调用语义。提交 `ade00349` 随后将它统一为common `HbgPrebuiltInvocationState`，在保持8216-byte大小和table offset 24不变的前提下加入完整函数表hash与重算校验。

A5对应HBG文件当前与A2/A3同构；实现时不得只修一个arch而依赖复制粘贴假设。

#### N.2.2 working image会被执行消费

下列字段会在一次execution中被改写：

- `pto_shared_memory.h:83-139`：completion flags与completed watermark；
- `pto_runtime2_types.h:305-409`：payload early-dispatch atomics；
- `pto_runtime2_types.h:445-543`：task state、wake list、completed subtasks、next block index；
- `pto_scheduler.h:521-583, 825-866`：wake-list registration/drain、completion publication和subtask counters；
- `scheduler_cold_path.cpp:1121-1157`：boot classify写ready queue和wake list；
- `aicpu/aicpu_executor.cpp:260-299`：attach/wire、mailbox和device-local runtime fixup；
- `runtime/shared/pto_runtime2_init.cpp:390-397`：`runtime_destroy` 清scheduler/orchestrator/mailbox/SM-handle pointer。

`pto_shared_memory.h:241-247` 明确规定 `attach_populated` 不重置已填充内容。因此每次ACLGraph replay必须恢复working state，只有graph-lifetime地址pin绝不足够。

#### N.2.3 image绑定具体destination addresses

- `runtime_maker.cpp:517-529` 按具体 `device_sm/device_arena` 计算relocation delta；
- `pto_runtime2_types.h:193-203` 保存GM heap的绝对 `packed_buffer_base/end`；
- `pto_runtime2_types.h:364-381` 将具体Tensor buffer地址写入payload；
- `pto_runtime2_init.cpp:347-387` 根据具体SM/GM heap base初始化runtime和wiring；
- `runtime_maker.cpp:876-888` 将具体runtime arena base/offset写入outer Runtime。

所以source blob可放在任意地址作copy source，但其bytes已绑定一组确定working destination bases。`host_build_graph/docs/RUNTIME_LOGIC.md` 中的“position-independent”只适用于fanin ID等局部表示，不适用于完整image。

#### N.2.4 旧L2 H2D和native pipeline lease都不是capture-lifetime解法

- `DeviceRunnerBase::copy_to_device` 当前使用同步 `rtMemcpy`；
- SM source是 `run_host_orchestration` 局部 `std::vector`，runtime-arena source是bind局部 `DeviceArena`；
- HBG `PipelineContract` 将GM heap/SM/runtime image定义为 `HOST_PER_RUN`，depth为2，但 `PipelineSlotLease` 只从host prepare存活到该native run finalize；
- ACLGraph replay不重新进入PyPTO host prepare/finalize，所以native lease、一次tail event或“host launch已返回”都无法证明captured package可回收。

HBG L1源码现已改用fresh inline HostArgs package作为首选task-owned source候选，并由AICPU leader在每次execution恢复working slot；这没有改变上面的否定结论：旧L2同步H2D/local vector和native pipeline lease仍不能充当captured-node owner，而新路径自身仍需N.10的CANN lifetime实证。

### N.3 五层对象和所有权模型

规范性结论：**每次dynamic host build产生的graph必须作为该次HBG launch task/captured node的tiling-like payload被管理**，不得实现为context-wide、可被下一次host build原地覆盖的 `current_graph`。随task/node生存的是第3层不可变pristine source；在v1无并发契约下可被多个package顺序共享的只是第4层mutable execution slot和context workspace，且它们必须在每次replay前从当前node的pristine source恢复。

| 层 | 名称 | 规范性内容 | 写入者 | owner/生命周期 |
| --- | --- | --- | --- | --- |
| 1 | `HbgGraphPlan` | host canonical immutable representation；pristine SM/runtime-arena bytes、optional GM initializer spans、topology、tensor/scalar/function bindings、generation/hash、restore/relocation metadata | host builder一次生成，随后只读 | PyPTO本次build，至少到第2层生成完成 |
| 2 | `HbgSerializedLaunchBlob` | 一次性writable `[header \| inline payloads]`；所有offset/size已校验，pointer slot可被placeholder patch | serializer写，CANN launch可原地patch | PyPTO host，到runtime完成snapshot；精确时点由P0证明 |
| 3 | `RuntimeOwnedHbgPayload` | CANN device task args中的immutable pristine source。AICPU只将它copy/parse到working slot，绝不在source上就地运行scheduler | runtime args loader创建；PyPTO device代码只读 | eager task或captured node/model；capture lifetime是P0门槛 |
| 4 | `HbgExecutionSlot` | working GM SM/runtime arena/GM heap、outer Runtime、KernelArgs、handshake/mailbox、restore/completion generation。这是会被scheduler消费的可写状态 | AICPU restore、scheduler、AICore | PyPTO context；所有graph不再replay且device externally quiescent后才能释放 |
| 5 | `HbgLifetimeRoots` | binary generations、context workspace、streams/events、external tensor storage，以及external-source fallback使用的explicit package lease | prepare/register/caller | PyPTO context + caller/ACLGraph owner + runtime按各自资源负责 |

这五层的核心不变量是：

1. 第1层永远不被placeholder或device execution改写；
2. 第2层必须是从第1层deep-serialize的writable scratch，不得与plan cache共享同一pointer字段；
3. 第3层是pristine source，不就地变成scheduler working memory；
4. 第4层每次execution/replay前都从该node的第3层恢复；
5. 第3层由runtime持有也不能延长第4/5层指针对应资源的生命；context close仍需graph destroyed + external quiescence。

### N.4 WithHostArgs inline payload ABI候选

#### N.4.1 CANN已证机制

CANN当前实现提供下列源码证据：

- `acl_rt.h:732-735`：`aclrtPlaceHolderInfo{addrOffset, dataOffset}`；
- `api_c_kernel.cc:42-77`：WithHostArgs将host pointer、args size和placeholder传入CPU/non-CPU args descriptor，并要求H2D copy；
- `arg_loader.hpp:29-40`：`UpdateAddrField` 将 `hostArgs + addrOffset` 原地写为 `kerArgs + dataOffset`；
- `stars_arg_manager.hpp:127-147`、`stars_arg_manager.cc:165-182`：分配args copy、patch placeholder并复制bytes；
- `aicpu_stars.cc:53-59, 105-113`：capture task强制走args-copy路径；
- `aicpu_starsv2.cc:47-53`：arg handle挂到task并交给postprocess；
- GE `executor_utils.cc:269-318`、`op_task.cc:619-646`：AscendC tiling使用“pointer + inline tiling bytes”同类布局。

这些证据能证明“inline payload + placeholder”是有源码基础的P0候选，但不能替代graph lifetime和大小上限的板上证明。

#### N.4.2 已落地的host ABI、device bridge与仍待证明的runtime ownership

提交 `11b7a4b1` 将host侧ABI具体化；提交 `2873feae` 把实际scheduler task数纳入task-owned identity；`6228b481` 把device实际消费的1024项函数地址表与同一task数收进pristine runtime-arena state；`ade00349` 将该state抽成A2/A3、A5共用的 `HbgPrebuiltInvocationState`，并加入完整固定长度函数表hash。`18b1fde9`建立callable registration；`74d0ff65`把 `callable_id`加入invocation identity、将launch blob ABI minor升级为2，并把fresh scratch/placeholder接到独立HBG AICPU run entry。host strong build要求输入identity、callable-local函数表与arena snapshot精确匹配；AICPU leader在每次restore后再次把runtime-owned header与restored state交叉验证。当前固定大小为：`HbgExecutionBinding=64` bytes、`HbgInvocationIdentity=40` bytes、`HbgLaunchRegion=40` bytes、`HbgLaunchBlobHeader=160` bytes、`HbgRestoreCommit=64` bytes、`HbgPrebuiltInvocationState=8216` bytes。在HBG capability开启前仍可按device probe结果做versioned演进，但任何变更必须同步magic/version、全部static assertions、plan hash和parser/restore测试：

```text
HbgLaunchBlobHeader
  magic / abi major+minor / exact header_size / exact total_size
  region_count / flags / plan_generation / plan_hash
  inline_payload_addr  ---- placeholder[0] ---> inline payload base
  inline_payload_size
  HbgExecutionBinding
    SM / runtime-arena / GM-heap bases and capacities
    runtime_offset / slot_generation
  HbgInvocationIdentity
    callable_hash / argument_snapshot_hash / function_binding_hash
    tensor_count / scalar_count / host_total_tasks / callable_id
HbgLaunchRegion[region_count]
  kind / flags / source_offset / size / destination_offset
zero alignment padding
aligned inline payload
  full pristine SM image
  full pristine runtime-arena image
  optional GM-heap initializer bytes
```

只用一个placeholder修补inline payload base，region的source offset全部相对该base。这避免为每个span建立独立pointer字段，也让region table本身仍位于runtime复制的固定header范围。canonical host blob中pointer必须为0；device-patched blob中必须精确等于runtime args base加 `header_size`。

实现必须：

- 使用独立HBG AICPU entry/ABI，不让当前固定 `L1AicpuInvocationArgs` 的 `struct_size == sizeof(...)` 验证接受variable tail；
- 在host上对 `total_size`、alignment、每个 `offset + size`、placeholder `addrOffset + 8`、`dataOffset`和region数量做overflow-safe校验；当前serializer/validator和真实placeholder bridge已接通，slot minimum capacity也使用相同padding算法；
- 显式限制 `total_size <= UINT32_MAX`，不接受public `size_t` 到runtime `uint32_t` 的静默截断；
- header在AICPU端通过对齐local copy或byte-safe parser读取，不假设runtime args base满足PyPTO `alignas(64)`；
- `LoadAicpuOp` 已增加接受writable host args和placeholder array的独立能力；HBG不使用 `const_cast + nullptr placeholder`的TRB helper形态；
- placeholder只指向本blob内部，任何external pointer都在ownership table中另行列出，不宣称被runtime拥有。

#### N.4.3 必须保持的未知项

- CANN public API未在header注释中承诺任意大小args的capture-lifetime deep snapshot；
- `GetClampedCpySize` 为某些旧backend保留allocated-entry clamp路径，必须验证目标板上是否完整copy；
- public `argsSize` 内部cast为32位，而 `UB_ARG_MAX_COPY_SIZE = 256 MiB` 等只是当前源码细节，不是可编码的产品规格；
- capture强制copy能证明host临时pointer不应被device task延迟读取，但还需证明arg handle究竟由capture graph、instantiate model还是首次replay task持有，以及graph destroy时何时回收。

### N.5 destination-address-bound和capacity freeze协议

`HbgGraphPlan` 和serialized header必须保存下列binding tuple：

```text
HbgWorkingBinding
  device_id
  context_generation
  working_sm_base / capacity
  working_runtime_arena_base / capacity
  gm_heap_base / capacity
  outer_runtime_base / size
  device_kernel_args_base / size
  binary_generation or function-address identity
```

AICPU leader恢复前必须比较header expected binding与当前context binding。任一base、capacity、device id、generation或必要binary identity不匹配都必须在放行scheduler/AICore前fail-closed，不得“先copy一部分再报错”。

prepare/init阶段必须根据ring task-window、heap和runtime-arena layout一次性确定最大capacity。从第一个HBG package可能被capture开始：

- 禁止 `setup_static_arena` 释放/重新commit任一被绑定region；
- 新graph需要更大SM/arena/heap时直接返回capacity error；
- 如果未来需扩容，创建新context generation和新stable slot，旧slot保留到旧graph全部销毁；
- 不得通过更新“current base”全局变量让旧captured package静默指向新allocation。

首版多个package可以绑定同一个working slot，前提是调用方保证不并发replay。未来多slot并发不可以直接复制当前address-bound bytes，必须实现per-slot materialization或完整relocation manifest。

### N.6 每次eager/replay的restore和执行时序

#### N.6.1 capture-time host路径

```text
caller invokes HBG L1 operator
  validate static metadata / device / context state
  run host orchestration and produce immutable HbgGraphPlan
  validate plan sizes and exact working binding
  deep-serialize a fresh writable HbgSerializedLaunchBlob
  prepare aclrtPlaceHolderInfo[] for inline spans
  enqueue caller-stream AICPU HBG entry with WithHostArgs
  enqueue the same closed caller/hidden-AICore event topology as L1
  return without stream/device synchronization
```

capture时host build执行一次，ACLGraph replay不重新进入host builder。所以这里的“dynamic graph”是capture/build-time dynamic，不是replay-time自动重新tiling。该captured node中的topology、tensor addresses、scalar和binding generation在v1都是静态snapshot；未来graph update/dynamic replay是另一项能力，不得暗含在v1。

#### N.6.2 每次device execution/replay

```text
AICPU launch node begins on caller stream
  all launched AICPU threads enter a generation gate
  exactly one leader:
    byte-safe parse and validate header
    validate exact destination binding and capacities
    restore every manifest span into working SM/runtime arena/initializers
    validate restored task-owned function-table metadata and require its host task数与header identity一致
    reset or reconstruct manifest-declared Runtime/KernelArgs/handshake/mailbox state
    perform required cache clean/invalidate
    publish restore success/failure with release semantics
  peers acquire the verdict
  on failure: no scheduler classify/dispatch; all peers and hidden AICore follow common safe epilogue
  on success:
    attach/wire working runtime
    attach_populated working SM
    classify roots, ready queues and wake lists
    dispatch and complete AICore work
  last AICPU teardown may mutate/destroy working state
  caller-stream op exit still waits hidden AICore done event
```

每次replay都要经过这个完整restore gate。首次执行之前恢复过一次、第一次结果数值正确、或working allocation从未free，都不能代替第二次replay restore。

restore manifest至少要覆盖pristine SM和runtime arena。完整runtime arena现在已包含 `HbgPrebuiltInvocationState`，即实际1024项callable-local函数地址表、完整函数表hash、非负host task数和magic/version/count校验头；因此每次restore自然同时恢复graph调度状态与函数分发语义。host strong build要求header identity候选中的 `function_binding_hash`与刚写入arena的完整表匹配；当前AICPU leader也已在每次restore之后、发布success verdict之前，要求header中的 `function_binding_hash/host_total_tasks`与runtime-owned source实际恢复出的state匹配，不能用host阶段通过过一次来替代device replay阶段的检查。GM heap不必默认全量copy，但host builder对GM heap产生的任何语义化初值都必须以initializer span明确列入manifest。outer Runtime、KernelArgs、handshake和mailbox中的per-execution字段必须通过字节恢复或确定re-init路径列入协议，不得依赖“上次deinit应该已清干净”。

#### N.6.3 为什么首选AICPU leader restore

runtime-owned inline source的device base在WithHostArgs launch内部才由CANN创建，host侧没有一个已知、可交给更早memcpy node的source address。因此这条ownership路径天然配AICPU leader从本次arg指针复制到working slot。restore发生在captured AICPU kernel node内，每次replay自然重复，不需要PyPTO感知capture/replay。

captured D2D restore node只与“external stable device pristine source”fallback搭配。若选用它，source allocation必须有独立graph-lifetime lease，不得使用临时host vector、单份可覆盖device buffer或native pipeline slot。

### N.7 资源所有权与close规则

| 资源 | 首选owner | 不得做的推断 | 释放条件 |
| --- | --- | --- | --- |
| host canonical plan | PyPTO host | placeholder API不会改它 | 序列化完成，或plan cache显式退出 |
| writable launch blob | PyPTO host | 不假设异步launch API返回前后边界，必须P0 | runtime snapshot已实证完成 |
| inline device pristine payload | CANN task/captured graph（待P0） | 不因capture强制copy就假定graph destroy lifetime已证 | eager task完成；graph destroy/model release的实证回收点 |
| working slot/workspace | PyPTO context | 不因package由runtime持有就提前free | 所有graph destroyed + external quiescence |
| incore/kernel binary | PyPTO/runtime append-only generation | 不依赖current callable map不再显示就free | 所有可引用它的graph不再replay；首版可到context close |
| external tensors | caller/torch/graph pool | runtime args拷贝pointer value不等于拥有storage | device最后使用完成 |

`close()` 仍不做sync，不猜测graph是否销毁。调用方必须先销毁所有可能replay的graph，再完成external quiescence，最后close HBG L1 context。在无graph retain/release hook的fallback中，PyPTO只能使用明确内存计数和上限的append-only/pin-until-context-close，不得在不能证明lifetime时复用source address。

### N.8 fallback决策树

```text
P0: WithHostArgs inline payload完整copy且随captured graph存活？
  yes
    -> runtime-owned inline source
    -> AICPU leader per-replay restore
    -> PyPTO不自建graph-payload device pool
  no
    -> 是否可用stable external device source + captured D2D restore？
       yes
         -> 是否有ACLGraph/HBG owner retain-release hook？
            yes -> explicit HbgPackageLease
            no  -> bounded append-only until context close
       no
         -> HBG ACLGraph保持unsupported，不退化为临时host source、同步copy或可覆盖单buffer
```

不论选择哪条source ownership路径，working slot都必须per replay restore；append-only只解决source lifetime，不会把被消费的working state变成可重复执行。

### N.9 第二阶段建议实施顺序

> **2026-08-18实施快照：** runtime提交 `11b7a4b1`、`10e69df6`、`de2aa0f9`、`f6ad61df`、`ee292037`、`2873feae`、`6228b481`、`ade00349`、`6b356c35` 与 `4a8c3964` 依次完成host-build/H2D拆分、variable HostArgs/placeholder bridge、transactional restore core、frozen execution-slot registration、真实working arena、immutable `HbgGraphPlan`、task-owned完整函数表与task数、函数表hash互证、context generation和device-side slot registry。`18b1fde9` 又增加独立HBG callable registration及binary-lifetime immutable registry；`74d0ff65` 已把此前分离的部件接成完整但尚未开放的per-invocation路径：DeviceRunner构造callable-local 1024项函数表与非零argument snapshot identity，调用A2/A3或A5 strong host builder生成本次plan，再从canonical plan复制fresh writable blob与placeholder，通过 `aclrtLaunchKernelWithHostArgs` 的独立HBG run symbol入队；AICPU先acquire slot/callable两份trust root并byte-copy fixed header，exactly-one leader验证完整variable blob、恢复SM/runtime arena、发布cache与统一verdict，peer在成功后才classify/dispatch。每次eager execution和每次ACLGraph replay都必须重新restore，绝不把已被scheduler/runtime_destroy消费的working image当作可直接复用的graph executable。`8427ffd7` 又把L2时代依赖reset的已知异常收尾改造成borrowed L1协议：独立prelaunch control、per-core pre-window CANCEL、两阶段completion gate、scheduler init最终裁决、prepare-time control fallback、affinity pre-barrier校验和AICore trusted Runtime override在A2/A3、A5同构实现。当前HBG runtime仍显式strong `l1_runtime_supported_impl() == 0`，Python也继续拒绝HBG，因为CANN runtime-owned source lifetime、大args、placeholder、hidden-stream capture/cache和上述错误路径的真实device行为仍缺device 1证据。详细过程见10.25～10.42；“源码路径已接通”不得改写成“HBG L1/ACLGraph已supported”。

当前进度不能解释成跳过H0：host-only ABI和边界拆分可以先写、先做无硬件fail-closed测试；任何关于CANN snapshot时点、captured-node lifetime、large args、cache可见性和replay恢复正确性的产品结论，仍必须由空闲device 1上的H0/P0实证给出。

#### N.9.1 HBG Phase H0：只做device 1 capability probes

1. AICPU WithHostArgs placeholder inline pointer、host原地patch和snapshot时点；
2. capture/instantiate/replay/destroy的args allocation lifetime；
3. large args完整copy、错误边界和capture延迟；
4. AICPU leader从args source到working GM的copy/cache可见性；
5. external tensor host-build data dependency分类。

H0不接入高层API，不声称HBG L1 supported。任一硬门槛失败都回到N.8决策树，不用隐式sync或private AICPU stream绕过。

#### N.9.2 HBG Phase H1：拆分host build与H2D

- 将当前 `run_host_orchestration` 从“本地构建后立即同步H2D”拆成“产生owning canonical `HbgGraphPlan`”；
- plan深拷贝SM、runtime arena和initializer spans，不保留local vector/arena的悬空pointer；
- 保留L2/L3原有bind/copy路径，不迫使它们经过L1 variable blob；
- 新增plan validation/hash/binding metadata UT。

当前已完成：`build_host_orchestration_image`不再执行H2D，调用方只在SM与runtime-arena两份host image都完整后越过显式上传边界；A2/A3与A5保持同构，旧L2仍在同一bind调用中同步上传。runtime `2873feae` 进一步增加immutable owning `HbgGraphPlan`和A2/A3、A5 strong host-build hook：host `DeviceArena`与SM vector只作为本次build的临时source，成功返回前被deep-copy进私有canonical blob；失败不覆盖caller原有plan owner。runtime `6228b481` 又在构图结束、H2D或plan deep-build之前，把当前完整函数地址表和实际task数写入runtime arena中的versioned invocation state；所以plan拥有的arena不再只有scheduler结构，也拥有实际dispatch所需的callable-local地址。当前plan已正式拥有完整SM与runtime-arena image，但GM heap中有语义的initializer spans尚未形成manifest/region，因而不能把“两个full image已有owner”扩张成“全部graph执行态都已可重复恢复”。captured-node lifetime还要由H2 runtime-owned device source和H0实证闭环。

#### N.9.3 HBG Phase H2：variable launch blob和placeholder bridge

- 定义独立HBG invocation header/ABI、serializer和parser；
- 增加writable args + placeholder array的AICPU launch helper；
- 实现完整size/offset/alignment/overflow validation；
- 保留TRB fixed WithHostArgs ABI不变；
- 对host patch污染canonical plan、截断blob、交叉span、错误placeholder做无硬件UT。

当前源码已完成：独立 `HbgLaunchBlobHeader/HbgLaunchRegion/HbgExecutionBinding/HbgInvocationIdentity` ABI、深拷贝serializer、host-unpatched/device-patched pointer状态、严格size/alignment/overflow/full-image/overlap/hash校验；runtime `10e69df6` 增加CANN-independent placeholder ABI、silent-narrowing/8-byte pointer-write校验和 `LoadAicpuOp::LaunchWithMutableHostArgs` 真实ACL API bridge。runtime `2873feae` 将serializer封进immutable `HbgGraphPlan`：canonical `HostUnpatched` bytes完全私有，每次 `serialize()` deep-copy出独立writable scratch；一份scratch被placeholder patch不影响plan或其他task。`74d0ff65` 已把fresh scratch真正接到独立 `simpler_aicpu_l1_hbg_exec`：DeviceRunner在每次host launch中持有plan和blob直到WithHostArgs API返回，placeholder只指向本次blob内的inline payload；AICPU先 `memcpy` fixed header到对齐local，再根据CANN patch后的inline地址验证整个variable package。ABI minor 2新增 `callable_id`，`argument_snapshot_hash`现在必须非零；minimum package capacity也计入header与各region之间的serializer padding，避免“各payload size相加但漏算alignment”的越界。

这里仍有一条不可偷换的证据边界：host源码和无硬件测试只能证明“传给CANN的bytes、offset、hash与placeholder是自洽的”，不能证明CANN在eager/capture中确实深拷贝任意实际HBG大小、何时完成snapshot、device args基址是否满足预期、captured node多久持有这份allocation。`RuntimeOwnedHbgPayload`这一层仍必须通过device 1 P0；在此之前不能用“API调用已接通”宣称task/captured node lifetime已经由产品契约保证。

#### N.9.4 HBG Phase H3：stable execution slot和capacity freeze

- prepare/init一次性分配working SM/runtime arena/GM heap/workspace/outer Runtime/KernelArgs；
- 记录 `HbgWorkingBinding`，capture可能开始后禁止增容/换址；
- 加入binding/capacity/generation fail-closed；
- 不对外暴露workspace pointer，延续当前“workspace由PyPTO内部prepare-time管理”结论。

当前源码已完成execution-slot和callable两类trust root。runtime `f6ad61df` 建立共享slot ABI/校验：`HbgExecutionSlotRegistration` 同时冻结device id、SM/runtime-arena/GM-heap、outer Runtime、device KernelArgs、slot/binary generation、最大package bytes和serial-only flag，校验所有window的overflow/alias后才发布registration hash；replay restore必须以该注册为trust root。runtime `ee292037` 让A2/A3与A5 HBG strong prepare真实计算、分配并冻结GM heap/SM/runtime arena，HostApi freeze要求精确base/capacity且冻结后只接受完全相同的setup请求；Runtime保留可信capacity。runtime `6b356c35` 由进程生命周期ChipWorker分配不复用的context generation；DeviceRunner在outer Runtime、device KernelArgs和通用AICore executor都稳定后补齐generation、executor content identity和精确package结构容量，transactionally seal并持有immutable registration。runtime `4a8c3964` 将slot registration发布进AICPU binary-lifetime registry；`18b1fde9` 为每个host-orchestration callable另建immutable registration，冻结 `callable_id`、callable/content identity、tensor/scalar count与完整函数绑定hash。

`74d0ff65` 的独立HBG run entry现在会先acquire两份registry，再接受task blob中的identity；blob不能再用自己的binding/callable hash自证。host prepare顺序固定为static slot与KernelArgs准备完成、slot registration生成、callable registration生成、AICPU init enqueue、slot registration enqueue、callable registration enqueue。重复registration只有逐byte相同才幂等，冲突fail-closed。close时binary unload失败必须保留registry引用的所有device window与host owner供重试。H3的源码所有权链已经形成，但production完成仍受H0与H4/H7硬件验证约束；尤其registry不拥有任何per-task graph package，它只证明package允许恢复到哪一组persistent destination以及由哪个callable消费。

`8427ffd7` 将registration原来的reserved字段versioned升级为outer Runtime内64-byte prelaunch control的offset，并把同一control地址随prepare-time `simpler_aicpu_init`另行驻留到AICPU SO。正常run仍以full registry为trust root；如果registry处于NotReady/Publishing/Corrupt或device校验失败，AICPU使用init-latched地址发布同值CANCEL，而不是从尚未校验的variable HostArgs里取device pointer。Host在caller stream上一次async memset连续清零control和active handshakes，且每次验证两段地址连续、容量不溢出。该fallback只解决“已经成功init的context后来无法acquire full registry”；若init task本身失败，则prepare必须失败并进入外部close/recovery，不能猜地址继续launch。

#### N.9.5 HBG Phase H4：AICPU leader per-replay restore

- 将restore放在当前classify-ready barrier之前，且exactly one leader执行；
- peers必须看到统一restore verdict，失败不attach/classify/dispatch；
- 明确A2/A3、A5 cache clean/invalidate和release/acquire协议；
- 与现有common epilogue整合，保证restore失败时hidden AICore不留在register-window轮询；
- 先用full-span correctness restore，再以profile数据决定是否拆分immutable/mutable物理layout。

runtime `de2aa0f9` 先提供不依赖device的transactional restore core：逐span copy并cache-publish，只有全部region成功才提交slot/plan generation；`f6ad61df` 又要求先通过prepare-time sealed slot registration和package capacity校验。无硬件反例覆盖同package重复恢复、A/B package交替、slot注册篡改以零copy拒绝，以及copy/publish中途失败不发布ready。`6228b481` 与 `ade00349` 让restored runtime arena自带完整callable-local函数表、真实task数和全表hash，scheduler不再回读outer Runtime中的调用级暂存表。

`74d0ff65` 已把这条restore core接进A2/A3与A5 HBG AICPU execution：run entry复制并校验fixed invocation view；唯一boot leader使用registry中的exact destination、package capacity、callable identity和function binding作为trust root，对Runtime-owned variable blob做full validation，完整恢复pristine SM与runtime arena并执行cache publish；restore success后再次比较header identity与restored `HbgPrebuiltInvocationState`；统一 `hbg_restore_error_` 以release/acquire发布，peer先invalidate working spans，只有错误为0才classify/dispatch。第一次执行结束后被scheduler和 `runtime_destroy`改写的queue、completion、pointer等状态，下一次execution/replay仍从同一task-owned pristine source完整恢复，不能依赖working slot残值。

`8427ffd7` 已完成当前能由源码协议与无硬件反例收口的no-reset路径：所有有效AICPU participant无论init/run共享错误均exactly-once进入arrive/finalize/snapshot/depart，last-depart才清代际状态；decoupled orchestrator必须在 `p_func` 前等待全部scheduler handshake/assign的最终裁决；已report core的physical id越界或register mapping为0时收到per-core CANCEL；slot/callable/blob/ABI/platform/affinity等generation前失败写独立prelaunch CANCEL；A5 PMU入口先保护physical-id索引。AICore launch又增加第二个Host直传Runtime pointer，HBG使用immutable slot registration中的outer Runtime，TRB/L2/L3传null继续旧路径，从而避免坏 `KernelArgs::runtime_args`让AICPU和AICore读取不同control。

H4仍不能标记production完成，原因现在收敛为三类硬件/产品事实，而不是仍有已知源码早退洞：A2/A3与A5真实cache可见性、两指针AICore launch和runtime args lifetime尚未上板；GM heap若将来出现有语义initializer span，必须进入manifest，不能把整块workspace盲目清零也不能漏恢复；完全不进入/不report的硬件core不在算子内可恢复模型，需CANN op timeout、driver fault containment或外部context/device recovery。restore/handshake错误仍必须通过N.10的device fault matrix证明hidden AICore实际退出，源码闭环不能替代上板证据。

#### N.9.6 HBG Phase H5：独立L1 registration/runtime路径

- 保留host orchestration DSO和host entry，不经由现有“host dlopen handle即拒绝”的TRB device-orchestration registration分支；
- 新增HBG对应 `l1_runtime_supported_impl/prepare_l1_runtime_impl` 及AICPU L1 entry，但只在H0-H4硬门槛通过后开启capability；
- caller stream运行AICPU主task，hidden stream只运行AICore，与本计划已确定单op fork/join边界一致；
- 不使用旧PyPTO的hidden AICPU stream、capture-aware early launch或model attach。

当前源码已经形成独立HBG L1 native路径：A2/A3与A5提供strong `prepare_l1_runtime_impl`、`query_l1_hbg_execution_binding_impl`、`build_l1_hbg_graph_plan_impl`，host callable不再被DeviceRunner的TRB分支拒绝；`CallableArtifacts`保存host orchestration entry bundle及显式destructor，失败/close时不会只 `dlclose` 而泄漏entry owner。DeviceRunner为每次launch从当前callable的kernel list构造一张先全清零的1024项表，拒绝越界、重复、零地址，再计算完整表hash；不同callable可以都使用 `func_id=0`，各自plan/restored arena只携带自己的表。`plan_generation`是独立单调 `uint64_t` identity，不依赖任何“约2048次kernel launch”内部规格。

task入队仍复用同一单算子拓扑：caller stream完成handshake memset、start record和 `simpler_aicpu_l1_hbg_exec` WithHostArgs；hidden stream只wait start、launch已注册AICore handle并record done；caller再wait done并recordtail。没有private AICPU stream、capture query、model attach、`rtStreamAddToModel`、early launch、launch-time binary registration或内部sync。HBG path与TRB fixed invocation共享外层fork/join，但使用完全不同的AICPU symbol和variable package ABI。

为防止“源码半接通就被误开放”，A2/A3与A5 HBG runtime maker当前都显式提供strong `l1_runtime_supported_impl() { return 0; }`；注释列出large variable HostArgs/placeholder、hidden-stream capture/replay、重复pristine restore、no-reset错误teardown的真实device证明以及Python高层支持等gate。Python `_RUNTIME_NAME`仍限定TRB并拒绝HBG。只有N.10硬件矩阵完成后，才能在单独提交中翻转capability与增加高层API；`8427ffd7`完成源码协议不等价于任何一项device checkbox已经通过。

#### N.9.7 HBG Phase H6：direct external tensors与host-build数据契约

- HBG L1 binder只借用调用方device tensor storage，不自行分配/搬运external input/output；
- 对每个host orchestration是否使用 `get_tensor_data/set_tensor_data` 建立静态标记或可审计分类；
- 只允许已证明capture-safe的host-known metadata/scalar/CPU control-data路径；
- 依赖device tensor value但无无sync host-view协议的callable必须fail-closed，不在capture中暗中D2H/sync；
- A2/A3 host mapping能力不能被默认等价到A5。

`74d0ff65` 将此前“若平台可map则建立read-only host view”的过渡方案进一步收紧为真正的异步L1规则：HBG L1 host build不注册、不map、不读取也不写入任何device tensor contents。原因是host launch发生时caller stream上的predecessor可能还没完成；即使平台能给出host virtual address，PyPTO也没有无同步的完成证明，读取会越过单算子入口依赖。strong builder只把调用方device地址原样写入tensor descriptor，允许host orchestration使用shape/dtype/stride/address、host scalar和拓扑；`host_tensor_access_reset_read_only()`在没有注册任何region的状态下运行，因此 `get_tensor_data`与 `set_tensor_data`都fail-closed。host orchestration结束后还检查fatal状态，避免把“data access失败后生成的半图”打包成有效plan。

这是当前v1的明确支持面，不再按A2/A3可映射、A5不可映射做条件分支。它仍不能替代编译期静态分类：最终transformed orchestration应携带 `requires_device_tensor_read/write` 等metadata，在任何host build/plan allocation前拒绝不支持callable。未来若HBG需要像传统tiling那样读取device-produced control data，必须另行设计caller-stream有序、无内部sync的异步control-data协议；不能恢复HAL map，也不能在capture中暗中D2H。

#### N.9.8 HBG Phase H7：ACLGraph、lifetime与回归

- 完成N.10 device 1矩阵；
- 对runtime-owned source、fallback source、working slot、binary和external tensor分别做memory accounting；
- 验证graph A/B顺序交替replay和无并发契约；
- 运行L2/L3 HBG/TRB回归，确认L1 host-plan拆分没有改变旧路径所有权/同步语义；
- 只在完成上述证据后将HBG L1 capability从unsupported切换为supported。

### N.10 device 1 P0/ST矩阵

本矩阵是第二阶段开工门槛，不并入当前TRB Phase 0的已实现声明。优先使用device 1，device 0留给并行会话。

> **2026-08-18探针快照：** runtime `3575f60b`已经把N.10.1～N.10.3中属于通用CANN WithHostArgs owner的部分做成独立可执行探针，路径为`tests/st/l1/host_args_probe`。默认eager size为64 KiB、1 MiB、16 MiB、64 MiB，默认双graph size为1 MiB/16 MiB、各100次交替replay，并可配置pressure count/size；另有纯Host self-test验证三指针解析、错误placeholder诊断和非对齐typed-access规避。runtime `eedfdc90`又把同一parser/self-test纳入常规无硬件CMake矩阵，当前C++结果为99/99通过。上述提交只证明探针本身可构建和Host解析逻辑正确，尚未产生任何device行为证据；以下checkbox只有真实device 1日志、数值结果、错误码、时延与memory accounting齐备后才能更新。
>
> **2026-08-18 restore反例快照：** runtime `620f1df4`补充跨越多条64-byte cache line的
> pristine SM/runtime-arena恢复测试。测试连续restore两次，在两次之间破坏首、中、尾字节，
> 并验证每一轮都重新复制、publish完整region；另验证一次copy/publish失败不发布commit，之后
> 只有完整成功restore才能覆盖整个被污染slot并发布正确generation/hash。完整no-hardware C++
> 矩阵仍为99/99通过。这是Host算法和事务边界证据，不证明AICPU cache maintenance、AICore
> 可见性或ACLGraph replay，因此N.10.4/N.10.5对应device checkbox仍保持未勾选。
>
> **2026-08-18 device1快照：** 修复探针descriptor与共享dispatcher的content-addressed
> basename契约后（runtime `6a5f70a9`），A2/A3 device1完成两组真实矩阵：默认矩阵覆盖
> 64 KiB/1 MiB/16 MiB/64 MiB eager、512次64 KiB压力，以及1 MiB/16 MiB双graph各
> 100次A/B交替replay；加强矩阵覆盖64 MiB captured graph、1 MiB companion graph各100次
> replay和2048次64 KiB压力。所有三placeholder实际地址、full checksum、首/中/尾样本和
> canonical hash均通过。正式PyPTO HBG扩展ST另有4/4通过，包括异步tensor/scalar快照、
> multi-output、multi-child/internal workspace、双graph和同进程第二context重新从
> `callable_id=0`注册。以下仅勾选这些证据直接证明的项目；A5、真实HBG最大image、显式
> mutable-state poison、cache可见性和no-reset fault hook仍未完成。

#### N.10.1 placeholder、snapshot与canonical immutability

- [x] AICPU最小kernel验证placeholder pointer等于runtime args base + inline offset。
- [x] 验证三个独立placeholder可分别代表SM、arena和restore manifest payload。
- [x] launch API返回立即poison/free/reuse host blob，eager仍读到原payload。
- [x] 证明CANN原地patch只改写第2层scratch，第1层canonical plan hash不变。
- [ ] 未对齐runtime args base不会导致typed UB，AICPU parser先做aligned local header copy。

#### N.10.2 large args和backend边界

- [ ] 至少扫描64 KiB、1 MiB、16 MiB、64 MiB、真实HBG常规image、真实最大image和失败边界。
- [x] 已扫描的64 KiB、1 MiB、16 MiB、64 MiB成功size均检查full checksum和tail bytes，防止old backend clamp后仅头部数值正确。
- [x] `argsSize > UINT32_MAX`、offset addition overflow、placeholder越界由PyPTO在host稳定拒绝。
- [ ] 记录A2/A3和A5各自错误码、内存占用、capture/instantiate延迟和replay延迟，不将内部常量固化为PyPTO限制。

#### N.10.3 captured-node lifetime

- [x] capture后立即销毁host blob，instantiate/replay至少100次仍正确。
- [x] capture后发射大量其他WithHostArgs task并制造args allocator环绕/压力，旧graph payload不串包。
- [x] graph A/B包含不同tail canary与topology/scalar，顺序交替replay仍持有各自payload。
- [x] 观察capture、instantiate、首次replay、graph destroy前后的arg handle/memory accounting，确认64 MiB captured args在graph destroy后回收；小allocation的runtime cache保留单独记录。
- [ ] graph销毁但device未external quiescent时，不提前释放working slot/context roots。

#### N.10.4 per-replay restore正确性

- [x] 同一ACLGraph不重新host build，至少连续replay两次。
- [x] graph A/B的callable都从 `func_id=0` 编号但绑定不同binary，交替replay必须分别命中各自runtime-arena中的完整函数表；不得回读outer Runtime的最后一次绑定。
- [ ] 篡改restored `HbgPrebuiltInvocationState` 的magic/version/count/task数/stored hash或任一函数地址，leader必须在wire/classify/dispatch前fail-closed；header identity中的task数/hash与arena state不一致也必须拒绝。
- [ ] 第一次执行后用test hook poison ready queue、wake list、completion flags、task state、runtime pointer、mailbox等known mutable spans。
- [ ] 第二次replay必须从该node的immutable source恢复，不是依赖某些field碰巧未变。
- [ ] 验证GM heap中所有有语义initializer span每次恢复，而不必要的workspace bytes不强制清零。
- [ ] restore失败时所有AICPU peer读取统一verdict，不classify/dispatch，hidden AICore安全退出。

#### N.10.5 address binding、capacity和cache/order

- [ ] 伪造wrong device/base/capacity/context generation/binary identity，leader在写working slot前fail-closed。
- [ ] capture后尝试请求更大SM/arena/heap，旧slot不被recommit，新请求明确失败。
- [ ] AICPU leader copy后多AICPU peer立即classify，AICore立即读descriptor/payload，A2/A3和A5都能看到完整同一generation。
- [ ] 将restore generation/canary分布在首、中、尾多个cache line，防止只验证header可见。

#### N.10.6 stream/op边界和禁止项

- [ ] trace中无PyPTO `aclrtSynchronizeStream/Device`或runtime同类sync。
- [ ] 无capture query、model attach、`rtStreamAddToModel`、private AICPU stream和capture-only early launch。
- [ ] 延迟predecessor能阻止restore/AICPU/AICore越过op entry；延迟AICore tail能阻止successor越过op exit。
- [ ] restore是每次captured AICPU node执行的内部阶段，不是capture之前提前启动的orchestrator。

#### N.10.7 close、fallback和无并发契约

- [ ] graph仍可replay时close fail-closed或按明确外部契约被拒绝，不free working slot/binary/workspace。
- [x] graph destroyed + external quiescence后close成功，所有资源只释放一次。
- [ ] runtime-owned路径失败时，external-source fallback有独立lease和memory accounting，不默默复用单buffer。
- [x] 两个graph顺序replay通过；并发replay不在v1 supported matrix中，文档和test均不声称支持。

#### N.10.8 no-reset故障注入与hidden AICore完成性

- [ ] slot registry分别处于NotReady、Publishing、CorruptState和wrong-device时，init-latched control仍收到CANCEL，hidden AICore完成，下一次合法调用无需reset。
- [ ] callable缺失、bad blob/header/identity/placeholder、platform bridge拒绝时，generation建立前CANCEL在A2/A3和A5都可见。
- [ ] `allowed_count/launch_count`为0、负数、超过 `MAX_GATE_THREADS`或allowed多于launched时，在任何线程进入affinity barrier前失败；合法over-subscription的dropped thread仍保持正常语义。
- [ ] 篡改device `KernelArgs::runtime_args`，AICPU向registered Runtime写CANCEL，AICore通过第二个trusted Runtime launch参数读取同一control并退出；验证AIC/AIV两种entry。
- [ ] physical core id越界和范围内但register address为0时，AICPU只写对应per-core CANCEL，不访问未知SPR；A5 PMU入口不会先OOB。
- [ ] restore、scheduler-init、assign、dispatch、shutdown和runtime-destroy各阶段注入错误，所有有效AICPU participant均完成arrive/finalize/snapshot/depart，只有last-depart清代际状态。
- [ ] 每个故障case后caller-stream tail event可达、context不依赖device reset，随后同context合法eager/capture/replay能够成功。
- [ ] 完全不进入或不report的硬件core按外部op-timeout/driver恢复边界记录，不伪装成PyPTO算子内可恢复case。

### N.11 独立硬阻塞：host orchestration的tensor-data依赖

当前L2 HBG在 `runtime_maker.cpp:695-766` 为external tensor自行分配device storage、H2D staging并注册host view；`host_tensor_access.cpp:60-84` 让host orchestration通过host view读写device-addressed tensor。这与L1“直接借用PyTorch device tensor、PyPTO不分配/搬运external input/output”的前提不同；而common onboard注释也明确A5当前没有A2/A3同类host-map path。

因此HBG L1 capability必须另外定义host build输入分类：

| 依赖 | capture-safe v1候选 | 处理 |
| --- | --- | --- |
| shape/dtype/stride/device address | 是 | 直接使用已校验metadata |
| scalar / host-known control value | 是 | 写入GraphPlan和captured payload snapshot |
| 显式CPU control tensor，且调用方保证capture期稳定 | 待定 | 定义独立API/签名与生命校验，不暗中D2H |
| device tensor value，无无sync host-view协议 | 否 | prepare/capture前fail-closed |
| host orchestration通过 `get_tensor_data`读取external device tensor | v1拒绝 | 当前HBG L1不建立任何host mapping；前序caller-stream task可能尚未完成，host没有无sync的可见性证明 |
| host orchestration通过 `set_tensor_data` 改写external device tensor | v1拒绝 | 当前HBG L1无host view，read/write都fail-closed；未来若开放必须另行证明caller-stream ordering、cache可见性和无sync协议 |

实现前必须统计/静态标记orchestration是否使用 `get_tensor_data/set_tensor_data`，不允许因为某个example只依赖shape就推断所有HBG都不读data。这是与graph payload lifetime平行的独立硬门槛：

- payload ownership解决“host已构建graph后，这些bytes如何随task/graph存活”；
- tensor-data protocol解决“host builder在不内部sync/搬运external tensor的前提下，是否有权取得构图所需数据”。

任一项未闭环，HBG L1 + ACLGraph都不得标记supported。

runtime `74d0ff65` 已把这条边界做成统一代码：L1 builder进入一个没有任何registered region的fail-closed host-access window，external tensor只以device address与metadata进入 `L2TaskArgs`；任何真实 `get_tensor_data/set_tensor_data` 都失败，host orchestration fatal状态也会让build整体失败。它不会因为A2/A3存在某种mapping能力就越过caller-stream predecessor，也不会为了A5做隐式D2H/sync。下一步仍需把data-read/write requirement写进final transformed callable metadata，使拒绝发生在host orchestration执行前；显式CPU control tensor若未来支持，必须有独立签名、snapshot和lifetime契约。

### N.12 第二阶段完成定义

HBG L1 + ACLGraph只在同时满足下列条件后才可以宣布完成：

1. 五层对象在代码、ABI、日志和memory accounting中可区分，没有模糊的global current graph pointer；
2. canonical plan与writable serialized blob完全分离，placeholder原地patch不污染plan/cache；
3. runtime-owned inline payload的大小、copy完整性、capture/replay/destroy lifetime有device 1证据，或者按N.8选用有完整lease的fallback；
4. graph package按exact destination binding生成，capacity在capture后freeze，错误binding不发生部分写入；
5. 每次eager/replay都restore SM/runtime arena及manifest中全部per-execution state；restored arena携带的完整callable-local函数表/task数与header identity一致，反复replay、同func-id graph A/B和poison test通过；
6. AICPU leader restore的A2/A3、A5 cache/order协议通过多cache-line可见性验证；
7. caller/AICPU/hidden-AICore entry/exit仍是一个闭合单算子，无sync、capture query、model attach或private AICPU early launch；
8. external tensor host-build data dependency已分类，不可capture的callable能在可控边界fail-closed；
9. v1无并发契约、graph destroyed + external quiescence + close顺序在上层文档和测试中一致；
10. N.10.8中所有可注入no-reset错误都证明hidden AICore完成、caller tail可达且下一次合法调用无需reset；完全失联core的外部恢复边界有明确结果。
11. TRB L1、HBG/TRB L2和L3回归通过，当前已确定的API与所有权不被暗中改写。

本附录明确把WithHostArgs inline payload定义为**已经接通源码、仍需要device P0证明的首选路径**，而不是已经得到CANN产品行为保证的能力。它遵循AscendC tiling data的核心所有权形态：每个launch task/captured node带一份immutable inline参数快照；但HBG graph image会被执行消费，所以还必须由该task在每次execution/replay把pristine source恢复到context-owned mutable working slot。任何只保持一份device graph buffer、只在capture时H2D一次、不在每次replay恢复working state，或用临时host source和隐式sync规避lifetime问题的实现，均不满足第二阶段完成定义。
