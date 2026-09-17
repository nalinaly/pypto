# PyPTO Attention 元数据的 Host/Device 语义与 vLLM 接入设计讨论

整理日期：2026-09-17。

本文整理本轮设计讨论，覆盖问题背景、源码核实、讨论中修正的认识、方案权衡和接入建议。文中的 PyPTO 接口是设计建议，不表示已经实现或最终定稿；性能取舍尚未经过硬件实验。

## 1. 问题背景与目标

LLM decode 通常把多个请求合并执行，各请求的 KV 长度不同。输入 Tensor 的形状、KV cache 的容量和本轮真正参与 attention 的长度并不相同，因此需要 actual sequence length（下文称 actual_seqlen）。

真实长度影响 attention 的工作量及切分方式。执行 attention 之前，可能需要先生成 tiling 或调度元数据。讨论的工程背景是：不同 GPU/NPU、算子实现及运行模式，可能分别选择 host CPU、设备 CPU 或 device kernel 执行这一步。

最初提供的参考包括 [FlashAttention](https://github.com/dao-ailab/flash-attention) 和 [CANN quant_flash_attn_metadata](https://gitcode.com/cann/ops-transformer/tree/master/attention/quant_flash_attn_metadata/op_kernel_aicpu)。A3 上追求性能的 host tiling，以及 GPU/A5 上的 device tiling，是本轮问题提出时的工程背景；本文不将其推广为所有硬件、CANN 版本、attention backend 的固定行为。

同样的矛盾进入 PyPTO 整网或 mega-kernel 后，需要同时满足：

1. Host Build Graph（HBG）能提前于 device 编排任务，保留 host/device 并发。
2. device kernel 能访问其需要的页表和真实长度。
3. host 提前编排时读取的是目标调用的数据，不能读取另一个 step 的内容。
4. 不要求普通用户手工传入、更新和维护两套 CPU/NPU Tensor。
5. 兼容 torch.compile、ACLGraph/CUDAGraph，以及 V1/V2 runner。
6. KV 分块变化导致任务划分变化时，能够按本轮数据重新生成任务。
7. 覆盖普通 decode、连续批处理、投机解码及整网内部生成元数据的场景。

本轮使用真实网络 [deepseek_v4_1_flash][network] 作为讨论对象，不以简单 page_attention 示例代替整网的数据依赖分析。

## 2. 源码依据与适用范围

这里的 V1/V2 均指 **Model Runner V1/V2**，不表示旧版 vLLM V0 引擎与 V1 引擎。

| 对象 | 本轮核对的版本 | 用途 |
| --- | --- | --- |
| vLLM main | `75c71390d5b399f5397a9166920fc45902f99f14` | 上游 V1/V2 的页表、长度和编译边界 |
| vLLM-Ascend main | `f035fa74062ba8395b5569c37a5183c33bc3ee4a` | Ascend V1/V2 适配行为 |
| Ascend 记录的已验证上游 commit | `84030bbe3d74d99bad477a3d2e37a973ccd8865c` | 检查 Ascend 实际继承的上游实现 |
| Ascend 记录的 release lane | `v0.28.0` | 说明其同时存在 release/main 兼容分支 |
| pypto-lib | `2602cbb6ffc7f2605a9ca48772b77519eaca62a7` | DeepSeek V4.1 Flash 网络的数据依赖 |

版本依据：[Ascend 上游 pin][asc-pin]、[release lane][asc-release]。此前读取的上游 `0ef8366b` 与 Ascend `50283947`，到上述 main 快照的增量不涉及本文分析的文件。

这些结论是上述版本的源码结论。没有运行模型、测量流水深度或验证每种硬件组合；不同 attention backend、投机算法、并行策略可能增加依赖，不能把某一路径概括为全部场景。

## 3. 先区分页表、有效长度和派生索引

| 数据 | 含义 | 典型维护方 |
| --- | --- | --- |
| `block_tables` | 请求的逻辑 KV 页到物理 block 的映射 | Host scheduler/KV cache manager 分配，runner 维护执行表示 |
| `seq_lens` | 本轮 attention 使用的有效 KV 逻辑长度 | Host 请求状态及调度信息，必要时结合 device 接受/拒绝结果 |
| `query_start_loc` | 各请求 query 在打包输入中的起止位置 | Runner 根据本轮 query 数量构造 |
| `slot_mapping` | 本轮新 KV 写入的物理槽位 | 由页表、position、请求映射等计算 |
| `compressed_lens` | 某个压缩 attention 分支的有效长度 | 从 position/逻辑长度按算法派生 |
| sparse/top-k 索引 | 实际选择哪些 KV 行参与计算 | 可能由网络内部 device 计算产生 |

普通、未进一步分片的本轮长度可以写为：

`seq_len = corrected_num_computed_tokens + current_query_len`

其中 corrected 表示已经处理先前投机拒绝等修正。多 query 的因果可见范围仍由 causal mask 等约束；压缩、滑窗、CP 分片后的长度还需各自转换。

**不能普遍从 block table 恢复精确 actual_seqlen。** 例如 block size 为 128，长度 129 和 200 都可能对应两个 block。投机解码还可能预留 lookahead 空间；分配容量不等于已有效计算的长度。[分配接口][kv-manager] 明确包含 `num_lookahead_tokens`。

页表也不一定每个 token 都新增条目：同一 block 内继续 decode 时映射可以不变，长度仍增长；跨 block、请求进入/退出、抢占恢复、batch 重排等会引起映射或行布局变化。

## 4. block_tables 的值由谁维护

### 4.1 共同的数据来源

在核对的标准路径中：

1. Host scheduler 决定本轮调度哪些请求和多少 token。
2. Host KV cache manager 管理物理 block 的分配、复用和释放。
3. Scheduler 将新请求的 block IDs、已有请求的新增 block IDs 等交给 worker。
4. Worker/runner 将这些 host 决策转换为本轮 device 可消费的页表及相关元数据。
5. Attention kernel 按页表读写 KV cache；写 KV 内容与修改调度器页表是不同操作。

因此，V2 用 device kernel 更新页表，并不表示物理 block 分配决策已经移到 device。更新位置的改变，与数据来源的改变是两回事。[Scheduler][scheduler]、[KV cache manager][kv-manager]

### 4.2 四条路径对比

| 路径 | Host 保存内容 | Device 页表生成方式 | 完整 CPU 页表是否现成可用 |
| --- | --- | --- | --- |
| vLLM V1 | 完整 CPU 页表，以及请求/batch 状态 | `commit_block_table()` 通过 `CpuGpuBuffer` 异步 H2D | 有；需固定本轮版本 |
| vLLM V2 | Block IDs 增量、写入行/起点、累计长度、block 数量等 | 更新 kernel 修改持久化 device 页表，再 gather 为本轮输入页表 | Runner 默认没有与本轮输入布局一致的完整镜像 |
| Ascend V1 | 完整 CPU 页表，以及请求/batch 状态 | `commit_block_table()` 异步 CPU→NPU | 有；需固定本轮版本 |
| Ascend V2 | 沿用上游 V2 的增量维护 | 适配后的 buffer 提供增量，NPU kernel 更新/gather | 默认没有新增完整镜像 |

源码：[上游 V1 页表][up-v1-table]、[CpuGpuBuffer][cpu-gpu-buffer]、[上游 V2 页表][up-v2-table]、[Ascend V1 页表][asc-v1-table]、[Ascend V2 页表][asc-v2-table]。

### 4.3 V1 的异步拷贝保证了什么

V1 在准备输入时较早提交页表拷贝，以便和后续 CPU 工作重叠。但 `non_blocking=True` 本身不保证正确性。

正确性还要求：

- Device 目标的覆盖排在旧消费者之后，并排在新消费者之前。
- 跨 stream 操作必须建立对应 event/stream 依赖。
- Host 源内存在 DMA 读完前不得被修改或复用。
- HBG 如果还在读取该 CPU 数据，也必须保留相应版本。

不能把“有 CPU Tensor”与“必然引入全局同步”等同。CPU 更新、快照和异步 H2D 可以流水执行，但需要明确的所有权和生命周期。[Ascend V1 输入准备][asc-v1-runner]

### 4.4 V2 的 UVA 到底是什么

V2 的 `UvaBuffer` 底层仍有 `pin_memory=True` 的 CPU `torch.Tensor`，并有 NumPy 视图。上游 GPU 路径通过 accelerator view 暴露同一 host 存储。

默认页表 `StagedWriteTensor` 的主要行为是：

1. Host 将本轮增量暂存为写入行、起始列、内容及累计长度。
2. 写入行、起始列、累计长度等描述信息进入 UVA buffer pool。
3. 默认路径的写入内容仍通过异步 H2D 传输。
4. `_apply_write_kernel` 或 fused writer 将增量写到 device 页表。
5. 另一 gather kernel 按本轮 request 映射形成输入页表。

所以，“V2 只有 UVA buffer，没有 host Tensor”不成立；“V2 有 UVA buffer，所以有完整 CPU 页表”也不成立。[V2 buffer 实现][up-v2-buffer]、[页表更新与 gather][up-v2-table]

UVA 也不自动提供按 step 的一致快照。Buffer pool 使用轮转槽位，并明确要求 GPU reader 结束后才能复用；并发深度必须覆盖实际在途消费者。不同 step 可以共享未修改数据，但不能覆写仍被异步消费者使用的存储。

### 4.5 Ascend V2 的实际 UVA 条件与启用状态

`UvaBufferWrapper` 始终分配 pinned CPU Tensor：

- 满足 `PYTORCH_NPU_ALLOC_CONF` 中的 `pinned_mem_register:True`，且版本检查通过时，走真实 UVA 路径。
- 否则维护额外 NPU buffer，在访问设备侧表示时异步复制修改部分。
- 当前代码明确禁用 `triton-ascend` 的 `3.2.1` 和 `3.2.2`。
- 本轮快照的 `requirements.txt` 锁定 `triton-ascend==3.2.2`，因此按此依赖安装的路径使用拷贝回退。

这不是“所有 Ascend 硬件永远不能 UVA”的结论。[UVA 适配][asc-uva]、[依赖版本][asc-requirements]

当前 Ascend 默认选择 V1；V2 需显式设置 `VLLM_USE_V2_MODEL_RUNNER=1`，其 README 仍标注 experimental。V2 已有实现和测试目录，但不能据此推断全部模型、backend 和执行模式已验证。[Runner 开关][asc-switch]、[V2 状态说明][asc-v2-readme]

## 5. 精确 actual_seqlen 的维护与修正

### 5.1 四条路径对比

| 路径 | 普通 decode | 异步投机或 device 决定长度时 | Host tiling 注意事项 |
| --- | --- | --- | --- |
| vLLM V1 | Host 计数加本轮调度数量可得到精确值；device 生成 `seq_lens` | CPU 可保留乐观值，GPU 根据先前实际接受数量修正 | CPU 长度不总是精确值 |
| vLLM V2 | Device `prepare_pos_seq_lens()` 根据请求状态生成长度；host 保留计数及上界 | Device 接受/拒绝后更新请求计数，host 上界不必立刻精确 | 不应假设每轮存在精确 CPU 长度 |
| Ascend V1 | 保留 CPU 长度供相关 attention backend 使用 | 需要精确 CPU 长度的路径复用接受数量的 D2H 结果，等待 event 后修正 | 校正后使用，不能越过结果依赖 |
| Ascend V2 | 额外维护 CPU 请求计数及 `seq_lens_cpu` | MTP 路径回传 device 修正后的 `num_computed_tokens`；下轮等待 event 后生成 CPU 长度 | 保留了必要的 host/device 依赖 |

上游 `seq_lens_cpu_upper_bound` 的注释明确指出：非异步投机路径可精确，异步投机 decode 行可能假设所有 draft 都被接受，不能直接交给要求精确逐请求长度的 kernel。[长度契约][up-length-contract]

上游 V1 的 device 修正见 [GPUModelRunner][up-v1-runner]；V2 根据 device 状态生成长度并构造 CPU 上界，见 [V2 runner][up-v2-runner] 和 [prepare_pos_seq_lens][up-v2-input]。特定 backend 还可能增加自己的元数据处理，不能把通用 runner 行为理解成所有 backend 都绝不回传。

### 5.2 Ascend V1 的 CPU 修正

当前实现对需要精确 CPU 长度的 backend/压缩 attention 路径，调用 `_correct_optimistic_seq_lens_cpu()`：

- 上一步通过侧流异步回传有效采样数量。
- CPU 读取之前等待对应 event。
- 结合上轮请求位置、draft 数量和实际接受结果，在 CPU 修正乐观长度。
- 避免为此额外回传整个 `seq_lens`。

这可以减少传输或隐藏等待，但 event 尚未完成时仍存在真实依赖，不能保证任意负载下零等待。也不能把该修正路径概括为所有 Ascend backend 都必经。[Ascend V1 实现][asc-v1-runner]

### 5.3 Ascend V2 的 CPU 长度

当前 V2 明确保留 CPU 长度需求：

- 非 MTP 路径，`update_requests` 更新共享 NumPy/CPU Tensor 状态。
- MTP 路径，在采样后回传 device 的 `num_computed_tokens`。
- 下一轮 `_update_seq_lens_cpu()` 等待回传 event，再计算本轮 CPU 长度。
- 对 adaptive verification 与 FIA 组合，代码还存在从本轮 device `seq_lens` 直接取回 CPU 值的路径。

因此，Ascend V2 并未消除 host tiling 的精确长度依赖。[Ascend V2 runner][asc-v2-runner]、[共享 CPU 请求状态][asc-v2-states]

## 6. Host 能提前多少 step，以及能否读取 device Tensor

### 6.1 提前量不能固定假设

普通 decode 的 token 内容通常来自前一次计算，但下一次所需的部分调度结构、预分配空间和长度变化可以提前确定。调度器可以使用受控的异步流水。

不过请求结束、新请求进入、抢占、资源压力、投机接受数量等，都可能改变后续调度。不能普遍宣称“第 s+1 轮所有元数据在第 s 轮计算之前就已确定”，也不能统一规定 host 必须或一定能领先多个 step。

应逐字段分析：

- 分配方案确定后，对应页表更新可在 host 获得。
- 精确长度可能还需要前一轮 device 的接受/拒绝结果。
- 网络内部生成的 sparse/top-k 索引可能要等本轮计算。
- 在途 batch 数量和 buffer 回收规则共同限制可提前的深度。

### 6.2 提交更新不等于更新完成

假设 host 准备第 s+1 轮，device 正在执行第 s 轮：

| 时刻 | Host | Device |
| --- | --- | --- |
| A | 已获得 s+1 的页表增量 | 仍使用 s 的数据 |
| B | 提交 s+1 的 H2D 或更新 kernel | 更新排队等待 |
| C | 开始 s+1 的 HBG/tiling | 目标页表可能尚未更新 |
| D | 已提交后续工作 | 依序执行更新、gather、s+1 forward |

C 时刻适合读取的是已固定的 s+1 host 元数据。Device Tensor 的地址不包含“目标 step”语义，也不保证它的内容已经切换到该版本。

### 6.3 读取规则

| 操作 | 正确性与代价 |
| --- | --- |
| 目标更新尚未提交就做 D2H | 取得的是更新之前的状态 |
| D2H 正确排在目标更新之后，并等待复制完成 | 可以取得新值，但 host 必须等到对应结果就绪 |
| 跨 stream 更新与读取没有依赖 | 没有一致快照保证，不能以概率判断正确性 |
| 读取已就绪、版本固定的 CPU 快照 | 可以继续提前编排，无需等 device 页表更新 |
| Device 直接通过 UVA 读 host 数据 | 仍需保护 host buffer，直至 device reader 完成 |

“提前一个 step 大概率读旧值”不是严格表述；严格表述是：**没有版本对应关系和执行依赖，就不能保证得到目标 step 的值。**

正常、具有正确依赖的阻塞 D2H 会通过等待取得正确值。Host 也不能把普通 device 指针当 CPU 内存直接解引用；UVA 等可访问机制与按轮次的数据一致性是不同问题。

建议的 HBG 默认规则：**不隐式读取可变 device Tensor 的内容；确需读取时，显式表达生产者依赖、传输、等待及版本绑定。**

## 7. torch.compile 与 graph replay 的边界

### 7.1 V2 页表更新为什么不要求把 CPU 页表作为编译图参数

上游 V2 在普通 runner 的 `execute_model()` 中提交 `apply_staged_writes()`，之后准备输入和 metadata，再进入 model forward 或 graph replay。这一预处理部分不等于被 torch.compile 捕获的模型主体；直接启动的 Triton JIT kernel 也不等于 Dynamo 对整个 runner 的追踪。[V2 调用顺序][asc-up-v2-runner]

标准 attention custom-op 路径中：

- Runner 通过 `set_forward_context` 设置本轮 attention metadata。
- 编译图调用 `torch.ops.vllm.unified_attention_with_output(...)`。
- Custom op 的运行时实现通过 `get_attention_context()` 取出 metadata，交给 backend。
- 页表和 CPU 长度等信息不必全部作为该编译图节点的显式 Tensor 参数。[Attention custom op][attention-context]

因此，不能把问题概括成“torch.compile 中所有 CPU/NPU Tensor 都绝对不能共同出现”。具体限制取决于算子 schema、fake/meta 实现、backend 和图边界；本轮核实的 vLLM 路径使用运行时上下文避免把全部 metadata 作为模型图的显式参数。

把值改成 `SymInt[]` 也不会自动解决数据就绪、版本或 graph task 更新问题。是否适用仍取决于具体算子接口。

### 7.2 上下文不是自动一致性机制

`forward_context` 展示了传递运行时信息的一种边界，不代表任意放入其中的对象都能被编译器安全忽略，也不代表退出 Python 的 with 块后异步消费者已经完成。

PyPTO 若借鉴此方式，必须明确运行时 wrapper、custom-op 边界、HBG 获取元数据的位置，以及异步任务持有快照的生命周期。图重放不会自动重新执行 capture 时的任意 Python 上下文读取。

### 7.3 ACLGraph/CUDAGraph 仍有独立约束

固定地址的 Tensor 内容更新，可以通过正确的 stream 顺序为 replay 提供新数据；host 生成的 tiling 参数、任务数量或依赖结构变化，还需要相应的更新机制。

Ascend V2 的 `ModelAclGraphManager.run_fullgraph()` 仍调用 `update_full_graph_params()`；attention backend 中仍有 `graph_task_update_begin/end`。这说明 V2 未消除 host 参数更新需求。调用提交顺序、update stream 和 graph 内 event 必须整体分析，不能简单宣称所有更新都在 replay 调用之前完成。[Graph manager][asc-graph]、[Attention task update][asc-attention]

本轮建议区分：每轮重新执行 HBG、更新已捕获任务参数、重建任务结构、重新编译算子、重新捕获外层图。这些操作的触发条件和代价不同。

## 8. 真实 DeepSeek V4.1 Flash 网络带来的约束

在核对的 `decode_c2a_full.py` 中，`index_select()` 的 `pl.read(compressed_lens)`、`pl.read(request_ids)` 和 `pl.read(block_table)` 位于 `pl.spmd` 区域内，是 device 计算区域的读取。

因此，这份源码中的这些 `pl.read` 不能直接当作“HBG 正在 host 解引用 device 页表”的证据。需要分析具体读取所在的执行区域，以及将其提升到 host 编排后新增的数据依赖。[实际网络代码][network]

该网络还区分：

- `index_block_table` 等外部传入的调度器映射。
- 由 position、长度、压缩比例等派生的 `compressed_lens`。
- 网络内部评分/选择后产生的 `topk_indices`。

`metadata.py` 的 `build_forward_metadata()` 从 query 起点、已有 KV 长度及各类 block table 构造派生信息，包括 `floor((position + 1) / ratio)` 形式的压缩长度。这说明元数据需要按网络实际语义转换，不能直接把普通页表行数当成有效长度。该辅助代码本身也不证明所有真实运行路径都在 CPU 上构造这些数据。[网络 metadata][network-metadata]

本轮核实的 `index_select` 使用输入页表进行地址映射，生成选择结果；不能由此推断整网修改了调度器页表。若将来把页表更新纳入 PyPTO，需显式声明其可修改状态。

## 9. PyPTO 接入建议：统一契约，按数据可用性选择路径

### 9.1 推荐方向及其边界

建议保持模型面向 device Tensor 的主要调用形式，由框架 adapter 为每次调用提供受控 host 元数据。普通用户不手工维护 CPU/NPU 两套 Tensor。

这是一项接口设计建议。无论使用调用上下文、内部参数包还是元数据绑定对象，host 信息都必须通过真实的运行时输入通道传入；不能从一个裸 device Tensor 自动推导其 host 来源和历史版本。

V1/V2 的适配差异应放在 adapter，核心契约按字段的就绪程度划分：

| Host 字段状态 | 合法执行方式 | 限制 |
| --- | --- | --- |
| 精确值且已就绪 | 提前 HBG/host tiling | 固定本轮版本，保持布局一致 |
| 安全上界 | 按上界生成覆盖任务，device 按真实值裁剪 | 算法必须支持；不能访问未分配页或未初始化 KV |
| 尚未就绪/device 决定 | 等待必要结果回传，或使用 device 编排/tiling | 不可能同时要求未知精确值与无等待 |
| 完全不被 host 编排使用 | 保持 device-only 输入 | 建立 kernel 间数据依赖即可 |

同一次调用可以同时具有精确 host 页表、只有上界的 host 长度，以及本轮 device 才产生的 sparse 索引。不能只给整次调用贴一个笼统的“host ready”标签。

### 9.2 元数据绑定的最小语义

| 内容 | 要求 |
| --- | --- |
| 调用标识 | 明确 invocation/step；多 microbatch 或 draft 子调用也需可区分 |
| Request 映射 | 与 device 本轮 batch 顺序、padding、KV group 对齐 |
| 布局信息 | Block size、逻辑/物理拆分、压缩比例和必要的分片语义一致 |
| 字段状态 | 逐字段区分精确值、上界、未就绪，并记录来源 |
| Host 版本 | 消费期间逻辑不可变；可用快照、轮转槽位或其他版本机制实现 |
| Device 依赖 | 明确页表更新、gather、kernel 消费的先后关系 |
| 生命周期 | HBG reader、DMA/UVA reader 和 device 消费者分别按实际需要保留资源 |
| 编译语义 | 每轮 metadata 是运行时数据，不能错误地常量折叠或复用首次值 |

不能只按 Tensor 地址查找 host 版本，因为不同 step 可以复用同一地址。也不能简单绑定到一个持续变化的“最新 CPU Tensor”。

### 9.3 调用形态示意

下面仅说明 adapter/runtime 的职责，不是已实现的 API，也不要求用户显式调用这些方法。

接入前，单独的 device 参数无法表达 host 数据来源：

```python
model(hidden, block_tables_device, seq_lens_device, output)
```

建议由框架内部增加本次调用绑定：

```python
# Conceptual adapter/runtime boundary; not an implemented PyPTO API.
metadata = adapter.freeze_metadata(
    invocation_id=invocation_id,
    request_order=request_order,
    required_host_fields=compiled_host_requirements,
)
with runtime.bind_invocation(metadata):
    model(hidden, block_tables_device, seq_lens_device, output)
```

Runtime 必须在调用提交后继续持有异步消费者需要的对象。绑定对象的创建、释放、字段校验和执行模式选择应由框架负责，不新增普通模型用户管理双份 Tensor 的负担。

公开类型、绑定入口和具体修改文件尚未决定；本轮仅形成语义建议，不改写现有 PyPTO API。

### 9.4 V1/V2 adapter 的不同职责

| Adapter | 可以复用的来源 | 需要补充的工作 |
| --- | --- | --- |
| 上游/Ascend V1 | 已有 CPU 页表、请求状态、长度元数据 | 固定本轮版本；标记或完成长度修正 |
| 上游/Ascend V2 | Host 增量、block 数量、请求映射及已有长度状态 | HBG 需要完整页表时维护 CPU 镜像并重排；只需要少数字段时避免完整镜像 |
| Device 派生元数据路径 | Device producer 的结果 | 保留 device 路径，或显式回传并等待 |

V2 镜像应从相同的 host 分配/更新事件维护，处理新增、删除、覆盖、重排及必要的物理/逻辑 block 转换；不能默认每轮从 device 拉回完整表。

CPU 镜像和快照有内存及复制开销，但不天然要求 device 同步。能否值得，取决于 HBG 所需字段、页表规模和编排收益，需要测量。

### 9.5 算子边界决定输入输出约束

| PyPTO 包含的范围 | 输入输出与状态约束 |
| --- | --- |
| 仅模型 forward，页表更新由 runner 完成 | 调度器页表通常为只读输入；KV cache 按实际计算声明 InOut；V1/V2 可统一契约 |
| 包含 V2 页表增量应用 | 增量是输入，持久化页表是可修改状态；明确别名、版本推进及外部镜像一致性 |
| 包含 sparse/top-k 等内部计算 | 派生索引建立 device 数据依赖；后续 host 消费需分阶段等待或调整边界 |

不应仅因为“V2”就要求把页表改成算子输出。只有算子范围确实包含页表更新时，输入/输出及状态修改约束才应随之改变。

## 10. 备选方案与取舍

| 方案 | 收益 | 代价或限制 | 本轮建议 |
| --- | --- | --- | --- |
| 用户显式传 CPU/NPU 两份 Tensor | 信息直观 | 用户同步两份值，易版本错配，脚本差异大 | 不作为普通用户默认形式 |
| 只传 device Tensor，HBG 隐式 D2H | 接口表面简单 | 等待隐藏，提前编排受限，依赖难审查 | 避免隐式行为 |
| 按调用绑定受控 host 元数据 | 保留用户 device 参数形式，支持 HBG 提前 | Adapter/runtime 必须管理版本和生命周期 | 推荐作为 host 可知数据的接入方向 |
| Device 编排/tiling，含 TRB 类路径 | 自然跟随 device 数据依赖 | 可能减少可利用的 host 提前工作，性能依硬件而定 | 覆盖 device 决定的数据 |
| Host 按上界编排，device 精确裁剪 | 可减少 host 等待 | 多余任务/资源，必须保证掩码与访问安全 | 算法支持且实测有收益时使用 |
| 回传最小修正量再 host tiling | 保留精确 host tiling，减少数据量 | 仍有结果依赖和潜在等待 | 参考 Ascend V1 的计数回传思路 |

HBG 与 device 路径可以在同一个整网内按阶段组合；不必把全部元数据都强制搬到同一侧。具体 fallback 是否自动发生、是否允许同步，应成为可审查的执行契约。

## 11. 本轮结论、尚未定稿项与后续验证

### 11.1 已有源码支持的结论

1. 标准调度器页表的分配决策主要来自 host；V2 的 device 更新 kernel 不改变这一来源。
2. V1 runner 有完整 CPU 页表，V2 runner 默认使用增量和 device 持久化表。
3. 有 CPU 页表不代表 host 已经获得本轮所有精确长度。
4. UVA 底层可以有 CPU Tensor；UVA 不自动提供完整页表或按 step 的一致性。
5. 上游 V1/V2 的异步投机 decode CPU 长度可能只是上界。
6. Ascend V1/V2 为需要精确 CPU 长度的路径保留不同的回传和修正机制。
7. 元数据预处理、torch.compile 模型图和 graph replay 具有不同边界。
8. 正确读取可变 device 数据需要版本及执行依赖；阻塞 D2H 的正确性来自等待。
9. V1/V2 本身不足以决定 PyPTO I/O 契约，还要明确算子范围和逐字段数据可用性。

### 11.2 推荐但未最终定稿的方向

- 保持统一的主要 device Tensor 调用形式，通过 adapter 提供本次调用的 host 元数据。
- HBG 默认不隐式读取可变 device Tensor；显式声明 host 编排所需字段。
- V1 复用已有 CPU 数据，V2 按需从 host 增量维护镜像。
- 对 device 才能确定的结果，允许 device 编排、显式等待或安全上界方案。
- 页表/长度变化引起任务变化时按本轮值重新编排，并分别处理 graph 更新机制。

这些是讨论建议，尚未授权或完成接口实现；本次产物仅为文档。

### 11.3 待决策问题

1. 哪些 DeepSeek 网络字段实际被提升为 HBG 的结构性输入，哪些继续在 device 读取？
2. PyPTO 是否包含页表更新/gather，还是只消费 runner 准备好的输入？
3. Host 元数据通过调用上下文、内部参数包还是统一 Tensor 包装暴露给编排入口？
4. 缺少精确 host 数据时，默认报错、等待还是选择 device 路径？
5. 哪些阶段允许按上界编排，额外工作量和 workspace 是否可接受？
6. 运行时如何表达跨 step/microbatch 的绑定、别名、回收和并发隔离？
7. ACLGraph 中哪些变化可 task update，哪些需要重建或重新捕获？
8. A3/A5、HBG/TRB、eager/graph 各组合的真实性能和支持边界如何验证？

### 11.4 实现后的验证重点

| 场景 | 需要验证的性质 |
| --- | --- |
| 不同长度混合、跨 block 边界 | 任务数量、末页有效范围及 KV 地址正确 |
| 请求进入/退出、batch 重排、抢占恢复 | Host 快照与 device request 顺序一致 |
| 连续多轮使用同一 device 地址 | 不按地址误取其他 step 的 host 版本 |
| 多 step 在途、延迟 HBG/UVA reader | Buffer 不提前覆写或回收 |
| 投机全部接受、部分拒绝、全部拒绝 | 上界与精确长度不混用 |
| 压缩、滑窗及必要的 CP 分片 | 派生长度和页表布局转换一致 |
| 网络内部 sparse/top-k 生成 | Device 依赖完整，不被 host 提前读取 |
| torch.compile 与 graph replay | 动态 metadata 不被固化，更新/重放顺序正确 |

性能验证应分别测 CPU 镜像与快照成本、host tiling 时间、H2D/D2H、event 等待、任务数以及端到端 decode 延迟，不能只比较单个 memcpy 或 tiling kernel。

## 12. 源码索引

下列链接固定到本轮核对的 commit。正文使用这些来源支撑实现事实；第 9～11 节的接口组织和取舍属于设计建议。

- 上游调度与分配：[Scheduler][scheduler]、[KV cache manager][kv-manager]。
- 上游 V1：[BlockTable][up-v1-table]、[CpuGpuBuffer][cpu-gpu-buffer]、[GPUModelRunner][up-v1-runner]。
- 上游 V2：[BlockTables][up-v2-table]、[buffer_utils][up-v2-buffer]、[runner][up-v2-runner]、[input_batch][up-v2-input]。
- 元数据与编译边界：[长度契约][up-length-contract]、[FlashAttention metadata][flash-metadata]、[attention custom op][attention-context]。
- Ascend V1：[页表][asc-v1-table]、[runner][asc-v1-runner]。
- Ascend V2：[页表][asc-v2-table]、[runner][asc-v2-runner]、[请求状态][asc-v2-states]、[UVA patch][asc-uva]。
- Ascend graph：[graph manager][asc-graph]、[attention backend][asc-attention]。
- 真实网络：[decode_c2a_full.py][network]、[metadata.py][network-metadata]。

[scheduler]: https://github.com/vllm-project/vllm/blob/75c71390d5b399f5397a9166920fc45902f99f14/vllm/v1/core/sched/scheduler.py
[kv-manager]: https://github.com/vllm-project/vllm/blob/75c71390d5b399f5397a9166920fc45902f99f14/vllm/v1/core/kv_cache_manager.py
[up-v1-table]: https://github.com/vllm-project/vllm/blob/75c71390d5b399f5397a9166920fc45902f99f14/vllm/v1/worker/block_table.py
[cpu-gpu-buffer]: https://github.com/vllm-project/vllm/blob/75c71390d5b399f5397a9166920fc45902f99f14/vllm/v1/utils.py
[up-v1-runner]: https://github.com/vllm-project/vllm/blob/75c71390d5b399f5397a9166920fc45902f99f14/vllm/v1/worker/gpu_model_runner.py
[up-v2-table]: https://github.com/vllm-project/vllm/blob/75c71390d5b399f5397a9166920fc45902f99f14/vllm/v1/worker/gpu/block_table.py
[up-v2-buffer]: https://github.com/vllm-project/vllm/blob/75c71390d5b399f5397a9166920fc45902f99f14/vllm/v1/worker/gpu/buffer_utils.py
[up-v2-runner]: https://github.com/vllm-project/vllm/blob/75c71390d5b399f5397a9166920fc45902f99f14/vllm/v1/worker/gpu/model_runner.py
[up-v2-input]: https://github.com/vllm-project/vllm/blob/75c71390d5b399f5397a9166920fc45902f99f14/vllm/v1/worker/gpu/input_batch.py
[up-length-contract]: https://github.com/vllm-project/vllm/blob/75c71390d5b399f5397a9166920fc45902f99f14/vllm/v1/attention/backend.py#L441-L445
[flash-metadata]: https://github.com/vllm-project/vllm/blob/75c71390d5b399f5397a9166920fc45902f99f14/vllm/v1/attention/backends/flash_attn.py#L577-L625
[asc-pin]: https://github.com/vllm-project/vllm-ascend/blob/f035fa74062ba8395b5569c37a5183c33bc3ee4a/.github/vllm-main-verified.commit
[asc-release]: https://github.com/vllm-project/vllm-ascend/blob/f035fa74062ba8395b5569c37a5183c33bc3ee4a/.github/vllm-release-tag.commit
[asc-up-v2-runner]: https://github.com/vllm-project/vllm/blob/84030bbe3d74d99bad477a3d2e37a973ccd8865c/vllm/v1/worker/gpu/model_runner.py#L1579-L1597
[attention-context]: https://github.com/vllm-project/vllm/blob/84030bbe3d74d99bad477a3d2e37a973ccd8865c/vllm/model_executor/layers/attention/attention.py#L673-L794
[asc-v1-table]: https://github.com/vllm-project/vllm-ascend/blob/f035fa74062ba8395b5569c37a5183c33bc3ee4a/vllm_ascend/worker/block_table.py
[asc-v1-runner]: https://github.com/vllm-project/vllm-ascend/blob/f035fa74062ba8395b5569c37a5183c33bc3ee4a/vllm_ascend/worker/model_runner_v1.py
[asc-v2-table]: https://github.com/vllm-project/vllm-ascend/blob/f035fa74062ba8395b5569c37a5183c33bc3ee4a/vllm_ascend/worker/v2/block_table.py
[asc-v2-runner]: https://github.com/vllm-project/vllm-ascend/blob/f035fa74062ba8395b5569c37a5183c33bc3ee4a/vllm_ascend/worker/v2/model_runner.py
[asc-v2-states]: https://github.com/vllm-project/vllm-ascend/blob/f035fa74062ba8395b5569c37a5183c33bc3ee4a/vllm_ascend/worker/v2/states.py
[asc-uva]: https://github.com/vllm-project/vllm-ascend/blob/f035fa74062ba8395b5569c37a5183c33bc3ee4a/vllm_ascend/patch/worker/patch_v2/patch_uva.py
[asc-requirements]: https://github.com/vllm-project/vllm-ascend/blob/f035fa74062ba8395b5569c37a5183c33bc3ee4a/requirements.txt
[asc-switch]: https://github.com/vllm-project/vllm-ascend/blob/f035fa74062ba8395b5569c37a5183c33bc3ee4a/vllm_ascend/patch/platform/patch_use_v2_model_runner.py
[asc-v2-readme]: https://github.com/vllm-project/vllm-ascend/blob/f035fa74062ba8395b5569c37a5183c33bc3ee4a/vllm_ascend/worker/v2/README.md
[asc-graph]: https://github.com/vllm-project/vllm-ascend/blob/f035fa74062ba8395b5569c37a5183c33bc3ee4a/vllm_ascend/worker/v2/aclgraph_utils.py#L146-L186
[asc-attention]: https://github.com/vllm-project/vllm-ascend/blob/f035fa74062ba8395b5569c37a5183c33bc3ee4a/vllm_ascend/attention/attention_v1.py#L867-L927
[network]: https://github.com/hw-native-sys/pypto-lib/blob/2602cbb6ffc7f2605a9ca48772b77519eaca62a7/models/deepseek_v4_1_flash/decode_c2a_full.py
[network-metadata]: https://github.com/hw-native-sys/pypto-lib/blob/2602cbb6ffc7f2605a9ca48772b77519eaca62a7/models/deepseek_v4_1_flash/metadata.py
