# Kernel 模式整网 decode：HBG 参数内存调查

更新日期：2026-09-14

状态：源码分析、有界 CANN 探针、定点分配失败注入及当前 Qwen3-14B 的 HBG host build 尺寸测量已完成；
未修改 PyPTO、simpler 或框架执行实现。
这不是 Qwen3 整网 OOM 复现，也不代表 A5、TRB 或所有 CANN 版本已验证。

## 1. 讨论边界

- 一次 decode 覆盖整网，只调用一次 PyPTO `kernel_launch`。
- 只考虑单模型多 step 串行执行，不考虑多模型并发。
- 内存爆炸是提前识别的风险，目前没有整网失败案例。
- 讨论中的约 2048 是 CANN stream 任务条目容量，不是 PyPTO 调用次数规格。
  一个 PyPTO 调用可能下发多个 CANN task，不能直接换算。
- 外部 Tensor 的原有存储不是本次增量；执行 heap/slot 在正确串行的前提下可复用。
- 重点是 HBG 构图后随每次提交携带的大参数图像。
- kernel 不控制外层 ACLGraph capture 边界；不改 vLLM/PyTorch 框架执行层。
- 保持既定接口方向：内部 lazy init，不增加公开 workspace allocate API，
  不把 program 的两槽位 pipeline 并入 `kernel_launch`。

## 2. 源码定位过程

### 2.1 本地 L1 demo 确实通过 RTS 参数上传 HBG 图

以下路径相对本地 simpler 源码根目录；这是 demo 实现，不把它等同于正式 main 已完成的接口。

- `src/common/platform/onboard/host/device_runner_base.cpp:965`：
  验证执行槽容量后，把 `HbgGraphPlan` 序列化成 `hbg_launch_blob`。
- 同文件 `:1077`：HBG 通过 `LaunchWithMutableHostArgs` 下发完整 blob；
  TRB 则传入较小的 invocation 参数结构。
- `src/common/aicpu_loader/host/load_aicpu_op.cpp:770`：
  最终调用 `aclrtLaunchKernelWithHostArgs`。
- `src/a2a3/runtime/host_build_graph/host/runtime_maker.cpp:1820`：
  注释明确区分共享的可变 execution slot 和每个 launch/captured node 的独立参数快照。

因此，共享 execution slot 与共享 launch 参数不是一回事。
Host 的 plan cache 命中也不意味着 RTS 不再复制该 plan 的参数图像。

### 2.2 RTS 大参数的申请与回收

以下路径相对本地 `torch_npu/third_party/acl_src/runtime` 源码根目录。
安装库的具体构建版本未与该源码逐项对应，设备现象另由下一节探针确认。

```text
aclrtLaunchKernelWithHostArgs
  → rtsLaunchKernelWithHostArgs
  → CPU_EX 参数路径
  → LoadArgsInfo(..., LP_CPU_KRN_EX)
  → 大参数选择 randomAllocator_
  → DevMemAlloc(size, RT_MEMORY_HBM)
```

- `src/runtime/api/api_c_kernel.cc:42`：CPU 注册类型进入 `RT_ARGS_CPU_EX`，
  `isNoNeedH2DCopy` 设置为 0。
- `src/runtime/core/src/launch/aicpu_stars.cc:53`：加载 AICPU 参数并把分配结果绑定到 task。
- `src/runtime/core/src/kernel/arg_loader/uma_arg_loader.cc:804`：
  `LP_CPU_KRN_EX` 的大参数选择 `randomAllocator_`。
- 同文件 `:885`：按实际大小分配，失败返回内存申请错误；`:747` 负责参数 handle 的释放。
- `src/runtime/core/src/pool/h2d_copy_mgr.cc:149`：大参数直接申请 HBM；
  `:175` 的 random allocator 释放路径调用设备内存释放。
- `src/runtime/core/src/task/task_info/davinci/davinci_kernel_task_v100.cc:488`：
  AICPU task 清理释放或转交参数回收；stream 另有延后回收队列。

这是随在途提交增长的真实分配路径，不是已经证明的 allocator 泄漏。
参数分配成功不意味着后续还有足够内存提交任意数量的大图。

### 2.3 Capture 与 replay 的生命周期不同

- `src/runtime/core/inc/model/model.hpp:117`：model 记录参数 handle。
- `src/runtime/feature/aclgraph/capture_model.cc:879`：捕获模型保留参数 handle。
- 同文件 `:38`、`:79`：模型析构释放保留的 handle。
- `src/runtime/feature/model/model.cc:268`：模型清理释放记录的参数。
- `src/runtime/feature/aclgraph/capture_model.cc:209`：replay 进入已建立模型的执行路径，
  不是重新调用每个节点的 HostArgs 上传入口。

已有图的重复 replay 不应按 eager 每次重新上传整图来估算。
但同一个 capture 内新增节点，以及另外创建的图实例，仍可能新增长期存活的参数图像。

## 3. 有界设备复现

### 方法与限制

- 环境：安装的 CANN 9.2.0；可见 A3，ACL 返回 2 个逻辑设备；使用逻辑 device 1。
- 原预检固定查询物理 NPU 0，在仅暴露物理 NPU 7 的容器中失败。
  通过可见卡信息及 CANN SoC 配置确认 A3 后，按用户明确许可直接运行。
- `task-submit` 不可用；运行前后 `npu-smi` 未显示其他运行进程。本次没有队列锁隔离。
- 独立 C++ Host 程序加最小 AICPU SO，使用与 HBG 相同的 HostArgs 入口。
  不加载模型、不调用 PyPTO 编译器、不修改 production runtime。
- 一个最长 2 秒的 AICPU 任务置于队首；后续连续提交，不在各次提交之间 synchronize。
  参数上传阶段完成于该时间窗口内，保证采样时确有排队。
- 最大 eager 压力为 64 份、每份 16 MiB，即约 1 GiB 参数；capture 独立扫描最多 512 MiB。
  入场要求至少 8 GiB 空闲内存，没有逼近 OOM，也没有测试 2048 条任务容量极限。
- 只采 `aclrtGetMemInfo`、提交数量、耗时和 API 状态，不做 payload hash 或精度校验。
  该探针验证参数内存生命周期，不验证整网业务结果或 HBG 序列化内容的正确性。
- 三次运行均正常完成；没有 timeout、launch 失败或设备 reset。

### Eager：在途参数按份累积

在第二次运行中，每份参数为 16 MiB。表中是采样 API 报告的进程 HBM 占用，包含运行时基线，四舍五入到整数 MiB。

| 已提交大参数任务 | HBM 占用（MiB） | 距本轮开始（ms） |
| --- | --- | --- |
| 16 | 435 | 86.242 |
| 32 | 691 | 156.937 |
| 48 | 947 | 227.058 |
| 64 | 1203 | 299.783 |

每增加 16 次提交，增加 256 MiB，等于 `16 × 16 MiB`。
1 MiB 参数的对照中，16/32/48/64 次提交对应 160/176/192/208 MiB，斜率同样符合每次一份参数。

### Capture：新增节点持有新增参数

单独启动新进程测 capture，避免上一轮 eager 的后台回收混入基线；每个节点参数 16 MiB。

| 同一图内已捕获节点 | 相对 capture 前的 HBM 增量（MiB） |
| --- | --- |
| 1 | 16 |
| 8 | 128 |
| 16 | 256 |
| 24 | 384 |
| 32 | 512 |

`CaptureEnd` 后这 512 MiB 仍然保留。

### Replay：不按次数复制整图参数

对上述 32 节点图执行一次 warmup，然后在队首有界任务之后连续异步 replay 64 次。

| 已提交 replay 次数 | HBM 占用（Byte） |
| --- | --- |
| 16 | 691990528 |
| 32 | 691990528 |
| 48 | 691990528 |
| 64 | 691990528 |

64 次提交耗时 1.536 ms，全部位于队首任务的 2 秒窗口内。
这是连续异步 replay，不是每次 replay 后 synchronize 的测试。
1 节点图和 8 节点图的对照也没有观察到按 replay 次数增长的大参数占用。

### 完成与回收不是同一采样时刻

16 MiB eager 组 synchronize 刚返回时报告 1244172288 Byte；
再过 250 ms 报告 271093760 Byte。说明任务完成后的设备内存回收/统计呈现不是瞬时归零。
结合 RTS 的 task/stream 回收路径，不能把 synchronize 后的第一个高值直接判为泄漏，
也不能据此承诺“完成 event 到达就立即归还等量物理 HBM”。

## 4. 对正式设计的影响

结论：需要解决的是 eager 在途图像及 capture 持有的图像数量/字节规模，
不是为同一个已捕获模型的每次 replay 再配置一份整图存储。

当前路径的大致占用可以分解为：

```text
共享执行 heap/slot
  + 未完成 eager 提交持有的参数图像
  + 存活 capture 节点持有的参数图像
  + RTS 执行元数据及待回收/缓存内存
```

这里的图像预算与此前讨论的代码注册条目数、2 GiB 代码容量不是同一个资源池。

用户进一步确认最关心整网 eager 的增长：即使设备严格串行，Host 仍可提前提交，
RTS 会先为后续任务上传参数。若每轮图像大小为 G，尚未回收的图像有 N 份，
这部分占用约为 N × G，而不是一份 G。任务条目容量不构成字节预算，
所以可能远未排满 stream 就遇到 device 内存不足。
这是大图参数传输及生命周期设计的问题；当前证据不支持把它叫作 CANN 内存泄漏。

以下是待决定的实现方向，不是已经修改的功能：

1. 优先分开不可变图结构与每次调用变化的 Tensor 地址、Scalar、执行状态。
   可复用的图结构上传一次，launch 只传引用和必要的调用参数。
   Scalar 改变任务拓扑时仍必须产生正确的新图实例，不能通过复用旧图改变算子语义。
2. 对仍需逐次生成的大图，eager 需要内部字节准入/背压，不能只依赖 CANN task 条目上限。
   正常调用保持异步；额度耗尽时是否允许 Host 等待旧提交完成，需用户确认。
   如果大图副本继续交给 RTS 逐次分配，仅按 event 归还软件额度并不能严格约束即时物理 HBM 峰值。
3. 值得单独验证的另一条路径：大图保留在 pinned Host 内存中，通过 stream 有序 H2D
   写入固定的 device staging 区，再启动本次执行。必须保证前次最后使用先于下次覆盖，
   并保留 Host 源直到复制完成。它把排队大图的占用移到 Host；本次尚未验证此替代传输路径。
   排序形态为 `H2D(G0 → 图像区) → 执行 G0 → H2D(G1 → 同一图像区) → 执行 G1`。
   需要确认 pinned H2D 的实际异步行为、源数据保活、稳定 device 地址/容量及 capture 语义。
4. Capture 不能照搬普通的“两槽位满了就在 Host 等完成”。
   被捕获节点尚未执行，第三次提交若等待前两个节点完成，可能使 Host 无法到达 CaptureEnd。
   捕获图引用的不可变数据也不能在某一次 replay 完成后就覆盖或释放。
5. kernel 不控制 capture 边界，不等于无法识别 capture。
   当前 CANN 头文件提供 `aclmdlRICaptureGetInfo` 和 `aclmdlRIDestroyRegisterCallback`，
   可作为内部关联图资源生命周期的实现依据，不要求改框架执行层。
   回调路径、失败 capture 清理及跨版本支持仍需单独验证，不能直接宣称已接通。
6. 不应为了 eager 的内存问题，默认把已捕获图的 device 常驻图像改成每次 replay 都从 Host 上传。
   那会额外引入整图 H2D；是否作为低显存策略，需要明确性能取舍。

后续重点确认：Host 在额度耗尽时可否背压；整网 HBG 的主要变化是参数绑定还是任务拓扑；
图像内存预算及 pinned Host 传输方案的取舍。

## 5. 复现材料

本次临时实验目录名为 `pypto-hbg-args-memory.uW4STR`，包含：

- `device.cpp`、`probe.cpp`、`descriptor.json`：独立探针源码及 AICPU 注册描述。
- `memory.jsonl`：首次 eager/capture/replay 对照。
- `memory_with_reclaim.jsonl`：增加完成后 250 ms 采样的对照。
- `capture_nodes.jsonl`：新进程中 32 节点 capture 与 64 次连续 replay。
- `run.stderr`、`reclaim.stderr`、`capture.stderr`：三次运行 stderr，均为空。

构建使用当前 CANN 的 Host 编译环境和 HCC AICPU 交叉编译器；
复用本地 simpler 的 dispatcher，仅编译实验 SO 与 Host 探针，不触发 PyPTO 全量构建。
记录保留原始采样；不把进程 HBM 总占用全都归因于参数，不据此报告整网 OOM 已复现。

## 6. 当前 Qwen3-14B：实际构图尺寸与对应设备内存

### 6.1 测量入口与方法

本轮使用 `pypto-lib/models/qwen3_14b/decode_fwd.py::decode_fwd`，
对应 vLLM 单卡接入的完整 decode：embedding、40 层 decoder、LM head、采样。
没有使用 simpler 中仅包含 decoder stack 的 `qwen3_14b_decode` GraphExecution 示例，
也没有把那份示例的 GraphDefinition 大小当成整个 launch blob。

配置：A2/A3、TP1，hidden 5120、intermediate 17408、40 个 Q head、8 个 KV head、
head_dim 128、内部计算 batch pad 16、词表 pad 152064。
按 vLLM demo 的 1024-token 容量构造每条序列 8 个 128-token page 的元数据；
RoPE 表沿用 4096 行。序列长度 tensor 的内容由设备 attention tiler 处理，不是 host 构图输入。

测量步骤：

1. 用外部 tensor 的 `meta` 形状/类型运行当前 PyPTO lowering，只生成 host orchestration C++；
   不分配模型权重和 KV 数据，不编译或执行实际 AICore 模型 kernel。
2. 用本地 HBG 头文件编译该 orchestration；通过实际
   `build_l1_hbg_graph_plan_impl` 运行 host 构图并读取
   `HbgGraphPlan::serialized_size()`、`identity().host_total_tasks`。
   仅供 host 构图使用的合成设备地址与 function table 地址不会提交到 NPU。
3. 从实际 `runtime_reserve_layout` 获取共享任务区、runtime arena 和恢复区尺寸；
   在 orchestration 返回时记录 `ring.task_allocator.heap_top()`。
4. 在 logical device 1 上申请相同尺寸的三块可复用区域，并通过前节 AICPU 探针
   连续提交同等字节长度的 HostArgs，读取实际 HBM 增量。
   这一步使用无模型计算的 payload，验证的是 RTS 参数分配，不是 Qwen 计算正确性或吞吐。

环境处理：本地 Python binding 需要重建；全平台构建遇到与本次 host 测量无关的
device 编译错误后，改为仅构建 `_task_interface` 目标并使用已增量构建的 A2/A3 HBG host 库。
外部 CCE 的 include 路径通过已有 `PTO_ISA_ROOT` 指向当前依赖；未修改模型或 runtime 源码。

### 6.2 实际构图结果

| Public batch | Runtime task 数 | Task window 容量 | 序列化图包 bytes | 图包 MiB | 构图后 heap top MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 11,171 | 16,384 | 83,400,640 | 79.5370 | 262.5459 |
| 16 | 11,201 | 16,384 | 83,400,640 | 79.5370 | 262.5459 |
| 32 | 22,400 | 32,768 | 164,812,736 | 157.1777 | 461.9990 |

B1/B16 使用 264 MiB heap 构图成功；B32 使用 512 MiB heap 构图成功。
B32 按当前算子写法走两个串行的 16-row window，不是只增加几个 scalar 参数。

B1/B16 图包的组成：

| 组成 | bytes | MiB |
| --- | ---: | ---: |
| SharedMemoryImage | 81,412,416 | 77.6409 |
| RuntimeArenaImage 恢复区 | 1,987,968 | 1.8959 |
| Header + 两个 region descriptor | 256 | 0.0002 |
| 合计 | 83,400,640 | 79.5370 |

这里是完整序列化图像，不是 `.so` 文件，也不是模型权重。
`SharedMemoryImage` 按整个 task window 容量打包，包含固定宽度的 descriptor、payload、
slot state 和 completion flag，未使用的任务槽也在镜像中。
因此 B1/B16 的有效 task 数不同，图包字节数仍相同。
当前 capacity 为 16384，而有效任务数已经超过 8192，不能简单把 capacity 减半。

### 6.3 默认 256 MiB heap 不足以完成这份 HBG 构图

第一次按默认 heap 256 MiB 进行 host build，在分配到第 10,611 个 task 时阻塞：

```text
tasks=10611/16384, heap=266153984/268435456, on=heap
```

随后触发 host 构图分配器的保护退出。此时尚未执行设备 kernel，
不是 vector core timeout，也不是 RTS 排队导致的设备 OOM。
改用更大的探针 heap 容量后，完整图构建成功，最终 heap top 为 275,299,328 bytes，
即 262.5459 MiB；264 MiB 容量也已确认可以完成本次构图。
这里的 heap top 是本次构图后的分配位置，不把它宣称为所有输入下的最小安全规格。
没有替用户修改默认配置，也没有调整算子内存复用逻辑。

### 6.4 Device 1：每个待执行图包实际增加 80 MiB HBM

设备探针先申请可复用的三个区域：

- Heap：264 MiB。
- Shared memory：77.6409 MiB。
- 完整 runtime arena：10.9901 MiB，其中只有 1.8959 MiB 随参数包重复携带。

三者申请字节总和为 352.6310 MiB；本次 `aclrtMalloc` 与分配器记账下，
`aclrtGetMemInfo` 观察到的 HBM 增量为 **360 MiB**。
这不是完整 PyPTO context 的总占用，未计入模型 kernel binary、其他 runtime 资源和外部 tensor。

随后用 2 秒有限 gate 保持队列，连续提交 8 个 83,400,640-byte 参数包：

| 已提交且尚待执行的同尺寸参数包 | 相比三块复用区申请后的 HBM 增量 |
| ---: | ---: |
| 2 | 160 MiB |
| 4 | 320 MiB |
| 6 | 480 MiB |
| 8 | 640 MiB |

8 次提交在基线计时约 233 ms 时已完成，仍处于 2 秒 gate 期间。
因此这不是按文件大小推算：本机 CANN 对此尺寸的每份待执行 HostArgs，
实测增加 **80 MiB** HBM。同步和分配回收后的变化另保留在原始 JSONL，
不把同步后的即时占用解释为泄漏。实验结束后 NPU 无残留进程，AICore 为 0%。

对当前 B1/B16 情形，可写出近似内存预算：

```text
本次所测图相关 HBM 增量 ≈ 360 MiB 可复用区 + 80 MiB × N 个未完成 decode
```

其中 32 个积压图包约 2.5 GiB、64 个约 5 GiB，是按测得的每包增量外推，
没有实际提交这么多 Qwen decode。N 也不能直接用 stream 的 2048 个 task 条目代替。
结论是：复用 heap 能限制一部分内存，但不会消除每次 eager 提交保存的图参数副本。

### 6.5 本轮复现材料

临时目录 `qwen-hbg-size.fU7qmW`：

- `lower_qwen.py`、`qwen.cpp`、`qwen.so`：当前完整模型的 host orchestration 生成入口与产物。
- `measure.cpp`、`measure`：调用实际 HBG host builder 的尺寸探针。
- `b1-h264.jsonl`、`b16-h264.jsonl`、`b32-h512.jsonl`：三种 batch 的最终构图结果。
- `b1-default.log`：默认 256 MiB heap 的 host 构图失败记录。
- `device_memory.cpp`、`device_memory`、`device-memory.jsonl`：同尺寸 HostArgs 的设备内存实验。
- `device-memory.log`：设备实验 stderr，运行成功且为空。

Host 构图探针的运行形式：

```bash
./measure ./qwen.so 1 16384 264
./measure ./qwen.so 16 16384 264
./measure ./qwen.so 32 32768 512
```

最后三个参数分别为 batch、task window、heap MiB。

## 7. Device 内存申请失败后会发生什么

本节追加于 2026-09-14。必须区分三条路径：RTS 保存每次 launch 的大参数、
simpler 申请固定执行区，以及已经申请好的 heap 内部子分配。
以下 simpler 行为指当前本地 L1 demo，不宣称是主仓正式 kernel API 的最终行为。

### 7.1 大 HostArgs 申请失败：本次 launch 返回错误，不自动等待或补交

本地 RTS 源码的错误链如下，路径相对 `torch_npu/third_party/acl_src/runtime`：

1. `src/runtime/core/src/pool/h2d_copy_mgr.cc:149`：大参数直接调用
   `Driver::DevMemAlloc`；失败返回空指针。
2. `src/runtime/core/src/kernel/arg_loader/uma_arg_loader.cc:916`：释放本次 host arg handle，
   返回 `RT_ERROR_MEMORY_ALLOCATION`。
3. `src/runtime/core/src/launch/aicpu_stars.cc:56`：参数加载失败进入 `ERROR_FREE`，
   回收本次 `TaskInfo`。真正的 `SubmitTask` 在第 82 行，因此本次 AICPU task 尚未提交。
4. ACL API 返回 `207001 / ACL_ERROR_RT_MEMORY_ALLOCATION`。
   此错误码含义也见 [CANN 官方错误码说明](https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/apiref/appdevgapi/aclcppdevg_03_1276.html)。

这条大参数路径没有“等前面 decode 完成并释放参数，再重试本次 launch”的机制。
驱动层的巨大页失败后尝试普通页，是页分配策略回退，不是按在途任务数进行背压。
此前已成功提交的 device task 不会因本次参数申请失败而整体回滚；
失败的 decode 也不会在内存恢复后自动执行。

本机安装库另做了定点验证，避免仅凭本地 RTS 源码推断已安装 CANN 的行为：

- 遵循仓库 `testing` 规程，使用 A3 逻辑 device 1；`task-submit` 不可用，直接运行，
  运行前后均无该 NPU 上的其他进程。
- 沿用前面的 AICPU 有限 gate / 空任务，不运行 Qwen AICore，也不耗尽整卡 HBM。
- `LD_PRELOAD` 仅在探针进程、指定调用期间，把 **83,400,640-byte** 的
  `halMemAlloc` 返回值设为驱动定义的 `DRV_ERROR_OUT_OF_MEMORY = 6`；其他申请正常透传。
- 先提交 2 秒 gate，再提交同尺寸 HostArgs。失败注入命中两次底层申请，
  launch 在 **0.411 ms** 内返回 **207001**，没有等待 gate 完成。
- 关闭注入后，在同一 stream 上**显式重新提交**同样大小的参数，launch 返回 0，
  synchronize 返回 0；从 gate 提交到排空约 **2001.040 ms**。

原始结果位于 PyPTO 的 `tests/build/hbg-oom.GZe3Hw/result.jsonl`：

```json
{"stage":"injected_launch","logical_device":1,"args_bytes":83400640,"launch_rc":207001,"elapsed_ms":0.411,"driver_failures":2}
{"stage":"explicit_resubmit_after_disarm","launch_rc":0,"sync_rc":0,"gate_to_drained_ms":2001.040}
```

同目录保留 `probe.cpp`、`fail_alloc.cpp`、`stderr.log` 和本进程 `ascend/` 日志。
结果证明本次内存分配错误没有使这个 CANN stream 必然失效，
但不代表真实整卡 OOM 下所有清理动作都能成功，也不代表 PyPTO 自动恢复了。

### 7.2 本地 simpler L1：取消已提交分支，然后将 context 标为失败

HBG 的提交顺序是 hidden stream 上的 AICore 在前、大参数 AICPU 在后。
所以“大参数 AICPU 未提交”不等于“一次 PyPTO 调用完全没有提交任何设备任务”。

当前本地 simpler 源码有如下补偿，路径相对 runtime 根目录：

- `src/common/platform/onboard/host/l1_launch_sequence.h:106`：AICPU launch 失败后，
  尝试提交 `cancel_waiting_aicore`，随后等待 `AicoreDone` 并记录 `SerialTail`；
  清理提交成功时返回原始错误，清理提交失败时返回清理错误。
- `src/common/platform/onboard/host/device_runner_base.cpp:1103`：取消通过 caller stream 上
  `aclrtMemsetAsync(..., 0xFF, ...)` 写入 GM 握手区。
- `src/a2a3/runtime/host_build_graph/aicore/aicore_executor.cpp:85` 起：
  尚未打开寄存器窗口的 AICore 轮询握手区，看到 host-cancel 标记后返回。
- `device_runner_base.cpp:1130`：提交序列返回非零即调用 `poison(rc)`。
- `src/common/platform/onboard/host/l1_execution_state.cpp:167` 将状态设置为 `Poisoned`；
  `accepts_dispatch()` 在第 231 行排除该状态。即使后来内存空出来，原 context 也拒绝继续 launch。

这是失败清理，不是成功执行当前 decode，也不是透明重试。
本轮上板只验证了 CANN 分配错误返回及重新提交；上述 AICore 取消和 context 状态为源码结论。
若真实资源耗尽还导致 cancel/event 提交失败，不能保证已提交 AICore 一定能退出，
因此不能把 OOM 当作现成的、安全的流控机制。

Python 直接调用路径会将非零返回转为 `RuntimeError`（`src/common/worker/chip_worker.cpp:794`）。
若启用 torch_npu host task queue，Python 入队可能先返回；错误在队列线程执行时发生，
后续入队或同步时才显现。`torch_npu/csrc/core/npu/NPUQueue.cpp` 的消费失败处理会
标记队列错误并清理尚未执行的 host 队列项，并非忽略失败后正常继续。

### 7.3 申请固定执行区失败：初始化/准备失败，回滚区域

这条路径与 RTS 大 HostArgs 不同：

- `src/a2a3/platform/onboard/host/memory_allocator.cpp:28`：`rtMalloc` 失败后返回 `nullptr`。
- `src/common/platform/onboard/host/device_runner_base.cpp:1487` 的 `setup_static_arena()`
  申请 heap、shared memory、runtime arena；申请阶段任一区域失败，就释放该 bank 的三个区域，
  清空容量缓存并返回 -1。已冻结布局的变更请求在申请前拒绝，不进入这段回滚。
- 上层 L1 prepare 返回失败并标记 context 失败；不自动缩小容量或重新初始化。
  三个 region 之外已经创建的 context 资源，不等于此处全部释放，仍由自身关闭路径管理。

也不能笼统说“CANN 所有申请都不重试”：本地 RTS 的
`src/runtime/core/src/api_impl/api_impl.cc:2224` 在显式 `DevMalloc` 首次失败后，
会调用 `MemPoolTrimImplicit(true)` 并再尝试一次。再次失败才向上返回错误。
7.1 的大 HostArgs 直接调用 driver，**没有经过这段 trim/retry**。

### 7.4 内部 heap 耗尽：不是向 CANN 申请新 HBM 失败

前面 Qwen B1 默认 256 MiB heap 的失败属于这一类：
预设 heap 容量已经不足，但并没有因此自动调用 `rtMalloc` 扩容。
`src/a2a3/runtime/host_build_graph/runtime/pto_ring_buffer.h:159` 起会等待回收水位推进；
约 500 ms 无回收进展后报告 heap 耗尽并返回失败。
HBG 在 host 构图期间没有 device task 完成来推进回收，所以这次等不到空间。

`b1-default.log` 记录了 `FATAL: Task Allocator Deadlock - Heap Exhausted!`，
随后生成的 orchestration 访问失败任务的输出，触发 `AssertionError(index < output_count_)`。
当时独立 host build 探针未捕获该 C++ 异常，进程以 134 退出；
不能把这个退出形式泛化成所有 Python/L1 入口都会 abort。

阶段判断：不能直接依赖 RTS 自带的分配失败处理；后续主动回收机制及用户决定见第 8 节。
当前“已有 heap 的内部回收等待”“CANN stream task 条目容量”“大参数申请失败”
是三个不同机制，前两者不会自动替第三者实现按 device 字节预算的背压。

## 8. OOM 后主动回收：普通 sync 与 timeline event 的区别

追加于 2026-09-14；按仓库 `testing` 规程，在 A3 逻辑 device 1 继续做有界 CANN 探针。
驱动注入对 83,400,640-byte 参数设置最多 8 份的预算，8 次真实申请后令第 9 次返回 OOM；
预算在同步及重试期间一直有效，只有真实 `halMemFree` 成功返回才归还额度，不耗尽整卡。
探针同时记录申请/释放计数和 HBM；两次采样不是原子操作，表中以底层释放完成数为准。

| OOM 后的操作 | 同步返回瞬间已释放的旧参数 | 立即重试 |
| --- | ---: | --- |
| 仅 stream synchronize | 0/8 | 再次 207001 |
| 记录 `ACL_EVENT_SYNC`，再同步 stream 或该 event | 两种各测一次，均为 1/8 | 成功，但返回时未释放完 |
| 记录 `ACL_EVENT_TIME_LINE`，再同步该 event | 三轮均为 8/8 | 三轮均成功 |

另把预算降为 1 份，timeline event 同步返回时 1/1 已释放，立即重试也成功。
普通 sync 的观察轮在约 10 ms 时释放到 7/8，此后到 1 秒仍保留最后一份；
新 task 的回收或销毁 stream 才释放尾部参数。1 份预算下，成功重试后空等 1 秒再普通同步，
下一次申请仍遇到预算 OOM，说明 sleep 不能替代正确的回收机制。

### 8.1 为什么 timeline event 有效

本地 RTS `src/runtime/core/src/event/event.cc:737` 明确区分：非 timeline event
向 `SynchronizeImpl` 传入“不等回收”的标记；timeline event 传入自身 task ID，
从而进入 `Stream::WaitConcernedTaskRecycled`。
`davinci_kernel_task_v100.cc:488` 会把刚完成的 AICPU arg handle 留在 stream；
`stars_engine.cc:1871` 回收下一 task 时先释放这份 handle。
尾部 timeline event 因而既提供下一 task，也让 host 等到回收推进至它。

```cpp
// 内部初始化时创建；下面所有 API 返回值都需要检查。
aclrtCreateEventExWithFlag(&reclaim_event, ACL_EVENT_TIME_LINE);
// eager 的回收慢路径：在待回收任务之后记录，不在 capture 中执行 host 等待。
aclrtRecordEvent(reclaim_event, caller_stream);
aclrtSynchronizeEventWithTimeout(reclaim_event, timeout_ms);
```

`ACL_EVENT_SYNC` 的公开含义是跨 stream 同步，不是“等待所有内存回收”；
`ACL_EVENT_TIME_LINE` 的公开含义是记录时间戳，见 [CANN Event 文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/API/appdevgapi/aclcppdevg_03_0081.html)。
**回收保证来自当前 RTS 实现及本机实验，不将它宣称为跨 CANN 版本/芯片的公开 API 契约。**
`aclrtMemPoolTrimTo` 面向指定 SOMA pool，不是上述 random allocator 的 args 回收入口。

### 8.2 用户确认的方案与内存边界（2026-09-14，待实施）

- simpler 内部首次 launch 遇到可恢复 args OOM，清理/汇合已提交 AICore 后记录并同步 timeline event；
  sync 成功则重试一次，清理/record/sync 或第二次 launch 失败才返回失败，不提前 poison、不无限重试。
- 8192 条目/2 GiB/2 MiB 粒度仅约束 **simpler 自管 callable 驻留代码**，不约束 RTS 管理的大 args。
  后者是 HBG build_graph 结果随 launch 下发的 device 副本；timeline sync 回收它，不卸载 callable。
- 整网 vLLM/PyTorch 接入继续采用 kernel 模式；正常路径异步，不改框架、不新增公开 API 或 args 配额。
- capture 不能走此 host 等待；单图无法容纳等持续性 OOM 仍返回失败。本轮未验证真实 L1/整网恢复。
- 完整已确认流程、失败边界及后续验收见[正式接口设计第 2.4 节](../../../../tests/pypto_formal_interface_change_scope.md)。

材料保留在 `tests/build/hbg-sync-reclaim.gMfXVR/`，包括探针源码、各模式 JSONL 和本进程日志。
timeline 首轮用 event synchronize，后续三轮使用带 15 秒超时的版本。
`task-submit` 不可用，本轮直接运行；结束后 NPU 无残留进程，未修改 runtime 实现。
