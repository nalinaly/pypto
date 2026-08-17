# PyPTO L1 与 ACLGraph 实现过程记录

## 1. 文档目的

本文档持续记录 PyPTO L1 单算子形态及 ACLGraph capture/replay 支持的实际实施过程，包括代码基线、已确认设计约束、源码审计结论、分阶段改动、验证证据、提交记录、硬件门槛与未解问题。

本文档不代替设计文档 `tests/pypto_l1_aclgraph_implementation_plan.md`。设计文档是实现契约和评审基准；本文档是与实际代码同步演进的工程记录。如果实现中发现源码事实与设计假设不一致，应先在本文档中记录证据和影响，再决定是修正实现还是更新设计。

## 2. 工作区、分支与基线

创建日期：2026-08-17。

本次实现与 Grok 的并行工作物理隔离：

- 顶层工作目录：`/mnt/workspace/inductor/pto/gpt_pypto`
- 顶层 Git 分支：`gpt/pypto-l1-aclgraph`
- 顶层基线提交：`8ead22af0e285b89a3675d32e4b7ad54367d1151`
- simpler/runtime 子模块工作目录：`/mnt/workspace/inductor/pto/gpt_pypto/runtime`
- simpler/runtime 子模块分支：`gpt/pypto-l1-aclgraph`
- simpler/runtime 子模块基线提交：`3165cc89b6ea6b58a0bc01cbec2d5f72f2029c35`
- Grok 工作目录：`/mnt/workspace/inductor/pto/pypto`，本实现不在该目录写入任何文件。
- 顶层与 runtime 子模块均使用独立 worktree gitdir，避免分支索引和子模块 HEAD 互相覆盖。
- 实现过程中只做本地分阶段提交，不 push。

## 3. 不可退化的架构约束

以下条目是已经确认的实现红线，不得为了尽快跑通测试而弱化：

1. L1 对外是一个普通、异步、带 caller stream 入参的 AscendC 风格单算子。
2. AICPU orchestrator 必须发射在 caller stream；AICore executor 发射在 PyPTO 内部持久 hidden stream；内部双 stream 不对用户暴露。
3. caller stream 在算子入口 record `start_event`，hidden stream 在 AICore 发射前 wait；hidden stream 在 AICore 返回后 record `aicore_done_event`，caller stream 在算子出口 wait。
4. 任何 AICPU/AICore 内部并行都必须封闭在单算子 entry/exit 之内，不允许 orchestrator 越过 caller stream 上本算子的入口提前展开。
5. 不查询 capture 状态，不获取或保存 graph/model handle，不调用 `rtStreamAddToModel`，不设计 capture-only early mode。
6. eager 与 capture 执行完全相同的 runtime enqueue 拓扑。hidden stream 必须只通过 event fork/join 被 ACLGraph 自然捕获；如果目标 CANN 不支持，Phase 0 应明确失败，不能回退到历史 early-launch 方案。
7. L1 launch/prepare/close 不调用 stream synchronize 或 device synchronize；不在 launch 中创建、销毁 stream/event；不在 launch 中做 device allocation/free。
8. L1 不为 input/output tensor 分配设备内存，不做 H2D staging、D2H copy-back 或 per-run free。调用方的 device tensor 地址必须直接进入本次 invocation 快照。
9. workspace 在 v1 继续由 PyPTO 内部管理，但必须在 capture 前统一申请并 pin 住；launch 中只能复用，不得增长或搬迁。
10. v1 为静态 metadata 契约，不支持同一 context 并发 invocation 或并发 graph replay。PyPTO 当前占满全部 AICore，合法执行之间共享一份 workspace/runtime arena。
11. AICPU 每次调用使用 `aclrtLaunchKernelWithHostArgs` 的 runtime-owned 参数快照；AICore 每次只接收同一个 context-lifetime persistent device `KernelArgs *`。
12. callable ID 和 func ID 在 context 生命周期内不重绑。child/incore binary、callable descriptor、persistent Runtime/KernelArgs、workspace 都必须至少活到所有可能 replay 它们的 graph 销毁之后。
13. L1 close 不调用 `rtDeviceReset`、`aclrtResetDevice`、`aclFinalize`，不销毁 caller stream，不为调用方建立隐式 quiescence。
14. 现有 L2/L3 的全资源掌控、prepare/launch/poll/wait/finalize、故障恢复与 reset 语义必须保持不变。
15. 跨算子提前展开 orchestration 的性能优化属于未来 `host_build_graph` 图级方案，不能在 L1 里复活 private-AICPU-stream/early-launch 路径。

## 4. 实施前源码审计

### 4.1 当前 L2/L3 路径

- `simpler_run` 是 `simpler_prepare_run -> simpler_launch_run -> simpler_wait_run -> simpler_finalize_run` 的同步组合。
- `simpler_prepare_run` 在 caller 提供的 opaque storage 上构造 `OnboardNativeRunState`，为本次 run 申请资源、构造 `Runtime`、绑定 tensor/scalar，并可能做 tensor staging。
- `simpler_launch_run` 创建 host executor thread，调用 arch-specific `DeviceRunner::run`；L3 progress loop 依赖 poll/finalize 管理这份 per-run state。
- A2/A3 和 A5 现有 `run()` 会构造 per-run device Runtime/KernelArgs，发射 AICore/AICPU，再同步 stream、D2H diagnostics/copy-back 并释放 per-run state。
- ACLGraph replay 不会重新进入 PyPTO host progress loop，因此不能直接暴露 L3 native-run token 来充当 L1 op。

### 4.2 当前 device 初始化和 callable 注册

- `simpler_init` 调用 `attach_current_thread`，然后调用 `ensure_device_initialized`。
- `ensure_device_initialized` 创建持久 AICPU/AICore stream，bootstrap dispatcher，通过 `rtsBinaryLoadFromFile`/`rtsFuncGetByName` 解析 AICPU entry，并调用 `simpler_aicpu_init`。
- `ensure_aicpu_init_launched` 在内部 AICPU stream 发射 init 后立即调用 `aclrtSynchronizeStreamWithTimeout`。这个入口不能原样用于 L1 prepare。
- `simpler_register_callable` 上传整个 `ChipCallable`，记录 orchestration SO 和 child binary 地址，TRB 路径再调用 `launch_device_register`。
- `launch_device_register` 在内部 AICPU stream 发射 `simpler_aicpu_register_callable`，然后立即做 stream sync。L1 需要可以在 caller stream 上异步 enqueue 的独立 prepare 变体。
- `record_device_orch_callable` 已经保存 orchestration SO 的稳定 device slice、entry/config name、child `(func_id, device_addr)` 以及 signature，是 L1 append-only callable state 的可复用基础。

### 4.3 AICPU 和 AICore 参数生命周期

- 现有 AICPU per-task 路径使用 `rtsLaunchCpuKernel`，payload 是 front-less `KernelArgs`。`LoadAicpuOp` 已缓存所有 `rtFuncHandle`，可在不改 bootstrap 的前提下增加 WithHostArgs launch 方法。
- 当前 `KernelArgsHelper` 每次 run 分配并 H2D 一份 `Runtime`，再分配并 H2D 一份 `KernelArgs`，结束后释放。L1 不能使用这个 per-run helper。
- AICore CANN launch 实际仅携带一个 device `KernelArgs *`。只要把其指向的 state 改成 context-lifetime persistent 并将每次变化的 callable/tensor/scalar 移入 AICPU invocation，就不需要 PyPTO 自建 AICore task-args pool。
- `rtRegisterAllKernel` 只有注册/发射路径，当前 CANN 没有可用的对称 unregister API。L1 必须在 prepare 中提前注册并在 context/process 生命周期 pin 住 handle，不能在 launch 中懒注册或在 close 中调用未验证的卸载 API。
- child/incore binary 不通过 CANN 普通 kernel registry，而是随 `ChipCallable` 上传到 GM，由 `Runtime::func_id_to_addr_` 和 device dispatcher 使用。v1 可继续累积/pin 住，不必在本阶段实现 binary 回收池。

### 4.4 运行时能力与必须 onboard 验证的部分

- 当前 simpler 正式路径使用 low-level `rt*`/`rts*` launch family，仓库中尚无 `aclrtLaunchKernelWithHostArgs` 生产调用点。
- `aclrtLaunchKernelWithHostArgs` 是 L1 的关键新能力，必须在目标 CANN 头文件和 onboard 环境核对 `aclrtFuncHandle`/`rtFuncHandle` 兼容性、AICPU launch count 语义和 host args 快照行为。
- `rtKernelLaunchWithHandleV2` 是当前 AICore executor 的已用路径，但它能否成为 ACLGraph 可 replay 节点必须用目标硬件和 CANN 版本实测。
- hidden stream 能否仅通过 caller record/start -> hidden wait/launch/record -> caller wait/done 的 event 闭环被 ACLGraph 捕获，以及 event 在连续 replay/多 graph 顺序 replay 中的 generation 语义，必须由 Phase 0 onboard probe 确认。
- simulator 可覆盖 host state machine、ABI 和 unsupported stub，但不能作为 stream/event/ACLGraph 语义证据。

## 5. 初始实施分层

实际代码将按下列可独立评审和回退的单元推进：

1. Phase 0 最小 onboard probe：WithHostArgs 快照、event-only hidden-stream capture、mixed AICPU/AICore launch API、entry/exit marker、event generation/reuse。
2. 独立 L1 C ABI 与 execution mode：所有 variant 都导出 symbol，TRB onboard 报告 supported，sim/HBG 稳定返回 unsupported，L1/L2 mode 互斥。
3. `L1ExecutionState`：借用 device/context/caller stream，只拥有 hidden AICore stream、events、persistent device state、workspace 和 append-only callable table。
4. prepare-time persistent resources：提前分配/pin Runtime、KernelArgs、workspace、arena、regs/FFTS，提前注册 AICore executor，launch 时不再懒初始化。
5. async callable prepare：复用 host parse/upload，变更为 caller-stream AICPU register task + prepare tail event，不内部 sync。
6. 版本化 `L1AicpuInvocationArgs` 和 AICPU L1 entry：每次的 callable/tensor/scalar 只来自 WithHostArgs snapshot。
7. AICore persistent state 字段审计：清理对 `active_callable_id_`/`orch_args_storage_` 等 per-call 字段的早期读取，必要时增加 device-side ready epoch。
8. direct-device binder 和固定 event enqueue 协议：全部 validation 先于第一个 device task，partial enqueue 失败后 sticky poison。
9. simpler low-level Python binding：raw `uintptr_t` stream、强所有权、显式 close，不把 torch 依赖泄漏到 simpler core。
10. PyPTO/torch_npu adapter：取 current stream，做 taskQueue 边界适配，提供 `pypto_init/operator/prepare/warmup/close`，ACLGraph 主验证使用显式 `out=`。
11. eager/ACLGraph onboard 测试与 L2/L3 回归。

## 6. 过程日志

### 2026-08-17：工作区隔离和实现前审计

已完成：

- 创建独立顶层 worktree 和同名 runtime 子模块分支，确认与 Grok 工作区的 gitdir 分离。
- 完整阅读 `tests/pypto_l1_aclgraph_implementation_plan.md`，确认没有遗漏附录中的 ABI、资源所有权、参数生命周期、stream 协议、负面方案和测试矩阵。
- 完整阅读仓库规则、测试流程和提交流程；后续 build/test 必须在本 worktree 中使用项目 `.venv`，显式设置 `PYPTO_BUILD_JOBS`/`PYPTO_TEST_JOBS`，onboard 测试使用 `task-submit`。
- 阅读 runtime 的 developer guide、L2 三程序架构、L0-L6 层级模型、task-flow、AICPU launch 机制和 capability survey，核对设计文档中对现有 L2/L3 路径的描述。
- 审计 `pto_runtime_c_api.h`、onboard `c_api_shared.cpp`、`DeviceRunnerBase`、A2/A3 与 A5 `DeviceRunner::run/finalize`、`LoadAicpuOp`、`KernelArgsHelper`、TRB runtime maker 的关键入口。

当前结论：

- L1 不应通过在现有 `run()` 上加布尔分支实现，因为现有路径将 per-run Runtime/KernelArgs、diagnostics、stream sync、故障 reset 和 copy-back/finalize 紧密绑定。
- 应在 `DeviceRunnerBase` 中引入显式 execution mode 和独立 `L1ExecutionState`，同时保留现有 L2 virtual `run/finalize` 路径。
- 初始的 ABI 和 mode skeleton 可在无硬件环境下用 compile/host UT 验证；但任何宣称 ACLGraph 可用的代码都必须以 Phase 0 onboard 证据为前提。

## 7. 验证记录

当前尚未运行 build 或 test。此时只完成设计和源码审计，未修改生产代码。

后续每个阶段记录必须包含：

- 运行的完整命令和 platform/runtime 选择；
- 通过、失败、跳过的测试数量；
- onboard 设备型号、CANN 版本、device id 和 `task-submit` 任务证据；
- 失败的原始 API/error code，不用模糊的“不支持”覆盖；
- launch-time forbidden API counters/trace；
- L2/L3 回归结果。

## 8. 提交记录

尚无实现提交。每次提交后在此补充顶层/runtime 提交 SHA、中文提交主题、完整变更范围和对应验证。

## 9. 待持续核对的实现问题

这些不是重新打开的用户选择，而是需要用源码或 onboard 实验回答的工程事实：

1. 本地/target CANN 头文件中 `aclrtLaunchKernelWithHostArgs`、event record/wait/create/destroy 的精确类型和参数顺序。
2. `rtFuncHandle` 到 `aclrtFuncHandle` 的 ABI 兼容方式，是否需要经过 public binary/function registration 才能调用 WithHostArgs。
3. `rtKernelLaunchWithHandleV2` 在 ACLGraph capture 中的可 replay 性。
4. event-only multi-stream capture 在 A2/A3 和 A5 上的实际支持范围与 repeated-record generation 语义。
5. AICore entry 在第一次 device-side 等待/开窗之前读取的 Runtime 字段全集，决定是否需要 `L1InvocationEpoch`。
6. A2/A3 与 A5 对 handshake invalidation 的最小区域是否都只需 `aicore_done`。
7. torch_npu 当前安装版本的 current-stream 获取和 taskQueue custom-op 注册形态。
8. 当前顶层 `CompiledProgram` 可稳定导出的 static signature、output spec、assembled callable 和 resource sizing metadata，以及哪些需要新增只读接口。
