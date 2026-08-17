# PyPTO L1 与 ACLGraph 实现过程记录

<!-- markdownlint-disable MD060 -->

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
12. callable ID 在 context 生命周期内不重绑；每个callable内的 `func_id -> child binary address` 快照不可变。不同callable可以复用同一func ID数值并指向不同binary。child/incore binary、callable descriptor、persistent Runtime/KernelArgs、workspace 都必须至少活到所有可能 replay 它们的 graph 销毁且device真实quiescent之后。
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

### 2026-08-17：确认现有 bootstrap 不能直接进入 L1

继续沿 `simpler_init -> ensure_device_initialized -> ensure_binaries_loaded` 检查后，确认 L1 不能只把现有初始化函数的 stream 参数替换为 caller stream：

- `LoadAicpuOp::BootstrapDispatcher` 先同步申请并拷贝 dispatcher、inner AICPU SO 和 device args，再调用 `rtAicpuKernelLaunchExWithArgs`，随后直接调用 `aclrtSynchronizeStream`。
- 这次同步有实际的 host 依赖：dispatcher AICPU task 会把 inner SO 写到 device preinstall 路径；同步完成后 host 才能执行 `rtsBinaryLoadFromFile` 和 `rtsFuncGetByName`。因此不能机械地删掉同步，否则 host 可能在文件产生之前注册 binary。
- `ensure_aicpu_init_launched` 和 `launch_device_register` 也分别在发射 AICPU init/register 后调用 `aclrtSynchronizeStreamWithTimeout`。它们都不能作为 L1 prepare 的内部实现。
- 当前 L1 init 因此只保留 executor/dispatcher 的 host bytes，并创建 context-lifetime stream/event；不会调用上述 bootstrap。异步 bootstrap、function handle 准备和 init/register 必须拆成 L1 专用 prepare 路径。

这个发现不会改变“L1 prepare 不同步”的约束。它说明下一步需要改变 bootstrap 的数据流和完成性协议，而不是绕回同步初始化。

### 2026-08-17：建立 L1 ABI、执行模式和资源生命周期骨架

已在 runtime 子模块完成第一组生产代码改动，但尚未声称 ACLGraph 已可用：

1. 在 `pto_runtime_c_api.h` 增加稳定的 L1 错误码和四个 C ABI：
   - `simpler_l1_supported`
   - `simpler_l1_init`
   - `simpler_l1_prepare_callable`
   - `simpler_l1_launch`
2. 所有 host runtime variant 都导出同一组 symbol，避免 Python/C++ loader 因 variant 不同而出现 dlsym 形态分叉：
   - A2/A3、A5 的 onboard `tensormap_and_ringbuffer` 通过强 `l1_runtime_supported_impl()` 返回支持；
   - onboard `host_build_graph` 使用 common weak default 返回不支持；
   - simulator 明确返回不支持，并为其余 L1 API 返回 `PTO_RUNTIME_ERR_UNSUPPORTED`。
3. `ChipWorker` 在加载 host runtime 时解析全部 L1 symbol，并在异常清理和 finalize 中清空函数指针。当前尚未对外暴露 public L1 method，避免在底层能力未完成时给 Python 一个半可用入口。
4. 新增 `DeviceExecutionModeState`，让同一个 `DeviceContextHandle` 在整个生命周期内只能选择一种资源所有权：
   - `L2Owned` 保留已有 owned-device/reset 语义；
   - `L1Borrowed` 只借用 caller 已设为 current 的 device；
   - 初始化失败且所有自有 handle 已回滚时才允许回到 `Uninitialized`；
   - 如果 rollback 本身失败并留下 handle，context 保持 `L1Borrowed`，只能显式 close 重试，绝不落入 arch destructor 的 device reset 路径。
5. 新增可注入、与硬件头文件解耦的 `L1ExecutionState`。当前由它持有：
   - 一个 persistent hidden AICore stream；
   - `prepare_tail`、`start`、`aicore_done`、`serial_tail` 四个 persistent event；
   - `New -> Initializing -> Collecting -> ReadyEnqueued -> Sealed` 正常阶段；
   - `Poisoned` 和 `Closed` 故障/终止阶段；
   - 初始化 device 一致性检查、partial-create 逆序回滚、destroy 失败保留 handle 并允许显式 close 重试。
   - **后续状态机修正：** 实现又增加了粘性 `Closing`。第一项destructive teardown前进入Closing；任何destroy/free失败仍保持Closing和device claim/DSO/handle owner，拒绝prepare/launch，只允许显式retry close。完整状态见10.23.2。
6. `L1RuntimeOps` 的类型定义里只有 get-current-device、create/destroy hidden stream、create/destroy event。该操作表有意不提供 stream/device sync、device reset、ACL finalize、capture query、model handle 或 add-to-model 操作，使无硬件单测能直接约束资源边界。
7. onboard `simpler_l1_init` 先用 `aclrtGetDevice` 检查 caller 当前 device，拒绝 mismatch 后才复制可能很大的 binary buffers；`L1ExecutionState::initialize` 在真正创建 runtime handle 前再次做事务性检查。init 不调用 `aclInit`、`aclrtSetDevice`、现有 binary bootstrap 或任何 synchronize。
8. 当前 `simpler_l1_prepare_callable` 和 `simpler_l1_launch` 只完成 support、null 参数和 mode 校验，然后返回 `PTO_RUNTIME_ERR_NOT_READY`。这是有意的 fail-closed 行为：绝不为“先跑起来”而复用带 sync、staging 和 per-run allocation 的 L2 路径。
9. `simpler_init`、`simpler_register_callable`、`simpler_prepare_run`、`simpler_unregister_callable` 和 DMA workspace provision 均增加 L2 mode guard，防止 L1 context 误入 owned-device 路径。
10. `finalize_device` 按 mode 分流：
    - 未初始化或已关闭 context 返回成功；
    - L1 只调用 `finalize_l1_borrowed`，释放 L1 创建的 event/hidden stream 和 host bytes，不 reset device、不调用 ACL finalize；
    - L2/L3 继续走原有 arch `finalize()`，完成后标记 mode closed。
11. `destroy_device_context` 拒绝删除尚未显式 close 的 L1 context。`L1ExecutionState` destructor 本身不调用 runtime API；这会在 API 误用时保守泄漏 context，而不是销毁仍可能被 ACLGraph replay 引用的 handle。A2/A3 与 A5 derived destructor 也在 borrowed context 未关闭时拒绝 owned-device finalize/reset。
12. 新增 `test_l1_execution_state`，用 fault-injection fake runtime 覆盖：
    - L1/L2 mode 互斥和终态不可逆；
    - 只创建一个 hidden stream 和四个 event；
    - device mismatch 在任何资源创建之前失败；
    - 任意创建步骤失败时只逆序释放已经拥有的 handle；
    - rollback destroy 失败时进入 poisoned，保留失败 handle 给 close 重试；
    - wrong-current-device close 不释放任何 handle；
    - poison 后不能 seal/ready，但仍可显式 close；
    - close 幂等且第二次不再触碰 runtime。

### 2026-08-17：核对本地 CANN 9.2.0 ABI

本地安装 `/home/developer/Ascend/cann-9.2.0` 中已经确认：

- `acl/acl_rt.h` 的 `aclrtLaunchKernelWithHostArgs` 签名为：

  ```cpp
  aclError aclrtLaunchKernelWithHostArgs(
      aclrtFuncHandle funcHandle,
      uint32_t numBlocks,
      aclrtStream stream,
      aclrtLaunchKernelCfg *cfg,
      void *hostArgs,
      size_t argsSize,
      aclrtPlaceHolderInfo *placeHolderArray,
      size_t placeHolderNum);
  ```

- public `aclrtFuncHandle` 和 low-level `rtFuncHandle` 在当前安装里都定义为 `void *`。这只证明 C++ 类型/ABI 表达兼容，不证明由 `rtsFuncGetByName` 得到的 handle 可以直接传给 `aclrtLaunchKernelWithHostArgs`；后者仍必须由 onboard probe 验证。
- 不应依赖某个固定的 AICPU kernel launch count 上限。launch count 应来自 PyPTO 已有的平台/拓扑结果，runtime 内部规格即使可观察也不进入 L1 API 或资源回收设计。

## 7. 验证记录

### 7.1 构建环境

- Python 虚拟环境：runtime 根目录 `.venv`，由 `python3 -m venv --system-site-packages .venv` 创建。
- CANN 环境：`source /home/developer/Ascend/cann-9.2.0/bin/setenv.bash`。
- 构建并行度：`PYPTO_BUILD_JOBS=16`。
- 测试并行度：`PYPTO_TEST_JOBS=8`。
- editable 安装命令：`python -m pip install --no-build-isolation -e .`。

editable 安装已运行两次，均成功构建全部 runtime variant 的 wheel/extension。第二次构建包含本节记录的最终 L1 scaffold。

### 7.2 C++ 无硬件单测

构建/测试命令：

```bash
cmake -B tests/ut/cpp/build -S tests/ut/cpp
cmake --build tests/ut/cpp/build -j 16
ctest --test-dir tests/ut/cpp/build -LE requires_hardware --output-on-failure -j 8
```

结果：80/80 通过。

第一次全量 link 暴露出 hierarchical test target 未链接 `${CMAKE_DL_LIBS}`，在 glibc 2.31 环境中对 `dlopen`/`dlsym` 产生 undefined reference。`add_hierarchical_test` 已补上这个 test-only link dependency，随后全量通过。该修改不进入生产 runtime，也不改变 L1 行为。

格式化后又直接运行：

```bash
./tests/ut/cpp/build/test_l1_execution_state
```

结果：7/7 通过，覆盖 2 个 test suite。

### 7.3 Python 单测

命令：

```bash
python -m pytest tests/ut -q
```

结果：1093 passed、6 skipped、14 warnings，用时 64.52 秒。验证命令使用 `.venv/bin/python -m pytest`，没有误用系统 `pytest` entry point。

### 7.4 runtime variant 和 ABI 检查

通过 `nm -D` 检查所有已构建 host runtime，全部导出四个 L1 symbol。用 `ctypes` 调用 `simpler_l1_supported` 得到以下矩阵：

| Arch | Platform | Runtime | 返回值 |
| --- | --- | --- | ---: |
| A2/A3 | onboard | host_build_graph | 0 |
| A2/A3 | onboard | tensormap_and_ringbuffer | 1 |
| A2/A3 | sim | host_build_graph | 0 |
| A2/A3 | sim | tensormap_and_ringbuffer | 0 |
| A5 | onboard | host_build_graph | 0 |
| A5 | onboard | tensormap_and_ringbuffer | 1 |
| A5 | sim | host_build_graph | 0 |
| A5 | sim | tensormap_and_ringbuffer | 0 |

`nm` 同时确认 HBG 使用 weak support hook，TRB 链接 strong override。对 `L1ExecutionState` 两个源文件的静态 grep 确认没有引用 synchronize、reset、ACL finalize、capture 或 add-to-model runtime API。

### 7.5 lint 和格式检查

- `git diff --check`：通过。
- `clang-format --dry-run --Werror`：所有本次改动的 C/C++ 文件通过。
- targeted `pre-commit run --files ...`：header、English-only、large-file、EOF、trailing-whitespace、clang-format、cpplint 均通过。
- `clang-tidy` hook 受当前工具链环境限制未通过：它对多个既有 translation unit 报 `'stddef.h' file not found`，并对未修改的 `SimDeviceRunnerBase` constructor 报 `modernize-use-equals-default`。这不是 L1 编译/单测失败；原始错误已保留在本次过程记录中，不通过修改无关代码掩盖。

### 7.6 simulator scene smoke 的环境限制

尝试对 A2/A3 simulator 的 HBG/TRB vector scene 做 smoke。正确使用 `.venv/bin/python -m pytest` 后，两条路径都在加载 PyPTO 之前停止：测试工具链硬编码要求 `g++-15`，当前机器只有 GCC 9.4，因而无法生成 scene 的 host 编译产物。

这个结果不能作为 L1 失败或成功证据。simulator 本来就不承担 ACLGraph/stream/event 语义验证；其 host ABI unsupported 行为已由 build、symbol 和 `ctypes` 检查覆盖。

### 7.7 尚未完成的硬件证据

当前没有运行 onboard 测试，也没有通过 Phase 0。机器上未找到仓库要求的 `task-submit` 命令，因此在没有确认合规的设备调度方式前没有直接占用 NPU。

以下结论目前都不能宣称成立：

- `aclrtLaunchKernelWithHostArgs` 对现有 AICPU function handle 的可用性及 host args snapshot 生命周期；
- `rtKernelLaunchWithHandleV2` 的 capture/replay；
- event-only hidden-stream capture；
- start/done/tail event 在重复 replay 和两个 graph 顺序 replay中的 generation 语义；
- eager 与 ACLGraph 的单算子 entry/exit 闭包。

后续每个阶段记录仍必须包含：

- 运行的完整命令和 platform/runtime 选择；
- 通过、失败、跳过的测试数量；
- onboard 设备型号、CANN 版本、device id 和 `task-submit` 任务证据；
- 失败的原始 API/error code，不用模糊的“不支持”覆盖；
- launch-time forbidden API counters/trace；
- L2/L3 回归结果。

## 8. 提交记录

- 顶层 `ba3c9e8d35e0d0d3e45850b7d2f00c75e3d897f0`：`Add: 建立PyPTO L1与ACLGraph实现基线文档`
  - 增加不压缩上下文的实现设计文档和本过程记录；
  - 固化用户确认的单算子边界、workspace、参数生命周期、并发限制、Phase 0 gate 和历史 pto2 反例结论。
- runtime `4d844e001dac88437693fb14a39619e6fa88b304`：`Add: 建立PyPTO L1借用设备执行边界`
  - 增加四个 L1 C ABI、跨 variant capability gate、ChipWorker symbol loader 和稳定错误码；
  - 增加 borrowed/owned execution mode、persistent hidden stream/event 生命周期和失败回滚；
  - 从 mode、finalize、destructor 三层隔离 L1 与已有 L2/L3 device reset 路径；
  - 增加 fault-injection C++ UT，并记录 80/80 C++、1093/6 Python UT 与八种 runtime variant ABI 检查结果。
- runtime `7e8d141b6d10b72302bc207d1881f25f0a316ac1`：`Update: 完善PyPTO借用流L1执行闭包`
  - AICPU直接使用caller stream，隐藏AICore分支通过可捕获event形成单op内fork/join，launch不做同步、分配、capture query或binary registration；
  - 为WithHostArgs和taskQueue调用保留不可变参数快照，按callable隔离从0重新编号的func-id/address表，并把scalar count放入旧 `ChipCallable` header padding；
  - 让init/close失败保留可重试owner，增加sticky `Closing`、两阶段AICPU completion gate、scheduler-init最终裁决和invalid/zero register mapping的WAIT/CANCEL收口；
  - 完整runtime pre-commit通过，定向Python 51项和无硬件C++ 85/85通过。
- 顶层 `015cb87d5f996ffd9c9052a92c889951b717f430`：`Add: 支持PyPTO L1与ACLGraph调用`
  - 增加 `pypto_init`、`L1Context`、`L1Operator`、显式prepare/warmup和静态tensor/scalar ABI校验；
  - 增加独立 `_torch_npu_l1` adapter，以 `.stream(false)`、`RunOpApiV2`、C++ Tensor lease和allocator `recordStream`维持taskQueue顺序与storage生命周期；
  - 增加runtime scalar metadata贯通、51项高层反例、最小ACLGraph真实ST及runtime子仓指针更新；
  - 顶层editable build、完整pre-commit和top/runtime组合350项回归通过。
- runtime `11b7a4b1`：`Add: 建立HBG图包序列化与上传边界`
  - A2/A3和A5同步把host orchestration拆成纯Host SM image构建与显式H2D边界，保持旧L2同步上传语义；
  - 增加destination-bound variable HBG launch blob、pristine SM/runtime-arena/heap manifest、slot binding和invocation identity；
  - 用完整镜像覆盖、容量、generation、地址、身份、区间重叠与content hash做fail-closed校验；
  - runtime完整pre-commit、editable build、无硬件C++ 86/86和HBG相关Python 112 passed/4 skipped通过；HBG L1 capability未开启。

每次后续HBG提交继续补充顶层/runtime SHA、中文提交主题、完整变更范围和对应验证；不 push。

## 9. 待持续核对的实现问题

这些不是重新打开的用户选择，而是需要用源码或 onboard 实验回答的工程事实：

1. target CANN 头文件中 `aclrtLaunchKernelWithHostArgs`、event record/wait/create/destroy 是否与本地 CANN 9.2.0 完全一致。
2. 虽然本地 `rtFuncHandle` 和 `aclrtFuncHandle` 都是 `void *`，由 `rtsFuncGetByName` 得到的 handle 是否可直接用于 WithHostArgs，还是必须经过 public binary/function registration。
3. `rtKernelLaunchWithHandleV2` 在 ACLGraph capture 中的可 replay 性。
4. event-only multi-stream capture 在 A2/A3 和 A5 上的实际支持范围与 repeated-record generation 语义。
5. AICore entry 在第一次 device-side 等待/开窗之前读取的 Runtime 字段全集，决定是否需要 `L1InvocationEpoch`。
6. A2/A3 与 A5 对 handshake invalidation 的最小区域是否都只需 `aicore_done`。
7. torch_npu 当前安装版本的 current-stream 获取和 taskQueue custom-op 注册形态。
8. 当前顶层 `CompiledProgram` 可稳定导出的 static signature、output spec、assembled callable 和 resource sizing metadata，以及哪些需要新增只读接口。

## 10. 2026-08-18：从 scaffold 进入完整 L1 协议实现

### 10.1 设备使用约束

用户补充了本轮上板资源约束：

- PyPTO/GPT 工作目录的验证默认只使用 device 1；
- device 0 主要留给 Grok 的并行实现；
- 任意硬件命令前先检查 device 1 是否空闲；
- device 1 不可用时不得自行退回 device 0，必须先向用户确认。

当前仍未执行本分支的上板命令，因此没有占用任何 NPU。后续命令、日志和测试报告必须显式记录 device id，不能只写“onboard 通过”。

### 10.2 提交前快照：已经落地的native主体

本节保留 `4d844e00` 之后、形成下一笔提交之前的工作树快照，便于追溯实现顺序。下列内容后来经过构建、单测和审查，已纳入runtime提交 `7e8d141b`；这里的“已经实现”不再表示未提交状态。

1. AICPU binary loader 支持从内存初始化，并新增 `aclrtLaunchKernelWithHostArgs` launch 路径；binary/function handle 仍由 context 持有，不在单次 launch 中注册或释放。
2. 新增版本化的 `L1AicpuInvocationArgs`。每次 host 调用都形成独立、按值复制的 `KernelArgs + callable_id + ChipStorageTaskArgs` 快照，交给 CANN host-args task；AICore 不保留这个 task image 的地址，而使用 context-lifetime device `KernelArgs`。
3. A2/A3 与 A5 的 onboard AICPU shell 新增普通 kernel 形态的 `simpler_aicpu_l1_exec` entry；AICPU task 发射在外部 caller stream，AICore task发射在 persistent hidden stream。
4. TRB runtime maker 增加 L1 persistent runtime、arena、register table 的准备入口；launch 不再复用 L2 的 H2D tensor staging、per-run allocation、private AICPU stream或内部同步。
5. `DeviceRunnerBase::launch_l1_callable` 已用一个纯操作表 helper 表达固定 fork/join：caller wait 前序 tail、异步清 handshake、record start、launch AICPU；hidden wait start、launch AICore、record done；caller wait done、record serial tail。helper 的类型系统中没有 allocation、sync、capture query、model attachment 等入口。
6. native binding 新增 `_ChipWorker.init_l1`、deferred prepare/launch capsule、raw-stream调用和 GIL 释放。`L1QueuedCall` 复制 callable bytes 或固定 POD invocation snapshot，并用 ref-counted dispatch state把 C++ worker 生命周期延长到 taskQueue callback 执行完毕。
7. 进程内 runtime registry 强制同一 device 同时只有一个 live L1 context；duplicate context 在 host UT 中被拒绝。该限制符合 v1 使用全部 AICore、禁止并发 context 的契约。
8. `Tensor.make_strided` 增加了直接 device pointer descriptor构造和整数溢出保护；`DataType.BOOL` 同步暴露给 Python。

这些实现仍遵守明确禁止项：没有 `aclrtSynchronize*`、没有 `aclrtResetDevice`、没有 capture query、没有 `rtStreamAddToModel`、没有 private AICPU stream、没有 prepare-time预启动 orchestrator。

### 10.3 taskQueue adapter 的本地源码结论

对同仓 TorchAir 与 torch_npu 源码的核对得到以下确定结论：

1. TorchAir custom op 示例实际使用：

   ```cpp
   auto npu_stream = c10_npu::getCurrentNPUStream();
   auto raw_stream = npu_stream.stream(false);
   at_npu::native::OpCommand::RunOpApiV2(...);
   ```

   这里必须是 `.stream(false)`。`.stream()` 等价于默认允许 drain taskQueue，会破坏“把本次调用作为一个 queue item 延迟执行”的边界。设计文档中残留的 `.stream()` 需要统一修正。
2. `RunOpApiV2` 在入队时移动并拥有 `std::function`，所以 adapter 可以在 lambda 中持有一个纯 C++ `shared_ptr<Lease>`；Lease 构造时 retain `simpler.l1.queue_call.v1` descriptor，析构时 release。
3. 只保留 descriptor snapshot 仍不足以保护真实 NPU storage。Python tensor 在 taskQueue callback 执行前被释放时，allocator可能重用其地址；callback把 task enqueue 到 caller stream之后，lambda销毁也早于 device 完成。因此 adapter 必须同时：
   - 在 deferred object 中持有 `std::vector<at::Tensor>`，直到 callback真正完成 native enqueue；
   - 对每个唯一 storage调用 `c10_npu::NPUCachingAllocator::recordStream(..., current_stream)`，把 allocator 生命周期延长到 caller stream完成；
   - callback 内不能捕获 `nb::object`、`PyObject *` 或其他需要 GIL 才能析构的 Python 对象。
4. 当前独立 `_torch_npu_l1` adapter 采用 nanobind module，但通过 `THPVariable_Check/Unpack` 在入口同步取得纯 C++ `at::Tensor`，deferred lambda只持有C++ tensor、descriptor lease和raw stream。该nanobind/torch Python ABI组合已通过editable top-level实际编译和无device import，因此不需要改成 `PYBIND11_MODULE + <torch/extension.h>`；仍保持“不捕获Python object到无GIL callback”这条硬约束。
5. adapter 是可选顶层 extension，不让 simpler core或 `_task_interface` 链接 torch_npu；CPU 环境 `import pypto` 不应隐式加载 torch_npu。

### 10.4 scalar 参数数量缺失及修复方向

审查发现原 `ChipCallable.signature` 只包含 tensor，orchestration codegen也明确排除 scalar。因此若 L1 从 signature推导 `expected_scalars`，任何 scalar program都会错误地得到 0。

本轮已经把 scalar count 作为静态 callable metadata 单独贯通：

- `ChipCallable` 增加不可变 `scalar_count` 字段和 ABI bounds检查；
- orchestration codegen 的 `OrchestrationResult` 输出 scalar count；
- backend config/assemble/runtime maker 将该值一直传到 `CallableArtifacts`；
- onboard `CallableState` 保存它，launch 分别用 `signature.size()` 与 `scalar_count` 校验 tensor/scalar数量。

后续必须补 bytes round-trip、codegen config和 L1 scalar launch 回归，防止只在 Python wrapper里绕过 native校验。

### 10.5 多 `@pl.program` 的 func_id 冲突：关键协议修正

审查发现这是一个 P0 结构性问题：每个独立编译的 program 都从 `func_id=0` 开始编号，而初版 L1 在 context中维护一份全局 `func_id -> device_addr`。这样第二个 program的 kernel 0 会与第一个 program的 kernel 0发生地址冲突，和 `pypto_init(programs=[a, b])` 的目标不相容。

当前选定并正在实现的长久方案是“callable-local AICore 地址快照”：

1. 保留 L2 的 `RegisterCallableArgs` 不变，新增只属于 L1 的版本化 `L1RegisterCallableArgs`；
2. registration payload 同时携带 callable id、device orchestration SO descriptor以及最多 `RUNTIME_MAX_FUNC_ID` 个 `{func_id, device_addr}`；固定 image约 16 KiB，仍低于当前 host-args 64 KiB契约；
3. AICPU `orch_so_table_[callable_id]` 在保存 SO handle/entry的同时保存这一份地址快照；
4. 每次串行 L1 invocation 选择 callable后，由唯一 orchestrator线程在 graph build前把该 callable的映射 replay到 persistent device Runtime；
5. 不再要求不同 callable之间同名 func_id具有相同地址。v1 invocation 已强制串行，因此这一步不会与另一个 callable的scheduler并发改表；
6. 新 symbol使用 `simpler_aicpu_l1_register_callable`，避免改变已有 L2 register ABI。

这个修正也解释了为什么 binary device内存可以 context-lifetime累积，但 Runtime中的 task/function地址表不能简单全局累加：binary地址本身稳定，选择哪个 program的局部 func_id语义则属于每次 invocation。

### 10.6 CANN task-args 对齐问题

本地 runtime实现只能证明 task arg pool按 8 byte推进，不能证明它满足 `L1AicpuInvocationArgs alignas(64)`。直接把 runtime给出的 `void *arg` reinterpret_cast成该结构并访问字段属于潜在未对齐 UB。

当前修正为 A2/A3、A5 的 `simpler_aicpu_l1_exec` 都先 `memcpy` 到自然满足 64-byte alignment的局部 `L1AicpuInvocationArgs`，验证 version/size/reserved后才做 typed access。新的 L1 registration entry同样先复制 raw bytes到局部结构。该修正不依赖 CANN内部 pool的具体 alignment，也不编码内部约 2048 launch规格。

### 10.7 ACLGraph event 的两个直接阻塞

另一份同仓上板过程证据暴露了初版 event实现的两个 P0 问题：

1. `aclrtCreateEvent` 创建的普通 event在 capture中 record曾返回 `207000`。L1的四个跨 stream event必须在 prepare/init阶段使用：

   ```cpp
   aclrtCreateEventExWithFlag(&event, ACL_EVENT_SYNC)
   ```

   本地 CANN 9.2头文件和 runtime文档均确认 `ACL_EVENT_SYNC` 是跨 stream wait的 event类型；当前实现已改为该 API，后续必须由 device 1 Phase-0再次验证 capture。
2. 初版每次 launch都无条件 wait `PrepareTail`。该 event在 capture外的 prepare stream上record；warmup/capture切换 stream时可能被 ACLGraph判定为外部 capture dependency并返回 `107024`。prepare链只需由第一次 warmup消费；成功 enqueue warmup后，后续 capture launch不再导入 PrepareTail，而靠 normal invocation tail做串行化。当前 helper已增加显式 `wait_for_prepare_tail`，并新增“warmup消费、capture不再等待”的 exact-order UT。

`SerialTail` 同样可能在 warmup stream上record、由另一个 capture stream等待。目前设计文档要求外部 synchronize后通过它衔接不同 stream，但需要进一步用已有上板证据和 device 1 probe确认“已完成 event 的外部依赖”是否可 capture；在这一事实确认前不能宣称 event-only Phase 0成立，也不能用 capture query绕过。

> **后续协议修正：** CANN源码确认capture stream对capture外已record的new-mode event会按event记录状态触发capture-isolation，外部stream/device synchronize并不清除这个状态。正式实现不再等待旧 `SerialTail`：同caller stream依赖FIFO；host换stream时在任何enqueue前用 `aclrtQueryEventStatus` 非阻塞证明上一tail已complete，not-ready则fail-closed并要求调用方外部quiesce后重试。query complete后也不把旧event wait导入capture。这只能保护再次进入host launch的eager/capture边界，不能观测graph replay并发；v1仍要求调用方保证graph间无并发。

### 10.8 尚待立即修复的 native 阻塞

> 本节保留当时的blocker清单作为实现演进记录。四项已在后续transaction lock、pointer attribute强校验、failure-safe close/cleanup owner和context-wide prepare中闭环；最新状态见10.21–10.24。

1. `simpler_l1_prepare_callable` 当前在进入 runner 的 L1 mutex和 Sealed检查之前，已经解析 identity、上传 binary并写入 `callables_`。这会让首次 launch seal后的一次错误 prepare先产生分配/状态修改再被拒绝，也让并发 prepare/launch对 map产生 data race。整个 parse/upload/record/prepare必须收拢为同一把 L1 transaction lock，并在任何 upload前检查 phase。
2. native L1 tensor校验目前只检查 null、shape/stride/extent。必须对每个非零地址调用 `aclrtPointerGetAttributes`，强制 `location.type == ACL_MEM_LOCATION_TYPE_DEVICE` 且 `location.id == context device_id`，并在第一个 enqueue前失败。Python wrapper的 device检查不是底层强制的替代品。
3. `ChipWorker::finalize` 当前忽略 native finalize返回值，随后即使 `destroy_device_context` 拒绝仍可能 `dlclose` host runtime。这会留下 live C++对象/handle却卸载其代码和 vtable。L1 close失败时必须保留 context、DSO和function pointers，向 Python抛错并允许显式重试；init rollback失败也要遵循同一原则。
4. 高层 `op.prepare()` 必须委托 context-wide `ctx.prepare()`，按稳定 callable id一次准备所有声明 programs。native第一次 launch会 seal；只准备当前 op会让后续 program永远无法注册。

### 10.9 Python v1 使用契约的早期提案与后续修正

本轮高层只读复核建议保持 API 很薄：

```python
ctx = pypto_init(
    programs=[compiled_a, compiled_b],
    device=torch_npu.npu.current_device(),
    config=L1Config(),
)
op = ctx.operator(compiled_a)
ctx.prepare()                  # 一次 enqueue 全部 program，幂等
op.warmup(x, scale, out=y)     # 不内部同步
torch_npu.npu.synchronize()    # 用户负责
graph = torch_npu.npu.NPUGraph()
capture_stream = torch_npu.npu.Stream(device=ctx.device)
with torch_npu.npu.graph(graph, stream=capture_stream):
    op(x, scale, out=y)
graph.replay()
torch_npu.npu.synchronize()
graph.reset()
ctx.close()
```

补充约束：

- v1 只做显式 `out=`，不在 capture-safe调用中自动分配输出；
- 普通eager首调可以自动prepare并记为warmed；ACLGraph支持契约上要求用户显式 `prepare -> warmup -> external synchronize -> capture`。wrapper故意不查询capture状态，因此无法也不应在首次capture调用时自动补救错误流程；
- init/prepare均汇总并拒绝 mixed arch/runtime、HBG、sim、distributed、CommCtx、SDMA和多 orchestration；
- tensor每次可换地址，scalar值可换；shape/dtype/stride metadata在第一次成功enqueue后绑定（不限于显式 `warmup()`），adapter/direct enqueue同步失败不提交候选layout或 `warmed=True`；
- 参数进入 `ChipStorageTaskArgs` 时必须先全部 tensor、再全部 scalar，各自在 IR 中保持相对顺序；
- wrapper、adapter和 native分别承担静态签名校验、torch storage生命周期、device pointer强校验，三层职责不能互相替代；
- `close()` 幂等但不 sync，不在 `__del__` 自动 close；graph销毁与外部 quiescence由用户负责。

### 10.10 本节验证状态

> 本节记录的是2026-08-18当时那一个实施切面，不是当前最终验证结论。后续已完成editable build、无硬件C++/Python回归及新增ST收集，详见10.24。

上述 func-id、alignment、event、scalar、adapter变更仍处于同一未提交工作树；目前只完成源码级审查和局部 UT编写，尚未运行新的全量 build/test。因此本节只记录“已实现/待编译”，不把它们写成验证通过。下一步依次是：完成 prepare transaction、pointer validation、failure-safe close，构建全部 runtime variants，运行无硬件 C++/Python回归，修正设计文档残留，再按中文详细提交信息分阶段 commit。

### 10.11 用户补充：第二阶段 HBG 动态 graph image 属于 task package

> **历史结论修正（以10.20为准）：** 本节保留了从“不能覆盖尚未消费的graph bytes”出发的第一轮推导，但其中“graph image是每次invocation的不可变task package”和单一 `HbgTaskPackage` 图并不是最终对象模型。后续源码证据确认：不可变的是每个captured node持有的 **pristine source payload**，实际被scheduler消费的 **working image** 必须是可写 `HbgExecutionSlot`；此外canonical plan、一次性writable host blob、runtime-owned device payload也必须分层。本节原始推导作为设计演进记录保留，不应直接照此实现。

用户明确指出：第二阶段 `host_build_graph` 场景中，每次动态 build 的 graph 在 H2D 到 device 后，本质上是该次 task 的 AscendC `tiling_data` 类入参。CANN 的 AscendC launch 会让 runtime 跟随 task 管理 `tiling_data` 生命周期；PyPTO HBG 也必须建立等价的所有权协议，不能把 graph image 当成普通、可被下一次 host 调用立即覆盖的 workspace。

这一补充把 HBG device graph image 的语义确定为：

1. graph image 是**每次 invocation 的不可变 task package**，不是 context-wide mutable state；
2. package 中至少包含 graph image 的 device address、size、generation/identity，以及 AICPU/AICore 消费它所需的其他本次调用快照；
3. H2D 只是发布 package 的一个步骤。其生命周期下界不是“H2D 返回”或“kernel launch API返回”，而是消费该 graph image 的 device task真正完成；
4. host 连续异步 build/launch时，下一次调用不得覆盖前一次仍在 device队列中的 graph image；
5. ACLGraph capture把 graph image地址和 H2D/launch依赖固化到 graph executable后，该 image还必须覆盖所有可能的 replay，不能在 capture结束或首次 replay结束时回收；
6. 如果两个 ACLGraph分别 capture了两个动态 graph generation，即使它们来自同一 PyPTO context，也必须拥有互不踩踏的 package lease；
7. context-wide workspace仍可按 TRB v1规则持久化，但 HBG dynamic graph image不能因为“当前 PyPTO占满全部 AICore、禁止并发”就退化成单份覆盖。host调用串行不等于device消费已经完成，graph replay更不会重新进入PyPTO host代码给出可见完成点。

由此，第二阶段不能直接照搬 TRB 的单份 persistent `Runtime/KernelArgs` 方案。至少要引入如下逻辑对象：

```text
HbgTaskPackage
  ├─ host graph image / build metadata（若 H2D API在返回前尚未完成host snapshot）
  ├─ immutable device graph allocation {addr, size}
  ├─ invocation args snapshot（指向该 generation，而非可变全局 current graph）
  ├─ enqueue/copy generation
  └─ lifetime lease
       ├─ eager: device tail完成后才可进入可复用池
       └─ captured: graph executable不再可能replay且device quiescent后才可释放
```

当前可预见的内存管理分层是：

- **eager package pool**：PyPTO可以像 CANN runtime内部 task-args pool一样按 generation分配；只有通过真实 completion可见性证明前一 task完成后，slot才回到free list。不能依赖固定的 kernel-launch数量或“约2048”规格，也不能仅凭host API返回回收。
- **captured package lease**：capture中的device地址会进入graph executable。一次 captured record/event 的host状态不能证明未来 replay已经结束，因此普通tail event pool不足以决定回收。package必须被graph级owner pin住，直到graph销毁并由调用方完成外部quiescence。
- **地址复用规则**：允许回收后复用allocation，但不允许在lease仍live时原地改写graph bytes。相同content的immutable image未来可以做content-addressed共享，但共享必须带引用计数且不能把“内容相同”误当成“生命周期相同”。
- **H2D顺序**：graph image的H2D必须是该单个task enqueue闭包的一部分，并在caller stream/AICPU消费前建立明确happens-before；不能在private stream提前上传后越过算子入口，也不能靠内部stream sync收口。

这里存在一个必须在第二阶段设计前回答的信息边界：TRB L1坚持不查询capture、不持有ACLGraph handle；但 HBG captured package 的最后一次 replay和graph销毁并不能从普通launch入口内部观察。长期方案只能在以下能力中选择并用源码/onboard证据证明，而不能假装一条tail event能解决：

1. CANN是否提供可让 runtime随captured task复制并拥有任意graph/tiling payload的公开launch API，使device image的所有权直接成为graph node的一部分；
2. torch_npu/ACLGraph wrapper是否能在graph object生命周期上持有一个纯C++ `HbgTaskPackageLease`，graph销毁后再配合外部quiescence归还pool；
3. 后续 `host_build_graph` 图级执行器是否本来就拥有graph executable生命周期，从而由它显式管理package，而不是让单算子TRB L1层感知capture；
4. 如果以上都不可用，首版HBG ACLGraph只能采用append-only/pin-until-context-close的保守策略，并明确内存上界；不能静默复用仍被graph引用的地址。

当前倾向是把这项职责放在未来的图级 `host_build_graph` owner，而不是破坏本轮TRB L1的capture透明单算子边界。也就是说：TRB L1的task package主要是runtime拥有的WithHostArgs快照；HBG的dynamic graph image则需要新增显式generation/lease层。两者都遵守“参数快照不能被下一次host调用提前覆盖”，但完成可见性和回收owner不同。

第二阶段至少需要新增以下验证，且不能只测数值：

- 两次连续eager动态build，第一份graph尚未消费时第二份image地址/内容不得覆盖第一份；
- H2D、AICPU读取graph和AICore执行的stream/event顺序可追踪，launch路径没有内部sync；
- capture后反复replay，在host侧制造allocator压力和新的dynamic build，旧graph结果仍稳定；
- 两个不同ACLGraph分别持有不同generation并交替顺序replay，不发生image串包；
- graph对象销毁但未外部quiescence时仍不回收；完成quiescence并释放graph lease后才允许pool复用；
- context close只在所有graph lease释放且设备已由调用方排空后成功，否则fail-closed并保留可重试ownership；
- memory accounting能分别报告active eager package、graph-pinned package和free pool，不依赖CANN内部launch-count常量。

本节目前是第二阶段的硬设计输入，还不是已实现结论。正式进入HBG前，要先沿现有 `host_build_graph` 的build、H2D、AICPU args、device Runtime和finalize路径画出真实ownership图，再决定是采用runtime task-owned payload、wrapper graph lease，还是首版append-only保守策略。

### 10.12 native 审查闭环、C ABI callable 长度与无硬件验证进展

10.7～10.10 记录的是当时审查快照。继续实现后，其中列出的 native 阻塞已经按如下方式逐项收口；保留旧记录是为了让下游能看到问题是怎样暴露和演进的，不应再把旧小节中的“尚未修复”理解为当前状态。

#### 10.12.1 SerialTail 的 capture isolation 与跨 stream fail-closed 协议

进一步核对 CANN runtime 源码后确认：capture stream 等待一个在 capture 外 record 的 `ACL_EVENT_SYNC` event 时，runtime依据 event 的 `HasRecord`/capture隔离状态返回 `107024`；外部 `stream/device synchronize` 只等待task完成，并不会清掉这一record状态。因此“warmup外部同步后，在capture里wait旧SerialTail”仍然错误，不能靠同步消除。

当前实现改成：

- 单次 launch 不再把上一 generation 的 `SerialTail` wait导入capture；同一 raw stream依赖FIFO；
- host准备切换到另一 raw stream时，先用非阻塞 `aclrtQueryEventStatus(SerialTail)`检查上一 eager generation是否真的完成；
- status complete时允许换流且不wait旧event，not-ready时直接失败，要求调用方外部quiescence后重试；
- 这个query不是capture query，也不做任何同步；
- capture内部会为public event建立图内generation，因此host query只能证明先前 eager/warmup task完成，不能证明未来graph replay完成。graph→eager、两个graph交替或并发replay仍属于v1外部quiescence契约。

这个协议解决标准 `warmup -> external sync -> different capture stream`，同时对普通eager跨stream fail-closed；它不是并发执行探测器，不能被描述成支持并发graph replay。

#### 10.12.2 prepare/launch/finalize 的同一生命周期锁

`ChipWorker` 的direct prepare、direct launch、deferred queue callback与finalize现在共同持有 `L1DispatchState::mutex`：

- dispatch state通过C++17 `atomic_load/atomic_store(shared_ptr)`发布和撤销；
- direct入口不再绕过deferred callback使用的生命周期锁；
- finalize在锁内将state标为closed，并保持native finalize、context destroy和DSO卸载的互斥；
- native close失败时不卸载DSO、不清空仍需重试的ownership；成功后才原子撤销dispatch state。

这避免了另一个线程已经通过Python释放GIL进入native入口，而finalize同时卸载function pointer/DSO的竞态。

#### 10.12.3 tensor descriptor 的底层重算与device归属校验

新增纯CPU `l1_tensor_validation.h`，不再信任C ABI传入的 `extent_elem_cache` 和 `is_contiguous`：

- 用shape/stride溢出安全地重算 `1 + Σ((shape[i]-1)*stride[i])`；
- 重算contiguous属性并要求与descriptor一致；
- 校验start offset、dtype byte width及实际可达extent均在buffer size内；
- v1仍按静态shape契约拒绝zero-sized tensor；
- onboard入口随后调用 `aclrtPointerGetAttributes`，强制非空地址属于device memory且device id等于context device。

Python wrapper的检查只改善错误信息，不能替代这一层native强制校验。

#### 10.12.4 L1可重试teardown与L2历史teardown语义隔离

failure-safe close不能通过修改公共allocator/loader语义而破坏L2。当前区分为：

- `LoadAicpuOp` 的 `AclData`（L1）unload失败保留binary handle，允许显式close重试；`RtsFile`（L2）仍在失败后清空host ownership，避免device reset/ACL finalize后析构再次进入已经死亡的RTS；
- `InitFromData` 在binary load成功后立即把handle纳入成员ownership。后续function registration或host容器操作失败时，unload失败也不会丢失handle；
- `MemoryAllocator::finalize(true)`只用于L1并保留free失败项；L2使用默认best-effort drop语义，保持reset后的析构为no-op；
- L1 binary/device allocation清理只要有一项失败，就不销毁hidden stream/event、不释放per-device唯一context claim；全部前置资源释放成功后才close execution state。

这保证close返回失败时资源和DSO仍由原context持有，既可重试，也不会让第二个L1 context趁机claim同一device。

#### 10.12.5 launch capture路径不再保留lazy registration与临时字符串分配

- L1调用新的 `launch_prepared_aicore_kernel`，只接受prepare阶段已经pin住的AICore binary handle；launch路径无法再表达 `rtRegisterAllKernel` 的lazy分支。L2/L3历史helper仍保留原lazy逻辑。
- `LoadAicpuOp::LaunchWithHostArgs` 改为接受稳定 `const char *` kernel name并扫描已注册的小表，避免每次L1 capture launch把二十余字节symbol隐式构造成可能触发heap allocation的临时 `std::string`。

#### 10.12.6 `simpler_l1_prepare_callable` 新增精确blob长度

公开C ABI此前只有 `const void *callable`，native会直接信任柔性数组中的count/offset并进入 `compute_chip_callable_layout`。Python生成的对象虽然正确，但任意C调用方可伪造child offset、binary size或count，使hash/upload前就越界解引用。

现在prepare ABI显式携带 `size_t callable_size`，含义是canonical `ChipCallable` image的**精确长度**。新增纯CPU validator，并在任何hash、device allocation、H2D或registration之前检查：

- host pointer满足 `ChipCallable`自然对齐，长度至少覆盖固定header；
- tensor/scalar/child count均在ABI容量内；
- chip signature只包含tensor方向，child signature的enum值合法；
- function/config name的声明长度、内嵌NUL和末尾NUL一致；
- orchestration binary非空且在storage内；
- child func id在范围内且同一callable内唯一；
- 每个child严格位于 `make_callable()` 的canonical 64-byte offset，天然对齐、不重叠、不跳过隐藏payload；
- CoreCallable header和binary完整落在blob内，child binary非空；
- 最后一个child结束位置必须恰好等于传入长度，truncation和trailing bytes都拒绝。

`scalar_count`继续放在历史 `ChipCallable` header尾部padding，并保留所有旧字段/storage offset的static assert；旧factory产生的zero padding自然解释为scalar count 0，没有改变L3 wire布局。

#### 10.12.7 本轮无硬件构建与回归

在 `runtime` 中按仓库要求执行：

```bash
source ../.claude/skills/testing/load-env.sh
cmake --build tests/ut/cpp/build --parallel "$PYPTO_BUILD_JOBS"
ctest --test-dir tests/ut/cpp/build --output-on-failure \
  -R 'test_l1_(callable_validation|tensor_validation|aicpu_args|launch_sequence|execution_state)|test_chip_max_tensor_args|test_memory_allocator'
```

第一次加入callable validator后构建正确暴露出测试目标不应通过 `l1_aicpu_args.h` 间接依赖平台 `common/kernel_args.h`。随后把唯一的 `L1_MAX_KERNELS_PER_CALLABLE` 抽到无平台依赖的 `l1_limits.h`，validator和AICPU ABI共同引用，避免复制常量。重跑结果：

- C++ UT build成功；
- 选定的7个test executable全部通过；
- 新增 `test_l1_callable_validation` 覆盖canonical image、legacy zero scalar padding、截断/尾随字节、非法count、chip scalar signature、名称终止、child offset、func id和child binary越界；
- `test_l1_tensor_validation`、`test_l1_aicpu_args`、`test_l1_launch_sequence`、`test_l1_execution_state`、`test_chip_max_tensor_args`、`test_memory_allocator`保持通过；
- 该轮没有运行NPU命令，没有占用device 0或device 1。

环境loader仍打印既有警告：它对worktree的 `git-common-dir --path-format=absolute` 输出做 `dirname` 时把整段当成option；随后仍正确设置了安全并发度，构建实际以 `PYPTO_BUILD_JOBS=2` 运行。这个warning需要单独修loader，不能误记为本轮C++失败。

下一步是用editable runtime build覆盖真正的onboard host/runtime targets（上述no-hardware UT不编译全部CANN路径），然后实现高层 `pypto_init/L1Context/L1Operator` 和可选torch_npu adapter，最后才进入device 1 Phase-0。

### 10.13 HBG P0新结论：device graph是会被消费的执行模板，不能只做地址pin

沿现有 A2/A3、A5 `host_build_graph` scheduler和runtime destroy路径继续画ownership图后，发现10.11对“immutable graph image”的表述还不够精确。当前HBG上传到device的整块image并非执行期间完全只读；它更接近**可被scheduler原地消费的初始执行模板**：

- scheduler会写 `wake_list`、`task_state`、completion flags、ready queues、completed-subtask计数和watermark；
- run结束的 `runtime_destroy` 还会清 scheduler queue pointer、shared-memory handle和mailbox等运行态指针；
- `attach_populated` 的语义是附着到已经填充的image，并不会为下一次执行自动恢复这些字段。

因此，如果ACLGraph只捕获一次H2D和一次launch、随后对同一device地址直接replay：

1. 第一次replay会消费/修改working graph；
2. 第一次结束又会清理若干runtime pointer；
3. 第二次replay不会重新进入PyPTO host build，也不会天然重做初始化；
4. 即使该地址从未free、内容没有被下一次host build覆盖，第二次replay仍可能读取“已经消费/销毁”的状态。

这意味着“graph address有graph-lifetime lease”只是必要条件，不是充分条件。第二阶段长期对象必须进一步拆为：

> **历史结论修正（以10.20为准）：** 下面的两层图首次区分了pristine source与working slot，方向正确，但它仍把host canonical `GraphPlan`、会被placeholder原地patch的 `HbgSerializedLaunchBlob`、以及CANN task/captured node持有的 `RuntimeOwnedHbgPayload` 合并成了一层。10.20将其拆成完整五层模型。

```text
Immutable GraphPlan / HbgInvocationPackage
  ├─ pristine scheduler/SM/runtime template
  ├─ topology + control values
  ├─ tensor addresses + scalar snapshot
  ├─ callable/function bindings
  ├─ generation/content identity
  └─ binary/workspace/package leases

Mutable HbgExecutionSlot
  ├─ working scheduler state / ready queues / completion flags
  ├─ working SM + Runtime + KernelArgs
  ├─ restore generation
  └─ completion/quiescence proof
```

每次eager执行或每次ACLGraph replay都必须在Start event/AICPU消费之前，把pristine template恢复到working slot。候选实现只有经过源码与上板证明后才能选定：

- capture中的D2D restore node：由graph固定一份immutable device source，每次replay先D2D到working image；
- AICPU leader restore：entry先从immutable package按明确范围重置working state，再release其他scheduler线程；
- 更长久的结构性拆分：真正只读的GraphPlan与小得多的mutable SchedulerExecutionState物理分离，避免每次复制整个image。

> **restore候选的适用边界修正（以10.20为准）：** 上述三项不是三种可随意替换的局部写法。WithHostArgs inline source的device args base在launch内部才产生，host通常不能让一个更早的独立D2D node引用它，因此它天然与AICPU leader restore配对；D2D restore只适用于存在另一份有graph-lifetime lease的external stable device source。物理拆分则是后续结构/性能方向。

v1仍然禁止并发，所以**working execution slot和workspace**原则上可以context-wide只有一份；但每个captured node/graph generation引用的**immutable source package**不能都退化成“最后一次build的那一份”。两个graph若含不同tensor地址、scalar或拓扑，必须各自持有独立package lease，或者通过content-addressed immutable共享证明内容完全相同。

现有L2之所以不会在host连续调用时踩踏，并不是因为launch API返回就可复用，而是其双stream运行流程会等真实device完成后才宣告execution done，arena reservation通常保持到finalize；这套同步式L2 ownership不能原封不动搬进“不允许内部sync”的L1/ACLGraph路径。

另一个容易误判的点是 `aclrtLaunchKernelWithHostArgs`：它只深拷贝**host args结构本身的bytes**。如果结构里只是 `{graph_device_addr, size}`，runtime并不会因此拥有这个地址指向的device allocation，更不会为未来graph replay恢复其内容。

> **WithHostArgs表述修正（以10.20为准）：** 上句对“结构中只有一个指向外部allocation的pointer”是正确的，但“只拷贝固定结构体、不可能拥有graph payload”的强表述不正确。CANN实现会对 `[hostArgs, hostArgs + argsSize)` 覆盖的inline bytes走args-copy流程，`aclrtPlaceHolderInfo{addrOffset, dataOffset}` 还可以把header中的pointer字段patch为 `runtime_device_args_base + dataOffset`。因此“header + inline pristine graph bytes”是一条需要上板验证的runtime-owned tiling-like payload路径；它仍不会自动拥有inline blob之外的tensor storage、working slot或binary。

H2D source也必须纳入package：普通 `aclrtMemcpyAsync` 从pinned host源通常会把host地址纳入captured node，而不是承诺深拷贝任意source bytes；pageable host源又可能退化成隐式stream sync/同步copy并在capture中失败。因此不能把临时 `std::vector<uint8_t>` 的data指针传给async H2D后立即析构。可接受方向是graph-lifetime immutable pinned staging、device pristine source，或有公开文档保证的runtime-owned payload机制。

据此，10.11测试矩阵增加两个P0：

- 同一个ACLGraph不重新进入host build，连续replay至少2次，证明每次都恢复ready queue、completion flags、task state和runtime pointer；只验证第一次结果不算通过；
- 故意在第一次replay后污染/消费working slot，再验证第二次replay由graph内restore重建状态，而不是碰巧依赖初值未变化。

在没有通用graph retain/release接口前，append-only pin到显式context close仍可作为**immutable package source**的保守ownership fallback，但它仍必须搭配每次replay restore，不能把“永不free”误写成“可直接重复执行”。

### 10.14 TRB AICPU多线程错误尾声与两阶段cleanup gate

对L1第二次warmup/replay的源码级审查发现，TRB原有AICPU executor有两个会跨generation污染状态的问题：

1. 正常路径最后一个线程先写 `finished_=true`，再destroy runtime；其他block从 `run()`返回后只要看到true也会进入deinit，因此可能多owner并发清理，而且cleanup publish早于runtime destroy完成。
2. orchestrator在invalid callable、missing SO、missing L1 address snapshot、arg mismatch、arena/SM reset失败等front-matter路径，会写 `runtime_init_ready_=true` 唤醒scheduler后直接return；它没有shutdown，也没有参与N-way `finished_count_`。peers最多到N-1，本轮不deinit，下一轮可能带着ready=true、N-1 count和旧pointer进入，出现提前destroy、跨launch race或永久挂死。

当前 A2/A3、A5 同构修正为：

- `run_error_`保存本generation第一个错误；orchestrator在release-acquire发布 `runtime_init_ready_` 前先latch错误；
- 所有通过init的显式front-matter失败都跳转到共同 `run_epilogue`，不再直接return；
- scheduler acquire ready后先读shared error；失败时不dispatch，但仍执行幂等shutdown并参与完成协议；
- `ThreadCompletionGate`替代 `finished_/finished_count_`，唯一last-arriver执行runtime unbind/destroy，destroy完成后才发布finalization-ready；
- 所有block等待finalization-ready，随后各自读取最终 `run_error` 和runtime error status；
- 第二个departure计数确保所有block完成这些post-finalization读取后，只有唯一last-departure进入deinit；不会再发生deinit清零error或cache invalidation早于迟到block读取；
- init leader在每个generation开始统一reset gate和error；deinit也reset下一轮状态。

进一步补了decoupled handshake失败路径：有效参与线程即使 `init()`返回错误，也会latch错误、release可能已经进入run的peer，并加入同一个N-way finalize/depart协议；无效的额外affinity index不计入barrier。这样不会再因为一个scheduler在init阶段退出而让已经提前进入run的orchestrator永远等不到N个arriver。

对应验证：

- `test_thread_completion_gate`升级为两阶段语义，证明finalizer尚未结束时waiter不能越过，且只有第二个/最后一个departure取得cleanup ownership；
- reset后第二个generation仍只能finalize/cleanup各一次；
- 测试目标通过；
- 修改后还需要再次运行editable runtime build，确保两个架构AICPU编译器都接受共同epilogue与新增gate API。

### 10.15 torch_npu taskQueue 适配层：raw stream 获取、deferred callback 与 tensor storage 生命周期

本轮把“PyPTO核心不依赖torch_npu、taskQueue逻辑放在近乎独立的PyTorch convenience wrapper”落实为一个可选扩展 `pypto._torch_npu_l1`，而不是把torch头文件和链接依赖塞进 `simpler`/`pypto_core`。

核对本机TorchAir和torch_npu源码后确认，ACLGraph kernel extension使用的是：

```cpp
auto npu_stream = c10_npu::getCurrentNPUStream();
auto raw_stream = npu_stream.stream(false);
at_npu::native::OpCommand::RunOpApiV2(...);
```

这里的 `false` 不能省略。`NPUStream::stream()` 的默认重载会处理/排空taskQueue；L1 adapter处于“把本次算子闭包加入taskQueue”的入口，若这里隐式flush，会改变异步顺序并破坏capture调用形态。设计文档中残留的 `.stream()` 必须统一改为 `.stream(false)`。

adapter与simpler之间采用命名capsule ABI `simpler.l1.queue_call.v1`：

- simpler core只生成一个纯C++ descriptor，包含固定POD参数快照、invoke函数和retain/release函数，不链接torch；
- adapter进入Python绑定时立即retain descriptor，并用 `shared_ptr<Lease>` 管理；taskQueue复制/move `std::function` 时仍只有一个明确的底层lease引用协议；
- deferred callback只捕获C++对象，不捕获capsule、nanobind object或其他需要GIL的Python对象；
- callback调用 `lease->invoke(raw_stream)`，最终仍走同一个native L1 lifecycle mutex；
- adapter仍使用本仓现有nanobind模块体系，但入口通过 `THPVariable_Check/THPVariable_Unpack` 立即把Python tensor转成 `std::vector<at::Tensor>`。无GIL的deferred callback只捕获C++ Tensor handle、descriptor lease和raw stream value，不捕获nanobind/Python对象；
- enqueue前对每个唯一storage调用 `c10_npu::NPUCachingAllocator::recordStream(..., current_npu_stream)`。仅让Tensor活到taskQueue callback结束还不够：callback返回只意味着AICPU/AICore任务已经入caller/hidden stream，device可能尚未消费input/output；allocator必须知道这些storage仍被当前stream使用。

`RunOpApiV2`在同步插入taskQueue时会把op name深拷贝并move callable到queue slot，因此adapter无需依赖Python对象寿命；但这个事实只覆盖host callback本身，不覆盖tensor storage或capsule指向的native context，所以lease与recordStream两层都必须保留。

当前扩展已经通过editable top-level build，并完成无设备import检查：

- `import pypto._torch_npu_l1` 成功；
- 暴露 `enqueue`、descriptor/API version和ABI version；
- 没有调用current stream、没有enqueue task，也没有占用device 0/1。

后续上板测试仍需覆盖：调用后立刻删除Python输入、输出和queue capsule，再同步外部stream；结果必须正确，ASAN/日志不能出现callback使用已释放descriptor。ACLGraph场景还要求用户持有graph绑定的tensor，adapter的单次recordStream不能替代graph executable自身对输入输出的正常生命周期契约。

### 10.16 高层 `pypto_init / L1Context / L1Operator` 首版API

新增 `python/pypto/runtime/l1.py` 及公共重导出 `python/pypto/l1.py`。当前首版API形态为：

```python
from pypto.l1 import L1Config, pypto_init

ctx = pypto_init(
    programs=[compiled_program],
    device=test_config.device_id,
    config=L1Config(),
)
op = ctx.operator(compiled_program)

ctx.prepare()                  # enqueue全部声明program，幂等
op.warmup(x, scale, out=y)     # 仍然异步，不内部sync
torch_npu.npu.synchronize()    # ACLGraph capture前由用户显式排空

graph = torch_npu.npu.NPUGraph()
capture_stream = torch_npu.npu.Stream(device=test_config.device_id)
with torch_npu.npu.graph(graph, stream=capture_stream):
    op(x, scale, out=y)

graph.replay()
torch_npu.npu.synchronize(test_config.device_id)  # 调用方保证device quiescent
graph.reset()                                    # graph不再可能replay
ctx.close()                                      # 最后释放PyPTO持久状态
```

关键实现选择如下：

1. `pypto_init`是整体context初始化入口，先完成program/materialization/静态能力检查，再创建borrowed L1 native context；不存在“每个operator偷偷初始化一份runtime”。
2. `L1Context.prepare()`按声明顺序一次准备全部program。native在第一次launch后seal，若 `op_a.prepare()`只准备a，之后b将无法注册；因此operator的prepare委托给context整体prepare。
3. 普通eager首调允许自动prepare/warmup；ACLGraph流程要求用户显式 `ctx.prepare()`、执行一次warmup，再做外部sync。wrapper不查询capture状态，错误流程不会靠capture感知补救。
4. 首版只接受TRB onboard、单机单device、静态shape、正stride、显式 `out=`；拒绝sim、distributed、HBG、SDMA、CommCtx、mixed arch/runtime和动态/zero shape。这是本阶段边界，不意味着PyPTO以后要感知ACLGraph replay时所谓“动态shape”；capture的仍然是已经compile/prepare好的静态task。
5. 参数ABI严格为所有tensor在前、scalar在后，各组保持IR相对顺序。scalar值每次可以变化；tensor地址每次可以变化；shape/dtype/stride metadata在第一次成功enqueue后绑定，后续必须一致。
6. Python层验证device、dtype、shape、stride和参数数量，native层仍重新验证descriptor及pointer属性，不能把Python检查当安全边界。
7. 默认调用可选taskQueue adapter；direct raw-stream路径只作为调试/低层入口。二者最终都走同一个L1 native dispatch state与生命周期锁。
8. context绑定创建线程和device；跨线程/错误current device直接拒绝。首版同一device只允许一个L1 context，且不宣称支持不同graph或不同stream并发。
9. `close()`不做任何stream/device sync。显式close失败会保留native ownership供重试；析构函数只告警，不在未知graph/device状态下偷偷同步或close。

实现时发现并修复一个Python包循环依赖：若从 `pypto.runtime.__init__` eager导入L1模块，会形成 `CompiledProgram -> runtime.device_tensor -> runtime.__init__ -> l1 -> CompiledProgram`。当前公共入口从 `pypto.l1` 和top-level `pypto`重导出，未把L1 eager塞回runtime initializer。

当前无硬件Python验证：

```bash
source .claude/skills/testing/load-env.sh
source runtime/.venv/bin/activate
export PYTHONPATH="/mnt/workspace/inductor/pto/gpt_pypto/python:/mnt/workspace/inductor/pto/gpt_pypto/runtime/python:/mnt/workspace/inductor/pto/gpt_pypto/runtime:$PYTHONPATH"
ASCEND_RT_VISIBLE_DEVICES=1 python -m pytest -q tests/ut/runtime/test_l1.py
```

结果为12 passed。runtime wrapper侧：

```bash
cd runtime
source ../.claude/skills/testing/load-env.sh
source .venv/bin/activate
export PYTHONPATH="/mnt/workspace/inductor/pto/gpt_pypto/runtime/python:/mnt/workspace/inductor/pto/gpt_pypto/runtime:$PYTHONPATH"
python -m pytest -q tests/ut/py/test_l1_chip_worker.py
```

结果为2 passed。这里的12/2是首轮API骨架的历史记录；后续新增了failure ownership、scalar bit ABI、layout commit、adapter version/direct-path等反例，最新合并结果见10.24。两个测试集均使用fake native worker/program检查API状态机，没有发NPU task；环境变量只约束未来若意外触达runtime时可见device为1。另完成相关Python `compileall`。裸执行 `pytest` 曾从环境中拾取到Grok工作区的editable package，之后统一使用显式 `PYTHONPATH` 和 `python -m pytest`，避免两个并行worktree互相污染验证结论。

当前仍需要真正device 1 warmup/capture/replay证据；program数量上限、assembled callable最终签名、taskQueue adapter错误传播和首次成功enqueue才绑定layout已在后续实现/反例中补齐。

### 10.17 decoupled scheduler-init最终裁决必须先于orchestrator `p_func`

10.14的两阶段completion gate修复了“进入run后的所有线程如何共同退出”，但继续审查发现更早的竞态：decoupled模式下orchestrator可以在scheduler handshake尚未完成时通过一次 `run_error_ == 0` 检查并进入 `p_func`；某个scheduler随后handshake/assign失败，即使正确latch错误，也没有消费者处理orchestrator继续提交的ring task。大图可能卡在flow-control等待，L1 borrowed模式又不能依赖后续device reset解围。

只在 `p_func` 前再读一次error仍有check-after-failure窗口。当前A2/A3和A5同构实现改为一个明确的scheduler-init verdict barrier：

- 每个scheduler完成自己的handshake和core assignment后，成功或失败都exactly-once增加 `hs_arrived_`；
- scheduler leader acquire等待全部scheduler到齐，汇聚所有register-window/core-state写入；
- leader唯一执行失败时的emergency shutdown，或成功时的post-handshake DFX init；随后用release store发布 `init_done_` 和最终成功/失败裁决；
- 其他scheduler acquire等待同一个 `init_done_`，所有scheduler返回一致结论；
- orchestrator仍可与handshake并行完成config、arena、shared-memory reset和runtime wiring，但在发布 `runtime_init_ready_`、bind orchestration SO和调用 `p_func`之前必须acquire等待scheduler verdict；失败则latch error并直接进入共同epilogue，绝不调用 `p_func`；
- serial路径若core assignment失败也先emergency shutdown，再返回错误。

重新执行editable runtime build后，A2/A3和A5目标均编译通过；review逐项检查normal、front-matter先失败、scheduler-init失败三条时序，未发现double-arrive、missing-arrive、generation reset或memory-order回归。`hs_arrived_`的acq_rel RMW链、leader acquire load及 `init_done_` release/acquire构成完整publication链。

### 10.18 fault-only问题及闭环范围：invalid physical core id的pre-window退出

上述barrier仍不能解决一个更底层的异常：AICore启动后在自己的handshake区域发布physical core id，然后等待AICPU通过该physical id打开register window、写入 `DATA_MAIN_BASE`。若AICPU读到的id超出平台上限，它会判scheduler init失败，但无法得到合法register address；现有emergency shutdown只遍历已经保存的非零 `reg_addr`，因此这个invalid core仍会停在 `DATA_MAIN_BASE == 0` 的AICore轮询里。历史L2注释把它交给后续device reset，而L1明确不能reset设备。

这是fault-only硬件异常路径，不影响目前的happy-path构建结果，但在声称L1“任何失败都可终止、close可重试”前必须处理。候选协议是复用现有64-byte `Handshake.aicpu_ready` 字段作为pre-window `WAIT/PROCEED/CANCEL`状态：AICPU发现invalid id后发布CANCEL并flush相应GM cache line；尚未观察到register window的AICore周期性invalidate该handshake line，观察到CANCEL后退出。是否需要AICore显式ACK、如何不改变L2/L3正常路径性能、以及如何确保A2/A3与A5 cache ordering完全一致，正在做专项只读评审，未贸然写代码。

**后续实现结论：** TRB L1中“AICore已进入且完成report，但reported physical id越界或对应register address为0”已闭环：

1. `aicpu_ready` 定义error-only `WAIT=0/CANCEL=2`语义；正常路径仍以AICPU打开 `DATA_MAIN_BASE` window作为放行条件，不恢复旧版每次额外AICPU→AICore ready round-trip。
2. 每轮L1整块async memset清零handshake；L2/L3历史路径也同时清 `aicpu_ready/aicore_done`，避免上一次错误CANCEL污染下一generation。
3. AICPU scheduler对physical id先做bounds-safe lookup；越界或在范围内但PG/register mapping为0都先对对应handshake cache line发布CANCEL并clean，绝不调用 `platform_init_aicore_regs(0)`。
4. AICore在 `DATA_MAIN_BASE == 0` 轮询中每256轮才invalidate/check一次CANCEL；CANCEL保持到kernel结束，因此不会漏掉，同时避免正常路径每轮DCCI重现历史preamble性能回退。观察到CANCEL后不访问任何SPR/register window，直接退出。
5. 不额外要求per-core ACK。CANCEL在本轮内持续有效，hidden AICore kernel完成event与caller-stream wait是整个kernel的collective completion/ack，重用/释放不会越过operator tail。
6. A5 PMU-enabled kernel原本在握手前直接以 `get_physical_core_id()` 索引PMU register table，intrinsic真返回越界id时CANCEL尚未生效就可能OOB。现已用host实际provision的108项表长做无符号边界检查；只有合法id才读表，越界/null把PMU base设0，再交由正常report/CANCEL协议收口。

该结论只覆盖TRB且core至少已report的异常。若某个AICore完全未进入/未report，`aicore_done` 永远为0，AICPU无从知道应对哪个handshake发CANCEL；在L1禁止device reset和内部sync的前提下，这属于CANN op timeout/driver fault containment/外部context-device recovery负责的硬件失联边界，不能声称已由单算子协议恢复。同样不将这一TRB闭环外推到尚未接入L1的HBG。

### 10.19 用户再次确认的HBG原则：graph就是每个task的tiling-like launch payload

用户进一步明确：第二阶段每次动态build并H2D的graph，在语义上就是该task类似AscendC `tiling_data` 的入参。这里要区分两件容易混为一谈的ownership：

- `aclrtLaunchKernelWithHostArgs`/runtime task-args池可以替PyPTO保存 `{graph_addr, graph_size, generation}` 这几个标量字段；
- 但除非存在公开且经验证的runtime-owned payload API，它不会因为保存了pointer，就自动拥有pointer指向的graph allocation、H2D source或可变working image。

> **后续源码结论修正（以10.20为准）：** 第二条对“外部pointer pointee”仍成立，但CANN已提供一个可用作P0候选的公开机制：把pristine graph bytes直接放入WithHostArgs的 `argsSize` 范围，再用placeholder把pointer指向inline bytes的runtime-owned device copy。这不等于已经证明“任意大小payload都会随captured graph完整存活”；大args完整copy、graph replay/destroy lifetime和A2/A3、A5不同backend行为仍是device 1 P0门槛。

因此第二阶段的长久接口目标应是“让一个HBG launch node拥有一个等价于AscendC tiling参数的 `HbgInvocationPackage` lease”，而不是“把graph pointer塞进host args”。这个package至少持有：

```text
HbgInvocationPackage（每个dynamic build/captured node）
  ├─ immutable GraphPlan / pristine device template
  ├─ 必要时的immutable pinned H2D staging
  ├─ tensor/scalar/function binding snapshot
  ├─ generation + size + integrity identity
  ├─ working-slot restore描述
  └─ lease
       ├─ eager：真实tail完成后可回收
       └─ ACLGraph：graph不再replay + 外部quiescence后才可回收

HbgExecutionSlot（执行时可写）
  ├─ scheduler queues / task states / completion flags
  ├─ Runtime / KernelArgs / working SM
  ├─ context-wide workspace（v1无并发时可共享）
  └─ 本次execution completion state
```

这个模型保留了用户指出的AscendC调用直觉：对外仍是一条普通单算子launch；graph package像tiling data一样跟着该node活着；AICPU/private AICore细节不暴露。但内部不能照搬“固定一份device task memory”的TRB实现，因为同一ACLGraph会反复replay，而当前HBG scheduler会消费/清空working image。每次replay前的restore必须成为该node的一部分。

第二阶段开始前需要优先核对CANN/torch_npu能否提供以下任一ownership hook：

1. launch API对任意payload做runtime-owned deep snapshot，并让snapshot随captured task/graph node存活；
2. ACLGraph wrapper能在graph executable上挂纯C++ lease与析构回调；
3. `host_build_graph`图级owner天然掌握graph executable生命周期，可承担package allocator；
4. 若都没有，先使用有内存计数和上限的append-only pin-until-context-close策略，绝不在未证明安全时复用地址。

在这个问题被验证前，不把HBG支持合入当前TRB L1的完成定义，也不为了提前支持HBG而让TRB单算子层感知capture/model handle。

### 10.20 HBG第二阶段完整修正：WithHostArgs inline payload、五层所有权与per-replay restore

#### 10.20.1 状态和范围

本节是对10.11、10.13、10.19历史推导的源码级补完，也是未来第二阶段 `host_build_graph + L1 + ACLGraph` 的指导性设计，**不是当前已实现能力**。当前完成定义仍只覆盖 `tensormap_and_ringbuffer` 的L1；HBG仍由common L1入口返回unsupported，它的host orchestration DSO也仍被现有L1 registration路径明确拒绝。本节不改动TRB L1高层API、workspace策略、stream边界或完成标准。

已核对的当前HBG事实是：

- host在本地 `std::vector<uint8_t> host_sm_buf` 中构建pristine SM，按最终 `device_sm/device_arena` 地址做relocation，再用同步 `rtMemcpy` 上传；
- host另外在本地 `DeviceArena host_arena` 中构建runtime arena，再上传到独立device runtime-arena region；
- GM heap、GM SM和runtime arena是三个独立device region，outer `Runtime`、device `KernelArgs`和handshake又是额外的device执行状态，不能统称为一块“graph image”；
- scheduler会改写ready queue、wake list、completion flag、watermark、task state、subtask counter等字段，`runtime_destroy` 还会清理多个runtime pointer；
- `attach_populated` 只重建wrapper/header指针，明确不重置已填充的flow-control和slot state；
- 因此当前HBG上传的不是可直接重复replay的immutable executable image，而是一份按特定device base完成relocation、随后会被执行消费的initial working template。

A2/A3和A5的HBG `runtime_maker.cpp`、`aicpu_executor.cpp`和 `pto_runtime2_init.cpp` 当前同构，下面的所有权问题同时适用两类平台；但CANN板上args-loader backend、cache行为和大args限制仍必须分平台验证。

#### 10.20.2 WithHostArgs的精确ownership边界

CANN源码将 `aclrtPlaceHolderInfo` 定义为：

```text
aclrtPlaceHolderInfo {
    uint32_t addrOffset;
    uint32_t dataOffset;
}
```

WithHostArgs的AICPU args-loader在分配runtime device args存储后，会将：

```text
*(uint64_t *)(hostArgs + addrOffset)
    = runtime_device_args_base + dataOffset
```

再将args bytes复制到device。GE的AscendC tiling打包同样使用“tiling pointer字段 + inline tiling bytes”布局，runtime将pointer指向args copy内的tiling bytes。由此必须区分两种情况：

1. `hostArgs` 中只有 `{external_graph_device_ptr, size}`：runtime只拥有pointer value，不拥有pointer指向的allocation；10.13和10.19原有结论对这种情况仍然正确。
2. `hostArgs` 直接布局为 `[header | inline pristine SM | inline pristine arena | optional initializer bytes]`，并用placeholder将header pointer指向inline区域：这些inline bytes会进入runtime args-copy流程，因此具备像AscendC tiling data一样由task/captured node持有的可能。

这条路径有四个不能模糊的边界：

- placeholder patch会**原地改写调用方传入的writable hostArgs**，因此canonical `GraphPlan` 不能直接作为launch buffer；
- runtime只可能拥有 `argsSize` 覆盖的inline bytes，不因此拥有external tensor storage、working execution slot、workspace、kernel binary或inline blob之外的任何pointee；
- public API的 `argsSize` 在当前CANN内部被cast为 `uint32_t`，placeholder offset也是32位；PyPTO必须独立校验所有size/offset/addition，不得依赖溢出后的runtime行为；
- 源码能证明capture task会强制走args copy，但不足以代替板上证明“该allocation在graph instantiate、多次replay、graph destroy前全程存活且不被池复用”。

当前PyPTO `LoadAicpuOp::LaunchWithHostArgs` 仍把 `const void *` 强转为writable pointer，并且把placeholder array固定为null。第二阶段不能偷用这个固定入口；应为HBG定义variable-length、writable、有严格ABI校验的launch blob入口。TRB `L1AicpuInvocationArgs <= 64 KiB` 是当前固定结构体的自我约束，不是HBG payload的CANN公开上限；CANN源码内部出现的256 MiB等常量也不是PyPTO可依赖规格。

#### 10.20.3 五层对象模型

第二阶段必须使用下列五层对象，不再用一个含义模糊的 `HbgTaskPackage` 同时指代五类内存：

这里的规范性结论是：**每次dynamic host build产生的graph是该次HBG launch task/captured node的tiling-like payload**，不是context-wide `current_graph` 或workspace。其中随task/node生存的是不可变pristine source；在v1无并发前提下可被多task顺序共享的只是每次replay都要重建的mutable execution slot/workspace。

| 层 | 对象 | 内容与可变性 | owner | 最短安全生命周期 |
| --- | --- | --- | --- | --- |
| 1 | `HbgGraphPlan` | host canonical immutable plan；pristine SM、pristine runtime arena、optional GM initializer spans、topology、tensor/scalar/function snapshot、generation/hash、relocation/binding metadata | PyPTO本次host build | 完成第2层序列化；调试/cache可更长 |
| 2 | `HbgSerializedLaunchBlob` | 一次性writable host bytes；`[header \| inline payloads]`，placeholder会原地patch pointer字段 | PyPTO host launch调用 | 直到WithHostArgs已完成snapshot；精确边界需P0证明 |
| 3 | `RuntimeOwnedHbgPayload` | runtime device args中的immutable pristine source；PyPTO/AICPU只读，绝不就地执行 | CANN eager task或captured node/model | eager到task完成；capture应到graph不再replay，待P0证明 |
| 4 | `HbgExecutionSlot` | device mutable working state；GM SM、runtime arena、GM heap/workspace、outer Runtime、KernelArgs、handshake/mailbox、restore/completion epoch | PyPTO L1 context | 所有可能引用它的graph销毁且device externally quiescent |
| 5 | `HbgLifetimeRoots` | context asset与caller asset的生命根；binary generation、stream/event、workspace、external tensor storage，以及fallback时的explicit package lease | PyPTO context + caller/ACLGraph owner + runtime | 各自最后一个possible replay真正结束 |

五层中只有第3层有机会完全借用CANN task-args所有权。第4层无论如何仍是PyPTO context-owned；如果context在captured graph仍可replay时释放第4/5层，runtime-owned的第3层也只会保留指向已释放working slot、binary或tensor的地址，不会自动防止use-after-free。

#### 10.20.4 graph package是destination-address-bound，不是自由relocatable image

当前host build在H2D前使用最终 `device_sm` 和 `device_arena` 计算delta，task descriptor直接保存GM heap的 `packed_buffer_base/end`，payload中的Tensor也保存实际device buffer地址。因此：

- pristine source bytes本身可以位于WithHostArgs args allocation中的任意source address；
- 但source bytes内的device pointer已绑定 `{working_sm_base, working_arena_base, gm_heap_base, external_tensor_bases, binary_bases}`；
- v1可以让多个不同captured package顺序恢复到一个context-wide working slot，但不得并发replay；
- 不能把一件已绑定slot A的payload直接D2D到slot B；未来多slot必须按slot重新生成template，或保留完整relocation table并在restore后重定patch；
- package header必须保存expected base tuple、required capacity和binding generation，AICPU leader在写working slot前fail-closed校验；
- 一旦某working slot被任何captured package引用，它的GM SM、runtime arena、GM heap/workspace和outer Runtime/KernelArgs地址都必须freeze。现有 `setup_static_arena` 在更大请求时会release/recommit region，第二阶段必须改为prepare时一次性容量规划或generation-specific stable slot，capture后增容必须报错。

HBG `RUNTIME_LOGIC.md` 现有“host/device boundary is POD and position-independent”只能改写为：fanin使用local producer ID，这一部分依赖表示与地址无关；完整graph image仍包含大量最终device pointer，是destination-address-bound。

#### 10.20.5 首选launch/replay方案：runtime-owned inline source + AICPU leader restore

当前首选P0方向是：

```text
capture-time / eager host invocation
  host build immutable HbgGraphPlan
      -> deep-serialize writable HbgSerializedLaunchBlob
      -> aclrtLaunchKernelWithHostArgs(placeholders...)
      -> CANN snapshot RuntimeOwnedHbgPayload

every eager execution / every ACLGraph replay
  caller stream reaches AICPU launch node
      -> one AICPU leader validates header/base/capacity/hash
      -> copy pristine SM spans into working GM SM
      -> copy pristine arena spans into working runtime arena
      -> restore/rebuild manifest-declared outer Runtime/KernelArgs/handshake/mailbox state
      -> required cache clean/invalidate
      -> release restore-ready barrier
  all AICPU scheduler threads
      -> attach/wire
      -> attach_populated
      -> classify roots and wake lists
      -> dispatch AICore work
  teardown may consume/destroy working state
  next replay repeats restore from immutable RuntimeOwnedHbgPayload
```

这个restore是单算子AICPU launch node内部的前置阶段，不是capture外提前启动的orchestrator，不跨越单算子边界，也不需要 `rtStreamAddToModel`、capture query或stream sync。AICPU仍在caller stream上，hidden AICore stream仍通过本算子内的start/done event与caller闭合。

首版应先使用“按restore manifest复制完整pristine spans”建立正确性，不应一开始就手工枚举所有mutable field。当前SM payload把静态tensor/scalar/descriptor与mutable early-dispatch atomics混在同一结构中，runtime arena也同时包含静态layout和mutable queue。未来可以物理拆分read-only `GraphPlanImage` 与小型 `SchedulerExecutionState`，以降低每次replay copy成本，但那是后续性能/结构优化，不是首版正确性前提。

inline runtime-owned source与独立D2D restore node不能被当作随意互换的两种写法：WithHostArgs的device args base在launch内部由runtime产生，host通常无法让一个更早的独立memcpy node引用这个地址；因此inline source天然配AICPU leader restore。只有PyPTO自己持有稳定external device pristine source时，captured D2D restore node才是合理候选。

#### 10.20.6 capture lifetime、capacity和fallback决策

若device 1 P0证明runtime-owned inline payload会随captured node/model存活到graph destroy，所有权应优先交给CANN：

- eager：runtime在device task真正完成后回收args storage，PyPTO不自建、不臆测task-args pool；
- capture：runtime-owned inline source跟captured launch node生存，每次replay由同一AICPU node用它恢复working slot；
- host `HbgSerializedLaunchBlob` 在runtime完成snapshot后可销毁，但该时点必须用“API返回后立即poison/free host blob”的P0证明，不只根据源码推测；
- PyPTO context仍必须保持working slot、workspace、binary和stream/event，直到所有graph不再replay且调用方已完成external quiescence。

若inline payload因大小、backend copy、capture lifetime或AICPU restore性能/可见性失败，fallback按以下顺序评估：

1. graph-lifetime immutable device source + caller-stream captured D2D restore node；
2. torch_npu/ACLGraph wrapper或未来HBG graph owner显式持有 `HbgPackageLease`；
3. 上述hook均不存在时，使用有memory accounting和明确上限的append-only/pin-until-context-close。

fallback不能使用当前native `PipelineSlotLease` 代替graph lease。该lease只覆盖host prepare到device finalize的in-flight run，ACLGraph replay不重新进入PyPTO host prepare/finalize，因此它看不到graph executable生命周期。同样，tail event只能证明某次已入队执行的tail，不能证明一个graph未来不再replay。

v1将“PyPTO占用全部AICore”转化为严格的无并发前置条件：多个captured package可以顺序共享一个working slot和workspace，但不同graph或同一graph的并发replay是unsupported。由于replay时不进入PyPTO host launch，这一点无法只靠普通host mutex/fail-fast覆盖，v1必须把它列为调用方可执行契约；如果未来要内部保证，需要graph owner级序列化或device-side execution gate，不能假装当前host代码可以侦测所有replay。

#### 10.20.7 device 1 P0验证矩阵

第二阶段实现前，先使用device 1做下列最小事实验证，device 0继续留给并行会话。所有限制都以实测capability/result记录，不将CANN内部常量写成PyPTO规格。

| 类别 | P0/ST | 必须证明的事实 |
| --- | --- | --- |
| placeholder ABI | 用最小AICPU kernel读取header pointer和inline bytes | `addrOffset/dataOffset` 在AICPU WithHostArgs路径生效，pointer落在本次runtime device args copy内 |
| host snapshot | launch API返回后立即poison/free/reuse host blob | eager仍读到原始payload，并确认placeholder导致的host原地patch不污染canonical plan |
| captured lifetime | capture一次，释放host blob，反复replay并制造大量其他WithHostArgs allocator压力 | 旧graph的runtime-owned payload不被池复用，每次replay内容一致 |
| graph destroy | 分别观察capture、instantiate、replay、graph destroy前后的args allocation/accounting | 确认captured args真实回收点，区分源码推测与公开可依赖行为 |
| large args | 扫描64 KiB、1 MiB、16 MiB、64 MiB、实际HBG SM+arena规模和失败边界 | 完整copy、无silent clamp/truncation、明确错误码、无隐式sync；不依赖256 MiB或任何内部常量 |
| replay restore | 同一graph至少连续replay两次，第一次执行自然消费状态，并用test hook额外poison known mutable spans | 第二次重建ready queue、wake list、completion flag、task state、runtime pointer和mailbox，不依赖未被碰巧改写的初值 |
| two generations | graph A/B带不同topology/scalar/tensor address，在外部保证无并发的前提下交替replay | 两份RuntimeOwnedHbgPayload不串包，一份working slot按generation正确恢复 |
| address binding | 故意传入错误working base、capacity、binding generation或大于freeze capacity的plan | AICPU leader在scheduler放行前fail-closed，不写越界，hidden AICore可安全收尾 |
| cache/order | leader restore后多AICPU thread classify，AICore立即读descriptor/payload | clean/invalidate和release/acquire链足以让A2/A3、A5都观察到完整新generation |
| stream boundary | trace capture/replay整个op | 无PyPTO stream/device sync、无capture query、无 `rtStreamAddToModel`、无private AICPU stream；restore与AICPU/AICore执行均在单op边界内 |
| close/lifetime | graph仍可replay时尝试close，graph destroy + external quiescence后重试 | 不释放仍被引用的working slot/binary/workspace，失败保留可重试ownership |

#### 10.20.8 独立阻塞：host build如何读取L1 external tensor data

graph payload所有权闭环后，HBG L1仍有一个不能混进同一问题的独立阻塞。当前L2 HBG会：

- 为输入/输出自行分配device tensor并做H2D/D2H staging；
- 尝试把device buffer映射为host view，失败则使用host staging copy；
- 让host orchestration通过 `get_tensor_data/set_tensor_data` 读写control tensor内容来构图。

L1不允许PyPTO替external tensor重新分配/staging，而A5当前又没有A2/A3同类host-map路径。如果某个 `@pl.program` 的host build依赖device tensor数值而不只是shape/stride/scalar/host-known metadata，capture过程中如何取得这些数据将可能要求D2H可见性或同步，与L1/ACLGraph限制直接冲突。

第二阶段必须独立完成下列决策：

1. 静态/capture-safe HBG是否限制host build只依赖host-known metadata、scalar和显式CPU control input；
2. 若允许读device control tensor，A2/A3与A5各自的无sync可见性协议是什么；
3. 无法在capture-safe边界内满足的callable是在prepare还是capture launch前fail-closed；
4. `get_tensor_data/set_tensor_data` 的使用是否能在compile/prepare阶段被静态标记，以便提前拒绝不可capture的HBG。

这个问题不能通过“把graph bytes放进WithHostArgs”解决；前者是host build数据依赖，后者是build完成后graph payload的device task ownership。两项P0必须分别通过，才能宣称HBG L1 + ACLGraph可实现。

#### 10.20.9 第二阶段实现准入结论

在上述P0之前，不实现HBG L1，不对当前TRB L1 ABI做大payload泛化，不在PyTorch convenience wrapper里添加未经证明的graph lease，也不把同步H2D、临时host vector或native pipeline slot包装成capture-safe实现。只有在WithHostArgs inline payload的copy/lifetime/size、AICPU leader restore的cache/order、working-slot capacity freeze、external tensor host-build数据依赖四类硬门槛都形成device 1证据后，才进入第二阶段实现。

### 10.21 scalar bit ABI、最终callable签名与L2/L3兼容

高层API落地后发现，“scalar数量已经传到native”不等于“scalar ABI已经正确”。这里实际有三层问题需要分开处理。

#### 10.21.1 `scalar_count` 不能破坏现有 `ChipCallable` wire layout

orchestration signature历史上只序列化tensor direction，scalar不在 `sig_count` 中。L1 native却必须在任何enqueue前同时校验tensor/scalar数量，因此codegen新增了 `ChipCallable.scalar_count`。第一轮尝试如果把它插在旧header中间，会让旧 `binary_size/func_name/child offset/config` 的字段语义全部错位，而L3 remote payload v1和local shared-memory都会直接 `ChipCallable.from_bytes` reinterpret这个blob。

最终方案是：

- `scalar_count` 放入旧header尾部、`storage_` 之前原有padding；
- 对所有legacy字段和 `storage_` offset增加 `static_assert`；
- 保留旧C++ factory overload，旧调用方默认 `scalar_count=0`；
- codegen和Python binding显式传递scalar count，native `CallableState` 使用它做底层强校验；
- L1的per-callable state独立保存scalar count，不用tensor-only signature反推。

这样不改变L2/L3的现行wire offset，也不需要为本次L1强行bump L3 payload protocol。仍保留一个非blocker的长期测试增强项：将冻结的旧raw blob真实经过Python `from_bytes` 和L3 descriptor/registration roundtrip，验证旧padding解读为scalar count 0。

#### 10.21.2 FP16/BF16不能用通用FP32 carrier的低16位

`_coerce_args` 为FP16/BF16 scalar使用 `ctypes.c_float`，而device codegen使用 `from_u64<declared_type>` 只读64-bit slot的低 `sizeof(type)` bytes。若简单将 `c_float(1.0)` 的 `0x3f800000` 塞进slot，FP16/BF16看到的低16位是 `0x0000`，会静默执行错误数值。

L1现在根据声明dtype做bit-exact packing：

- FP16：IEEE binary16的低16位；
- BF16：从已coerce的exact FP32 bit做round-to-nearest-even，NaN保留payload高位并强制quiet bit，避免截断成infinity；
- FP32/FP64：分别写exact 32/64-bit IEEE pattern；
- signed/unsigned integer、bool、index：先由ctypes应用声明宽度/符号转换，再将有效位写入64-bit carrier。

首版支持 `fp16/fp32/fp64/bfloat16/int8/16/32/64/uint8/16/32/64/bool/index`。FP4/FP8/HF等未定义scalar ABI的dtype在context init阶段fail-fast，不推迟到首次launch。无硬件UT覆盖FP16 1.0=`0x3c00`、BF16 1.0=`0x3f80`、非整数舍入、所有整数宽度及unsupported dtype init rejection。

#### 10.21.3 必须校验assembled callable，不只看优化前IR

CommCtx/distributed materialization可能在后续lowering阶段改变最终orchestration签名。Python init现在对 `CompiledProgram._get_metadata()` 与assembled `ChipCallable` 的tensor count、scalar count及每个tensor direction再做一次对照。若最终callable出现未被高层原始IR扫描捕获的unsupported argument，在native init/allocation前直接拒绝。

### 10.22 Python/init ownership、layout commit与taskQueue storage lifetime

#### 10.22.1 init失败后必须仍有可达cleanup owner

native L1 init在rollback自身也失败时，会故意保留device context、device claim和DSO，要求调用方显式retry close。如果Python只在 `init_l1()` 成功后才把worker挂到context，或只catch `Exception`，就会有两类泄漏：

- native已建立owner后返回RuntimeError，局部worker随栈丢失；
- native binding释放GIL期间收到KeyboardInterrupt/SystemExit，`BaseException` 越过两层handler，留下永久device claim。

当前两层闭环是：

1. simpler `ChipWorker.init_l1` 调用native前先确认worker是fresh，已L2初始化或double-L1直接拒绝，不会把原 `_execution_mode` 误改为L1；
2. simpler和PyPTO context都catch `BaseException`；
3. 只要native worker报告 `initialized=True`，Python就先接管为L1 owner；
4. 高层抛出 `L1InitializationError`，其 `cleanup_context` 只允许close，不允许prepare/launch；
5. native cleanup没有留下owner时，才将worker置空并按原异常返回。

UT分别注入“先将initialized置真再抛KeyboardInterrupt”、L2→L1二次init和double-L1，确认cleanup owner可达、mode不被污染、close可重试。

#### 10.22.2 首次成功enqueue才是metadata commit point

v1允许tensor address和scalar value每次改变，但首次成功enqueue后锁定tensor shape/dtype/stride。“首次host尝试”不能等于“首次成功”：adapter enqueue若同步抛错，本次候选layout不能污染operator。

Python `_build_args` 先构建完整candidate metadata并做全部host validation；只在adapter/direct native enqueue成功返回后才写 `bound_tensor_metadata` 和 `warmed=True`。UT注入第一次adapter错误，确认之后不同合法stride仍能成为第一个成功layout；另覆盖input和output stride在commit后变化均在任何新capsule/enqueue前拒绝。

#### 10.22.3 adapter和allocator的精确保活边界

`pypto._torch_npu_l1` 使用nanobind定义模块，入口通过 `THPVariable_Unpack` 得到C++ Tensor handle。它保持：

- named `simpler.l1.queue_call.v1` descriptor lease到taskQueue callback完成；
- `std::vector<at::Tensor>` 到callback真正把完整L1 fork/join序列enqueue到raw stream；
- 每个唯一default NPU caching-allocator storage通过 `recordStream(current_npu_stream)` 保活到device真正用完。

callback不捕获capsule、nanobind/Python object或任何需要GIL的状态。adapter在native init前校验queue-call ABI version及build/runtime torch、torch_npu version，避免包装不匹配时已占有device资源。direct debug路径不生成capsule，通过current stream raw handle直接launch，同时对输入/输出调用Tensor `record_stream`；它的Python `.npu_stream` getter在全局taskQueue开启时可能drain host queue，因此只是bring-up/debug能力，不是默认production path。

`recordStream` 对external/from-blob/custom allocator storage可能因为不是local allocator deleter而无法接管lifetime。v1不伪装这类storage已被adapter保护：调用方必须保持外部owner到真实stream completion，ACLGraph场景还要持有到graph已销毁且外部quiescent。当前这是显式契约，不在adapter中捕获Python owner到未知graph生命期。

### 10.23 native transaction、failure-safe close和AICPU/AICore异常收口

#### 10.23.1 prepare/launch的单一transaction边界

`prepare_l1_callable_from_blob` 现在用同一把L1 operation mutex覆盖phase检查、exact blob length/identity validation、binary upload、per-callable address snapshot、AICPU registration和prepare tail enqueue。纯input错误在任何upload/allocation前返回，不poison context；已开始enqueue后的partial failure才进入sticky poison。第一次launch后seal，任何新callable prepare在产生side effect前被拒绝。

native tensor validation不信任Python/POD cache：它overflow-safe重算shape/stride可达extent和contiguous性，验证buffer size/start offset/element bytes，并用 `aclrtPointerGetAttributes` 强制non-empty pointer是预期device上的device memory。因此伪造较小 `extent_elem_cache`、host pointer或wrong-device pointer都不会到达AICPU/AICore。

多 `@pl.program` 不再共享一张context-global `func_id -> addr`表；每个callable保持自己的immutable func-id/address snapshot，AICPU每次根据callable id重放对应快照。这与每个独立编译program的func-id都从0开始的事实一致。L1 launch只调用已prepare的AICore handle，不再经过可lazy `rtRegisterAllKernel` 的通用helper，从代码结构上保证capture launch无法表达binary registration。

#### 10.23.2 `Closing` 是粘性状态，不是一个best-effort日志

close必须处理一个比“不内部sync”更难的问题：unload/free/destroy可能部分成功、部分失败。如果失败后context还显示Collecting/Ready/Sealed，公开C ABI就可能再次prepare/launch已释放一半的Runtime，形成UAF。

当前状态机在确认current device后、第一项destructive teardown之前调用 `begin_close()` 进入 `Closing`。该intent是粘性的：

- C ABI和runner内部prepare/launch都在operation mutex内再检查 `accepts_l1_dispatch()`，Closing/Poisoned/Closed一律拒绝；
- unload/free失败不清理相应owner table/handle，context、DSO和per-device claim保留；
- `ChipWorker::finalize` 不在native failure后 `destroy_device_context/dlclose`，而是抛错并允许显式retry；
- 只有AICPU binary、KernelArgs/runtime args、register mapping、callable buffer、arena/allocation全部释放成功后，才销毁events/hidden stream、释放device claim并进入Closed；
- close路径仍完全不做stream/device synchronize，调用方必须先销毁graph并外部证明quiescence。

`LoadAicpuOp::Finalize` 和allocator finalize的“失败保留owner以便retry”语义被限定在borrowed L1；历史L2/L3在RTS reset/aclFinalize后不能再访问旧runtime handle，因此保持其best-effort/no-retry teardown语义，避免本次修复反向制造L2析构回归。

#### 10.23.3 AICPU multi-thread completion和scheduler-init verdict

A2/A3和A5的AICPU执行器都改为双阶段completion gate：所有有效参与者无论init/run成功失败都exactly-once arrive；唯一last-arriver做runtime destroy/final verdict；各线程snapshot最终error/runtime status后exactly-once depart；唯一last-departure才deinit/reset generation。这避免了早到线程清零error、销毁runtime或cache-invalidate早于迟到线程读取。

decoupled scheduler另增加最终init verdict barrier：所有scheduler完成handshake/assignment后到齐，leader唯一发布success/failure；orchestrator可与这些front matter并行准备，但在 `p_func` 前必须acquire最终verdict。任一scheduler init失败都不再让orchestrator进入无消费者的ring submit等待，serial assignment失败也先emergency shutdown再返回。

pre-window WAIT/CANCEL的闭环、A5 PMU越界guard及完全不report的外部故障边界详见10.18。

### 10.24 当前验证证据、device 1状态和尚未通过的Phase-0

#### 10.24.1 无硬件构建与回归

所有命令都在 `/mnt/workspace/inductor/pto/gpt_pypto` 或其 `runtime` 子仓执行，并显式设置GPT worktree的 `PYTHONPATH`，避免editable finder拾取同级Grok checkout。当前证据为：

- top/runtime两层L1 Python反例：`tests/ut/runtime/test_l1.py + runtime/tests/ut/py/test_l1_chip_worker.py` 合计 **51 passed**；
- L1 + backend signature + orchestration codegen：**90 passed, 17 warnings**；
- runtime L1/task-interface/callable identity/L3 remote protocol/lifecycle：**257 passed, 14 warnings**；
- runtime scene cache：**3 passed**；
- 将上述top/runtime L1、backend/codegen、task-interface、callable identity、L3 protocol/lifecycle和scene cache一次性合并重跑：**350 passed, 17 warnings**；
- runtime editable全量build通过，A2/A3和A5的onboard/sim TRB scheduler对象及A5 TRB/HBG AIC/AIV kernel对象已重编；
- top-level `pip install --no-build-isolation -e .` 再次增量构建通过；显式GPT `PYTHONPATH` 下 `pypto`、`simpler`和 `pypto._torch_npu_l1` 均从 `gpt_pypto`/其venv加载，adapter报告queue-call ABI 1、build torch `2.12.0+cpu`、torch_npu `2.12.0+git5462a1b`；该import没有获取stream或发NPU task；
- `ctest -LE requires_hardware`：**85/85 passed**，其中69个显式标记 `no_hardware`；
- Python `compileall`、top/runtime完整pre-commit、native clang-format/cpplint/clang-tidy、ruff、pyright及两仓 `git diff --check` 均通过。首轮runtime hook误用了本机 `/usr/local/bin/clang-tidy` 21.1.4，但该安装缺少匹配的Clang resource headers，产生 `stddef.h/cstdint not found` 和后续不完整AST伪报；未跳过hook、未修改正确代码，而是临时解包与CI同系列的Ubuntu LLVM 18.1.8，以匹配resource headers重跑完整hook并通过。顶层ruff使用pre-commit锁定的0.14.8环境通过，不再以本机0.15.14的required-version拒绝替代正式门禁。
- runtime提交后，旧editable `_task_interface` 按设计检测到内嵌source SHA仍为 `4d844e00`、当前源码已是 `7e8d141b`，在任何测试体执行前拒绝加载；重新执行runtime editable build后再运行同一组合命令，最终得到上述350项通过。这证明ABI source-tree guard生效，不是测试失败或绕过。

51个高层/wrapper测试包含了过去缺失的关键反例：KeyboardInterrupt owner adoption、L2→L1/double-L1 mode不污染、FP16/BF16和整数scalar bit pattern、unsupported scalar init rejection、首次adapter enqueue失败不绑定layout/warmed、output stride变化、adapter ABI/framework version mismatch在native init前失败、direct launch不建capsule并对输入/输出record stream。

#### 10.24.2 新增真实ST验收面，但未上板执行

新增 `tests/st/runtime/l1/test_l1_aclgraph.py`，不复制Grok实现，使用本文件test-local `@pl.jit` 64x128 FP32 add和当前GPT API：

```python
compiled = kernel.compile(config=RunConfig(platform=platform, device_id=device_id))
context = pypto_init(programs=[compiled], device=device_id)
op = context.operator(compiled)
```

该测试在一个端到端case中覆盖：

1. `context.prepare()` + 普通stream `op.warmup()` 的eager数值；
2. warmup stream与独立capture stream的raw handle确实不同；
3. caller显式external synchronize后才开始capture；
4. 图内 `torch.add(out=) -> PyPTO L1(out=) -> torch.mul(out=)` 的predecessor/operator/successor顺序；
5. 三组不同输入值连续replay并逐次验数；
6. graph-bound tensors全程保持强引用；finally中先external device quiescence，再 `graph.reset()`，最后 `context.close()`；init rollback若抛 `L1InitializationError`，也接管 `cleanup_context` 完成close retry。

当前只做了AST、a2a3/a5 pure-host lowering、isolated ruff、format/diff检查和pytest collect-only。`--platform=a2a3 --device=1` 正确收集1个a2a3 item并deselect a5，collect-only不进入测试体、不初始化NPU。

#### 10.24.3 为什么本轮不运行device 1

按用户约束只考虑device 1，不使用device 0。只读环境检查显示：

- 逻辑device 1映射到 `/dev/davinci15`，对应card/NPU ID 7、chip logic ID 1、physical ID 15，不是device 0；
- device 1当时HBM约2873/65536 MiB，AICore 100%、AIVector 90%；
- `npu-smi` 虽未列出process，但本仓调度规则要求idle HBM为0，所以该卡明确非空闲，不能抢占；
- `task-submit` 不在PATH，无法通过合规队列借卡；
- 现有mandatory onboard arch precheck还硬编码查询 `npu-smi ... -i 0 -c 0`，而本机card ID为7，该查询会返回invalid card id。真正执行前需要合规调度或修正预检入口，不能因此fallback到device 0。

因此本轮没有发送任何ACL/NPU task，没有触碰device 0。目前只能宣称“实现、产物和ST已进入Phase-0候选状态”，不能宣称“ACLGraph已在device 1通过”。真正发布门槛仍是在空闲device 1上通过event-only hidden-stream capture、WithHostArgs snapshot、AICore handle launch、连续replay、entry/exit顺序和无forbidden API trace。

### 10.25 第一笔HBG实现：把dynamic graph落实为tiling-like invocation package

#### 10.25.1 为什么先做host基础而不伪装H0已经通过

用户再次强调，第二阶段每次动态build并H2D到device的graph，本质上是这次task类似AscendC `tiling_data` 的入参。由此实现顺序必须同时满足两个条件：

1. 先在代码中消除“host builder一边构建、一边立即把唯一current image写进共享device buffer”的模糊边界，产出可以被一个launch task独立持有的pristine bytes；
2. 在不知道CANN对large WithHostArgs、placeholder、capture/replay和graph destroy实际ownership之前，不能把host-only serializer写成“ACLGraph已经会替PyPTO管理graph image”的完成结论。

device 1本轮仍非空闲，且没有可用的合规 `task-submit`。因此先实施不依赖device事实的H1/H2基础：拆host-build与H2D边界、定义variable task package ABI、写严格validator和无硬件UT；H0涉及的CANN snapshot/lifetime/cache结论继续保持未知。这个顺序不降低H0门槛，也没有触碰device 0。

#### 10.25.2 A2/A3和A5的host-build/H2D边界已经同步拆开

修改两份同构实现：

- `runtime/src/a2a3/runtime/host_build_graph/host/runtime_maker.cpp`；
- `runtime/src/a5/runtime/host_build_graph/host/runtime_maker.cpp`。

原 `run_host_orchestration` 同时承担四件事：分配host SM mirror、运行host orchestration、按最终device bases做relocation、立即把SM同步H2D。它返回以后，调用方再单独上传runtime arena。这种写法虽然对当前L2可用，却没有一个明确时刻能说“两份构成graph初始状态的host image都已经完成，现在可以把它们交给task/package owner”。

当前改为 `build_host_orchestration_image`：

1. 接收caller-owned `std::vector<uint8_t> *out_sm_image`；
2. 在host上完成SM清零、orchestrator执行、dependency graph收集和destination-address relocation；
3. 只返回 `host_total_tasks` 和owning SM bytes，函数内部不再调用 `copy_to_device`；
4. bind caller在host tensor-view window关闭、`PTO2Runtime::prebuilt_layout`写完、SM与arena两份host image都完成后，才进入一段显式H2D区域；
5. 先上传完整SM，再上传完整runtime arena；两次copy都成功后才发布 `runtime->host_total_tasks` 和prebuilt arena binding，避免把半完成graph标记成ready。

这一笔刻意不改变旧L2的外部时序：它仍在同一个bind调用中做同步H2D，仍使用原来的pooled GM SM/runtime arena/GM heap，AICPU执行入口也未改变。拆分的价值是建立可审计的ownership boundary，并不是提前把L2改成异步L1，也不是已经创建完整 `HbgGraphPlan`。当前host `DeviceArena`仍是bind栈内对象，下一阶段还要把SM、arena和initializer spans一起提升为正式canonical plan owner。

#### 10.25.3 variable launch blob的内存布局

新增：

- `runtime/src/common/task_interface/hbg_launch_blob.h`；
- `runtime/src/common/worker/hbg_launch_blob_builder.h`。

host serializer产生一份writable、逐launch独立的深拷贝，布局固定为：

```text
HbgLaunchBlobHeader
HbgLaunchRegion[region_count]
zero alignment padding
inline pristine payload bytes
  - full SharedMemory image
  - full RuntimeArena image
  - zero or more GM heap initializer spans
```

这里继续严格区分五层owner：canonical graph plan不是这个writable blob；writable blob也不是CANN runtime-owned device args；runtime-owned inline source又不是scheduler会改写的working slot。serializer从caller source做深拷贝，后续CANN即使按placeholder规则原地修改writable blob，也不能污染canonical plan。

`HbgLaunchBlobHeader::inline_payload_addr` 在canonical host状态必须为0。未来HBG专用 `aclrtLaunchKernelWithHostArgs` bridge可用 `aclrtPlaceHolderInfo{addrOffset, dataOffset}`，把该字段修补为runtime-owned device args blob中inline payload的device地址。header、region descriptors和payload全部包含在 `argsSize` 范围，外部pointer pointee没有被错误地当成runtime owner。当前代码只定义并验证host-unpatched/device-patched两种状态，尚未调用CANN placeholder API。

#### 10.25.4 destination binding和invocation identity不能混成一个hash

一个HBG source image内含最终device绝对地址，所以blob明确标记 `HBG_LAUNCH_DESTINATION_BOUND`，并携带 `HbgExecutionBinding`：

- working SM base/capacity；
- runtime arena base/capacity和runtime offset；
- GM heap base/capacity；
- execution slot generation。

slot generation与地址/capacity分别比较：generation不匹配返回generation错误，任何base、capacity或runtime offset变化返回binding错误。plan content hash故意不把working binding混进去；同一canonical content未来若要materialize到另一slot，必须经过显式relocation/materialization并产生匹配的新binding，不能通过改header假装原bytes可自由搬家。

另外新增32-byte `HbgInvocationIdentity`，把三个生命周期域分开：

- `callable_hash`：这份graph属于哪个编译operator；
- `argument_snapshot_hash`：host build时被烘焙进image的tensor地址、scalar和控制参数快照；
- `function_binding_hash`：这份image依赖哪张已解析AICore function table；
- `tensor_count/scalar_count`：对照本次callable签名，且不能超过现有chip ABI容量。

identity参与plan hash，同时validator还可接收expected identity单独比较。这样未来AICPU restore入口可以在写working slot之前拒绝“地址碰巧相同但属于另一个callable/args/function generation”的旧package。header保持8-byte alignment，并用 `sizeof/offsetof/trivially_copyable` static assertions锁定placeholder、binding和identity ABI offset；没有依赖未证实的64-byte CANN args alignment。

#### 10.25.5 restore manifest的fail-closed规则

`validate_hbg_launch_blob`把输入视为不可信variable bytes，在任何restore前验证：

1. magic、ABI major/minor、exact canonical header size、exact total size和zero padding；
2. blob/header alignment、region数量、所有size/offset加法和 `uint32_t argsSize` 表示范围；
3. plan generation、slot generation、destination base/capacity和runtime offset；
4. invocation identity非空关键hash、tensor/scalar容量及可选expected identity；
5. 恰好一份完整覆盖frozen capacity的SM image和一份完整runtime-arena image，拒绝partial restore；
6. 所有source spans落在inline payload内且互不重叠；同类destination spans也不能重叠；
7. GM heap initializer只能写进声明的heap window；没有heap binding时不得偷放initializer；
8. region flags和reserved字段必须canonical；
9. identity、region descriptors和全部payload bytes的FNV-1a content hash必须一致；placeholder pointer本身不参与hash，因为它属于runtime每次launch的patch结果。

要求SM和runtime arena完整覆盖不是为了追求序列化简单，而是当前HBG scheduler会原地消费ready queue、wake list、completion flags、task state和runtime pointers。只恢复“看起来变化的字段”很容易漏掉跨代状态；首版先用full-image restore证明正确性，之后只能在profile和逐字段ownership证据支持下优化。

builder具有transaction语义：先在local candidate中完成allocation、descriptor构造、deep copy、hash和全量validate，成功后才替换caller的 `out`。无论非法identity、缺失full image、空source、越界还是span overlap，已有canonical blob保持不变，避免失败build把上一次可用package清空或部分覆盖。

实现早期UT实际暴露过一个有价值的问题：`candidate.assign(total_size, 0)` 只产生零字节，随后把它reinterpret为header却没有执行default member initializers，magic/version/flags仍是0，测试在继续解释非法header时触发崩溃。修正为在写字段前显式执行 `*header = HbgLaunchBlobHeader{}`。这也说明variable ABI不能依赖“零内存等于默认对象”的偶然假设。

#### 10.25.6 本阶段验证证据

新增 `runtime/tests/ut/cpp/types/test_hbg_launch_blob.cpp` 和独立 `no_hardware` CMake target，10个case覆盖：

- source被修改后deep snapshot不变；
- host-unpatched、device-patched和错误pointer状态；
- truncated/header/generation/identity损坏；
- partial full-image、source/destination overlap和capacity越界；
- hash覆盖identity/descriptors/payload但排除placeholder；
- stale slot generation/address和另一次argument/function identity被拒绝；
- 多个不重叠heap initializer及重叠负例；
- invalid input/build失败不覆盖上一份canonical blob。

验证结果：

- runtime完整pre-commit通过，包括header检查、English-only、clang-format、clang-tidy、cpplint；仍使用与CI同系列且resource headers完整的LLVM 18.1.8临时工具链，没有跳过hook；
- runtime editable build通过，并重新编译A2/A3与A5的onboard/sim HBG host runtime对象；
- 新target：10/10 passed；
- `ctest --test-dir tests/ut/cpp/build -LE requires_hardware --output-on-failure`：**86/86 passed**，其中70项标记 `no_hardware`；
- `test_runtime_builder.py + test_host_runtime_abi.py + test_worker_reuse.py + test_chip_worker_explicit_dispatch.py + test_binary_cache_context.py`：**112 passed, 4 skipped, 14 warnings**；
- 两仓 `git diff --check`通过；没有运行NPU命令。

第一次Python回归命令从runtime子仓执行时误写成 `runtime/tests/...`，pytest只报告file not found、没有执行测试体；随后用正确的 `tests/ut/py/...` 路径重跑，得到上述112/4结果。过程记录保留这一点，避免以后把错误命令的空跑当成验证。

runtime阶段提交为：

```text
11b7a4b1 Add: 建立HBG图包序列化与上传边界
```

提交仅在GPT工作区和 `gpt/pypto-l1-aclgraph` 分支，本地commit，未push；没有修改Grok工作目录。

#### 10.25.7 当前明确没有完成的能力

这一阶段不能被表述为“HBG已经支持L1/ACLGraph”。仍未完成的关键闭环是：

1. current HBG host builder尚未把SM、runtime arena和heap initializers发布成正式、可缓存、带owner的 `HbgGraphPlan`；新serializer当前是common基础设施，尚未接入production HBG bind；
2. 尚未新增writable args + placeholder array的CANN launch helper，未证明CANN何时原地patch host blob、何时完成snapshot、captured graph销毁时何时释放runtime-owned device args；
3. 尚未有HBG独立AICPU L1 entry，也未把parser放到device侧，更没有exactly-one leader在每次eager/replay前恢复working SM/runtime arena/initializer；
4. 尚未建立HBG stable execution slot、prepare-time capacity freeze、generation owner、outer Runtime/KernelArgs/handshake的per-execution reset清单；
5. 尚未解决host orchestration读取tensor data时，L1 direct external NPU tensor如何在不D2H、不sync的前提下提供host-known control data；
6. HBG所有variant的 `simpler_l1_supported()`仍返回false，高层 `pypto_init`仍拒绝HBG；TRB L1路径和fixed `L1AicpuInvocationArgs`完全未改；
7. device 1上的placeholder/large-args/capture lifetime、AICPU restore cache/order、同图第二次及后续replay、graph A/B交替replay和memory accounting全部还是发布硬门槛。

当前可以确认的只是：代码中已经有一个与用户“graph像tiling参数随task管理”原则一致的host表示——每次launch的writable blob深拷贝并封装自己的pristine graph bytes、identity和destination generation，working slot则明确属于context且必须per replay restore。谁最终拥有runtime-owned source、什么时候可释放、device如何恢复，仍由后续H0与H3/H4实现回答，不能从本轮host UT推断。
