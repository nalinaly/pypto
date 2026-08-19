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
- runtime `10e69df66a71ce752bf1ef58c8dd9147a8de775e`：`Add: 建立HBG可写HostArgs占位符桥接`
  - 增加CANN-independent placeholder POD及args-size/count/offset/alignment/overlap校验；
  - `LoadAicpuOp`新增无分配、无同步的mutable HostArgs + placeholder入口，并静态锁定CANN ABI布局；
  - HBG blob生成单一inline payload placeholder，不修改canonical bytes，13项定向UT覆盖runtime-patched和全部负例；
  - 完整pre-commit、editable build、无硬件C++ 86/86和相关Python 112 passed/4 skipped通过；仍未注册HBG L1 AICPU entry。

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
2. 当时尚未新增writable args + placeholder array的CANN launch helper；该host bridge随后已由10.26和runtime `10e69df6`补齐，但仍未接入HBG AICPU entry，也未证明CANN何时patch/snapshot或何时释放captured runtime-owned args；
3. 尚未有HBG独立AICPU L1 entry，也未把parser放到device侧，更没有exactly-one leader在每次eager/replay前恢复working SM/runtime arena/initializer；
4. 尚未建立HBG stable execution slot、prepare-time capacity freeze、generation owner、outer Runtime/KernelArgs/handshake的per-execution reset清单；
5. 尚未解决host orchestration读取tensor data时，L1 direct external NPU tensor如何在不D2H、不sync的前提下提供host-known control data；
6. HBG所有variant的 `simpler_l1_supported()`仍返回false，高层 `pypto_init`仍拒绝HBG；TRB L1路径和fixed `L1AicpuInvocationArgs`完全未改；
7. device 1上的placeholder/large-args/capture lifetime、AICPU restore cache/order、同图第二次及后续replay、graph A/B交替replay和memory accounting全部还是发布硬门槛。

当前可以确认的只是：代码中已经有一个与用户“graph像tiling参数随task管理”原则一致的host表示——每次launch的writable blob深拷贝并封装自己的pristine graph bytes、identity和destination generation，working slot则明确属于context且必须per replay restore。谁最终拥有runtime-owned source、什么时候可释放、device如何恢复，仍由后续H0与H3/H4实现回答，不能从本轮host UT推断。

### 10.26 HBG H2继续实现：writable HostArgs与placeholder bridge

#### 10.26.1 现有TRB helper为什么不能直接复用

现有 `LoadAicpuOp::LaunchWithHostArgs` 面向固定大小TRB结构体：参数类型是 `const void *`，内部对CANN做 `const_cast`，placeholder array固定传null。它适合不含inline pointer的 `InitArgs/L1RegisterCallableArgs/L1AicpuInvocationArgs`，不适合HBG variable blob，原因有四个：

1. HBG必须明确要求每次launch使用fresh writable serialization，不能让CANN可能的原地patch污染canonical graph plan；
2. `aclrtPlaceHolderInfo.addrOffset`指向一个8-byte pointer字段，CANN公开检查只保证offset小于args size；PyPTO还必须保证完整8 bytes不越界且正确对齐；
3. 本机CANN 9.2.0 public API的 `argsSize/placeHolderNum` 是 `size_t`，实现中分别窄化到 `uint32_t`，后续内部descriptor又以 `uint16_t`携带placeholder数量。调用前不校验会允许静默截断；
4. HBG capture launch路径不能为了转换placeholder临时分配vector，也不能做同步或capture query。

因此没有把TRB方法改成一个含HBG分支的大入口，而是保留固定overload并新增显式mutable bridge。

#### 10.26.2 CANN-independent placeholder ABI与通用校验

新增 `runtime/src/common/task_interface/host_args_launch.h`，定义8-byte `HostArgsPlaceholder{addr_offset, data_offset}`。它不include ACL头文件，因此serializer、parser和no-hardware UT可独立编译；onboard `load_aicpu_op.cpp`再用static assertions逐项比对：

- `sizeof(HostArgsPlaceholder) == sizeof(aclrtPlaceHolderInfo)`；
- alignment一致；
- `addr_offset`与CANN `addrOffset`的 `offsetof`一致；
- `data_offset`与CANN `dataOffset`的 `offsetof`一致。

`validate_host_args_launch_layout`在CANN调用之前拒绝：

- null或empty host args；
- `args_size > UINT32_MAX`；
- placeholder pointer/count不一致；
- placeholder count不能由当前runtime carrier无损表示；
- pointer field offset未按8 bytes对齐；
- `addr_offset + sizeof(uint64_t)`越界；
- `data_offset >= args_size`；
- 两个pointer write ranges重叠。

这里的32/16-bit检查只是防止本机已核对ABI发生silent narrowing，不被写成HBG产品size/count规格。HBG当前只用一个placeholder；large args真正可用边界仍必须扫描device 1，不能从 `UINT32_MAX`反推出CANN一定支持这么大。

#### 10.26.3 实际runtime launch桥接

`LoadAicpuOp`新增：

```cpp
int LaunchWithMutableHostArgs(
    rtStream_t stream,
    void *host_args,
    size_t args_size,
    HostArgsPlaceholder *placeholders,
    size_t placeholder_count,
    int aicpu_num,
    const char *func_name);
```

它先执行全部纯host layout validation，再沿已有不构造临时 `std::string` 的function-handle查找，最后直接调用：

```cpp
aclrtLaunchKernelWithHostArgs(
    func_handle, num_blocks, caller_stream, nullptr,
    writable_host_args, args_size,
    acl_placeholders, placeholder_count);
```

该路径没有host/device allocation、stream/event创建、synchronize、capture-state query或model attach。原 `LaunchWithHostArgs(const void *)`委托到同一实现但传零placeholder，所以TRB launch拓扑和参数ABI不变，同时也获得args-size不截断检查。

四个onboard产物都实际包含新symbol：A2/A3和A5的TRB/HBG `libhost_runtime.so`经 `nm -C`确认同时导出fixed和mutable两个C++方法。sim不编译该onboard loader，不受ACL header依赖影响。

#### 10.26.4 HBG单placeholder生成与canonical immutability

`make_hbg_launch_placeholder`先以host-unpatched模式全量验证blob，并可同时比较expected execution binding与expected invocation identity；成功后只输出一个descriptor：

```text
addr_offset = offsetof(HbgLaunchBlobHeader, inline_payload_addr)
data_offset = header.header_size
```

函数不修改blob，失败也不修改caller提供的output descriptor。CANN拿到的必须是canonical plan深拷贝得到的fresh writable blob；canonical plan继续保持 `inline_payload_addr == 0`。UT模拟runtime-owned copy后按这两个offset写入 `runtime_args_base + data_offset`，再用device-patched模式验证，证明host/device两种ABI状态能够闭合；这只是等价字节行为测试，不是CANN lifetime实证。

#### 10.26.5 验证和提交

定向HBG target从10项增至13项，新增覆盖：

1. placeholder生成不修改canonical bytes；
2. expected identity不匹配和null output fail-closed且保留原output；
3. runtime-owned copy的pointer patch结果可按device模式解析；
4. null/empty/too-large args、pointer/count mismatch、过多placeholder、misaligned/OOB address field、OOB data offset和overlapping pointer writes全部返回精确错误。

验证结果：

- runtime editable全量build通过，A2/A3与A5的onboard TRB/HBG host runtime均重编；
- runtime完整pre-commit通过；
- 定向13/13通过；
- no-hardware C++全量仍为 **86/86 passed**；
- HBG/runtime相关Python仍为 **112 passed, 4 skipped, 14 warnings**；
- `git diff --check`通过；未执行NPU任务。

runtime阶段提交：

```text
10e69df66a71ce752bf1ef58c8dd9147a8de775e Add: 建立HBG可写HostArgs占位符桥接
```

#### 10.26.6 H2完成到哪里、下一步为什么仍是H3/H4

现在host侧已经能构造fresh writable package、生成严格placeholder，并通过真实CANN API入口enqueue；但production HBG尚无调用者，因为还没有独立HBG L1 AICPU symbol和stable execution slot。此时若直接把mutable helper接到现有HBG L2 `simpler_aicpu_exec`，AICPU会把blob误解释成arch-specific `KernelArgs`，属于明确ABI错误。

下一阶段必须先定义HBG prepare-time stable slot与可信expected binding，再定义独立AICPU entry：device侧先把固定header复制到对齐local、比较slot generation/address/capacity/callable identity，exactly-one leader从runtime-owned inline source恢复full SM/runtime arena和initializer spans，所有peer看到同一restore verdict后才attach/classify/dispatch。只有这条路径存在，mutable HostArgs helper才有合法production call site。

因此当前HBG capability继续为false；不能因为“已经调用得了带placeholder的runtime API”就忽略per-replay restore，也不能把CANN会复制args的源码行为扩大成captured graph lifetime已经通过。

### 10.27 HBG H4正确性核心：每次replay恢复pristine image，完整成功后才提交epoch

#### 10.27.1 为什么在stable execution slot完整接线前先落一层common restore core

10.26结尾指出production顺序仍然是先建立H3 stable slot，再把H4 leader restore接到AICPU。这次没有颠倒该依赖：本阶段实现的只是不持有device资源、不选leader、不放行scheduler的common正确性核心，让后续H3/H4接线时不再重新发明“什么可以复制、什么时候才算一代可执行”。

新增 `runtime/src/common/task_interface/hbg_restore.h`，其输入明确分成两个信任域：

- `blob/blob_size`是本次task/captured node持有的variable runtime-owned bytes，必须当作不可信输入全量校验；
- `expected_binding/expected_identity`必须来自prepare-time context/callable state，绝对不能先从blob中读出再原样传回validator；
- `HbgRestoreOps`只提供无分配的copy和publish回调，以后A2/A3、A5可分别将其实现为device-memory copy与cache clean/barrier；
- `HbgRestoreCommit`是完整恢复后的host/AICPU控制面候选commit，记录slot generation、plan generation/hash和invocation identity。

这层不调用ACL/runtime API，不分配内存，不创建stream/event，不查capture状态，也不假设自己是leader。它是将来AICPU entry在“exactly-one leader已经选出、trusted slot registration已经查到”之后调用的纯恢复原语，不是完整HBG L1 runtime。

#### 10.27.2 恢复前先验证task-owned source确实属于这个slot和这次invocation

`restore_hbg_launch_blob` 只接受 `DevicePatched` 状态的blob：`inline_payload_addr`必须精确等于 `runtime_device_args_base + header_size`。这使恢复source不能偷换成blob外的某块device allocation，也不能把host-unpatched canonical bytes直接当成device args运行。`hbg_launch_blob.h`同时补了计算expected payload address时的无符号溢出检查。

在任何working byte被改写前，validator会完成以下强校验：

1. blob长度、header/region canonical layout、source/destination边界和全量SM/runtime-arena覆盖；
2. blob中slot generation与prepare-time generation完全相等，所有base/capacity/runtime offset与trusted binding完全相等；
3. callable/arguments/function-table identity与trusted identity完全相等；
4. identity、restore descriptors和全部pristine payload的content hash仍然一致。

因此stale captured node、另一个operator的package、已换址或已增容的slot、被截断/篡改的runtime args都在第一次copy前fail-closed。当前完整hash校验会在copy前额外扫描一遍大payload，这是首版correctness-first取舍，不被描述为最终性能形态。后续只能在device profile证明成本不可接受后，再设计fused copy/hash或更强的prepare-time trust token，不先删掉完整性门禁。

#### 10.27.3 “复制成功”不等于“调度器可以看见新epoch”

恢复核心按manifest顺序处理每个region：

1. 用checked arithmetic计算runtime-owned inline source和trusted working destination；
2. 调用copy回调恢复该region的完整bytes；
3. 调用publish回调，为将来A2/A3、A5 cache clean/可见性协议保留明确边界；
4. 所有region全部成功后，才一次性覆盖caller的 `HbgRestoreCommit`。

任一copy或publish失败时，之前的working bytes可能已经被部分改写，这是不可能通过一个普通memcpy回调回滚的。协议不假装回滚成功，而是保证旧commit不变，并要求该slot保持non-dispatchable，直到下一次完整restore成功或context teardown。这个区分很重要：不允许scheduler仅看到某些cache line已经更新就开始classify/dispatch。

`HbgRestoreCommit` 当前只是common层的transaction result，尚未形成AICPU peer可见的release-store gate。后续接入production时还必须明确：

- leader将commit/cache clean完成后，以release语义发布restore verdict/generation；
- peers以acquire语义等待同一verdict，只有success才能attach/classify；
- failure必须走统一epilogue，不得放行hidden AICore继续等register window；
- `blob_size`必须由prepare-time registration的可信max/package length约束，不能只相信untrusted header声明的 `total_size`。

#### 10.27.4 无硬件反例要直接对应ACLGraph replay语义

`test_hbg_launch_blob` 从13项增至16项，新增的三组用例不只检查“一次memcpy对不对”：

1. `RestoresThePristineWorkingImageOnEveryReplay`：第一次恢复后主动将working SM/arena填成其他值，再次执行同一package，要求两个full image都恢复到初始快照；这对应“同一ACLGraph第二次replay不得沿用第一次已消费状态”。
2. `AlternatingCapturedPackagesRestoreTheirOwnSnapshotIntoOneSlot`：A/B两个package携带不同argument identity和plan generation，按A→B→A顺序恢复到同一trusted slot，每次都必须看到当前node自己的SM/arena和commit；这对应v1无并发下多captured node共享working slot但不共享pristine source。
3. `FailedRestoreNeverPublishesAReadyCommit`：stale slot generation在0-copy时拒绝；第二个region copy失败和第一个region publish失败都保持sentinel commit不变，即使working slot已被部分改写也不能发布ready。

验证结果：

- runtime完整pre-commit通过，包括LLVM 18 clang-tidy、clang-format、cpplint和header检查；
- runtime editable build通过；
- 定向HBG restore/blob target：**16/16 passed**；
- `ctest --test-dir tests/ut/cpp/build -LE requires_hardware --output-on-failure`：**86/86 passed**，其中70项标记 `no_hardware`；
- HBG/runtime相关Python回归：**112 passed, 4 skipped, 14 warnings**；
- `git diff --check`通过；没有运行NPU任务，没有使用device 0。

runtime阶段提交：

```text
de2aa0f9 Add: 建立HBG逐回放工作镜像恢复核心
```

#### 10.27.5 本提交与“graph像AscendC tiling参数被task管理”的精确关系

当前四层关系已在common code中有了可执行的两段闭环：

```text
canonical HbgGraphPlan（host owner，尚未完整production化）
  -> fresh writable HbgSerializedLaunchBlob
  -> aclrtLaunchKernelWithHostArgs + placeholder
  -> RuntimeOwnedHbgPayload（task/captured-node owner，待device 1证明）
  -> restore_hbg_launch_blob on every invocation/replay
  -> context-owned mutable HbgExecutionSlot（H3尚未接线）
```

这与AscendC tiling data相似的部分是：每个launch node拥有一份不会被下一次host build覆盖的参数快照，capture/replay时仍由该node的runtime args owner保持。与普通小tiling结构不同的部分是：HBG pristine payload较大且含有destination-bound device pointers，scheduler又会消费working image；因此它不能直接在runtime-owned source上执行，必须每次恢复到另一个稳定working slot。

仍未完成的产品闭环包括：

1. H3 prepare-time slot allocation/capacity freeze/device id与generation registration；
2. HBG独立AICPU entry、exactly-one leader和peer acquire/release verdict；
3. A2/A3、A5的真实copy/cache clean/invalidate协议；
4. restore失败时AICPU/AICore common epilogue及operator tail的可终止性；
5. CANN runtime对inline large args的snapshot时点、captured-node lifetime、graph destroy回收点和大小边界；
6. host build对external tensor value的依赖分类，以及不D2H/不sync时哪些HBG callable可以被capture；
7. device 1上同图多次replay、A/B两图交替、主动poison working state、cache-line canary和memory accounting实证。

因此这次提交只把H4的“每次完整恢复、失败不提交”协议变成了可测代码。HBG `simpler_l1_supported()`仍为false，高层 `pypto_init` 仍必须拒绝HBG；当前不存在用host-only UT替代device 1 P0的结论。

### 10.28 HBG H3信任根：task-owned blob不能用自己的binding自证可写

#### 10.28.1 10.27的expected binding仍然缺少一个可传递的owner

10.27中 `restore_hbg_launch_blob` 已要求caller另行提供 `expected_binding/expected_identity`，方向比“直接信任blob header”正确，但它还没有回答这份expected binding到底是谁创建、包含哪些必须freeze的地址、如何证明没有被中途改写。如果未来AICPU entry只是把blob中的 `header.binding` 复制一份当expected再调restore，形式上有两个参数，实际仍然是不可信source自证。

因此H3先新增一个固定interface的prepare-time trust root：

```text
HbgExecutionSlotRegistration (144 bytes, align 8)
  magic / ABI major+minor / exact struct_size
  device_id
  required flags:
    CAPACITY_FROZEN
    SERIAL_ONLY
  max_launch_blob_size
  HbgExecutionBinding:
    working SM base/capacity
    working runtime-arena base/capacity/runtime_offset
    working GM-heap base/capacity
    slot_generation
  outer Runtime base/size
  device KernelArgs base/size
  binary_generation
  registration_hash
```

这份记录将由未来HBG L1 prepare在实际device allocation全部成功后构造，host保留一份copy，并将完全相同的immutable bytes注册给HBG AICPU state。task/captured node只拥有graph package，不拥有也不能更新slot registration。

#### 10.28.2 为什么不只记SM/arena/heap三个base

HBG一次执行真正依赖的persistent device state不只有scheduler working image。outer `Runtime`保存handshake、worker topology、SM/heap/prebuilt-arena pointer与function table；device `KernelArgs`又持有outer Runtime、register window和AICore launch参数。如果只验证SM/arena/heap，旧captured package仍可能在context重建后指向新的working image，却与旧Runtime/KernelArgs/binary generation混用。

新注册记录因此同时freeze：

- 三个working windows和runtime offset；
- outer Runtime device copy的完整window；
- device KernelArgs copy的完整window；
- slot generation与binary generation；
- 本context首版允许的最大serialized launch bytes；
- `SERIAL_ONLY`硬契约，明确v1的一个working slot不可支持并发graph replay。

validator还拒绝五类window之间的任何地址重叠。当前 `KernelArgsHelper` 的outer Runtime和device KernelArgs本来就是两次独立device allocation，HBG pooled SM/arena/heap也分别是独立arena，所以这不改变现有合法layout；它防止未来为省一个allocation而未经设计地让restore destination覆盖control block。

#### 10.28.3 capacity freeze的数学边界

`max_launch_blob_size` 首先必须能容纳：

```text
canonical header
+ at least two region descriptors
+ alignment padding
+ full working SM capacity
+ full runtime-arena capacity
```

因为这两个full image是每次restore的强制项，小于这个下界的注册没有任何合法package，必须在prepare阶段拒绝。上界暂时不超过 `UINT32_MAX`，原因是当前HBG blob的 `total_size`和已核对CANN args-copy内部carrier都是32-bit；这只是防silent narrowing的PyPTO ABI边界，**不是CANN承诺支持接近4 GiB args**。真正可用64 KiB、1 MiB、16 MiB、64 MiB或更大尺寸仍必须由device 1 H0扫描决定。

注册只保存max，不会把max假装成每个CANN task args allocation的真实size。HBG host bridge未来仍必须在launch前验证：

1. fresh writable blob的实际 `args_size` 与header `total_size` 完全一致；
2. 实际size不超过registration max；
3. runtime确实复制了这么多bytes，而不是在某个backend被clamp；
4. AICPU固定header parser只在上述host/runtime合约下使用 `total_size`，不声称kernel入参本身能查询backing allocation长度。

第四点是 `aclrtLaunchKernelWithHostArgs` 入口只给AICPU function一个 `void *arg`而不额外传 `args_size` 的真实限制；注册max能给出private protocol的上界，但不能对一个绕过host bridge的恶意裸symbol调用独立证明device allocation真有那么大。这个信任边界必须在H5 AICPU entry注释和P0 probe中保留，不可在文档中隐去。

#### 10.28.4 registration seal与restore的新关系

`seal_hbg_execution_slot_registration` 在写hash之前先检查：

- magic/version/exact struct size/reserved/required flags；
- device id和可选expected current device；
- 全部window非空、加法不溢出，runtime offset在arena内；
- slot/binary generation非0；
- package capacity同时满足full-image下界和32-bit carrier上界；
- 五个persistent/mutable windows互不重叠。

全部成功后才使用FNV-1a封存registration hash；失败seal保留caller原有hash，避免一份半更新记录被误发布。这个hash是防止stale/corrupt host-device protocol的一致性token，不被当作针对恶意device代码的密码学认证。

`restore_hbg_launch_blob` 的API同步改为只接受完整 `HbgExecutionSlotRegistration`，顺序变为：

```text
validate sealed slot registration
  -> validate actual blob size <= frozen max
  -> validate device-patched blob against registration.binding
  -> validate invocation identity/hash/full restore manifest
  -> copy and cache-publish every region
  -> commit registration.slot_generation + plan generation/hash/identity
```

所以现在有三种不同的失败面：slot registration被拒绝时0-copy；slot合法但blob与之不匹配时0-copy；copy/publish中途失败时slot可能partial-dirty但commit不变。三者不再被一个模糊错误码混在一起。

#### 10.28.5 验证、提交和未完成边界

新增 `test_hbg_execution_slot` 6项无硬件反例：

1. 封存完整注册、expected device mismatch及seal后字段篡改hash mismatch；
2. 缺frozen/serial flag、非法device、零slot/binary generation、缺outer Runtime或KernelArgs window；
3. 小于full-image下界、超过32-bit carrier上界、为0或超过registered max的本launch size；
4. SM/arena、heap/Runtime、Runtime/KernelArgs的三类alias；
5. device window和minimum blob size的溢出；
6. 失败seal不覆盖旧hash。

`test_hbg_launch_blob` 另加了篡改binary generation但不重算registration hash的restore反例，确认在任何copy前返回 `SlotRejected/HashMismatch`且不更新旧commit。

验证结果：

- runtime完整pre-commit通过；
- runtime editable build通过；
- HBG定向测试：slot registration **6/6 passed**，blob/restore **16/16 passed**；
- no-hardware C++全量：**87/87 passed**，其中71项标记 `no_hardware`；
- HBG/runtime相关Python：**112 passed, 4 skipped, 14 warnings**；
- `git diff --check`通过；未运行NPU任务，未使用device 0。

runtime阶段提交：

```text
f6ad61df Add: 建立HBG稳定执行槽可信注册协议
```

这份提交是H3的共享ABI/校验基础，不是H3 production完成。截至 `f6ad61df`，DeviceRunner尚未在HBG L1 prepare中一次性分配五个window，也尚未把注册bytes通过独立AICPU registration task发布到device；当时现有HBG L2 `setup_static_arena` 遇到更大request仍可以release/recommit，不能直接当成capture-safe H3 owner。后续10.29已经补上三块working arena的真实分配/冻结，但完整registration与AICPU发布仍未完成。HBG capability继续为false。

### 10.29 HBG H3继续实现：由DeviceRunner冻结真实working slot，而不是让graph package自报地址

#### 10.29.1 本次解决的是mutable destination所有权，不是pristine source所有权

用户对第二阶段HBG内存管理给出的核心原则是：每次dynamic host build得到并H2D到device的graph，本质上是本次task类似AscendC `tiling_data` 的入参。它必须随launch task或captured node保活，不能放在下一次host build会原地覆盖的context-wide `current_graph` 中。

这个原则同时要求另一块独立内存：HBG scheduler会原地消费SM、runtime arena和GM heap中的ready queue、task state、completion flags与runtime指针，所以runtime-owned pristine source不能直接作为执行区。每次eager调用或ACLGraph replay必须先把当前node自己的source恢复到一个地址稳定的mutable working slot，再放行scheduler。

因此10.29只实现后半条生命周期：

```text
task/captured-node-owned pristine graph package
  -- 每次调用/replay restore，尚未接入AICPU -->
context-owned mutable HBG working slot
  = GM heap + shared memory + runtime arena
```

本提交没有把pristine graph塞入这三块context arena，也没有因此宣称CANN已经替PyPTO管理了graph source。source的WithHostArgs inline snapshot、capture lifetime和destroy回收点仍必须由H0/device 1证明。

#### 10.29.2 为什么旧 `setup_static_arena` 不能直接作为capture-safe owner

原有L2/L3 arena语义允许后续更大的request触发release/recommit。这对一次native run绑定是合理的，因为pipeline lease和同步finalize能界定使用期；对可能已经被ACLGraph node记录的L1地址则不成立：

1. graph package中的SM、arena和heap绝对地址已经在host relocation时固化；
2. captured node可能在任意未来replay，PyPTO看不到最后一次replay完成时点；
3. 如果后续prepare静默增容并换址，旧node会把pristine bytes恢复到已经释放或属于新generation的地址；
4. 只比较request size不能证明调用方拿到的base仍然由当前DeviceRunner arena bank持有。

所以 `HostApi` 新增 `freeze_static_arena`，参数不是单纯三个size，而是三组精确的 `{base, capacity}`。platform owner只有在以下条件全部成立时才冻结：

- 三个region都已经commit且base非空；
- caller提供的base与当前arena bank实际base逐项相等；
- caller提供的capacity与DeviceRunner缓存capacity逐项相等且非0；
- 三个region属于当前选中的同一arena bank。

冻结后，`setup_static_arena` 只接受完全相同的容量三元组；任何增容、缩容、将某个region改为0或间接换址都在修改任何arena前返回失败。相同请求保持幂等且不重新分配。L1显式close和L2/L3 finalize释放arena时同步清除freeze状态，使owner状态与实际allocation生命周期一致。

#### 10.29.3 common helper的transaction边界

新增 `runtime/src/common/task_interface/hbg_static_execution_slot.h`，把platform-independent顺序固定为：

1. 检查三块capacity非0、可无损转换为 `size_t`，且 `runtime_offset < runtime_arena_capacity`；
2. 检查HostApi确实提供setup、三项acquire和freeze能力；
3. 一次setup三块arena，再分别取得实际base；
4. 用checked device-window规则拒绝地址加法溢出和三块window互相重叠；
5. 把刚取得的精确base/capacity交回platform owner冻结；
6. 只有freeze成功后才把candidate `HbgExecutionBinding` 发布给caller。

失败事务不假装撤销已经完成的device allocation：setup成功而后续acquire/layout/freeze失败时，allocation仍由context持有并由显式close回收；但helper绝不发布半可信binding。这个边界与10.27 restore失败的处理一致——物理bytes可能已经存在或部分变化，逻辑ready/registration必须保持未发布。

`HbgPreparedStaticExecutionSlot::binding.slot_generation`在这一层故意保持0。generation不是arena helper可以自行发明的值；它必须由DeviceRunner在outer Runtime、device KernelArgs、AICore binary和所有callable function binding都准备完成后统一生成，并与完整 `HbgExecutionSlotRegistration` 一起seal。

#### 10.29.4 A2/A3与A5 HBG prepare现在真实做了什么

A2/A3和A5的 `host_build_graph/host/runtime_maker.cpp` 都新增同名strong `prepare_l1_runtime_impl`，两份实现保持逐行同构：

1. 按现有优先级解析 `ring_task_window/ring_heap`（task config覆盖环境变量，环境变量覆盖编译默认值）；
2. overflow-safe累加全部ring heap容量；
3. 通过 `PTO2SharedMemoryHandle::calculate_size_per_ring` 计算完整working SM容量；
4. 只在host `DeviceArena` 上重放 `runtime_reserve_layout` 以得到runtime arena容量与inner runtime offset，不commit host image、不做H2D；
5. 调用common helper分配、核对并冻结三块真实device working region；
6. 将三个base、runtime offset和精确capacity写入host `Runtime`；
7. 保持空的orchestration args和 `host_total_tasks == 0`，等待后续每个invocation自己的host build/package。

这里没有构建某个callable的graph，更没有把某次graph同步上传到working slot。prepare只建立地址和容量稳定的destination。以后HBG host builder必须针对这些最终base做relocation并序列化pristine package；AICPU必须在每次调用/replay从该package恢复，二者不能颠倒。

#### 10.29.5 与L2/L3、TRB和当前公开capability的隔离

- `freeze_static_arena` 是HostApi内部能力，onboard与sim platform都提供同一签名；A2/A3、A5、TRB、HBG四类host runtime在同一次editable build中重新编译，避免一侧按旧struct offset读取函数指针。
- 既有TRB bind继续只调用setup/acquire，不调用freeze；对应fake HostApi显式保留空freeze callback，原有temp-buffer测试继续通过。
- HBG L2/L3仍走原有per-run bind/build/H2D和pipeline lease，不会因为新增callback自动进入L1 freeze协议。
- HBG strong `prepare_l1_runtime_impl` 已存在，但public `simpler_l1_supported()`仍为false；common L1 registration也仍对 `host_dlopen_handle` 返回unsupported。因此当前用户入口不会半途进入“slot已分配但AICPU entry不存在”的错误形态。
- 本阶段没有创建stream/event，没有sync/reset，没有capture query/model attach，也没有触碰TRB fixed `L1AicpuInvocationArgs`。

#### 10.29.6 无硬件验证、提交与仍未完成的H3部分

新增 `test_hbg_static_execution_slot` 七组反例：

1. 成功路径只在setup、三项acquire和exact freeze都成功后发布base/capacity；
2. runtime offset越界在任何platform side effect前拒绝；
3. 缺少freeze callback在setup前拒绝；
4. setup失败后不acquire、不freeze；
5. 任一arena base为空时不freeze；
6. 三个device window重叠时不freeze；
7. freeze失败时保留caller原有sentinel output，不发布candidate binding。

验证结果：

- 先确认新测试在HostApi尚无freeze字段时编译失败，再补齐实现；
- runtime editable build通过，A2/A3与A5的onboard/sim HBG/TRB host runtime全部完成增量重编；
- runtime完整pre-commit通过，包括LLVM 18 clang-tidy、clang-format、cpplint和header检查；
- `test_hbg_static_execution_slot` **7/7 passed**，TRB temp-buffer定向回归通过；
- no-hardware C++全量：**88/88 passed**，其中71项标记 `no_hardware`；
- HBG/runtime相关Python：**112 passed, 4 skipped, 14 warnings**；
- `git diff --check`通过；没有运行NPU任务，没有使用device 0。

runtime阶段提交：

```text
ee29203770b1a71d89747a216283c937b9b02ccc Add: 建立HBG L1稳定执行槽分配与冻结边界
```

10.29仍不是H3 production完成。剩余关键闭环是：

1. DeviceRunner在所有callable prepare完成后生成process内不复用的slot generation和binary generation；
2. 把working slot、outer Runtime、device KernelArgs和capacity ceiling组合成完整registration并seal；
3. 通过独立HBG AICPU registration entry发布同一份immutable registration；
4. 由每次host build生成owning `HbgGraphPlan`与fresh writable task package，而不是直接写working slot；
5. AICPU exactly-one leader在每次调用/replay执行full restore，peers只在统一success verdict后进入scheduler；
6. device 1证明CANN确实按task/captured-node持有inline graph source，并确定large-args上限和graph destroy回收行为。

这些完成前，HBG capability继续保持false；本次提交只能表述为“真实working destination已经在prepare阶段可分配且可冻结”，不能表述为“dynamic graph已经被ACLGraph安全管理”。

### 10.30 HBG H1/H2继续实现：把一次dynamic build变成task级immutable plan

#### 10.30.1 本轮直接回应的所有权问题

用户进一步明确：第二阶段HBG每次动态build出来、随后需要送往device的graph，本质上就是这个launch task的AscendC `tiling_data`。因此它的生命周期不能由“最近一次build”或“一份context current graph buffer”决定，而必须跟随具体task；进入ACLGraph后，还必须跟随具体captured node/model。

10.29已经建立了地址稳定的mutable working destination，但当时仍缺两件事：

1. host builder产生的SM/runtime arena仍只是局部vector/arena，没有正式owner；
2. 同一份host bytes若直接交给CANN placeholder，runtime允许原地patch其中的pointer field，canonical cache可能被污染，下一次launch也可能复用到已patch的device地址。

本轮将host侧source拆成两个明确层次：

```text
one dynamic host build
  -> immutable HbgGraphPlan
       private canonical HostUnpatched blob
       owns pristine full SM + full runtime-arena bytes
       owns destination binding + invocation identity + plan hash
       never exposes mutable bytes

one eager launch / one captured node
  -> HbgGraphPlan::serialize()
       fresh writable HostArgs scratch
       CANN may patch inline_payload_addr in this scratch
       scratch A patch cannot modify plan or scratch B

CANN WithHostArgs args loader（尚待device 1证明）
  -> runtime-owned device args source
       AICPU leader restores it into the context working slot on every replay
```

这意味着“图计划”和“launch参数”不是同一个对象。plan可以缓存或被host owner长期持有；scratch只是一次CANN调用的可写输入；未来runtime-owned device copy才是实际跟随task/captured node的tiling-like source。三层不能用同一vector偷懒合并。

#### 10.30.2 `HbgGraphPlan`的不可变与事务边界

新增 `runtime/src/common/worker/hbg_graph_plan.h`：

- `HbgGraphPlan`不可copy、不可move，只能由 `std::unique_ptr<const HbgGraphPlan>` 持有；
- 私有 `canonical_blob_` 是一份已经完成严格校验的 `HostUnpatched` variable blob；
- build时从所有临时region source做deep copy，因此成功返回后不再引用host SM vector、host `DeviceArena`或caller buffer；
- class只暴露binding、identity、generation、hash、serialized size和region count等只读metadata，不暴露mutable byte view；
- `serialize(out)`每次先复制canonical bytes，再以 `HostUnpatched` 模式复验完整blob，最后transactionally替换caller output；
- build或serialize失败时保留caller原有owner/output，不发布半成品。

`build_hbg_graph_plan(...)`仍复用既有 `build_hbg_launch_blob` 的所有overflow、alignment、full-image、region overlap、destination binding和hash规则。这里没有另造第二套宽松serializer；immutable owner只是把已经验证过的canonical representation封装成不可变对象。

需要特别说明其内存成本：每次dynamic build至少产生一次deep copy，把局部SM/runtime-arena source收进canonical plan；每次真正launch再从plan复制一份writable scratch。第一份copy换取明确的host owner，第二份copy换取不同task之间的placeholder隔离。后续只有在profile证明这是瓶颈且能保持相同ownership时才考虑优化，不能先用共享mutable blob换性能。

#### 10.30.3 `host_total_tasks`为什么必须进入task identity

HBG host orchestration结束后会得到实际scheduler task数。原L2路径把它写入context-wide `Runtime::host_total_tasks`，这是一次拥有型run内可用的做法；在异步L1/ACLGraph里却不成立：

```text
host build A -> host_total_tasks = 37 -> enqueue/capture A
host build B -> host_total_tasks = 52 -> overwrite shared Runtime
device later consumes/replays A
```

如果AICPU仍从共享Runtime读取，A会错误地按52个task初始化/判断完成。即使A和B不并发执行，host enqueue返回也不代表device已消费参数；captured A更可能在B build之后才replay。

所以 `HbgInvocationIdentity` 新增：

```text
int32_t host_total_tasks
uint32_t reserved
```

host调用strong build hook时必须把 `host_total_tasks` 初值置0；host orchestration成功后才将实际非负值写入本次plan identity。该字段参与identity equality和plan hash，restore commit也携带同一identity，不能被后续build从context状态覆盖。`reserved`当前必须为0，为versioned演进保留fail-closed空间。

这次ABI是尚未开放的HBG private ABI，因此同步把minor从0升级为1，并锁定：

- `HbgExecutionBinding`：64 bytes；
- `HbgInvocationIdentity`：40 bytes；
- `HbgLaunchRegion`：40 bytes；
- `HbgLaunchBlobHeader`：160 bytes；
- `HbgRestoreCommit`：64 bytes；
- `inline_payload_addr` placeholder offset仍为40；
- binding offset仍为56；identity offset变为120。

不能把上述数值变成CANN large-args上限；它们只是PyPTO自有header ABI。整个variable payload的可接受大小仍要由public API carrier检查和device 1 H0实证共同决定。

#### 10.30.4 A2/A3与A5 strong host-build hook

新增common内部声明 `runtime/src/common/worker/hbg_l1_host_build.h`，并在common `DeviceRunnerBase`提供weak unsupported定义。A2/A3和A5 HBG `runtime_maker.cpp`各自提供同签名strong实现；TRB没有strong覆盖，因此不会误把固定TRB L1 ABI当成HBG variable package。

strong `build_l1_hbg_graph_plan_impl` 的输入包括：

- context-owned host `Runtime`和 `HostApi`；
- 本次 `ChipStorageTaskArgs`与host orchestration entry；
- DeviceRunner未来生成的可信frozen `HbgExecutionBinding`；
- 本次预校验identity和非0 plan generation；
- ring task-window/heap/dep配置；
- transactionally返回的 `unique_ptr<const HbgGraphPlan>`。

它按以下顺序执行：

1. 在任何构图前拒绝null、零generation、未冻结slot、零slot generation；
2. 逐项确认binding中的GM heap、SM、runtime arena base/capacity/offset与prepare后 `Runtime`真实记录完全相等；
3. 确认identity的tensor/scalar count与本次args相等、hash字段合法、输入 `host_total_tasks==0`、reserved为0；
4. 重新解析ring配置并重放host-only layout sizing，要求SM size、arena size和runtime offset与frozen slot完全相等；
5. 在host `DeviceArena` commit临时runtime image，但不commit任何device working slot，也不执行H2D；
6. 直接借用输入输出tensor的device地址建立 `L2TaskArgs`；不为external tensor分配storage，不做H2D/D2H，不做stream/device sync；
7. 建立本次host runtime与orchestration binding，运行host orchestration，得到owning host SM image和实际task count；
8. 把实际task count写入本次identity，并将完整SM与完整runtime-arena作为两个required immutable region deep-build为 `HbgGraphPlan`；
9. RAII清除framework/orchestration runtime binding、host tensor mapping并成对unregister；
10. 只有全部步骤成功后才替换caller plan owner。

这里刻意没有把plan bytes写入10.29分配的working slot。working slot是scheduler会修改的execution state；在真正执行前，只能由未来AICPU entry从本次runtime-owned task source恢复。host build直接H2D会重新引入capture前一次性写入和replay第二次读脏状态的问题。

当前plan包含两个full-image region：SM和runtime arena。GM heap目前只冻结了base/capacity，但尚未生成“哪些bytes是每次execution有语义initializer”的manifest；因此不能简单把整块workspace都复制，也不能漏掉确实需要逐回放初始化的heap span。GM initializer manifest仍是后续H4前的必做项。

#### 10.30.5 external tensor只借用，host view强制read-only

> **10.35后的实现修正：** 本节记录的是 `2873feae` 当时的过渡方案。`74d0ff65` 已进一步取消HBG L1 host build中的全部device-tensor host mapping：read-only mapping也不够，因为host launch时caller stream上的predecessor可能尚未完成，地址可映射不等于数据已经按stream语义可读。当前L1 builder不注册任何host tensor region，`get_tensor_data/set_tensor_data`都会fail-closed，只允许metadata、device address、host scalar和拓扑参与构图。L2 HBG原有staging/mapping路径不变。后续应以10.39的现行结论为准。

L1约束要求输入输出storage由调用方拥有。HBG host orchestration又可能调用 `get_tensor_data/set_tensor_data`，这两者必须和graph payload lifetime分开处理。

本轮strong build hook采取保守规则：

- `ChipStorageTaskArgs`中的device地址原样进入host orchestration参数，不分配或替换；
- platform必须同时提供register/unregister，只有一侧存在时直接拒绝，避免泄漏或不对称ownership；
- 若两者都提供，则逐tensor尝试建立临时host view；成功mapping由RAII成对注销；
- host access进入read-only模式：`host_tensor_read`可读已注册region，`host_tensor_write`在查找region、修改direct bytes或copy mirrored bytes之前统一返回false；
- 没有mapping或某个tensor mapping失败时，不伪造数据；真正依赖该device value的host orchestration通过现有read接口失败；只依赖shape/dtype/stride/address、scalar和拓扑的program仍可build。

这项实现只证明“PyPTO不会在HBG L1 host build偷偷改写调用方tensor”，没有证明任意device tensor value都可在无sync条件下供host读取。尤其A2/A3可用的host mapping不能默认外推到A5。长期仍需在final transformed orchestration metadata中静态标记data-read/data-write requirement：不支持的callable应在进入host build前fail-fast，而不是依赖执行到某个host access时才失败。

#### 10.30.6 canonical、scratch和runtime-owned source的精确边界

截至本轮，三层状态分别是：

| 层 | 当前实现状态 | 可以宣称什么 | 不能宣称什么 |
| --- | --- | --- | --- |
| immutable `HbgGraphPlan` | 已实现 | 一次build的SM/arena bytes有独立host owner，不受后续source修改影响 | 不能说明ACLGraph持有它 |
| fresh writable HostArgs scratch | 已实现 | 每次serialize互相独立，placeholder patch不污染canonical/其他scratch | 尚未接production HBG launch symbol |
| CANN runtime-owned device args | 仅有API bridge与源码依据 | 是首选tiling-like source候选 | copy大小、snapshot时点、capture/replay/destroy lifetime尚无device 1证据 |

`aclrtLaunchKernelWithHostArgs`只会拥有 `argsSize`范围内被copy的bytes。把external tensor地址、working slot地址或binary地址放在header里，不会让runtime接管这些pointer指向的allocation；调用方和PyPTO context仍必须分别保活它们。反过来，plan也不能直接交给CANN原地patch：canonical和writable scratch分层是必要条件，不是多余copy。

#### 10.30.7 无硬件测试与符号边界

本轮新增或扩展的no-hardware测试包括：

- `test_hbg_launch_blob`共18项；新增plan ownership反例证明：build后修改原SM/arena source不影响plan；两份serialize内容初始相等但storage独立；patch scratch A不修改plan或scratch B；随后scratch C仍是canonical；失败build保留旧owner；
- task count负值、reserved非0被拒绝，`host_total_tasks`变化会改变identity/hash；
- `test_hbg_tensor_access`共11项；read-only window仍允许read，但direct和mirrored write都在修改/copy前失败；
- A2/A3与A5 onboard/sim HBG host runtime都重新编译；符号检查确认HBG产物导出strong `T build_l1_hbg_graph_plan_impl`，TRB产物只导出common weak `W` fallback。

验证结果：

- 测试优先：先看到缺少 `hbg_graph_plan.h`和 `host_tensor_access_reset_read_only` 导致的预期编译失败，再补实现；
- HBG定向两个test executable全部通过；
- runtime editable build通过；
- no-hardware C++全量：**88/88 passed**，其中71项标记 `no_hardware`；
- HBG/runtime相关Python：**112 passed, 4 skipped, 14 warnings**；第一次从runtime目录误用了顶层相对测试路径，只得到file-not-found/no-tests，随后回到顶层仓并显式设置当前worktree `PYTHONPATH`重跑，以上结果才是有效回归；
- runtime完整pre-commit通过，包括header、英文字符、large-file、whitespace、clang-format、LLVM 18 clang-tidy和cpplint；
- `git diff --check`通过；
- 没有运行任何NPU task，没有使用device 0。device 1当时仍非空闲，不能据无硬件测试宣称H0通过。

runtime阶段提交：

```text
2873feae Add: 建立HBG任务级图计划所有权
```

#### 10.30.8 对10.29待办的更新和下一步

10.29末尾第4项“由每次host build生成owning `HbgGraphPlan`与fresh writable task package”现在已完成host侧实现，但只完成到CANN调用之前。其余关键闭环仍是：

1. DeviceRunner在executor、outer Runtime、device KernelArgs和callable function binding稳定后生成process内不复用的slot generation、binary generation和plan generation；
2. 组合并seal完整 `HbgExecutionSlotRegistration`，通过独立HBG AICPU registration entry把同一trust root发布到device；
3. 将strong host-build hook接到HBG callable prepare/launch路径，以每次args生成plan，再serialize fresh writable scratch；
4. 使用 `LaunchWithMutableHostArgs`和placeholder调用独立HBG AICPU run symbol，不能让TRB固定entry误解析variable blob；
5. AICPU侧先byte-safe parse header，再用device registration而不是blob自报binding完成fail-closed验证；
6. exactly-one leader在每次eager/replay恢复full SM/runtime arena和GM initializer manifest，所有peer通过统一release/acquire verdict后才attach/classify/dispatch；
7. restore失败进入共同epilogue，hidden AICore不能遗留在window/handshake等待；
8. 在空闲device 1完成placeholder、large args、snapshot时点、captured lifetime、连续replay、graph A/B、cache/order和destroy回收H0矩阵；
9. 完成host tensor data依赖的静态分类，分别记录A2/A3和A5支持边界；
10. 只有上述device闭环和L2/L3回归通过后才把HBG L1 capability从false改为true。

因此当前准确结论是：**HBG dynamic graph已经有task级host canonical owner和逐launch writable参数快照，但还没有runtime-owned captured-node source的板上证据，也没有device逐回放执行闭环。** 这比10.29前进了一层，但仍不能对外宣称HBG L1或HBG ACLGraph可用。

### 10.31 HBG H2/H4前置收口：函数绑定与task数进入pristine runtime arena

#### 10.31.1 继续审查10.30后发现的关键缺口

10.30解决了host graph bytes由谁拥有，但继续沿device消费链检查时发现：当时的plan还不是一个真正可以独立执行的图计划。`HbgInvocationIdentity`虽然已有 `function_binding_hash`和 `host_total_tasks`，但hash只能说明“期望哪份绑定”，不能给scheduler提供实际要跳转的AICore函数地址。

改造前，HBG实际执行仍有两条context-wide读取：

1. `SchedulerContext::post_handshake_init(Runtime *runtime)`直接执行 `func_id_to_addr_ = runtime->func_id_to_addr_`；
2. HBG AICPU boot把 `runtime->host_total_tasks`传给 `on_orchestration_done`。

这里的 `Runtime`是outer Runtime。L2一次拥有型run中，host在同步H2D之后立即执行，后续build不会越过当前run，所以这个结构长期没有暴露问题；L1/ACLGraph却完全不同：

```text
build callable A
  A.func_id=0 -> binary_A
  A.host_total_tasks=37
  enqueue/capture task A

host returns before device consumes A

build callable B
  B.func_id=0 -> binary_B
  B.host_total_tasks=52
  overwrite context-wide outer Runtime

device executes/replays A
  if scheduler reads outer Runtime, it sees B's function/task semantics
```

PyPTO禁止同context合法并发，只能说明A与B不应该同时占用working slot；它不能把异步task参数的ownership降级成“host调用返回即可覆盖”。captured A甚至可以在B完成build很久以后才replay。因此函数地址表和task数都必须和A自己的pristine graph package一起生存，并在每次A replay时一起恢复。

这也是用户提出的AscendC tiling类比在更深一层的具体结果：graph topology bytes、用于解释task中 `func_id` 的真实地址表、以及scheduler完成条件所需的task count，都是同一个task的tiling-like invocation package，不能拆一部分放task、一部分留在“current context”。

#### 10.31.2 outer Runtime与inner PTO2Runtime重新划界

本轮没有把整个outer Runtime再复制一份塞进graph package，而是把状态按语义分为：

```text
context-persistent outer Runtime
  worker handshakes / core geometry
  frozen GM SM / runtime arena / heap bases
  prebuilt arena base + runtime offset
  active platform execution control
  host build阶段可暂存的function table（不是device invocation source）

task-owned pristine PTO2Runtime inside runtime-arena image
  orchestrator/scheduler state
  ready queues / mailbox / arena layout
  PTO2PrebuiltInvocationState
    exact full func_id -> device address table
    exact host_total_tasks
    magic / ABI / count / reserved validation metadata
```

之所以选择inner `PTO2Runtime`，而不是给launch blob再增加一个第三份独立function-table region，有三个原因：

1. 10.30的plan已经完整拥有runtime-arena image，表进入arena后自动参与canonical deep copy和未来full-span per-replay restore；
2. scheduler本来就会在AICPU boot得到restored `PTO2Runtime *`，无需再保留一个外部source pointer或第二套lifetime；
3. graph execution结束时 `runtime_destroy`会修改inner runtime的pointer/state，下一次replay恢复整份arena即可同时恢复scheduler state和函数分发语义，不会出现两种generation各自推进的问题。

新增的内部ABI在10.31提交时为：

> **10.32演进说明：** 以下version 1布局完整保留为实现演进证据，但已不是当前代码。10.32将两架构定义收敛为common `HbgPrebuiltInvocationState`，ABI version升级到2，并用原 `reserved[2]` 的8 bytes保存完整函数表hash；state总大小8216 bytes与函数表offset 24均保持不变。当前规范见10.32.2。

```text
PTO2PrebuiltInvocationState  // 8216 bytes, align 8
  magic = "HBGI"
  abi_version = 1
  func_id_count = 1024
  host_total_tasks >= 0
  reserved[2] == {0, 0}
  func_id_to_addr[1024]      // offset 24
```

固定复制完整1024项而不是只序列化稀疏 `(id, addr)` 对，是有意的correctness选择：

- 新generation会整体替换旧generation，未使用entry自然恢复为0，不会保留上一callable的尾部槽；
- device不需要解析变长稀疏表或另外分配展开buffer；
- `RUNTIME_MAX_FUNC_ID`和task-owned capacity有编译期 `static_assert`，两边不能静默漂移；
- 约8 KiB开销会自动进入 `sizeof(PTO2Runtime)`、arena layout、capacity freeze和plan size，不依赖隐藏常量。

这份state是HBG尚未公开的private ABI，A2/A3与A5必须成对重编，不能混用旧host image与新AICPU binary。本轮双架构文件保持逐行同构，并完成了两套产物的完整editable重编。

#### 10.31.3 host build发布事务

新增 `runtime_set_prebuilt_invocation_state`，其边界是：

1. null runtime、null source、不是精确1024项或负task count时，在任何字段变化前拒绝；
2. 成功输入先把magic清0，使正在构造的state不呈现valid；
3. 深拷贝完整函数表，而不是保留outer Runtime pointer；
4. 写task count、count、reserved和ABI；
5. 最后写magic，发布完整candidate。

当前没有并发reader进入host临时arena，所以这里不需要device atomic publication；“magic最后写”的目的主要是把构造顺序和未来device validator语义固定下来。真正的AICPU peer publication仍必须走后续H4 exactly-one leader + release/acquire gate。

两条host路径都在graph完整构建之后、任何device upload或plan deep-build之前调用该helper：

- 既有L2 HBG：host orchestration得到实际task数后，把当前outer Runtime函数表与task数写进host arena，随后SM与完整arena同步H2D；outer `host_total_tasks`暂时继续赋值以保持legacy host状态可观测，但device scheduler不再读它；
- HBG L1 strong plan hook：同样在host orchestration成功后写inner state，再把SM与完整arena deep-build进immutable `HbgGraphPlan`，不触碰working device slot。

要特别保留一个后续接入要求：strong hook目前只看到outer Runtime的完整staging table。公开HBG L1 DeviceRunner真正调用它之前，必须先清空该表，只重放当前callable全部 `(func_id, addr)`，并对同一份精确全表计算/核对 `function_binding_hash`。本轮没有为了抢跑而修改common L2/L3 binding语义，也不把“snapshot已经task-owned”误写成“identity与精确current-callable表已全部接通”。

#### 10.31.4 device消费路径不再回读context-wide调用语义

HBG AICPU boot现在按以下顺序工作：

```text
attach stable runtime arena base
  -> locate restored PTO2Runtime
  -> validate HBGI magic / ABI / exact count / nonnegative task count / reserved
  -> validation failure: rt=null, run_rc=-1, no wire/classify/dispatch
  -> validation success: wire arena pointers
  -> attach populated SM
  -> finalize device-only runtime fields
  -> scheduler.bind_runtime(restored rt)
       func_id_to_addr_ = rt->prebuilt_invocation.func_id_to_addr
  -> on_orchestration_done(rt->prebuilt_invocation.host_total_tasks)
  -> publish existing classify-ready barrier
```

原来 `post_handshake_init` 中从outer Runtime抓函数表的语句已经删除。函数表只在leader绑定restored runtime时确定；已有 `classify_ready_` / `runtime_init_ready_` release-acquire顺序使peer在dispatch前看到该绑定。后续peer仍会调用现有幂等 `bind_runtime(rt)`，写入的是同一个restored地址，不会重新回到outer Runtime。

这个改动对L2的执行结果保持等价：L2 host H2D已经把snapshot放入arena，AICPU仍得到相同函数地址与task数，只是source从outer Runtime换成inner pristine image。它对未来L1的价值则是决定性的：当AICPU leader每次从runtime-owned HostArgs source恢复full runtime arena后，scheduler天然得到captured node自己的地址表；不需要PyPTO知道这是首次eager、capture执行还是第N次replay。

10.31提交时AICPU只校验inner state自身，还没有拿独立HBG launch header中的 `function_binding_hash/host_total_tasks`做交叉核对，因为production HBG AICPU entry尚未接入。10.32已经在host strong build阶段加入identity hash与实际完整表的源头互证，但后续独立entry仍必须在每次device restore success发布前再次满足：

```text
header.identity.host_total_tasks == restored_state.host_total_tasks
header.identity.function_binding_hash == hash(restored exact full table)
registration binary generation == header/plan expected generation
```

任一不一致都必须在wire/classify/dispatch前失败，不能把两个都来自host当作可跳过验证的理由。

#### 10.31.5 测试先行与新增反例

本轮先增加 `test_hbg_prebuilt_invocation.cpp`，并让同一source分别通过A2/A3和A5 HBG include/runtime配置编译。第一次构建按预期因缺少constant、state字段和helper而失败，证明测试不是在验证已有行为。

实现后四组反例覆盖：

1. null runtime、null function source、错误table count、负task count全部拒绝，且已有snapshot逐byte不变；
2. 成功snapshot持有完整函数表和task数，调用方随后清零原source不会修改arena；
3. 第二generation用新的 `func_id=0`地址整体替换第一generation，同时把旧generation独有的 `func_id=19`清0；
4. magic/ABI/count/task metadata被破坏后validator拒绝。

此外，既有HBG `run_stream_reuse`场景本来已有add/sub两个callable：两者都从 `func_id=0`编号，但0分别绑定ADD与SUB binary。旧用例只在A2/A3 onboard检查AICore instruction-cache/新stream问题；本轮增加独立的sim/onboard通用交替用例：

```text
register add callable  (func_id 0 -> ADD)
register sub callable  (func_id 0 -> SUB)
run add -> run sub -> run add -> run sub
each result compared with its own golden
```

该scene test和结构性snapshot UT解决不同问题：scene test证明两个callable共享worker时不会把数值func ID当全局identity；UT证明host source变化和新generation替换不会修改旧task snapshot。真正的“两个captured graph异步交替replay”仍需device 1 HBG L1路径完成后验证，不能由同步L2 sim替代。

#### 10.31.6 验证过程中的环境问题也保留记录

本轮验证结果：

- A2/A3与A5的新prebuilt invocation C++ target：**2/2 passed**；
- runtime editable build通过，A2/A3与A5 onboard/sim HBG产物均完成重编；
- no-hardware C++全量：**90/90 passed**，其中73项标记 `no_hardware`；
- A2/A3 HBG sim定向：vector + prepared callable + native lifecycle，**8/8 passed**；
- A5 HBG sim定向：vector + prepared callable，**7/7 passed**；
- callable-local `func_id=0` add/sub交替专项：**1/1 passed**；
- runtime完整pre-commit通过，包括headers、English-only、platform literal、clang-format、LLVM 18 clang-tidy、cpplint、ruff和pyright；
- `git diff --check`通过。

仿真第一次运行的8个用例全部在编译SimKernel之前失败，错误统一为 `g++-15 not found`。这不是代码回归：机器已有 `/mnt/workspace/inductor/toolchains/gcc15`与 `gcc15-shims/g++-15`，只是默认PATH未包含。仅为测试进程临时加入shim和对应lib目录后，上述A2/A3与A5结果全部通过；没有修改仓库或系统toolchain配置。

pre-commit第一次也因默认 `/usr/local` LLVM 21缺少resource headers而在 `unistd.h/stddef.h`、`cstdint`处误报。改用此前已解包的LLVM 18及其库目录后，完整hook通过。这里保留失败原因，避免以后把“工具根目录未选中”误诊成HBG头文件破坏。

本轮没有运行任何NPU task，没有使用device 0。device 1仍不满足空闲条件，因此没有新增WithHostArgs/ACLGraph板上证据。

runtime阶段提交：

```text
6228b481 Add: 将HBG函数绑定收进任务级图镜像
```

#### 10.31.7 当前可以宣称与仍不可宣称的边界

现在可以准确宣称：

- HBG pristine runtime-arena image已经自包含scheduler实际要使用的完整function address table和host task count；
- 现有L2 HBG AICPU/scheduler已经从该task-owned state消费，不再从outer Runtime读取这两项调用级语义；
- 新graph generation能整体替换旧函数表，旧host source修改不影响已构建snapshot；
- A2/A3与A5实现同构，并通过各自重编和无硬件/仿真回归。

仍然不能宣称：

- HBG L1已supported；support query仍为false；
- CANN已经随eager task或captured node持有大尺寸inline graph source；
- AICPU leader已经从WithHostArgs source执行per-replay restore；当前L2仍是host同步H2D预填arena；
- header identity、sealed registration与restored function table已经完成三方交叉验证；
- GM heap initializer manifest已经完整；
- 同一ACLGraph连续replay或graph A/B交替replay已经在device 1通过。

下一阶段仍按原顺序推进：生成/seal完整slot registration并发布给独立HBG AICPU entry；接入fresh mutable HostArgs scratch；leader以registration为trust root恢复SM/runtime arena并验证本轮task-owned invocation state；peer在统一verdict后才进入scheduler。完成这些代码也不等于可开放capability，最后仍需空闲device 1上的snapshot/lifetime/large-args/cache/order/replay矩阵。

### 10.32 HBG H4前置收口：让launch identity与scheduler实际函数表互证

#### 10.32.1 为什么10.31的“完整表进入graph package”仍不够

10.31已经解决了最主要的所有权问题：每次dynamic host build生成的pristine runtime-arena image自己携带完整1024项函数地址表，后续build不会再通过outer `Runtime::func_id_to_addr_`覆盖旧task/captured node的分发语义。继续沿未来L1 restore路径检查时，仍有一条没有闭合的身份链：

```text
HbgLaunchBlobHeader.identity.function_binding_hash
  -> 声明这次launch期望哪份函数绑定

pristine runtime arena中的func_id_to_addr[1024]
  -> scheduler真正会解引用的数据
```

此前两者虽然都由host构建，却没有共同算法和强制比较。header hash若算错、调用者传错、outer Runtime staging table混入另一callable残留，或者runtime-owned source中的arena bytes被破坏，scheduler仍可能拿到一张和header身份不一致的表。只校验magic/version/count/task数不能发现这种差异；只信header hash更不能替代实际地址数据。

这与用户强调的“dynamic graph是task自己的tiling参数”直接相关：tiling-like package不仅要拥有graph bytes，也必须能证明package header所选择的调用身份与package内真正执行的数据属于同一代。否则两个captured node即使各自有独立source，仍可能出现“node A的header + node B的函数表”这种串包。

因此本轮没有先急着生成slot registration，而是先补齐更靠前的函数绑定互证。slot registration负责证明destination/context/binary可信；它不能替代invocation内部header与restored data的一致性。

#### 10.32.2 common `HbgPrebuiltInvocationState`与ABI处理

新增 `runtime/src/common/task_interface/hbg_prebuilt_invocation.h`，将A2/A3和A5此前各自复制在 `pto_runtime2.h`中的prebuilt invocation ABI收敛为common定义：

```text
HbgPrebuiltInvocationState, 8216 bytes, align 8
  offset 0   magic = "HBGI"
  offset 4   abi_version = 2
  offset 8   func_id_count = 1024
  offset 12  host_total_tasks
  offset 16  function_binding_hash
  offset 24  func_id_to_addr[1024]
```

这里没有扩大runtime arena中的state，也没有移动函数表。原version 1在offset 16有两个必须为0的 `uint32_t reserved`；version 2正好将这8 bytes解释为一个 `uint64_t function_binding_hash`。所以：

- state总大小仍为8216 bytes；
- scheduler现有的函数表offset仍为24；
- A2/A3与A5的 `PTO2Runtime`后续字段相对布局不因本次变化扩大；
- ABI version明确从1升到2，旧binary与新binary混用会因version不符fail-closed，而不是把旧reserved静默当成有效hash；
- `sizeof`、两个关键 `offsetof`、standard-layout和trivially-copyable均有static assertion。

这项ABI尚未对外开放HBG L1，所以可以versioned演进；但AICPU、host runtime和AICore相关产物仍必须成套重编，不能依赖布局碰巧不变而混用不同version。

#### 10.32.3 完整函数表hash的定义与所有权

common helper `hbg_function_binding_hash`只接受精确1024项表。hash输入为：

```text
fixed-width uint64 function_count (=1024)
followed by 1024 * uint64 function addresses
```

实现继续使用项目已有FNV-1a 64工具，但count先转换为固定宽度 `uint64_t`，不把host `size_t`字节表示偷偷变成ABI的一部分。null table或非精确count直接返回0；0不是合法binding identity。

必须hash完整固定长度表，而不是只hash非零entry，原因有三：

1. 第二个callable可能复用 `func_id=0`，同时必须把第一代独有的 `func_id=19`恢复为0；
2. 稀疏表中“没有绑定”也是本次调用语义，不能继承上一代尾部entry；
3. full-table hash与full-table deep copy使用相同边界，后续device交叉校验无需再发明稀疏排序或entry count协议。

hash本身不是地址表owner。真正的scheduler data仍是pristine runtime arena内的1024项副本；hash只用于证明header、state和未来registration选择的binary generation没有相互串代。CANN WithHostArgs即使deep-copy整个blob，也只拥有这份task参数bytes，不会因此拥有地址指向的AICore binary或external tensor storage。

#### 10.32.4 snapshot发布与校验事务

`hbg_set_prebuilt_invocation_state`保持10.31的事务边界：

1. null source、错误count、负task数或非法hash在任何state mutation前拒绝；
2. 先将magic清0，使构建中的state不可能被validator误认为ready；
3. 完整复制1024项表，写task count、count和hash；
4. 写ABI version；
5. 最后发布magic。

当前该helper只在host构建局部pristine arena时调用，没有并发reader；“magic最后发布”主要是清晰的image publication协议，不应误写成已经替代未来AICPU leader/peer的release-acquire gate。真正device replay仍须由exactly-one leader完成restore和cache publish，再通过独立verdict原子发布给peers。

`hbg_has_valid_prebuilt_invocation_state`现在不仅检查magic/version/count/task数，还会重算完整表hash并与state字段比较。因此runtime arena中的任一函数地址被破坏，即使其他metadata未变，也会在现有HBG AICPU leader进入wire/classify前被拒绝。

`hbg_prebuilt_invocation_matches`再增加expected hash与expected task count比较，作为未来device header交叉校验的共同语义入口。

#### 10.32.5 host strong build现在完成哪一级交叉校验

A2/A3与A5的 `build_l1_hbg_graph_plan_impl`在host orchestration完成、实际task数已知后按下列顺序处理：

```text
copy exact outer staging table into local pristine arena state
  -> common helper computes/stores full-table hash
  -> compare state.hash with input identity.function_binding_hash
  -> compare state.host_total_tasks with actual host_total_tasks
  -> only then finish runtime layout and build immutable HbgGraphPlan
```

这意味着future DeviceRunner若没有先清空outer staging table、只重放当前callable全部 `(func_id, addr)`并用同一common算法生成identity，strong hook会直接失败，不会产生一个header与arena data不一致的plan。它也把10.31文档里的“后续必须核对hash”从纯设计要求推进成了host build的硬门槛。

但这仍不是device replay闭环。capture之后CANN持有的是runtime-owned args source；每次replay恢复的是那份source中的runtime arena。未来独立HBG AICPU entry仍必须在每次restore后重新比较：

```text
runtime-owned header.identity.function_binding_hash
    == restored_state.function_binding_hash
    == hash(restored_state.func_id_to_addr[1024])

runtime-owned header.identity.host_total_tasks
    == restored_state.host_total_tasks

sealed registration.binary_generation
    matches the executor/function-address generation selected by this plan
```

host阶段曾经通过不能替代device阶段，因为args copy、placeholder patch、captured-node持有、device restore与cache可见性都位于host plan构建之后。只有device阶段也在publish restore success之前完成互证，才能阻止被破坏或串代的runtime-owned source进入scheduler。

#### 10.32.6 对现有L2/L3和两个架构的影响

现有HBG L2仍在同步H2D前调用相同snapshot helper，所以它现在也将hash写进pristine arena；AICPU boot会重算并验证。scheduler仍读取完全相同的1024项地址，task数语义不变。该变化没有修改：

- L2/L3的API和stream/resource ownership；
- HBG L2的host build与同步H2D时序；
- TRB fixed L1 ABI和 `L1AicpuInvocationArgs`；
- 当前HBG capability query（仍为false）；
- working slot、workspace或external tensor的owner。

A2/A3与A5的arch header只保留同名type alias和count alias，实际结构、hash、set/validate/match实现均来自common header，避免未来一个架构升级ABI而另一个仍校验旧reserved字段。

#### 10.32.7 测试与验证

扩展同一份 `test_hbg_prebuilt_invocation.cpp`，继续由A2/A3和A5各自编译。新增反例覆盖：

1. snapshot保存的hash等于common算法对原完整表的结果；
2. expected task数错误或expected hash错误时match失败；
3. 第二generation替换完整表后hash也整体替换，不沿用第一代；
4. snapshot完成后单独篡改 `func_id=0`地址，metadata本身不变，validator仍因重算hash失败；
5. null source与非精确1024项count不能生成hash；
6. 只修改最后一项地址也会改变hash，证明覆盖完整固定长度表而不是前缀。

验证结果：

- A2/A3与A5 prebuilt invocation定向target：**2/2 passed**；
- runtime editable build通过，A2/A3、A5的onboard/sim HBG与TRB产物完成重编；
- no-hardware C++全量：**90/90 passed**，其中73项标记 `no_hardware`；
- A2/A3 HBG sim中add/sub两个callable都从 `func_id=0`起编号的交替专项：**1/1 passed**；
- changed-files pre-commit全部通过，包括headers、English-only、clang-format、LLVM 18 clang-tidy和cpplint；
- `git diff --check`通过。

完整pre-commit第一次运行暴露的是构建缓存工具竞态，不是代码错误：多个clang-tidy worker同时发现若干空 `compile_commands.json`并发重建/删除同一cache目录，分别出现 `FileNotFoundError`和 `Directory not empty`；同一轮markdown hook还格式化了两个与本阶段无关的skill文档。处理方式是：

1. 精确检查worktree；
2. 用 `apply_patch`还原两个无关markdown副作用；
3. 等editable build结束后单独重跑clang-tidy，结果通过；
4. 再对本阶段changed files运行完整hook集合，全部通过。

没有用 `git checkout/reset`覆盖用户文件，也没有把lint工具副作用提交。没有运行NPU任务，没有使用device 0；device 1仍没有满足空闲门槛，因此本阶段没有新增CANN snapshot/capture lifetime证据。

runtime阶段提交：

```text
ade00349 Add: 固化HBG任务级函数绑定身份
```

#### 10.32.8 当前结论和下一阶段

现在可以新增宣称：

- HBG pristine runtime arena中的函数表、task数和完整表hash属于同一个task-owned image；
- A2/A3与A5共享同一ABI和hash算法；
- host strong build不会生成identity hash与实际scheduler函数表不一致的plan；
- 现有HBG AICPU boot会发现runtime arena函数地址被破坏；
- 结构总大小与函数表offset保持，L2同步路径通过双架构编译、无硬件和sim专项回归。

仍然不能新增宣称：

- device leader已经从runtime-owned WithHostArgs source执行restore；
- runtime-owned header与restored state已经在每次ACLGraph replay互证；
- slot registration、binary generation和函数绑定identity已经三方闭环；
- CANN确实以captured-node lifetime持有大尺寸graph package；
- HBG L1或HBG ACLGraph已经supported。

下一阶段回到H3/H5主线：由DeviceRunner在frozen working arenas、outer Runtime、device KernelArgs和已注册AICore executor全部稳定后，形成完整 `HbgExecutionSlotRegistration`的host owner并seal；generation与package capacity必须有明确来源，不能使用CANN内部“约2048次launch”或256 MiB实现常量。registration发布给独立HBG AICPU entry之后，才能把本轮的函数表互证放进每次device restore的success gate。

### 10.33 HBG H3继续实现：完整slot registration的host trust-root owner

#### 10.33.1 本阶段要解决的不是“再做一份binding”

10.29已经分配并冻结了working GM heap、shared-memory和runtime arena；10.30/10.31/10.32又建立了task-owned graph plan、函数表和identity。此前仍没有一个由context owner统一认可的对象，能够回答未来AICPU restore前必须校验的全部persistent事实：

```text
这是不是同一个device？
这是不是同一代working slot？
SM / runtime arena / GM heap的base与capacity是否仍是prepare时那组？
outer Runtime和device KernelArgs是否仍是同一组allocation？
通用AICore executor是否仍是同一份binary？
这个runtime-owned graph package是否超过prepare时允许的结构容量？
这个registration本身是否被修改？
```

launch blob不能靠自己携带的binding回答这些问题，否则任何stale/corrupt blob都可以“自报一个匹配自己的destination”。`HbgExecutionSlotRegistration`在10.28已经定义了144-byte ABI和校验器，但此前只有测试构造的样例，没有production host owner。

本阶段将“arena事实由谁提供”“generation由谁生成”“何时seal”“失败后谁保留”分别落到明确owner上。仍没有把registration交给AICPU，因此这一步是H3 host trust-root完成，不是H3/H5 device闭环完成。

#### 10.33.2 generation不能由host-runtime DSO或launch次数生成

一个容易实现但不可靠的方案是在 `libhost_runtime.so`中放static atomic。`ChipWorker::finalize()`会 `dlclose`该DSO；重新加载同一runtime后static counter可以从头开始。若将这种值写进captured package，generation的“不复用”就依赖DSO加载历史，而不是进程owner。

另一个错误方案是使用kernel launch次数、所谓“约2048次launch”上限或某个args pool index。这些值既不是公开稳定规格，也不能证明context/resource lifetime；host launch返回更不代表device已消费task。

现有 `ChipWorker`本来就为native run维护一个进程生命周期单调 `run_epoch`，其代码位于常驻的 `_task_interface`侧而非动态host runtime。此次将该source重命名为更准确的 `next_process_epoch()`，并让下列对象共用同一非零唯一性域：

```text
L1 context generation
native-run epoch
```

实现使用原子compare-exchange，从1单调递增；达到 `UINT64_MAX`后直接抛出overflow，不回绕复用。L1 context在调用host runtime init前领取一个generation；失败init可以烧掉编号，但不能重复使用旧编号。

内部 `simpler_l1_init` ABI新增最后一个 `uint64_t context_generation`参数：

- `ChipWorker`自动生成并传入，Python `pypto_init`和operator API不增加参数；
- onboard C API和 `DeviceRunnerBase::initialize_l1_borrowed`同时拒绝0；
- simulator的unsupported stub保持同签名，避免host runtime产物间ABI漂移；
- L2/L3 `simpler_init`签名完全不变；
- TRB L1也收到generation，但不为此分配HBG registration，只把它作为dormant context identity保留到close。

generation在execution mode成功claim后、stream/event init之前进入DeviceRunner。若init失败且全部回滚到 `New`，它被清0；若stream/event rollback失败导致context必须保留给显式close，generation也随owner保留。这样失败初始化不会留下“有资源但generation owner丢失”的半状态。

#### 10.33.3 HBG strong query只陈述arena事实，不发明generation

common `hbg_l1_host_build.h`新增runtime-specific query：

```text
query_l1_hbg_execution_binding_impl(const Runtime *, HbgExecutionBinding *)
```

职责边界刻意很窄：

- common/TRB只有weak unsupported实现；
- A2/A3和A5 HBG各自提供strong实现；
- strong实现只在 `prepare_l1_runtime_impl`已经成功冻结static arena后可用；
- 它从host `Runtime`读取精确SM、runtime arena、GM heap base/capacity和runtime offset；
- 它要求三块device window与offset自身有效，失败时不替换caller output；
- 输出的 `slot_generation`必须为0。

最后一点是所有权防线：runtime maker知道自己分配了哪些arena，但不知道进程context identity。若strong hook自行填写generation，DeviceRunner就无法区分“runtime事实”和“caller自报identity”，也会重新引入DSO lifetime问题。

editable build后的符号核对结果符合隔离预期：

```text
A2/A3 HBG onboard:  T query_l1_hbg_execution_binding_impl
A2/A3 TRB onboard:  W query_l1_hbg_execution_binding_impl
A5 HBG onboard:     T query_l1_hbg_execution_binding_impl
A5 TRB onboard:     W query_l1_hbg_execution_binding_impl
```

sim产物也成套重编；公开HBG support query仍为false，因此这些strong prepare/query基础不会被用户入口半途调用。

#### 10.33.4 registration只能在所有persistent window稳定后seal

`DeviceRunnerBase::prepare_l1_callable_locked`原来的静态prepare顺序是：

```text
prepare platform/runtime state
  -> allocate/copy outer Runtime
  -> allocate/copy device KernelArgs
  -> register generic AICore executor binary
  -> enqueue AICPU init/register callable
```

本阶段把registration构建插在“executor注册完成”和“任何AICPU init/register enqueue”之间。此时：

- HBG working GM heap/SM/runtime arena已经分配并freeze；
- outer Runtime device allocation及其精确copy size已知；
- device KernelArgs allocation及 `sizeof(KernelArgs)`已知；
- `rtRegisterAllKernel`已经成功，capture-time launch不需要lazy registration；
- AICore executor原始bytes仍由context持有，可以计算稳定内容身份；
- caller stream上还没有发布任何HBG registration task。

如果runtime query返回unsupported，方法直接成功返回且不分配owner，这就是TRB路径。如果HBG query成功，DeviceRunner要求hook给出的generation仍为0，再填入自己的 `l1_context_generation_`。

#### 10.33.5 binary identity与function binding identity不是同一个东西

registration中的 `binary_generation`本阶段写入通用AICore executor ELF的build-id/FNV fallback内容身份。它证明context准备并pin住的是哪份executor binary。

每个callable的child AICore binaries则不属于这一个字段：它们已经在每次graph plan的完整 `func_id_to_addr[1024]`和 `function_binding_hash`中表达。二者的生命周期根不同：

```text
registration.binary_generation
  -> context-pinned generic executor / launch entry identity

plan.identity.function_binding_hash
  -> this task/captured node's exact child function address table
```

如果把二者合并成一个hash，要么每注册一个callable就必须重建context registration并使旧captured node stale，要么registration无法证明通用executor是否变化。当前拆分允许registration在context内保持immutable，同时每个plan拥有自己的callable-local table identity。

字段名暂时保留既有ABI中的 `binary_generation`；过程记录明确它当前是content identity，而非另一个会随每次launch递增的counter。

#### 10.33.6 package capacity如何确定，以及它不代表什么

当前HBG plan只允许两个mandatory full-image region：

1. 完整shared-memory image；
2. 完整runtime-arena image。

所以DeviceRunner使用 `hbg_minimum_launch_blob_size(binding)`推导当前registration的 `max_launch_blob_size`：

```text
aligned HbgLaunchBlobHeader
+ 2 * HbgLaunchRegion
+ frozen shared_memory_capacity
+ frozen runtime_arena_capacity
```

对当前manifest而言，这既是minimum也是exact structural capacity。它不是：

- CANN公开支持的大参数上限；
- runtime源码中的256 MiB内部常量；
- launch次数或task args pool容量；
- 未来GM heap initializer manifest的预留。

如果后续加入GM initializer region，当前registration会因blob超过capacity而fail-closed，必须在capability开放前明确新的prepare-time预算并生成新context registration；不能捕获后再偷偷增大。即使当前结构size小于 `UINT32_MAX`并能seal，也仍必须由device 1 H0证明目标CANN路径完整copy、capture持有和replay这类实际大小的HostArgs。

#### 10.33.7 transactional builder与immutable host owner

common `hbg_execution_slot.h`新增 `HbgExecutionSlotRegistrationSpec`和 `build_hbg_execution_slot_registration`：

- spec只包含DeviceRunner已经拥有的事实；
- helper先构造local candidate；
- 复用既有validator检查magic/version/flags、device、非零generation、全部window、capacity、overflow/alias和binary identity；
- 只有seal成功、registration hash写入candidate后才替换caller output；
- 任何失败保留原owner逐byte不变。

DeviceRunner随后在prepare阶段分配：

```text
unique_ptr<const HbgExecutionSlotRegistration>
```

这份host owner只在HBG strong query成功时存在，TRB每个callable不增加大块按callable状态。`const`防止seal后被普通代码修改；未来独立HBG AICPU registration entry必须接收这同一份bytes，而不是从launch blob重新拼一份“看起来等价”的registration。

registration host owner当前没有device copy，也没有AICPU static registry slot。过程文档使用“host trust root”而不是“device registration完成”，避免过度宣称。

#### 10.33.8 close失败与重试所有权

`finalize_l1_borrowed`仍在任何destructive teardown前进入Closing；prepare/launch不能再进入。registration owner和context generation不在teardown中途清除：

- AICPU unload、KernelArgs/Runtime free、arena/memory allocator finalize任一失败时，close返回首个错误；
- immutable registration与generation继续留在DeviceRunner中，host runtime DSO也由ChipWorker保留，可显式重试；
- 只有所有device资源释放成功、L1ExecutionState实际到达Closed后，才reset registration owner和generation；
- close不增加stream/device sync，调用方仍必须先graph destroy并external quiescence。

这与未来device registration owner的要求一致：当device/static registration发布实现加入时，失败释放必须纳入同一事务，不能host owner先消失而device仍持有旧trust root。

#### 10.33.9 测试、构建和仍未运行的硬件验证

`test_hbg_execution_slot`先增加两个测试，再实现builder；第一次target build按预期因缺少spec与builder符号失败。完成后共8项通过，新增覆盖：

1. spec构建出的registration逐项等于输入事实，registration hash非零且完整validator通过；
2. 零slot generation导致build失败，caller原有registration逐byte不变；
3. null output明确返回 `NullArgument`。

本阶段验证结果：

- runtime完整editable build通过，A2/A3、A5的onboard/sim、HBG/TRB产物成套重编；
- no-hardware C++全量：**90/90 passed**，其中73项标记 `no_hardware`；
- PyPTO L1与simpler ChipWorker高层无硬件UT：**51/51 passed**；
- HBG/TRB query strong/weak符号核对符合预期；
- changed-files pre-commit全部通过，包括headers、English-only、clang-format、LLVM 18 clang-tidy和cpplint；
- `git diff --check`通过。

没有运行任何NPU task，没有使用device 0。device 1仍没有新的空闲授权，因此registration是否能通过未来AICPU entry异步发布、其HostArgs大小能否capture、以及captured graph多次replay时是否保持同一trust root，均没有板上证据。

runtime阶段提交：

```text
6b356c35 Add: 封装HBG稳定执行槽注册所有权
```

#### 10.33.10 当前可以宣称与下一步

现在可以准确宣称：

- L1 context拥有进程生命周期不复用、非零且不对Python暴露的generation；
- HBG runtime maker只提供冻结arena事实，不能自行选择generation；
- DeviceRunner在所有persistent window和通用executor稳定后，生成并seal完整registration；
- registration冻结device、三块working arena、outer Runtime、device KernelArgs、context generation、executor内容身份、当前package结构容量和serial-only flag；
- host registration owner immutable，并遵守close失败保留/成功释放语义；
- TRB weak路径不创建HBG registration owner，L2/L3 API不变。

仍然不能宣称：

- registration已经通过独立HBG AICPU entry发布到device；
- AICPU restore已经以这份registration为trust root；
- `max_launch_blob_size`已经由CANN device capability证明；
- HBG public callable registration已接受host orchestration DSO；
- HBG L1/ACLGraph已supported。

下一阶段需要实现独立HBG AICPU registration ABI和device-side immutable registry owner。它至少要：

1. 与TRB `L1RegisterCallableArgs`分离，不能让fixed callable ABI误解析144-byte slot registration；
2. 在prepare caller stream上使用WithHostArgs异步发布，不引入sync；
3. device端先byte-safe复制/校验registration，再一次性publish ready；
4. 重复注册必须要求bytes/hash完全一致，冲突fail-closed；
5. close/failure时明确AICPU static state与binary unload的顺序；
6. 随后独立HBG run entry才能用这份trust root校验runtime-owned graph blob并执行per-replay restore。

在registration device owner和run entry完成前，不能为了“先跑起来”让AICPU直接信launch blob里的binding，也不能复用TRB固定run entry偷渡variable tail。

### 10.34 HBG H3继续实现：独立AICPU registration与binary-lifetime device owner

#### 10.34.1 先再次区分context trust root与task graph package

本阶段直接承接用户对HBG graph内存的要求：每次dynamic host build产生的graph，在H2D并作为一次kernel task的输入时，本质上等价于AscendC每次launch的tiling参数。由此有两条不能互相替代的lifetime：

```text
context lifetime
  HbgExecutionSlotRegistration
    -> 哪个device / slot generation
    -> working SM/runtime arena/GM heap window
    -> outer Runtime / device KernelArgs
    -> generic AICore executor identity
    -> package capacity ceiling

task / captured-node lifetime
  runtime-owned HbgLaunchBlob
    -> 本次host build的pristine SM image
    -> 本次host build的pristine runtime-arena image
    -> 本次host_total_tasks
    -> 本次callable-local function table/hash
    -> 本次tensor addresses / scalars / plan generation
```

context registration可以且应该只有一份；task package绝不能因为当前“全核占用、禁止并发”就只有一份可覆盖buffer。禁止并发只允许多个执行复用同一个mutable working slot，不允许后一次host launch改写前一个尚未被CANN消费、或已经被ACLGraph捕获的pristine source。

所以本阶段只建立第一条lifetime在AICPU侧的owner。它没有把graph blob存进registry，也没有让后一次build覆盖前一次package；真正类似tiling_data的runtime-owned task package仍留给下一阶段的独立run entry。

#### 10.34.2 为什么registry跟随AICPU binary lifetime，而不提供reset

> **2026-08-18 后续上板纠正：**本节记录的是当时的设计假设，不是最终事实。后续在同一Host进程中顺序创建第二个HBG L1 context时，`aclrtBinaryUnLoad`已经成功，但标准AICPU scheduler内部加载的runtime DSO及其static registry仍然resident，第二个context的execution-slot registration因而返回`Conflict(status=7)`。最终协议改为“context内immutable，新context在上一context已外部quiesce并close的前提下，用新generation有序reset resident registry”，详见10.45。下文保留用于说明为什么原假设在没有真实上板证据时看似合理，以及为什么后来必须修正。

registration内包含多个device地址。AICPU restore未来会在每次replay前读取它，因此device侧必须存在一个跨task稳定、不可被普通run修改的owner。

本阶段选择把它放在HBG AICPU runtime DSO的静态存储中：

- `aclrtBinaryLoadFromData`成功后，DSO与静态registry一起存在；
- prepare registration task只负责一次性发布immutable bytes；
- 所有后续HBG run task只读acquire该registry；
- caller完成外部quiescence后，显式close先unload AICPU binary；
- binary unload成功即结束device registry lifetime，之后才允许释放它引用的Runtime/KernelArgs/arena；
- binary unload失败则registry仍可能存在，host不能先释放其引用对象。

registry没有公开或内部reset函数。若允许单独把Ready改回Empty，会出现旧captured node仍可replay、但同一DSO静态slot已经接受另一代registration的ABA问题。context generation虽然能检测generation不一致，却不能让已经释放并可能复用的旧device地址重新安全。正确的代际边界是整个L1 context与其AICPU binary owner，而不是一条可重置全局变量。

#### 10.34.3 `Empty -> Publishing -> Ready`的一次性发布协议

common新增 `hbg_execution_slot_registry.h`，定义device-process owner：

```text
HbgExecutionSlotRegistry
  atomic phase
  HbgExecutionSlotRegistration registration
```

phase只有三个合法值：

```text
Empty      尚无registration
Publishing 唯一writer已经取得发布权，但完整bytes尚不可读
Ready      immutable registration已经release-publish
```

注册流程严格按以下顺序：

1. 从task args复制出本地 `HbgExecutionSlotRegistration candidate`；
2. 用expected device id执行完整magic/version/size/flags/generation/window/capacity/overlap/hash校验；
3. 校验失败直接返回 `SlotRejected`，registry仍为Empty，registration bytes不变；
4. CAS从Empty取得Publishing；
5. 复制完整144-byte candidate到registry；
6. release-store Ready；
7. reader只有在acquire-load看到Ready后才复制registration，并再次运行完整validator；
8. reader验证失败返回 `CorruptState`，不向caller发布部分output。

Ready之后不再写任何registration field。再次注册时：

- candidate逐byte完全相同：返回 `AlreadyRegistered`，作为幂等成功；
- 任一byte不同：返回 `Conflict`，旧owner保持不变；
- 观察到Publishing：返回Publishing并fail-closed，不等待、不覆盖；
- phase为任何未知值：返回CorruptState。

这里使用逐byte相等不是为了替代validator，而是在两份都已经通过validator后，进一步要求duplicate registration真的是同一个context trust root。仅比较generation或hash字段会把碰撞、字段遗漏或错误重建降级成隐式覆盖。

#### 10.34.4 独立HBG AICPU entry，不复用TRB callable ABI

A2/A3和A5的HBG AICPU DSO各自新增并导出：

```text
simpler_aicpu_l1_hbg_register_execution_slot
```

它与TRB的下列入口没有ABI关系：

```text
simpler_aicpu_l1_register_callable
  payload = L1RegisterCallableArgs
  内容 = callable id + device orchestration SO + callable-local kernel addresses
```

HBG新入口的payload就是完整sealed `HbgExecutionSlotRegistration`。这样144-byte context registration不会被TRB fixed callable parser误解释，未来variable graph blob也不会借TRB `L1AicpuInvocationArgs`尾部偷渡。

CANN task-argument地址不保证满足C++对象的自然对齐，所以entry不直接cast/dereference `void *arg`。它先 `memcpy` 到正确对齐的本地对象，再调用common registry publisher。这个规则与现有TRB L1 entry对 `L1RegisterCallableArgs`和 `L1AicpuInvocationArgs`的处理保持一致。

entry使用此前 `simpler_aicpu_init`在同一caller stream上发布的device id作为expected device，registration自己的device id不能自证。首次Published和逐byte相同的AlreadyRegistered都返回成功；SlotRejected/Conflict/Publishing/CorruptState均返回错误。

#### 10.34.5 host prepare发布顺序与canonical owner保护

DeviceRunner在静态资源prepare中的关键顺序现在是：

```text
prepare/freeze HBG working arenas
  -> upload outer Runtime
  -> upload device KernelArgs
  -> register generic AICore executor
  -> build/seal immutable host slot registration
  -> enqueue simpler_aicpu_init on caller stream
  -> enqueue HBG slot registration on the same caller stream
  -> 后续才允许进入runtime-specific callable/plan prepare
  -> record PrepareTail
```

registration必须排在init之后，因为AICPU侧要用init latched的device id校验；必须排在任何future HBG run之前，因为run只能acquire Ready trust root。两条task都使用用户传入的caller stream，没有新增private AICPU stream、event绕行、stream sync或device sync。

host owner是 `unique_ptr<const HbgExecutionSlotRegistration>`。虽然registration launch当前没有placeholder，`aclrtLaunchKernelWithHostArgs`的C接口仍接收可写pointer。为避免未来runtime patch行为或接口实现修改污染canonical trust root，DeviceRunner每次发布使用一个fresh writable栈副本：

```text
immutable canonical owner
  -> byte copy to fresh launch_args
  -> aclrtLaunchKernelWithHostArgs snapshots launch_args
  -> host call return后launch_args可销毁
```

只有CANN接受enqueue后，host才置 `l1_hbg_execution_slot_registration_enqueued_ = true`。该bool只表示“device/static owner可能已经或将要出现”，不表示AICPU task已经执行，更不表示device已消费后续graph package。prepare整体仍通过caller-stream FIFO与PrepareTail定义顺序。

#### 10.34.6 close失败必须在释放被引用window之前停止

此前close即使 `LoadAicpuOp::Finalize()`失败，仍会继续尝试释放device KernelArgs、outer Runtime和arena。对没有persistent device registration的旧路径，这至少不会留下一个仍可读这些地址的HBG registry；一旦本阶段把registration发布给resident DSO，该顺序就不再安全。

现在close执行：

1. 先进入粘性Closing，禁止新的prepare/launch；
2. 调用AICPU binary unload；
3. 若unload成功，清除“registration已入队”标记，device registry lifetime结束；
4. 若unload失败且HBG registration曾入队，立即返回首个错误；
5. 不释放device KernelArgs、outer Runtime、working arenas、host registration或context generation；
6. ChipWorker保留host runtime DSO与DeviceRunner owner，允许显式close retry；
7. 只有后续unload成功后才继续其他资源释放；
8. 全部资源与stream/event close成功后，最后清host registration与generation。

即使registration AICPU task最终在device上返回校验错误，host也无法从异步launch返回值同步获知；只要task已经入队，就按“可能已发布”保守保留资源。这比根据host launch返回推断device状态更安全，也不需要引入内部sync。

#### 10.34.7 symbol与capability隔离

HBG host runtime新增strong `runtime_l1_extra_aicpu_symbols`，当前只报告：

```text
simpler_aicpu_l1_hbg_register_execution_slot
```

构建后的符号核对：

```text
A2/A3 HBG onboard AICPU: 导出HBG registration entry
A5 HBG onboard AICPU:    导出HBG registration entry
A2/A3 HBG sim AICPU:     导出HBG registration entry
A5 HBG sim AICPU:        导出HBG registration entry
TRB AICPU:               不导出该HBG entry

A2/A3 HBG host runtime:  strong runtime_l1_extra_aicpu_symbols
A5 HBG host runtime:     strong runtime_l1_extra_aicpu_symbols
HBG l1_runtime_supported_impl: 仍为common weak/false
TRB l1_runtime_supported_impl: 仍为strong/true
```

因此这一阶段的代码能被四变体编译和符号检查，但公开 `simpler_l1_prepare_callable`仍会在support gate前拒绝HBG。不会出现“registration能发、run entry不存在，却被Python误认为supported”的半开放状态。

#### 10.34.8 无硬件反例与验证结果

新增 `test_hbg_execution_slot_registry`，先在header不存在时确认target按预期编译失败，再实现状态机。5组测试覆盖：

1. 有效registration只在完整校验后发布Ready，reader acquire后获得逐byte相同快照；
2. hash篡改和expected device mismatch都返回SlotRejected，phase仍Empty，owner bytes不变；
3. 同一registration重复发布返回AlreadyRegistered；
4. 不同slot generation且重新seal的另一份有效registration返回Conflict，旧owner不变；
5. Publishing、未知phase、空registry与null参数全部fail-closed，失败acquire不覆盖caller原output。

本阶段验证结果：

- runtime editable全构建通过，A2/A3和A5的onboard/sim、HBG/TRB全部重编；
- no-hardware C++全量：**91/91 passed**，其中74项标记 `no_hardware`；
- PyPTO L1与simpler ChipWorker Python回归：**51/51 passed**；
- A2/A3、A5的onboard与sim HBG AICPU DSO均导出新entry，TRB不导出；
- HBG host runtime的L1 symbol list为strong，support query仍是weak/false；
- changed-files pre-commit通过headers、English-only、clang-format、LLVM 18 clang-tidy与cpplint；
- `git diff --check`通过。

没有运行任何NPU task，没有触碰device 0。device 1仍没有新的空闲授权；因此本阶段只证明ABI、host/device静态协议和构建完整性，没有证明CANN实际执行registration task或跨capture持有graph args。

runtime阶段提交：

```text
4a8c3964 Add: 建立HBG执行槽设备注册所有权
```

#### 10.34.9 当前结论与下一阶段不可绕过的工作

现在可以新增宣称：

- HBG slot registration已经有独立于TRB callable ABI的AICPU entry；
- device侧有binary-lifetime immutable registry，完整校验后才release-publish；
- duplicate仅允许逐byte相同，冲突不覆盖；
- host在caller stream上按init -> registration顺序异步发布，不创建private AICPU stream、不sync；
- canonical host trust root不传给可写CANN API；
- binary unload失败时，close不会释放registry引用的device window；
- HBG capability仍关闭，L2/L3/TRB行为与symbol set保持隔离。

仍然不能宣称：

- 每个dynamic graph package已经交给CANN runtime-owned task args；
- ACLGraph captured node已经独立持有其graph package；
- HBG AICPU run entry已经acquire registry并restore working slot；
- restore的cache publish与peer barrier已在A2/A3、A5设备上成立；
- large HostArgs、placeholder patch、capture/replay lifetime已有板上证据；
- HBG L1或HBG ACLGraph已经supported。

下一阶段要实现的是独立HBG per-invocation run ABI，而不是继续扩registry。至少需要：

1. DeviceRunner为本次callable与本次tensor/scalar args构建task-local `HbgGraphPlan`；
2. 每次launch从immutable plan生成fresh writable HostArgs scratch和placeholder；
3. CANN runtime-owned snapshot成为该eager task/captured node的pristine source owner；
4. HBG AICPU run entry从task args取得variable blob，先acquire本阶段的device registration；
5. exactly-one leader校验slot、package、plan identity与restored invocation state，再完整恢复working SM/runtime arena；
6. cache publish和统一restore verdict完成后，peers才允许wire/classify/dispatch；
7. 任一失败进入共同epilogue，不能留下AICore等待或半恢复slot；
8. capability继续关闭，直到device 1完成同graph多次replay、A/B captured node交替、host owner释放压力和large-args边界实证。

这里尤其不能把 `l1_hbg_execution_slot_registration_enqueued_`当作task package lease。它只保护context registration与其引用window；每个graph blob的lifetime仍必须由CANN task/graph或显式package lease独立承担。

### 10.35 HBG任务级graph package正式接入L1 launch主线

#### 10.35.1 本阶段回答的核心问题

用户补充的第二阶段原则是：每次动态build出来并H2D到device的graph，本质上是本次HBG task的tiling-like入参。AscendC的tiling data由CANN runtime随launch task管理；HBG graph bytes也应尽量获得同样的task/captured-node lifetime，而不是由PyPTO维护一块会被下一次host调用覆盖的“current graph device buffer”。

这个类比成立，但不能停在“把一个graph pointer传给kernel”这一层。HBG与普通只读tiling data有一个决定性的差异：

- tiling bytes通常只读；
- 当前HBG SM/runtime-arena image会被scheduler执行原地修改；
- `runtime_destroy`还会清理queue pointer、mailbox、runtime attachment等字段；
- 因此task-owned graph bytes必须是immutable pristine source，不能直接作为working state执行；
- 每次eager execution和每次ACLGraph replay都必须先把这份source恢复到mutable execution slot，再允许scheduler工作。

本阶段把下面这条所有权链真正接进了源码：

```text
一次Python/native L1 host call
  -> 当前callable + 当前tensor/scalar snapshot
  -> immutable HbgGraphPlan
  -> fresh writable HbgSerializedLaunchBlob
  -> aclrtLaunchKernelWithHostArgs + placeholder
  -> CANN runtime-owned task args（候选的task/node lifetime source）
  -> 每次task execution/replay由AICPU leader restore
  -> context-owned mutable HbgExecutionSlot
  -> peer classify/dispatch + hidden AICore execution
```

这里没有把workspace改为外部入参。GM heap、working SM、working runtime arena、outer Runtime、KernelArgs、handshake和workspace仍由PyPTO context在prepare阶段分配、冻结并持有。由于v1占用全部AICore并禁止并发，多个task/captured node可以顺序共享这一份mutable slot；它们不能共享或覆盖彼此的immutable source。

#### 10.35.2 五类内存在当前实现中的落点

本阶段之后，五层对象不再只是设计名词，而是可以对应到具体代码和owner：

| 层 | 当前对象 | owner与生命周期 | 当前可变性 |
| --- | --- | --- | --- |
| canonical host plan | `std::unique_ptr<const HbgGraphPlan>` | 仅覆盖本次host build到launch序列enqueue；plan已deep-copy，不依赖builder临时arena | immutable |
| writable host launch scratch | `std::vector<uint8_t> hbg_launch_blob` | 本次 `launch_l1_callable` 栈帧持有到WithHostArgs API返回 | placeholder可原地patch |
| runtime-owned task source | CANN从完整HostArgs复制出的device args blob | eager应到task消费结束；capture应到captured node不再replay，仍待device 1实证 | AICPU只读 |
| mutable execution slot | frozen GM SM + runtime arena + GM heap等 | L1 context持有到graph销毁、外部quiescence并成功close | 每次execution会被修改，下一次必须restore |
| lifetime roots | slot/callable registrations、binary、KernelArgs、external tensor owner、stream/event | PyPTO context与调用方分别持有 | registration immutable，working state mutable |

这张表也说明了为什么CANN拥有HostArgs bytes仍不等于CANN拥有整个算子状态：blob中的external tensor地址、working slot地址和child binary地址只是pointer value。CANN复制该数值，不会替PyPTO保活pointee。调用方仍必须持有graph-bound tensors；PyPTO context仍必须持有working slot、workspace、binary、Runtime/KernelArgs和hidden stream/events。

#### 10.35.3 本阶段提交和范围

runtime提交：

```text
18b1fde9 Add: 建立HBG调用身份注册协议
74d0ff65 Add: 接通HBG任务级图快照恢复链路
```

`18b1fde9`完成per-callable trust root；`74d0ff65`完成host launch、独立AICPU run entry和leader/peer restore的整条静态链。两次提交都没有翻转HBG capability，没有改Python公开支持面，也没有声称ACLGraph已上板通过。

### 10.36 callable-global身份与callable-local函数表

#### 10.36.1 为什么不能继续使用context-global `func_id -> addr`

每个独立编译的 `@pl.program` 都可能从 `func_id=0` 开始编号。若HBG host builder从context-wide `Runtime::func_id_to_addr_`直接取表，第二个callable会与第一个callable发生数值ID冲突；即使host侧临时覆盖全局表，已经capture的旧node也可能在replay时看到最后一次build留下的地址。

最终协议明确区分：

- `callable_id`：context-global，append-only，不允许改指另一份callable identity；
- `func_id`：callable-local，只要求在同一callable内唯一、有效且地址非零；
- `function_binding_hash`：对完整固定长度1024项表计算，零项也参与identity；
- restored runtime arena中的 `HbgPrebuiltInvocationState`：task-owned，携带本次完整表、实际task数和同一hash；
- outer `Runtime::func_id_to_addr_`：只可作为旧L2暂存/历史状态，不能作为L1 task的execution source。

新增 `hbg_callable_function_binding.h` 的transactional builder每次都先创建全零1024项candidate，再写入本callable的 `(func_id, device_addr)`。它拒绝：

1. output/table/hash空指针；
2. table capacity不是精确1024；
3. func ID负数或越界；
4. device地址为0；
5. 同一callable内重复func ID；
6. 最终hash为0。

只有全部校验成功才commit output table和hash。这样从callable A切到B时，A独有的func ID一定回到0；两个callable都使用 `func_id=0` 且地址不同是合法场景，而不是全局冲突。

#### 10.36.2 callable registration的device trust root

host-orchestration callable现在持有独立 `HbgCallableRegistration`，其核心字段包括：

- `callable_id`；
- callable/content hash；
- function binding hash；
- tensor count与scalar count；
- registration自身的magic/version/size/hash。

prepare在host所有字段稳定后生成immutable owner，再通过独立 `simpler_aicpu_l1_hbg_register_callable` entry发布到AICPU binary-lifetime registry。device registry只允许：

- Empty到Publishing再到Ready的一次性发布；
- 已Ready时逐byte相同的重复注册返回幂等成功；
- 同一ID内容冲突、未知phase、中间态或坏hash全部fail-closed；
- run只能按本次header中的 `callable_id` acquire已经Ready的registration。

slot registration回答“恢复到哪一组persistent device window”；callable registration回答“哪一个compiled callable、签名和函数绑定有权使用该slot”。两者都不能代替per-task package owner。

#### 10.36.3 invocation identity不依赖launch次数规格

`HbgInvocationIdentity`当前包含：

```text
callable_hash
argument_snapshot_hash
function_binding_hash
tensor_count
scalar_count
host_total_tasks
callable_id
```

三个hash分别描述三个不同ownership domain：compiled callable、会固化进graph image的本次参数语义、实际child function binding。`argument_snapshot_hash`现在必须非零；tensor/scalar count必须同时与native args、callable registration和plan结果一致；host builder负责在构图后把真实 `host_total_tasks`填回plan identity。

`plan_generation`是L1 context内独立单调 `uint64_t`。0保留为invalid；溢出后停止接受新plan。它不使用、不查询也不推断CANN kernel-launch内部可能存在的约2048等实现规格。generation是PyPTO task identity，不是slot recycle计数或CANN参数池索引。

### 10.37 每次host launch生成runtime-owned候选package

#### 10.37.1 DeviceRunner中的HBG分支

`DeviceRunnerBase::launch_l1_callable`完成通用device、phase、stream、tensor/scalar count与静态layout校验后，根据callable是否持有host orchestration entry选择TRB或HBG：

- TRB继续构造固定 `L1AicpuInvocationArgs`；
- HBG要求slot registration与本callable registration都已生成且enqueue；
- HBG构造callable-local完整函数表和hash；
- 对本次 `ChipStorageTaskArgs`计算argument snapshot hash；
- 生成初始identity与非零plan generation；
- 调用架构strong `build_l1_hbg_graph_plan_impl`；
- 交叉检查builder返回的callable、args、binding和真实task数；
- 校验serialized size不超过prepare冻结的package capacity；
- 从immutable plan生成fresh writable blob；
- 生成只指向本blob inline payload的placeholder；
- 在统一fork/join序列的AICPU节点调用 `LaunchWithMutableHostArgs`。

`hbg_plan`和 `hbg_launch_blob`都活到完整enqueue函数返回。WithHostArgs返回后PyPTO不保留该host scratch；当前设计依赖CANN的task-args snapshot成为device source。正因为这一点尚未板上证明，capability仍关闭。

#### 10.37.2 strong host builder的输入与事务边界

A2/A3、A5的 `build_l1_hbg_graph_plan_impl` 现在额外接收显式的callable-local函数表和精确count，不再读取context-global函数表。它按以下顺序工作：

1. 拒绝null runtime/API/args/host entry/binding/identity/function table/out owner与零generation；
2. 校验binding精确命中prepare冻结的GM heap、SM、runtime arena和runtime offset；
3. 校验tensor/scalar count、非零identity字段和初始 `host_total_tasks == 0`；
4. 重新计算ring/heap layout，要求host image大小与frozen slot capacity精确一致；
5. 只commit临时host `DeviceArena`，不commit或改写device working slot；
6. 建立host runtime/host orchestration binding并运行host builder；
7. 捕获host orchestration fatal，失败不发布半成品plan；
8. 将显式callable-local函数表和真实task数写入host runtime-arena image；
9. 重算/校验 `HbgPrebuiltInvocationState`的function hash与header identity；
10. 把完整SM image和完整runtime-arena image作为required immutable regions deep-copy进 `HbgGraphPlan`；
11. 只有所有步骤成功才transactionally替换out owner。

builder不执行H2D，不调用stream/device sync，也不读取capture状态。它生成的是destination-bound pristine bytes：image内已经带有working SM/runtime arena/GM heap/external tensor/binary的最终device地址，因此只能恢复到registration指明的同一slot，不能复制到任意新slot后直接执行。

#### 10.37.3 canonical plan、writable scratch和placeholder

`HbgGraphPlan`私有持有canonical `HostUnpatched` bytes。每次 `serialize()`都deep-copy为新vector，避免：

- CANN placeholder原地patch污染plan cache；
- 第二次host launch覆盖第一次尚未被device消费的bytes；
- graph A/B交替capture时共享同一writable staging；
- caller释放builder临时vector后plan引用悬空。

当前variable blob布局是：

```text
HbgLaunchBlobHeader
HbgLaunchRegion[region_count]
alignment padding through header_size
pristine SM bytes
per-region alignment padding
pristine runtime-arena bytes
future optional initializer spans
```

placeholder把header中的 `inline_payload_addr`改写为runtime device args base加 `header_size`。region的source offset都相对这个地址。capacity计算已经修正为使用与serializer完全相同的padding规则，而不是简单相加payload size；反例固定覆盖 `240 + align_up(15, 8) + 9 = 265`，防止slot registration低估最大package。

当前header ABI是major保持、minor 2；固定类型仍使用8-byte alignment，AICPU不假设CANN args allocation满足更强C++对齐。`argsSize`和placeholder offset的32位carrier在host桥接前做lossless/overflow/bounds校验，但32位carrier本身不能被解释成已验证的产品最大payload规格。

### 10.38 AICPU从task package恢复共享working slot

#### 10.38.1 独立run entry的fixed-prefix校验

A2/A3与A5 HBG AICPU DSO都导出：

```text
simpler_aicpu_l1_hbg_register_execution_slot
simpler_aicpu_l1_hbg_register_callable
simpler_aicpu_l1_hbg_exec
```

`simpler_aicpu_l1_hbg_exec`不复用TRB fixed invocation parser。它先：

1. acquire当前device对应的slot registration；
2. 用 `memcpy`把可能未对齐的HostArgs fixed header复制到对齐local；
3. 校验header magic、ABI、header/total size、region count、非零generation和identity；
4. 以header `callable_id` acquire callable registration；
5. 校验package size未超过slot capacity；
6. 校验placeholder patched地址精确等于 `blob_base + header_size`；
7. 校验header binding与registered slot逐项相等；
8. 校验callable id/hash/function hash/tensor count/scalar count与registration相等；
9. 校验registered device KernelArgs仍指向registered outer Runtime；
10. 通过平台bridge进入现有AICPU affinity/executor。

这一阶段只扫fixed prefix，目的是在进入多线程平台路径前选择可信Runtime/KernelArgs。完整descriptor table、payload hash、region overlap和restore span由唯一leader校验，避免每个AICPU peer重复扫描大package。

#### 10.38.2 exactly-one leader restore和peer barrier

当前HBG AICPU executor把最后一个AICPU worker作为boot leader。L1 invocation存在时，leader在wire、attach、classify和dispatch之前：

1. 调用full `restore_hbg_launch_blob`；
2. 以slot registration而不是blob自身作为destination trust root；
3. 重新校验header、全部region descriptor、source/destination window、hash、capacity与identity；
4. 将pristine SM region复制到registered working SM；
5. 将pristine runtime-arena region复制到registered working arena；
6. 对每个destination span执行架构cache publish；
7. 只有全部copy/publish成功才形成 `HbgRestoreCommit`；
8. 定位restored `PTO2Runtime`，校验prebuilt invocation magic/version/count/task数和完整函数表hash；
9. 比较restored state的task数/hash与runtime-owned header identity；
10. 成功后wire arena pointer、attach populated SM并初始化device-only字段；
11. 把统一 `hbg_restore_error_`以release语义发布，再放行classify-ready。

非leader peer等待classify-ready后，对完整working SM/runtime arena做cache invalidate，再以acquire读取统一restore verdict。任何restore error都会阻止所有peer执行classify/dispatch；不能出现leader失败但某个peer继续消费半恢复graph的情况。

#### 10.38.3 为什么每次replay都必须走这条restore

一次HBG执行会修改至少以下状态：

- ready queue头尾与slot；
- wake list与fanin/fanout完成状态；
- task state与completion flags；
- completed subtasks与watermark；
- scheduler queue pointer；
- runtime mailbox、SM handle和attach字段；
- `runtime_destroy`清理的若干runtime pointer。

因此第一次capture-time execution、第一次replay和第N次replay不能共享“执行后的working bytes”。每次CANN重新执行同一个AICPU captured node时，entry仍会收到该node自己的RuntimeOwnedHbgPayload，并再次完整restore。无硬件UT中的两类反例已经锁定这一点：

- 同一个package在第一次restore后把working state全部poison，再restore仍回到原pristine bytes；
- A/B两个package交替恢复到一个slot，A、B、A每次都得到各自image与generation，不落入“最后一次host build覆盖所有node”。

这正是“graph package像tiling参数一样随task管理”与“graph working state必须单独恢复”两条原则的结合，而不是二选一。

#### 10.38.4 单算子stream边界没有为HBG破例

HBG launch仍使用与TRB相同的外层顺序：

```text
caller stream:
  optional wait PrepareTail（仅第一次完整成功launch）
  async memset handshake
  record Start
  launch HBG AICPU WithHostArgs
  wait AicoreDone
  record SerialTail

hidden AICore stream:
  wait Start
  launch prepared AICore executor handle
  record AicoreDone
```

HBG restore是caller-stream AICPU task内部的第一阶段；它没有提前于本算子入口启动，也不会在capture外预跑orchestrator。代码中没有：

- private AICPU run stream；
- `rtStreamAddToModel`；
- capture/model query；
- capture-only early launch；
- launch-time device allocation/free；
- launch-time binary registration；
- PyPTO内部stream/device synchronize或reset。

host build本身在host launch调用中执行一次；ACLGraph replay不会再次进入Python/PyPTO host builder。replay执行的是已capture的AICPU task和其runtime-owned参数，因此每个captured node必须已经持有自己完整的pristine source。

### 10.39 HBG host build禁止读取device tensor contents

#### 10.39.1 为什么“read-only host mapping”仍然越过L1边界

`2873feae`阶段曾保守地允许：平台若同时提供register/unregister，就为external device tensor建立临时read-only host view；写入统一拒绝。继续把这条路径放进真实异步launch后，发现它仍然不正确。

典型时序是：

```text
caller stream:
  predecessor writes control/input tensor
  PyPTO L1 op entry
     host thread executes dynamic HBG build
     device stream尚未必执行完predecessor
```

L1禁止内部stream sync；PyPTO也不允许为了host builder查询capture或偷建等待流。因此即便某平台能把device allocation映射到host地址，host也没有证据证明前序device write已经对CPU可见。读取该地址会越过单算子entry的stream happens-before。A2/A3能否map、A5能否map不是关键；关键是映射不提供caller-stream completion。

#### 10.39.2 当前实现的fail-closed规则

`74d0ff65`在A2/A3和A5 strong HBG builder中统一采用：

- external tensor device地址原样进入descriptor；
- 不为input/output重新分配storage；
- 不做H2D/D2H；
- 不调用tensor register/unregister；
- 不建立任何host access region；
- 进入no-registration read-only access模式，因此read和write都会因region不存在而失败；
- host orchestration若触发data access fatal，builder返回失败，不生成或launch半成品plan；
- 只依赖shape、dtype、stride、device address、host scalar与拓扑的program可继续build。

L2 HBG原有staging/mapping路径不改。L2掌控资源并能在自己的whole-run生命周期里完成数据搬运；L1不能因为复用同一host orchestration实现就继承这套同步/所有权假设。

#### 10.39.3 长期支持device-produced control data需要新协议

当前运行时fail-closed是安全底线，但长期还应把unsupported要求前移到compile/prepare：final transformed orchestration metadata需要标记是否调用 `get_tensor_data/set_tensor_data`，或更直接记录 `requires_device_tensor_value`。这样在任何host graph build和plan allocation前即可拒绝不支持callable。

如果未来HBG必须读取前序device task产生的control/tiling数据，需要单独设计异步协议，例如让device侧builder或受caller stream排序的专用task消费control data。不能采用：

- host HAL map后直接读；
- capture中暗中D2H；
- PyPTO内部stream/device sync；
- private AICPU stream提前构图；
- 用“capture时scalar稳定”替代真实device-value ordering。

显式CPU control tensor将来若开放，也必须有独立参数类型、snapshot语义和graph lifetime契约，不能混在普通NPU tensor输入里猜测。

### 10.40 当前验证、明确关闭的capability与剩余硬门槛

#### 10.40.1 无硬件验证结果

`74d0ff65`完成后执行了以下验证：

- runtime editable全量构建通过；
- A2/A3与A5、onboard与sim、TRB与HBG相关host/AICPU/AICore目标均完成编译/链接；
- no-hardware CTest：**95/95 passed**；
- PyPTO L1与simpler ChipWorker Python回归：**51/51 passed**；
- HBG定向测试：**7/7 passed**，覆盖launch blob、execution slot、slot/callable registry、callable-local function binding、fixed AICPU invocation、prebuilt invocation和tensor-access边界；
- `git diff --check`与相关格式检查通过。

新增或加强的关键反例包括：

1. 两个callable均使用 `func_id=0`但地址不同，完整表和hash互不污染；
2. table输出只在全部binding合法后commit；
3. bad callable ID/hash/count、bad slot binding/capacity在restore前拒绝；
4. `argument_snapshot_hash == 0`拒绝；
5. serializer padding计入package capacity；
6. fixed prefix从未对齐HostArgs地址先copy再typed access；
7. same package重复restore；
8. A/B package交替restore；
9. copy/cache publish中途失败不发布commit；
10. restored function table/task count/hash与header identity不一致时拒绝；
11. HBG L1 host builder无device tensor mapping，data read/write失败而metadata-only路径可构图；
12. host orchestration entry bundle在失败与close路径有显式destructor，不依赖仅 `dlclose`回收C++ owner。

这些结果证明源码协议、ABI、transaction边界和双架构构建自洽，不证明CANN runtime/ACLGraph设备行为。

#### 10.40.2 device 1本轮没有运行NPU任务

本轮只读检查发现当前逻辑device 1对应 `/dev/davinci15`，`npu-smi`观察到约2873 MiB HBM占用、AICore 100%、AIVector 90%。仓库约定空闲设备必须满足HBM为0；即使进程列表未显示普通用户进程，也不能据此抢占。

因此本轮：

- 没有向device 1提交ACL/NPU task；
- 没有触碰device 0；
- 没有用device 0作为fallback；
- 没有把collect-only、host lowering或sim结果写成onboard结果；
- `task-submit`当前不在PATH，仓库mandatory arch precheck还硬编码card 0，而本机card ID为7；正式上板前还要通过合规调度并修正/绕开错误预检入口。

已经新增但未执行的ST是 `tests/st/runtime/l1/test_l1_aclgraph.py`。它使用显式device参数、普通warmup stream和独立capture stream，覆盖：

1. `context.prepare()`与eager warmup数值；
2. warmup/capture raw stream确实不同；
3. caller外部synchronize后capture；
4. graph内 `torch.add(out=) -> L1(out=) -> torch.mul(out=)`顺序；
5. 三组输入连续replay并逐次验数；
6. teardown严格执行外部device sync、`graph.reset()`、`context.close()`；
7. `pypto_init`失败时接管 `cleanup_context`并保证close retry owner可达。

该ST已通过A2/A3与A5纯host lowering、collect-only、ruff等静态检查，但**尚未在任何NPU上执行**。

#### 10.40.3 HBG capability被显式strong关闭

此前HBG依赖common weak unsupported hook。随着HBG源码路径逐步接通，只依赖weak default容易让未来链接或symbol变动意外翻转capability。当前A2/A3和A5 HBG `runtime_maker.cpp`都显式定义：

```cpp
extern "C" int l1_runtime_supported_impl(void) { return 0; }
```

对应注释列出未满足门槛：

- large variable HostArgs和placeholder真实行为；
- hidden-stream event-only capture/replay；
- runtime-owned source在同graph反复replay中的lifetime；
- repeated pristine restore和跨cache可见性；
- HBG AICPU/AICore no-reset错误收尾；
- Python高层当前仍只允许TRB。

所以当前正确表述是：

```text
TRB L1 native/Python path: 源码实现完成，等待device 1 ACLGraph Phase-0实证
HBG L1 native path: 源码链已接通但capability显式false
HBG Python path: 未开放，仍fail-closed
HBG L1 + ACLGraph: unsupported
```

#### 10.40.4 HBG开放前仍必须解决的P0/P1

第一组是CANN task-args/ACLGraph事实：

1. `aclrtLaunchKernelWithHostArgs`对实际HBG大小是否完整deep-copy，不能只测小header；
2. placeholder是否在A2/A3与A5都把pointer patch到runtime-owned inline payload；
3. API返回后立即poison/free/reuse host scratch，device仍读取原bytes；
4. capture/instantiate后，graph多次replay期间runtime-owned source不被参数池回收或串包；
5. device args base至少满足当前8-byte parser要求；若不能，所有variable table也必须byte-copy解析；
6. 同graph至少连续replay两次、A/B不同package交替replay，证明restore每次执行；
7. hidden AICore event-only branch无需model attach也确实进入captured graph。

第二组是HBG executor no-reset错误收尾：

- HBG `AicpuExecutor::init()`仍有若干早退路径，需要证明所有有效AICPU participant exactly-once进入共同completion/shutdown协议；
- HBG scheduler对invalid physical core ID、范围内但register address为0等异常仍没有TRB已经实现的WAIT/CANCEL同等级闭环；
- `emergency_shutdown`只能关闭已经打开register window的core，不能把“未打开window但仍在AICore等待”的情况交给L2时代的host device reset；
- restore failure、handshake failure、scheduler failure都必须保证hidden AICore kernel结束、caller tail最终可完成，不能让borrowed L1 context依赖reset恢复；
- 完全不report的硬件core属于更外层driver/op-timeout/fault-containment边界，但必须在support矩阵中明确，不得静默挂死。

第三组是graph state完整性：

- 当前package完整恢复SM与runtime arena；
- GM heap暂时只作为冻结capacity/address root；
- 如果某类HBG program在GM heap中存在每次execution必须重置的initializer bytes，必须增加明确的 `GmHeapInitializer` region manifest；
- 不能每次把整个workspace清零，因为workspace可能包含无需初始化或具有不同语义的scratch；
- 也不能假设heap永远无初始语义，必须由builder/runtime结构审计和poison test证明。

第四组是上层生命周期与无并发契约：

- graph不自动持有PyPTO context；调用方必须强引用context、graph-bound tensors和external/custom storage owner；
- default torch_npu allocator的 `recordStream`只覆盖实际stream use，不替代graph owner；
- external/from-blob/custom storage owner必须活到graph销毁且最后一次device use真正完成；
- teardown必须是外部quiescence、`graph.reset()`、`context.close()`；
- v1禁止同一context的并发replay。host mutex只能串行host enqueue，不能侦测未来两个graph的并发device replay；长期若要支持并发，需要per-node execution slot或device execution gate。

#### 10.40.5 HBG graph package的长期内存管理原则

结合用户关于AscendC tiling data的补充，长期策略按以下优先级执行：

1. **首选：Runtime-owned inline package。** 每个WithHostArgs task/captured node拥有自己的immutable pristine source；PyPTO不实现看不见completion的device task-args pool。
2. **共享的只有working slot。** v1无并发时，context可以只有一份mutable SM/runtime arena/GM heap/workspace；每次task必须restore，不能把共享slot误当作package owner。
3. **binary继续context-lifetime pin。** child/incore binary地址进入task-owned函数表，但allocation本身由context append-only持有；本阶段不做binary device内存复用。
4. **若Runtime-owned路径板上失败，才评估external immutable source。** 每个captured node必须有独立device source或显式graph lease，不能退化为一份可覆盖buffer。
5. **没有graph retain/release hook时，fallback只能append-only pin到context close。** 必须有memory accounting和limit；宁可明确内存增长，也不能在不知道graph是否未来replay时回收。
6. **任何source方案都不取消per-replay restore。** source lifetime和working-state reset是两个正交问题。
7. **不依赖固定launch数量。** 2048、args pool环大小或任何源码内部常量都不能作为回收条件；只有CANN task/graph owner或显式外部lifetime协议可以证明回收安全。

如果未来HBG通过 `host_build_graph`获得性能优化，也仍必须保持一个算子边界：host build可以把更大范围的调度变成一个明确graph operator，但不能在当前L1 op到达caller stream之前，借private stream/model attach提前启动orchestrator。性能方案必须改变公开抽象，而不是悄悄越过已有抽象。

#### 10.40.6 当前阶段结论

现在可以准确新增的结论是：

- HBG每次dynamic build的graph已经在代码中被建模为task-local、tiling-like immutable package；
- callable identity、argument snapshot、callable-local函数表和真实task数都进入该package的identity或pristine arena；
- fresh writable host scratch已经接到独立HBG WithHostArgs run entry；
- AICPU run entry会acquire slot/callable trust roots，unique leader每次完整restore，peer只在统一success verdict后dispatch；
- HBG restore仍处于同一个caller-stream AICPU op节点内，外层AICPU/AICore fork/join没有破坏单算子边界；
- HBG L1不再读取或写入device tensor contents；
- A2/A3和A5保持同构，TRB fixed invocation和L2/L3旧路径没有被泛化成variable package；
- HBG capability仍显式false，Python仍拒绝HBG。

现在仍然不能宣称：

- CANN已经以captured-node lifetime可靠持有真实大小HBG package；
- HBG L1在device 1 eager执行成功；
- HBG ACLGraph capture/replay成功；
- A2/A3或A5真实cache/order已验证；
- HBG no-reset错误路径已完整闭环；
- GM heap initializer manifest对所有program完整；
- 同一context并发graph replay得到支持。

本阶段的正确定位是：**源码已经具备进入HBG device capability probe的结构，不再缺“task package如何传到device并在replay前restore”的主链；但产品capability必须保持关闭，直到device 1和错误路径证据完成。**

### 10.41 HBG L1 no-reset收口：把“已经排队的hidden AICore”纳入每条失败路径

#### 10.41.1 继续审查10.40后暴露出的真正问题

10.40已经把HBG dynamic graph建模为task-local immutable package，也把每次执行前的restore接到AICPU run entry。但L1外层fork/join还有一个比graph package本身更基础的硬约束：Host一旦执行到operator launch，就会固定排入以下device工作：

```text
caller stream:
  clear launch state
  record Start
  launch HBG AICPU task
  wait AicoreDone

hidden AICore stream:
  wait Start
  launch persistent AICore executor
  record AicoreDone
```

因此AICPU在任何校验点返回错误，并不等于本算子已经安全失败。只要hidden AICore已经排队，它就必须能够退出并最终record `AicoreDone`；否则caller stream会永远等在fork/join尾部。L2时代可以在外层失败后reset device，但L1借用外部资源，明确不允许把reset或内部sync当作错误收尾。

最终审查把早退分成两类：

1. **已经进入scheduler generation之后的失败。** 这类失败可以依靠N-way completion gate、统一shutdown、最后参与者cleanup以及已经打开register window的EXIT/ACK协议收口。
2. **scheduler generation建立之前的失败。** 例如slot/callable/blob/ABI/KernelArgs/platform bridge/affinity校验失败；这时没有generation owner，也可能没有任何per-core handshake owner，不能强行跳入后半段barrier。

第二类正是本阶段新增独立prelaunch cancellation协议的原因。它不是“再加一个错误码”，而是确保任何已经排队的AICore kernel都有一个不依赖scheduler generation的退出通道。

#### 10.41.2 为什么不能只复用每个core的Handshake

每个AICore会先把physical core id、core type和 `aicore_done` 作为一整条cache line report出去，然后等待AICPU打开register window。若AICPU在AICore report之前就往同一条Handshake写CANCEL，稍后AICore的whole-cache-line `CACHELINE_OUT` 可能把这个早期CANCEL覆盖掉。

所以当前协议保留两级取消面：

- `Runtime::l1_launch_control`：独立64-byte cache line，处理generation建立前的整次invocation拒绝；
- `Handshake::aicpu_ready = AICORE_PRE_WINDOW_CANCEL`：处理AICPU已经看到本core report，但physical id越界、对应register address为0等无法打开window的per-core拒绝。

Host在caller stream上用一次async memset连续清零：

```text
[ HbgL1LaunchControl: 64 bytes ][ active Handshake array ]
```

这里要求control恰好位于active handshakes之前，Host在prepare时根据HBG Runtime的strong query取得offset，并在每次launch前重新校验两段地址连续、大小不溢出。它没有引入额外stream、event、sync或capture query；clear仍是本算子caller-stream序列的一部分。

AICPU写整次invocation CANCEL时使用同值atomic store并对独立control cache line执行flush。多个AICPU线程同时发现同一错误时可以重复写同一个值，不需要选举一个可能根本进不了generation的leader。AICore在等待window的循环中低频invalidate该control line；只有错误路径命中CANCEL，正常路径仍以register window打开为唯一放行信号，避免恢复历史上被移除的AICPU→AICore正常路径round trip。

#### 10.41.3 generation内部的exactly-once完成协议

本阶段同时把A2/A3和A5 HBG `AicpuExecutor`的错误收尾改为同构的两阶段参与者协议：

```text
每个有效AICPU participant：
  init / scheduler verdict
  -> 无论共享run_error是否已经置位，都进入共同run epilogue
  -> exactly once arrive
  -> 唯一last-arriver执行runtime finalize并发布final verdict
  -> 每个participant snapshot最终error/runtime status
  -> exactly once depart
  -> 唯一last-depart执行deinit/reset generation-local host state
```

之所以arrival之后还需要departure，是因为旧实现可能由某个线程先deinit并清 `run_error_` 或invalid runtime cache，另一个已经arrive但尚未读取最终状态的线程随后看到“成功”或访问已清理状态。现在所有participant先完成final snapshot，再允许last-depart清代际状态。

decoupled模式也不再允许orchestrator在scheduler handshake最终裁决之前进入 `p_func`。所有scheduler完成handshake/assign后汇合，唯一leader发布 `init_done/init_failed`；orchestrator可以与前半段配置、arena和SM初始化重叠，但必须在bind和调用host-built graph入口之前等待最终裁决。这样某个scheduler晚到的handshake失败不会让orchestrator继续向无人消费的ring提交任务。

这部分修复不改变正常graph task调度语义；它只把旧L2路径中隐含依赖device reset的异常清理，改成borrowed L1可以证明的generation-local收尾。

#### 10.41.4 physical core报告异常的no-reset边界

A2/A3与A5 scheduler现在把以下两种报告统一视为invalid mapping：

- `physical_core_id >= register_address_count`；
- id虽然在范围内，但Host按PG/topology构造的 `regs[physical_core_id] == 0`。

对每个已经report的invalid core，AICPU先发布per-core pre-window CANCEL并flush，再汇总handshake失败；绝不调用 `platform_init_aicore_regs(0)`，也不对未知core猜测、mask或clamp物理id。AICore低频invalidate自己的Handshake，看到CANCEL后在访问任何SPR前直接退出。

A5还补了更早的PMU入口保护：PMU enabled时，kernel entry在索引per-core register table之前先用Host实际分配的精确table长度检查 `get_physical_core_id()`；越界时发布0 PMU base，继续交给正常report→CANCEL协议裁决，避免在取消协议生效前已经OOB。

这条闭环只覆盖“core已经进入kernel并完成report，但report的physical id不可用”。如果某个硬件core完全不进入、不report，AICPU没有安全证据判断哪个Handshake可以取消；在禁止内部reset/sync的L1算子协议内不能伪造恢复。这仍属于CANN op timeout、driver fault containment或外部context/device recovery边界，不能在文档中写成所有硬件失联都由PyPTO恢复。

#### 10.41.5 最终审查发现的三个条件式P1

独立control line与generation completion都实现后，最终逐路径审查仍发现三个“只有异常输入才触发，但一旦触发hidden AICore可能不退出”的P1：

1. **execution-slot registry acquire失败。** HBG outer AICPU entry在 `NotReady/Publishing/CorruptState/device mismatch` 时还拿不到完整slot，原实现直接return，也没有可信control地址。
2. **affinity正数但越界。** `allowed_count`或 `launch_count`大于固定 `MAX_GATE_THREADS` 时，公共gate返回false；平台wrapper把false当成普通dropped thread，可能所有线程都返回0且无人写CANCEL。
3. **device KernelArgs中的 `runtime_args` 与registered outer Runtime失配。** AICPU能根据可信slot向正确Runtime写CANCEL，但hidden AICore原来仍从同一个错误KernelArgs读取另一个Runtime地址；两边会轮询不同control line，错误地址不可读时还可能先产生device fault。

这三项不能靠“通常Host不会传错”忽略。C ABI、device memory corruption和故障注入都可以到达这些分支，而当前工作的目标正是让L1错误路径不借reset收尾。

#### 10.41.6 registry失败时的prepare-time独立信任根

不能在slot registry失败后回头信任本次variable HostArgs blob里的pointer。那正是尚未通过identity/ABI/slot校验的输入；拿其中地址做device write会把fail-closed校验变成任意地址写。

当前做法是在 `simpler_aicpu_init` 的 `InitArgs` 末尾增加：

```text
hbg_l1_prelaunch_control_addr
```

Host在prepare期间已经完成outer Runtime/device KernelArgs分配并seal immutable execution-slot registration，此时从registration解析control device地址，随init task发布到resident AICPU SO的device-config全局。顺序是：

```text
prepare/freeze persistent windows
  -> seal host registration
  -> enqueue simpler_aicpu_init(control address)
  -> enqueue full slot registration
  -> enqueue callable registration
  -> record PrepareTail
  -> future invocation task
```

所有任务仍在用户caller stream上；没有内部sync。正常路径继续使用完整slot registration。只有full registry acquire本身失败时，HBG outer entry才读取prepare-time resident control地址并发布CANCEL。common resolver `hbg_l1_launch_control_or_fallback`明确表达优先级：valid registration优先，registration缺失或字段校验失败才使用init-latched fallback。

TRB L1和L2/L3构造的 `InitArgs` 该字段保持0；它们不会解析或使用HBG fallback。因此这是同一内部InitArgs构建版本上的零值扩展，不会把HBG control语义注入旧执行模式。

这里仍有一个不可伪装成可恢复的前提：如果连prepare-time init task都没有执行成功，则resident SO根本没有可靠信任根，caller stream本身也已处于初始化失败状态。PyPTO不会从后续坏task猜一个地址继续运行；init/prepare失败必须由外部同步观察并走context close/recovery。

#### 10.41.7 affinity必须在进入barrier之前区分“invalid”和“dropped”

公共平台层新增纯函数：

```text
platform_aicpu_affinity_config_valid(allowed_count, total_launched)
```

合法条件不是只有两者大于0，还包括：

- `allowed_count <= MAX_GATE_THREADS`；
- `total_launched <= MAX_GATE_THREADS`；
- `allowed_count <= total_launched`，否则至少一个scheduler/orchestrator角色永远没有participant。

A2/A3与A5 platform wrapper在任何线程进入filter gate barrier前执行同一校验。invalid config统一记录错误、发布HBG prelaunch CANCEL并返回失败；只有config合法后gate返回false，才解释为本线程被正常淘汰。公共gate内部也复用同一predicate，避免caller校验和数组/barrier实现漂移。

这项区分很重要：`false`不是天然等于error。正常runtime可能有CANN over-subscription，部分线程确实应该drop；只有全局shape非法时才必须让本算子失败并释放hidden AICore。

#### 10.41.8 AICore Runtime不能由正在被校验的KernelArgs决定

第三个P1的本质是一个split-brain：

```text
AICPU trust root: slot.outer_runtime_base
AICore old trust root: device KernelArgs.runtime_args
```

当两者失配时，即使AICPU正确写CANCEL，也无法证明AICore在读同一地址。把第二份pointer再塞回KernelArgs没有解决信任问题；它仍位于同一块被判定异常的device POD里。当前实现改为扩展generic AICore kernel launch ABI：

```text
arg0: KernelArgs *
arg1: Runtime *trusted_l1_runtime_override
```

Host的 `launch_prepared_aicore_kernel`固定构造两个相邻pointer，并用static assert锁定size与offset。HBG L1从immutable host-side slot registration取得 `outer_runtime_base`，作为arg1直接交给CANN kernel launch；AICore entry在访问Runtime/Handshake/prelaunch control之前选择：

```text
runtime = trusted_l1_runtime_override != nullptr
            ? trusted_l1_runtime_override
            : k_args->runtime_args
```

所以HBG L1的AICPU与AICore都以同一host-sealed Runtime为根。若device KernelArgs里的 `runtime_args` 被修改，AICPU校验失败并向可信Runtime写CANCEL，AICore也在该可信Runtime上读到CANCEL后退出。

TRB L1和所有L2/L3 launch传 `nullptr` override，继续走历史 `KernelArgs::runtime_args`。AICore binary仍由PyPTO context在prepare/legacy init中注册并与Host实现成套构建；没有引入外部public ABI或graph可见参数。

本轮不仅依赖“编译通过”。A2/A3和A5、HBG与TRB四份onboard AICore产物的ELF `__CCE_KernelArgSize` section都检查为：

```text
0x10, 0x10
```

分别对应AIC/AIV entry的16-byte参数区，与Host的两个64-bit pointer完全一致。真实CANN launch/capture是否同样消费第二个pointer仍必须在device 1验证，但源码产物和Host blob已不存在8-byte/16-byte静态失配。

#### 10.41.9 为什么没有选择timeout、reset或HostArgs pointer

本阶段明确没有采用以下看似简单的方案：

- **固定spin timeout后AICore自行退出。** 这会把合法但启动较慢的AICPU误判为失败，且阈值跨A2/A3、A5和系统负载不稳定。
- **失败后Host reset device。** L1不掌控device资源，reset越过单算子边界并破坏同device上的其他工作。
- **AICPU内部stream sync确认谁没启动。** 违反L1禁止内部sync，也无法在capture中安全使用。
- **从未校验HostArgs blob取control pointer。** 失败路径会产生不受信任的device write target。
- **把per-core CANCEL提前写进Handshake。** 可能被稍后的AICore whole-line report覆盖。
- **恢复旧PyPTO private AICPU stream或model attach。** 这会重新跨越单算子边界；性能应由未来显式HBG operator抽象获得。

当前协议选择的是prepare-time信任根、caller-stream clear、AICPU cache publish、AICore低频cache acquire与完整fork/join完成证明。它更啰嗦，但每一步都能落在L1允许的资源边界内。

#### 10.41.10 本轮验证结果与没有算作通过的测试

当前代码完成后执行并通过：

- runtime editable全量构建；
- A2/A3、A5 × onboard/sim × HBG/TRB相关目标重新编译和链接；
- no-hardware CTest：**97/97 passed**，其中**80**项标记 `no_hardware`；
- PyPTO L1 + simpler ChipWorker Python回归：**51/51 passed**；
- 新增 `test_platform_aicpu_affinity_config`，覆盖0、负数、超 `MAX_GATE_THREADS`、allowed多于launched和合法边界；
- 新增 `test_aicpu_device_config`，覆盖HBG control地址在resident invocation之间保持；
- `test_hbg_execution_slot`新增valid registration优先、坏registration回退、无registration回退和zero fallback反例；
- completion gate、HBG slot/blob/registry、AICore handshake等定向测试继续通过；
- `clang-format --dry-run --Werror`与 `git diff --check`通过；
- 四份onboard AICore产物的kernel arg size均静态核对为16 bytes。

本轮尝试重跑A2/A3与A5 HBG validation simulator时，过程里出现两类环境/调用问题，不能写成产品测试失败，也不能写成通过：

1. 第一次从top repo目录启动，resource child的pytest root落在top repo，没有加载runtime自己的 `conftest.py`，child因此不认识 `--runtime/--device/--platform`；修正为从runtime目录启动。
2. 修正root后，当前机器PATH没有 `g++-15`，四个case在编译sim incore kernel前即报 `g++-15 not found`。确认是工具链缺失后主动终止剩余重复case。

所以本轮的准确记录是：此前阶段曾完成A2/A3与A5 HBG simulator的exact-error定向回归；当前最终diff完成后，sim目标在editable build中编译通过，但完整scene执行没有因缺 `g++-15`重跑成功。它不能替代device 1，也不会被计入上面的97/97或51/51。

本轮仍没有向任何NPU提交task，device 0没有被使用；命令中的 `--device 0` 只属于CPU simulator的逻辑slot，不是NPU device 0。

最终收口前又只读查询了一次真实device 1（card 7 / chip 1 / physical id 15）：HBM usage仍为4%，AICore 100%，AIVector 90%。它继续不满足仓库“HBM为0才视为空闲”的上板条件，所以没有抢占、没有把device 0当fallback，也没有启动TRB或HBG ACLGraph ST。

#### 10.41.11 本节更新10.40中的哪些结论

10.40.4记录的是当时尚未完成的审查清单。到本节为止，可以把以下源码级问题更新为已闭环：

- HBG有效AICPU participant的exactly-once arrive/finalize/snapshot/depart；
- scheduler最终init verdict对orchestrator `p_func` 的硬门槛；
- 已report core的physical id越界或register mapping为0时的pre-window CANCEL；
- A5 PMU在handshake前按physical id索引的OOB保护；
- generation建立前slot/callable/blob/ABI/KernelArgs/platform/affinity失败对hidden AICore的独立CANCEL；
- registry acquire失败的prepare-timecontrol fallback；
- affinity全员silent-drop与角色数不足；
- `KernelArgs::runtime_args`失配导致AICPU/AICore control地址分叉。

仍然不能把HBG capability翻为true，原因没有变化：

- large variable `aclrtLaunchKernelWithHostArgs`与placeholder的真实runtime-owned lifetime未上板；
- hidden-stream event-only ACLGraph capture/replay未上板；
- repeated pristine restore和cache可见性未上板；
- graph多次replay期间task package owner未上板；
- 完全不report的硬件core不在算子内可恢复模型；
- HBG Python公开路径仍然关闭；
- GM heap是否存在需显式initializer region的program类别还需审计与poison test。

所以当前新的准确表述是：**HBG L1 no-reset错误路径的已知源码协议缺口已经收口，但HBG L1/ACLGraph仍是capability=false；硬件与CANN所有权事实没有因为源码审查完成而被假定成立。**

### 10.42 第二阶段HBG graph内存继续按tiling-like task参数建模

#### 10.42.1 用户补充原则的正式落点

用户明确指出：第二阶段每次dynamic build得到的graph，在H2D到device时，本质上就是本次task的AscendC tiling-like输入。AscendC tiling data由CANN runtime随launch task管理；HBG graph内存也应尽量获得同等级的task/captured-node lifetime，而不是让PyPTO维护一个无法知道何时安全复用的device task buffer池。

当前设计继续接受这个原则，并把它拆成四个不可混淆的owner：

| 对象 | owner/lifetime | 是否会被执行修改 | 是否允许多个captured node共享 |
|---|---|---:|---:|
| canonical `HbgGraphPlan` | 本次Host build到launch enqueue | 否 | 否；每次build独立 |
| writable serialized HostArgs scratch | 仅到CANN成功snapshot本次launch参数 | CANN可能原地patch placeholder | 否 |
| runtime-owned immutable graph package | eager task或ACLGraph captured node | 否，只作为pristine source | 否；每个node必须拥有自己的bytes |
| context-owned mutable execution slot | PyPTO context到外部quiescence、graph销毁、close成功 | 是 | v1只允许串行共享 |

这里最关键的不是“H2D一份graph”这一动作，而是**source与working state分离**。当前HBG graph image包含scheduler queue、task state、completion flag、mailbox、runtime pointer等执行中原地变化的字段；它不像普通只读tiling bytes那样可以直接反复执行。正确序列必须是：

```text
captured-node-owned immutable package
  -> 每次eager execution / 每次graph replay
  -> restore pristine SM + runtime arena（未来可能还有显式GM initializer region）
  -> attach/wire mutable working slot
  -> scheduler execution mutates only working slot
```

因此“让CANN拥有task参数”和“PyPTO每次restore working state”不是替代关系，而是两条正交要求：前者解决immutable source活多久，后者解决同一node第二次replay为什么不会读到第一次执行后的残留状态。

#### 10.42.2 `aclrtLaunchKernelWithHostArgs`当前承担的角色

当前首选路径把variable-length HBG header、region descriptors、pristine SM和pristine runtime-arena bytes序列化到一份fresh writable HostArgs scratch。placeholder把blob里的source pointer patch到CANN runtime-owned device args blob内部的inline payload；AICPU收到的pointer因此应指向本次task自己的snapshot，而不是PyPTO共享device buffer。

这个方案与AscendC tiling data最接近，但仍必须通过device事实验证：

- API是否对实际HBG大小完整deep-copy，而不是只复制fixed header；
- placeholder是否在A2/A3和A5均按期望patch；
- API返回后立即poison/free/reuseHost scratch，device仍读到原始package；
- capture/instantiate后多次replay，captured node仍持有package；
- 两个不同node的A/B package不会落到同一CANN args pool slot后互相覆盖；
- internal args size上限、对齐和capture行为不能从源码常量推断，必须用真实大小probe。

canonical plan本身不能直接交给可写C API。即使当前placeholder实现只应修改scratch，接口签名和Runtime内部实现都允许原地patch；所以必须保持：

```text
const canonical plan
  -> fresh writable serialized scratch
  -> CANN snapshot / placeholder patch
  -> runtime-owned task package
```

Host canonical、一次性scratch和device task owner三层不能折叠成一层。

#### 10.42.3 workspace继续内部管理并不与task package冲突

当前PyPTO仍内部申请workspace、outer Runtime、KernelArgs、working SM、runtime arena和GM heap。用户已经明确本阶段不要求workspace外传；同时PyPTO占满全部AICore，v1禁止并发执行，所以单context只有一个mutable working slot不会产生合法并发踩踏。

但“workspace可以共享”不能外推成“graph package也可以共享”：

- workspace/working slot只在执行期间被当前串行task修改；
- graph package属于未来仍可能replay的captured node；
- PyPTO不感知graph何时销毁，不能在launch返回、event query成功或某次replay结束后回收另一个node仍可能引用的source；
- graph replay可能绕过Python和PyPTO Host入口，Host mutex看不到它。

因此v1可以继续共享一份context working slot，但每个captured node必须有独立immutable package owner。若未来开放并发graph replay，共享working slot本身也必须升级为per-node/per-flight execution slot或device-side串行gate；仅靠“PyPTO占满AICore”并不能自动防止两个graph从不同外部stream并发replay。

#### 10.42.4 如果runtime-owned inline package上板失败

fallback优先级仍保持保守：

1. 先确认是否有CANN正式的task/graph retain-release或等价tiling owner接口；若有，PyPTO只保存opaque lease，不猜completion。
2. 若只能传external device source，则每个captured node需要独立immutable allocation，并由graph lifetime lease持有。
3. 若当前wrapper/runtime没有graph销毁回调，唯一正确的临时fallback是append-only pin到context close，同时提供明确memory accounting、limit和OOM错误。
4. 在任何fallback中，binary仍可按context lifetime append-only持有；本阶段不做incore binary device内存复用。
5. 无论source owner为何，每次execution/replay的restore都不能删除。

明确禁止以下回收依据：

- 固定约2048次kernel launch；
- CANN args pool观察到的slot数量；
- Host调用已经返回；
- AICPU task已经enqueue；
- 某次event query显示完成；
- 当前没有Python引用某个input tensor。

这些事实都不能证明ACLGraph未来不会再replay该node。只有CANN task/graph owner或显式外部lease release能成为回收边界。

#### 10.42.5 HBG capability probe必须同时验证lifetime与no-reset

未来device 1的HBG probe不能只验一个数值结果。至少要形成以下矩阵：

1. eager单次：真实大小package、placeholder、restore、数值正确；
2. eager连续A/B/A：不同package交替恢复同一working slot；
3. 单graph连续replay至少两次：第二次不读取第一次的mutable残留；
4. 同graph多个不同HBG node：各node保持自己的package，不退化为最后一次build；
5. Host scratch在launch API返回后立即poison/free/reuse；
6. capture完成后销毁Host canonical/scratch，保留graph/context owner仍可replay；
7. 坏blob、缺callable、registry不可用、非法affinity、Runtime pointer失配、zero register mapping分别失败；
8. 每个失败后hidden AICore完成、caller tail可达，下一次合法调用无需reset；
9. graph teardown严格外部quiescence、graph reset/destroy、context close；
10. 全过程不出现PyPTO内部stream/device sync、reset、capture query或model attach。

只有lifetime和no-reset两组同时成立，HBG graph package才真正具备“像AscendC tiling参数一样随task进入ACLGraph”的语义。单纯证明H2D成功、AICPU能读到一次bytes，远远不够。

#### 10.42.6 当前阶段的最终状态

截至本节：

- graph package已经按task-local immutable tiling-like输入建模；
- working state与source严格分离，每次execution/replay都要求restore；
- known pre-generation与generation-internal no-reset源码路径已经收口；
- HBG AICPU/AICore对Runtime cancellation root不再split-brain；
- workspace继续由PyPTO context内部持有，符合当前决策；
- TRB L1、L2/L3不采用HBG task package或control fallback语义；
- HBG capability仍显式false，Python仍fail-closed；
- device 1 ACLGraph、CANN args owner和真实cache/order证据仍未取得。

这意味着第二阶段的架构方向已经清楚：**graph package跟task/captured node走，mutable execution state跟context slot走，二者由每次replay restore连接；任何性能优化都必须留在显式HBG operator边界内，不能重新引入旧PyPTO那种提前启动并跨越单算子边界的hidden行为。**

### 10.43 GPT/Grok隔离、device 1首次TRB L1实测与独立report缓存行

#### 10.43.1 第一阶段当前不能宣称完成

2026-08-18再次向用户汇报时，第一阶段的准确状态是：Host API、Python wrapper、taskQueue adapter、borrowed caller stream、hidden AICore stream、prepare/close生命周期和大量无硬件契约已经落地，但TRB L1 eager在device 1上的首次真实执行仍停在AICPU/AICore startup handshake；因此ACLGraph capture和多次replay还没有通过证据。

不能因为以下事实就把第一阶段写成完成：

- editable build通过；
- no-hardware UT通过；
- L2同一套AICore binary和KernelArgs可以运行；
- ACLGraph ST已经写好并能collect；
- 源码审查没有发现新的普通路径P0。

阶段完成门槛仍然是device 1上按真实torch/taskQueue调用顺序完成：prepare、eager warmup、外部同步、独立capture stream、图内前后torch算子排序、多次replay数值验证和显式graph/context teardown。当前只完成了到真实eager故障定位这一步。

#### 10.43.2 两个session的五层隔离

Grok和GPT没有在同一个working tree里直接改文件。实测核验结果如下：

| 层次 | GPT session | Grok session |
|---|---|---|
| top worktree | `/mnt/workspace/inductor/pto/gpt_pypto` | `/mnt/workspace/inductor/pto/pypto` |
| top branch | `gpt/pypto-l1-aclgraph` | `main` |
| nested simpler/runtime | `/mnt/workspace/inductor/pto/gpt_pypto/runtime` | `/mnt/workspace/inductor/pto/pypto/runtime` |
| nested branch | `gpt/pypto-l1-aclgraph` | `l1-aclgraph` |
| Python/native owner | GPT runtime自己的 `.venv`、editable `.pth`和 `_task_interface.so` | 不从GPT venv加载 |
| NPU约定 | 只允许device 1 | 主要使用device 0 |

Git worktree可以共享object database，但working tree、index、当前branch和未提交文件彼此独立。核验时Grok nested runtime为clean；当前几十个L1/HBG runtime修改全部只存在于GPT nested runtime。GPT top repo自己的Python wrapper和测试修改也只存在于`gpt_pypto`。

真正危险的不是Git，而是Python editable安装。用户级文件：

```text
/home/developer/.local/lib/python3.11/site-packages/_pypto_editable.pth
```

仍明确指向Grok的：

```text
/mnt/workspace/inductor/pto/pypto/python
```

它不能被GPT session删除或改写，否则会反向破坏Grok环境。因此GPT所有构建、pytest和上板命令必须同时满足：

```bash
PYTHONNOUSERSITE=1
PYTHONSAFEPATH=1
PYTHONPATH=/mnt/workspace/inductor/pto/gpt_pypto/python:\
/mnt/workspace/inductor/pto/gpt_pypto/runtime/python:\
/mnt/workspace/inductor/pto/gpt_pypto/runtime
```

解释器必须使用：

```text
/mnt/workspace/inductor/pto/gpt_pypto/runtime/.venv/bin/python
```

不再把整个用户级`site-packages`追加到`PYTHONPATH`；即使普通目录方式通常不会处理其中的`.pth`，也没有必要扩大串仓面。第三方依赖由GPT venv和系统site-packages提供。禁止使用裸`python`、裸`pytest`或向用户级site-packages执行editable install。

本次实际import证明为：

```text
pypto   = /mnt/workspace/inductor/pto/gpt_pypto/python/pypto/__init__.py
simpler = /mnt/workspace/inductor/pto/gpt_pypto/runtime/python/simpler/__init__.py
native  = /mnt/workspace/inductor/pto/gpt_pypto/runtime/.venv/lib/python3.11/site-packages/
          _task_interface.cpython-311-aarch64-linux-gnu.so
adapter = /mnt/workspace/inductor/pto/gpt_pypto/runtime/.venv/lib/python3.11/site-packages/
          pypto/_torch_npu_l1.cpython-311-aarch64-linux-gnu.so
ENABLE_USER_SITE = False
```

native editable build目录也位于GPT runtime自己的`build/`。2026-08-18再次核验时，两边已stage的A2/A3 TRB AICPU产物具有不同inode、size和SHA-256：GPT为`3816736`字节、`44fef471...`，Grok为`3720400`字节、`69a429da...`，证明不是同一个文件或软链接。

还必须区分GPT runtime内部的两层构建产物：

- `runtime/build/cache/...`是增量编译输出；
- `runtime/build/lib/...`才是`RuntimeBuilder.get_binaries(build=False)`和上板ST实际打包的staged产物。

只重编`build/cache`而没有将产物stage进本工作树自己的`build/lib`，会让GPT测试加载GPT目录中的旧二进制；这不是串到Grok仓，但会造成“源码已经修改、设备仍运行旧实现”的假象。因此每次runtime改动后必须在GPT目录显式执行本工作树的`RuntimeBuilder(...).get_binaries(..., build=True)`，再核对cache/staged文件的时间戳、size或hash。绝不能复制或引用Grok的`runtime/build/lib`。

PTOAS是双方只读共享的工具，不是Python/native产物；GPT固定使用`PTOAS/build-v0.57-llvm21-cann9.2-clean/tools/ptoas`，并清除可能指向不兼容工具的`PTOAS_ROOT`。

#### 10.43.3 当前没有NPU进程，不能把历史占用归因于Grok

再次查询host进程时，没有发现命令行指向`gpt_pypto`、`pypto`、pytest或正在运行的NPU测试进程。此前`npu-smi`曾显示某device有非零HBM和AICore/AIV利用率，但没有对应PID；这类观测不能证明Grok正在运行，也不能作为抢占或reset设备的依据。

因此本节采用严格表述：

- 当前查询时没有发现任何人正在提交NPU任务；
- 先前的utilization/HBM读数不能归因于Grok；
- device 0无论是否显示进程都不由GPT session使用或reset；
- GPT仅在确认目标为device 1后显式`aclrtSetDevice(1)`/`--device=1`；
- 每次上板前打印`pypto.__file__`、`simpler.__file__`、`_task_interface.__file__`和目标device，任一不匹配就拒绝执行。

#### 10.43.4 device 1首次TRB L1故障事实与反证

真实A2/A3 L1 ST在device 1进入warmup后，外部sync报507018/AICPU 0x2a。Host侧确认caller stream、hidden stream、Start event和两类kernel task均成功进入提交路径；device日志显示custom AICPU worker进入startup handshake，但并非所有scheduler slice都完成收集。

进一步诊断日志中，全部6个custom AICPU线程都进入affinity gate，其中4个有效角色为3个scheduler和1个orchestrator；三个scheduler均进入handshake，但只有其中一个收齐其21个core，另外两个一直等待本slice的report。这个形态排除了“缺少某个AICPU affinity worker”这一直接解释。

同一分支、同一device 1执行L2反证程序，连续4次dispatch通过。这证明至少以下基础事实成立：

- 当前AICore binary能够在device 1执行；
- KernelArgs和Runtime的基本H2D地址链在L2成立；
- physical-core register mapping不是全面失效；
- 不是简单的编译架构或device选择错误。

随后通过源码、产物和小改动逐项排除了若干L1-only候选：

1. `L1AicpuInvocationArgs`约33KiB导致每worker大栈：入口改成只拷贝小prefix，实产物栈帧从约34KiB降到224B，故障仍存在。
2. CANN HostArgs存在32KiB硬截断：本机runtime的CPU_EX参数池规格和copy路径可以容纳约33KiB，未发现16-bit SQE length截断。
3. persistent Runtime或device KernelArgs H2D损坏：prepare后D2H逐字段校验通过。
4. AICPU和AICore取到不同Runtime地址：所有L1 AICore launch增加host prepare信任的Runtime override后，故障形态未消失。
5. 仅仅把AICore launch放到AICPU launch之前：提交顺序改变后仍出现部分slice收不到report，说明顺序最多是调度影响，不是完整正确性证明。
6. AICPU入口对整个Runtime做一次cache invalidate：入口可能早于AICore report，首次读0后仍可缓存；一次invalidate无法建立持续可见性。

507018/0x2a也不能简单写成“AICPU代码发生fault”。仓内既有记录证明STARS watchdog可把纯握手自旋表现为同类错误码；必须结合最后可见phase和per-thread进度判断。

#### 10.43.5 legacy Handshake为什么不能靠循环`dc civac`修复

旧`Handshake`把两种所有权混在同一个64B cache line：

- AICore写`aicore_done/physical_core_id/core_type`；
- AICPU写`aicpu_ready/task`等控制字段。

L1中custom AICPU可能不在AICore HBM write的自动snoop域，AICPU轮询前确实需要cache maintenance。但当前`cache_invalidate_range`在A2/A3实际使用`dc civac`，语义不只是丢弃本地line，还可能把AICPU本地dirty stale line clean回HBM。若AICore刚把同一line上的report写回，AICPU的CIVAC就可能用旧整行覆盖新report。多个scheduler并行扫描不同slice时，“一个slice全收齐，另两个slice永久缺若干report”与这种mixed-owner whole-line clobber完全吻合。

所以不能继续在旧Handshake上增加更频繁的flush/invalidate；那会把竞态窗口放大。正确性要求把report和control的cache-line所有权物理拆开。

#### 10.43.6 独立`L1AicoreReport`的当前实现

当前实现新增：

```cpp
struct alignas(64) L1AicoreReport {
    volatile uint32_t aicore_done;
    volatile uint32_t physical_core_id;
    volatile uint32_t core_type;
    // padding to exactly one 64-byte cache line
};
```

核心约束如下：

- 每个launched AICore独占一条64B report line，不能把多个core压到同一line；
- AICore是唯一writer，AICPU只读并可安全CIVAC；
- legacy Handshake继续承载AICPU task和pre-window CANCEL，不再被L1依赖为report source；
- L2/L3 Runtime构造时report pointer显式为`nullptr`，继续走旧协议；
- L1在prepare阶段按`worker_count * 64`分配并验证64B alignment；
- report base只存进persistent device Runtime，AICPU和AICore从同一信任根读取，避免新增双源地址；
- 每次launch/replay由caller stream在Start event前异步clear report span；
- allocation由现有L1 context allocator持有到显式close，地址在capture/replay期间稳定；
- scheduler completion不再回读旧Handshake的`core_type`，而使用handshake阶段已经固化的`core_type_compact_`。

没有选择把report base放进resident AICPU SO global，也没有增加第三个AICore kernel pointer参数。前者会引入同device多context覆盖问题，后者会让AICPU和AICore再次出现两个传参源并扩大wire ABI。Runtime pointer方案只在四套Runtime布局尾部追加字段，并将L2/L3默认值固定为null。

截至本次记录，以下验证已经完成：

- GPT runtime editable全量构建通过，A2/A3、A5 × TRB/HBG × onboard/sim相关产物均编译；
- 四份scheduler独立include和DFX `-Werror=unused-parameter`问题已修；
- report的64B size/alignment、相邻元素独占cache line和字段边界UT通过；
- A2 TRB Runtime默认null及setter round-trip UT通过；
- launch sequence、AICore handshake、TRB Runtime和orchestration requirements四组定向CTest共4/4通过；
- `git diff --check`通过。

但必须保留最后一句：**独立report尚未完成device 1复测，因此它当前是由日志形态、cache ownership和反证链支持的最强根因修复，不是已经由硬件确认的最终结论。**

#### 10.43.7 上板前仍需闭合的fork提交失败窗口

caller-stream AICPU与hidden-stream AICore需要两次独立runtime enqueue；它们天然不是一个原子事务。当前临时顺序先enqueue hidden AICore，再enqueue caller AICPU。若第二个API同步返回失败，AICore已经在等待AICPU打开register window，却没有AICPU发布task/CANCEL；仅把context poison并返回会留下orphan AICore，与L1 no-reset目标冲突。

反过来恢复AICPU-first也不自动解决：若随后hidden AICore enqueue失败，AICPU同样会等待永远不会出现的report。所以最终协议不能只争论谁先launch，必须满足“第二分支失败时，第一分支有可靠、capture-compatible、无sync的退出路径”。当前正在审查的候选是显式launch commit/cancel control或失败时可enqueue的device cancel closure；在该错误闭包确定并有无硬件反例测试前，不把新的happy-path report实现直接当成可交付状态。

#### 10.43.8 独立report上板后的反证：cache line不是本次507018的直接根因

在把独立`L1AicoreReport`完整编译、stage并送到device 1后，第一次诊断运行仍然在eager warmup的外部同步点报507018/AICPU 0x2a。但这一次探针已经能够证明启动report链本身全部成功：

- 6个AICPU worker全部进入L1 HostArgs prefix，affinity保留3个scheduler和1个orchestrator；
- persistent Runtime为`0x12c2000cd000`，独立report数组为`0x12c2000cc000`，worker count为60；
- 三个scheduler都完成各自report Phase 1/Phase 2；
- `tidx0`汇总的60-bit可见mask为`0x0fffffffffffffff`，即60个AICore全部完成report；
- 因此“某些AICore report被mixed-owner Handshake cache line覆盖”并不是这一次507018的直接根因。

真正的最后进度停在AICPU第一次打开AICore寄存器窗口：

```text
physical_core_id = 2
reg_addr         = 0x3ffbc7ff000
phase            = platform_init_aicore_regs() / window open begin
```

没有对应的`window open done`。另外一个scheduler也停在physical core 4的`0x3ffbd7ff000`。这把故障从“report/cache可见性”进一步收敛为“当前L1 AICPU进程不能访问正确的AIC_CTRL MMIO窗口”。

这个反证修正了10.43.5/10.43.6当时的根因判断，但不要求回退独立report：每核一条AICore-only cache line仍然是更清晰、可证明的ownership协议，避免未来AICPU cache maintenance覆盖AICore report；其A2/A3真实硬件路径也已证明能够连续warmup/capture/replay。准确表述应改为：**独立report是正确性hardening和故障排除工具，但不是本次custom scheduler MMIO fault的直接修复。**

#### 10.43.9 与L2同地址逐项对照：custom scheduler缺少MMIO执行能力

随后在同一GPT工作树、同一device 1、同一A2/A3 runtime binary上执行L2 control：

```text
tests/st/runtime/framework_and_models/test_compiled_program.py::
TestJitCompiledProgram::test_inplace_add
```

L2通过。device日志给出了决定性对照：

| 路径 | device进程标签 | physical core | AIC_CTRL地址 | 写窗口结果 |
|---|---|---:|---:|---|
| L1旧路径 | `aicpu_custom_scheduler` | 2 | `0x3ffbc7ff000` | begin后fault/卡住 |
| L2路径 | `aicpu_scheduler` | 2 | `0x3ffbc7ff000` | begin与done均出现 |

相同地址在L2标准scheduler进程可写，在L1 custom scheduler进程不可写，因此可以排除host register table、physical-core映射和地址计算错误。差异来自AICPU binary的加载模式：

- 旧L1调用`aclrtBinaryLoadFromData(cpuKernelMode=2)`，再`aclrtRegisterCpuFunc`；task运行于`aicpu_custom_scheduler`；
- L2先用KFC dispatcher把SO写进preinstall目录，再用mode0 JSON注册；task运行于标准`aicpu_scheduler`；
- `platform_init_aicore_regs`要求标准scheduler拥有的AIC_CTRL MMIO映射/权限，custom scheduler不具备该执行环境。

所以不能通过继续调整report轮询、HostArgs大小、Runtime地址或AICore/AICPU launch先后顺序来修复。L1必须改为标准AICPU scheduler，同时仍然遵守caller stream和不内部同步的单算子边界。

#### 10.43.10 为什么mode0注册不需要等待dispatcher写完SO

原设计记录在10.1曾保守认为：dispatcher必须先完成device文件写入，host才能mode0注册，因此bootstrap内的stream sync可能不可避免。核对本机CANN 9.2 runtime源码后，这个假设被更精确地拆开：

1. `BinaryLoader::LoadCpuMode0Program()`只调用`ParseJsonAndRegisterCpuKernel()`；注释明确是“只加载json”。
2. `aclrtBinaryGetFunction`/`rtsFuncGetByName`会为Kernel保存SO名、函数名，并把这两个literal name复制到device；它不打开或读取JSON里指向的preinstall SO。
3. mode0下`Program::ProcCpuKernelH2DMem`直接走“不需要处理CPU SO H2D”的分支；只有mode1/mode2才触发SO复制/加载。
4. 真正按SO名进入标准AICPU scheduler并`dlopen`目标SO发生在后续kernel task执行阶段。

因此host可以在dispatcher task尚未device执行时立即完成mode0 JSON解析和function-handle解析。只要dispatcher task与后续init/register/run task都在同一个caller stream，device FIFO自然建立：

```text
caller stream:
  KFC dispatcher bootstrap（写preinstall SO）
    -> simpler_aicpu_init
    -> simpler_aicpu_l1_register_callable
    -> simpler_aicpu_l1_exec / capture node
```

这条顺序不要求`aclrtSynchronizeStream`，不需要private AICPU stream，也不让orchestrator越过caller-stream predecessor。mode0 host注册期间发生的literal-name device allocation/copy属于capture外prepare-time metadata准备，不读取inner SO，也没有新增stream/device synchronize API。

#### 10.43.11 L1异步bootstrap的ownership事务

`LoadAicpuOp`新增一条与L2/L3明确隔离的路径：

```cpp
BootstrapDispatcherAsync(..., caller_stream, device_id);
InitPreinstalledAcl(extra_symbols);
```

L2/L3继续使用原来的：

```cpp
BootstrapDispatcher(...);  // 内部同步，随后局部RAII释放bootstrap输入
Init(...);                 // RTS mode0
```

异步L1路径的关键不只是删除一行sync，而是重新定义三块device输入的owner：

- dispatcher SO device copy；
- inner runtime SO device copy；
- dispatcher `DeviceArgs` device copy。

同步L2路径可在`aclrtSynchronizeStream`之后让局部`DeviceBuf`析构；L1路径在enqueue API返回时device可能尚未读取这些地址，绝不能复用该局部RAII。当前实现把三块指针在launch前转交给`LoadAicpuOp`成员，直到调用方已经外部quiesce graph/stream后显式`ctx.close()`，才由`Finalize()`释放。

失败事务同样保守：

- H2D或launch之前失败，没有device task引用的局部allocation可立即清理；
- 调用launch API之后，即使API同步返回错误，也保守持有三块地址，避免“部分接收task”语义不清时UAF；
- mode0 load成功后立即接管ACL binary handle；后续任一function解析失败都不在prepare栈上卸载/释放，而是让poisoned context保留close owner；
- `aclrtBinaryUnLoad`失败时保留handle和全部bootstrap allocation，允许显式close重试；
- unload成功后逐个`aclrtFree`，成功的指针清空，失败的精确指针保留供重试；
- L2的`RtsFile` unload仍保持历史best-effort语义，不把L1可重试teardown反向扩散到owned-device reset流程。

异步路径不读取也不写process级“已完成bootstrap”cache。host在没有同步的情况下不能声称某次enqueue已经完成；L1 v1又已有每device单context约束，所以每个context明确enqueue一次bootstrap并持有自己的输入，比猜测另一个context/device task是否完成更安全。device dispatcher按`fingerprint + device_id`生成目标名，并用temp-file + atomic rename写入，同内容重复执行是幂等的。

JSON临时文件名也从`fingerprint + pid`加强为`fingerprint + device + pid + loader地址`。这是因为同一进程同时注册两个device时，inner SO fingerprint可能相同，但JSON里的device-suffixed `kernelSo`不同；唯一文件名避免一方覆盖或删除另一方尚未读取的descriptor。

#### 10.43.12 fork失败闭包已经落地

10.43.7记录的窗口已经在本次正式上板前闭合，旧段落保留为问题发现的时间线。最终顺序仍采用AICore-first，但增加host可提交的pre-window cancel：

1. caller clear Handshake/report并record Start；
2. hidden wait Start，enqueue AICore，成功后立即record AICoreDone；
3. caller enqueue AICPU；
4. 若AICPU enqueue同步失败，caller对整段legacy Handshake执行`aclrtMemsetAsync(..., 0xFF, ...)`；
5. AICore pre-window poll把`UINT32_MAX`识别为host cancel，全部退出；
6. caller wait已经排好的AICoreDone，再record serial tail并返回原始错误/poison。

这里用byte-fill `0xFF`是为了让每个`aicpu_ready`首word精确得到`0xffffffff`，避免`aclrtMemsetD32Async`可能申请临时pinned buffer。正常路径没有额外节点；补偿分支仍只有async enqueue，没有内部sync/reset。若cancel enqueue或join本身也失败，context保持poisoned/Closing并保留资源，不伪装成已安全回收。

反向改成AICPU-first并不更正确：hidden wait、AICore launch或done-event record任一同步失败都会留下等待report的AICPU orphan，而且Host没有同样简单的标准scheduler cancel通道。AICore-first把所有这些风险点放在AICPU入队前，只留下一个已经有Handshake pre-window cancel协议的分支。

#### 10.43.13 device 1正式无探针ACLGraph验收

完成异步mode0路径后，先保留诊断探针运行一次。Host日志显示：

```text
BootstrapDispatcherAsync: queued ... target=simpler_inner_a48eda407420804c_1.so
InitPreinstalledAcl: resolved preinstalled ACL handle ...
```

device日志对应进程标签已经从`aicpu_custom_scheduler`变为`aicpu_scheduler`：

```text
[simpler-dispatcher] Init: wrote ...simpler_inner_a48eda407420804c_1.so
AICPU(...,aicpu_scheduler) ... simpler_aicpu_l1_exec
```

60个report全部可见，所有`platform_init_aicore_regs` window open都有done，首次完整ST通过。随后删除源码中的全部`[L1_PROBE]`，重新编译并stage A2/A3 TRB AICPU SO，确认正式SO中不存在探针字符串；使用正式产物再次执行：

```bash
env -u PTOAS_ROOT \
  PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
  PYTHONPATH=/mnt/workspace/inductor/pto/gpt_pypto/python:\
/mnt/workspace/inductor/pto/gpt_pypto/runtime/python:\
/mnt/workspace/inductor/pto/gpt_pypto/runtime:\
/mnt/workspace/inductor/pto/gpt_pypto/tests/st \
  PATH=/mnt/workspace/inductor/pto/PTOAS/build-v0.57-llvm21-cann9.2-clean/tools/ptoas:\
/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  timeout 240s runtime/.venv/bin/python -m pytest \
  tests/st/runtime/l1/test_l1_aclgraph.py \
  --platform=a2a3 --device=1 --runtime-log-level=debug -q
```

结果：

```text
1 passed, 1 deselected, 1 warning in 3.79s
```

这个单个ST不是只测一个add结果，而是完整覆盖：

1. test-local `@pl.jit` 64x128 FP32 add编译；
2. `pypto_init(programs=[compiled], device=1)`；
3. `context.prepare()`与普通stream eager `op.warmup()`；
4. 调用方外部device synchronize；
5. warmup raw stream与独立capture stream不同；
6. graph内`torch.add(out=) -> L1(out=) -> torch.mul(out=)`的上下游顺序；
7. 三组不同输入连续graph replay并逐次验数；
8. graph-bound tensor保持强引用；
9. finally中调用方外部quiescence，随后`graph.reset()`，最后`context.close()`。

PyPTO内部仍没有stream/device synchronize、capture query、`rtStreamAddToModel`、model attach、private AICPU stream或early orchestrator launch。测试中的同步属于用户明确执行的warmup/teardown边界。

同一正式产物随后执行L2 control：

```bash
runtime/.venv/bin/python -m pytest \
  tests/st/runtime/framework_and_models/test_compiled_program.py::\
TestJitCompiledProgram::test_inplace_add \
  --platform=a2a3 --device=1 --runtime-log-level=debug -q
```

结果：

```text
1 passed, 1 warning in 2.91s
```

这证明新增L1异步bootstrap/ACL mode0分支没有替换或破坏L2原有同步bootstrap/RTS mode0路径。

#### 10.43.14 第一阶段现在可以声明到什么范围

截至本节，A2/A3 TRB L1的核心硬件门槛已经首次闭环：

- caller stream AICPU + hidden AICore fork/join可被ACLGraph自然捕获；
- warmup与capture使用不同raw stream；
- capture前外部同步，不把capture外event依赖带入图；
- 每次WithHostArgs参数由CANN task持有；
- capture后连续三次replay数值正确；
- graph/context按外部quiescence契约销毁；
- AICPU实际运行于具备AIC_CTRL能力的标准scheduler；
- L2 control继续通过。

但阶段commit前仍要完成：完整no-hardware CTest、Python L1 UT、A2/A3+A5 TRB/HBG所有onboard构建、loader失败/无sync结构性护栏、`git diff --check`和文档状态更新。A5没有当前硬件可做同等上板，因此只能用双架构编译和共享协议审计作为本机证据，不能把A2/A3实测文字泛化成“A5已上板”。HBG capability仍然显式关闭；这次通过只证明第一阶段TRB L1/ACLGraph，不证明第二阶段HBG package/restore已经可用。

#### 10.43.15 第一阶段提交前完整回归结果

在删除全部诊断探针、完成标准AICPU scheduler异步bootstrap和独立AICore report协议后，使用GPT工作树的最新源码重新执行完整无硬件回归：

```bash
cd /mnt/workspace/inductor/pto/gpt_pypto/runtime
cmake --build tests/ut/cpp/build -j4
ctest --test-dir tests/ut/cpp/build --output-on-failure -LE requires_hardware
```

结果为：

```text
100% tests passed, 0 tests failed out of 98
```

Python侧强制关闭user-site，并显式指定GPT的PyPTO、simpler和runtime路径：

```bash
cd /mnt/workspace/inductor/pto/gpt_pypto
PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
PYTHONPATH=/mnt/workspace/inductor/pto/gpt_pypto/python:\
/mnt/workspace/inductor/pto/gpt_pypto/runtime/python:\
/mnt/workspace/inductor/pto/gpt_pypto/runtime \
runtime/.venv/bin/python -m pytest -q \
  tests/ut/runtime/test_l1.py \
  runtime/tests/ut/py/test_l1_chip_worker.py
```

结果为：

```text
55 passed, 1 warning in 0.37s
```

随后对四个`arch × runtime`组合分别重编host、AICPU、AICore，共12套onboard产物：

| 架构 | runtime | host | AICPU | AICore |
|---|---|---|---|---|
| A2/A3 | TRB | PASS | PASS | PASS |
| A2/A3 | HBG | PASS | PASS | PASS |
| A5 | TRB | PASS | PASS | PASS |
| A5 | HBG | PASS | PASS | PASS |

这些结果把10.43.14中列出的三类回归门槛全部关闭。`BootstrapDispatcherAsync`在成功提交后、调用`aclrtSynchronizeStream`之前直接返回；三块dispatcher输入在调用launch API前转交给`LoadAicpuOp`，ACL unload/free失败时保留原handle或原指针供显式close重试。正式device 1 ST又直接覆盖了这条异步分支，因此当前证据由源码结构、双架构全编译、无硬件错误分支UT以及真实ACLGraph执行共同组成，而不是依赖单一字符串检查。

上述runtime收口已经形成独立中文提交：

```text
3631ea0d Fix: 完成TRB L1标准AICPU调度与ACLGraph闭环
```

该提交没有包含仍未完成的HBG orchestration requirements emitter；相关validator WIP继续留在工作区，避免把“能fail-closed”误写成“HBG已经可用”。

#### 10.43.16 GPT/Grok隔离复核

阶段提交前重新核对两个并行session，结论如下：

- Grok顶层工作树是`/mnt/workspace/inductor/pto/pypto`，分支`main`；其runtime分支是`l1-aclgraph`；
- GPT顶层工作树是`/mnt/workspace/inductor/pto/gpt_pypto`，顶层和runtime均为`gpt/pypto-l1-aclgraph`；
- 两边工作目录、runtime目录、build/cache和虚拟环境均是不同inode；
- GPT的`_task_interface.so`与`pypto/_torch_npu_l1.so`位于`gpt_pypto/runtime/.venv`；
- 实际导入路径逐项确认：`pypto`来自`gpt_pypto/python/pypto`，`simpler`来自`gpt_pypto/runtime/python/simpler`，两个native扩展也都来自GPT虚拟环境；
- 用户级`_pypto_editable.pth`仍指向Grok工作树。不能修改它，否则会反向影响另一个session；GPT所有构建和测试因此必须同时设置`PYTHONNOUSERSITE=1`、`PYTHONSAFEPATH=1`和显式GPT `PYTHONPATH`；
- NPU硬件没有进程级命名空间隔离。GPT命令只传`--device=1`，不使用device 0；运行前仍需检查设备占用。一次复核中`npu-smi`没有列出host进程，`fuser /dev/davinci15`也为空，但两颗device仍显示AICore 100%和残留HBM；这只能表明存在设备侧残留状态，不能归因于Grok正在使用NPU。

因此“代码、Python包、native产物、build cache”已经物理隔离；“device 0/device 1”是明确的session约定和运行前门禁，而不是硬件级namespace。

#### 10.43.17 隔离门禁的实际触发与当前状态

在runtime继续提交到`80615b1e`后，再次使用GPT专属环境导入时，源码/native哈希门禁按预期拒绝了旧产物：

```text
_task_interface was built from 3631ea0d0a39,
but this source tree is at 80615b1ef126
```

这次失败不是导入了Grok的SO，而是GPT自己的`_task_interface`落后于GPT runtime源码。该门禁的价值正是防止Python以一个版本的struct布局驱动另一个版本的native扩展，避免把ABI错配误判为L1运行时故障。

随后只在GPT虚拟环境中，使用独立构建目录重建runtime editable产物：

```bash
cd /mnt/workspace/inductor/pto/gpt_pypto/runtime
PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
PYTHONPATH=/mnt/workspace/inductor/pto/gpt_pypto/python:\
/mnt/workspace/inductor/pto/gpt_pypto/runtime/python:\
/mnt/workspace/inductor/pto/gpt_pypto/runtime \
.venv/bin/python -m pip install --no-build-isolation \
  --config-settings=build-dir=build/editable-gpt-runtime-80615b1e -e .
```

重建后逐项核对的实际加载路径为：

```text
pypto:          /mnt/workspace/inductor/pto/gpt_pypto/python/pypto/__init__.py
simpler:        /mnt/workspace/inductor/pto/gpt_pypto/runtime/python/simpler/__init__.py
pypto_core:     /mnt/workspace/inductor/pto/gpt_pypto/runtime/.venv/.../pypto_core...so
_task_interface:/mnt/workspace/inductor/pto/gpt_pypto/runtime/.venv/.../_task_interface...so
runtime source/native hash guard: PASS
```

四者都在`gpt_pypto`树下，没有任何一项位于Grok的`/mnt/workspace/inductor/pto/pypto`。用户级editable仍指向Grok，但`PYTHONNOUSERSITE=1`使它对GPT命令不可见；该用户级文件不作任何修改。

### 10.44 HBG orchestration requirements生产/消费闭环

#### 10.44.1 为什么这不是一个可选的优化项

HBG L1的Host graph build与历史HBG L2有一个关键差别：L2拥有tensor staging、D2H/H2D以及整次run的同步边界，而L1借用的是caller传入的device tensor，prepare/capture launch均不允许为了Host读取tensor内容而做stream/device synchronize或隐式D2H。因此，HBG L1在调用Host orchestration生成graph image之前，必须能证明该orchestration只使用shape/stride/dtype、scalar、callable identity和可静态获取的元数据，不会解引用外部tensor数据。

在本节修复前，runtime已经有fail-closed validator，但PyPTO codegen没有任何producer导出它所要求的符号。这意味着并非“某些有Host tensor read的程序被拒绝”，而是所有现有PyPTO生成的HBG orchestration SO都会因`MetadataUnavailable`被拒绝。在开启HBG capability之前，producer/consumer必须一起完成，不能以“runtime已经fail-closed”替代端到端协议。

#### 10.44.2 元数据的判定点必须是真正的Host数据访问生成点

version 1定义两个bit：

- bit 0：Host orchestration读取tensor contents；
- bit 1：Host orchestration写入tensor contents。

不能简单遍历IR中所有`tensor.read`文本就置bit 0。`Submit`的predicate也会以tensor element为条件，但这类read会被`EmitPredicateHint`编码成device scheduler使用的predicate metadata，Host graph build并不调用`get_tensor_data`。如果在过早的IR visitor中置位，会错误禁用本来是HBG的重要使用场景。

本次因此把置位放在`OrchestrationStmtCodegen::GenerateTensorOpCode`的真正发射点：

- 只有正常`tensor.read`即将生成`get_tensor_data<T>`时置read bit；
- 只有正常`tensor.write`即将生成`set_tensor_data<T>`时置write bit；
- predicate hint不经过这一发射路径，因此requirements保持为0。

`OrchestrationResult` 同时增加`orchestration_requirements_v1`字段，并经nanobind暴露给Python。这一字段用于单元测试和诊断；runtime不信任Python侧另行传入的一份flag，它信任的是最终被编译进orchestration SO的versioned符号。

#### 10.44.3 SO ABI与L2兼容性

新codegen在每个orchestration source中生成：

```cpp
__attribute__((visibility("default")))
uint64_t pypto_orchestration_requirements_v1(void) {
    return UINT64_C(flags);
}
```

采用独立符号，而不是修改既有`PTO2OrchestrationConfig`返回struct，目的是避免破坏现有orchestration SO ABI。runtime A2/A3与A5的HBG loader在`dlopen`时用`dlsym`可选解析该符号：

- 历史SO没有该符号时，仍可正常注册并走L2；
- 只有`build_l1_hbg_graph_plan_impl`会强制调用validator；
- metadata缺失、未知future bit、read bit、write bit全部fail-closed；
- 当前只有明确存在且flags为0的version 1 metadata允许HBG L1继续构图。

这个设计保持了L2历史产物的可加载性，同时不把“缺少证据”当成HBG L1安全。runtime消费侧已形成独立提交：

```text
80615b1e Add: 建立HBG L1 orchestration需求元数据门禁
```

#### 10.44.4 当前验证证据

runtime validator反例覆盖了metadata缺失、空位图、单独read、单独write、read+write和未知高位bit。使用当前runtime HEAD执行：

```bash
ctest --test-dir runtime/tests/ut/cpp/build \
  --output-on-failure -R '^test_orchestration_requirements$'
```

结果：

```text
1/1 Test #20: test_orchestration_requirements ... Passed
100% tests passed, 0 tests failed out of 1
```

PyPTO producer侧使用GPT隔离环境执行三个codegen文件：

```bash
PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
PYTHONPATH=/mnt/workspace/inductor/pto/gpt_pypto/python:\
/mnt/workspace/inductor/pto/gpt_pypto/runtime/python:\
/mnt/workspace/inductor/pto/gpt_pypto/runtime \
runtime/.venv/bin/python -m pytest -q \
  tests/ut/codegen/test_orchestration_codegen.py \
  tests/ut/codegen/test_orchestration_tensor_rw.py \
  tests/ut/codegen/test_predicate_codegen.py
```

结果：

```text
79 passed, 10 warnings in 40.86s
```

关键断言包括：

- 无Host tensor access的基本orchestration为0；
- 普通tensor read为1；
- 同时read/write为3；
- predicate tensor read为0，且生成源码中没有`get_tensor_data`。

为避免只验证C++源码字符串，又通过`pl.parse_program(string)`构造无Host tensor access的program，用`KernelCompiler(platform="a2a3").compile_orchestration("host_build_graph", ...)`真实生成Host HBG SO，再查动态符号表。结果为：

```text
0000000000003f10 T pypto_orchestration_requirements_v1
requirements_value=0
```

这证明符号具有default visibility，位于最终SO的dynamic symbol table，runtime `dlsym`能够消费，而不是只存在于中间源码。首次尝试直接在stdin中使用`@pl.program`时，DSL因`inspect`无法回取stdin class源码而拒绝；改用正式支持的`pl.parse_program(string)`后完成真实链接验证。

#### 10.44.5 完成这一节后仍不能宣称的事情

requirements门禁只解决“Host graph build是否会解引用借用tensor数据”。它不能证明以下device语义：

- `aclrtLaunchKernelWithHostArgs`对大型可变HostArgs blob和placeholder的snapshot/patch在ACLGraph capture与多次replay中符合预期；
- 每个captured node持有自己的immutable graph package，不会被后续Host launch覆盖；
- 每次replay在AICore/AICPU放行前完整恢复working shared-memory image与runtime arena；
- restore的cache visibility、completion/control reset和错误路径在真实device上能无reset收口；
- graph/context/package的销毁顺序满足“外部quiescence后才释放”。

因此当前HBG的`l1_runtime_supported_impl()`仍必须保持0，Python也必须继续拒绝HBG L1。下一步将以这些未证明项为门禁，逐项审计已有`HbgGraphPlan`/launch blob/execution slot/restore链，然后才在device 1上开启能力并执行eager与ACLGraph replay验收。

> **后续状态：**10.44完成requirements producer/consumer闭环后，10.45按上述门禁开启了HBG L1 capability并进行首轮device 1验证。本段“仍必须保持0”是10.44当时的阶段结论，不代表后续工作树的当前开关状态。

### 10.45 HBG L1首轮device 1闭环：ELF namespace隔离与resident registry代际

#### 10.45.1 本轮开启capability的前提与验收拓扑

10.44已经让PyPTO codegen在最终Host orchestration SO中导出`pypto_orchestration_requirements_v1`，且HBG loader只允许“元数据存在、ABI可识别、无Host tensor content read/write”的program进入L1 graph build。在这个fail-closed前提下，本轮才将A2/A3与A5 HBG runtime的`l1_runtime_supported_impl()`开启，并让Python公共路径可以通过`RunConfig(runtime="host_build_graph")`显式选择HBG。默认runtime与旧L2/L3路径不因此改变。

`tests/st/runtime/l1/test_l1_aclgraph.py`的上板验收扩展为两层：

1. 同一用例分别选择TRB与HBG，执行`context.prepare()`、普通eager warmup、外部device synchronize、独立ccapture stream上的`torch.add -> PyPTO L1 -> torch.mul`、三组输入的连续ACLGraph replay；
2. HBG专项在同一context注册`add`/`mul`两个`@pl.jit` program。两个callable的incore `func_id`都从0开始，但必须保持各自的function-binding snapshot和immutable graph package；两个HBG node在一张ACLGraph中串联，连续replay不得被后一package覆盖。

销毁顺序仍严格保持：先由调用方外部quiesce device，再`graph.reset()`，最后`context.close()`。PyPTO本身不在launch/capture内增加stream/device synchronize。

#### 10.45.2 第一个真实根因：TRB与HBG在标准AICPU scheduler的全局ELF namespace中抢占符号

第一次在同一pytest进程中按“TRB用例在前、HBG用例在后”执行时，HBG AICPU日志报告affinity thread index超出`[0, 1)`。HBG的Host config和invocation bytes本身并没有把worker count设为1，所以不能继续把现象简化成HostArgs copy或cache visibility问题。

对A2/A3 TRB与HBG的实际`libaicpu_kernel.so`执行dynamic-symbol差集后发现，两个SO对外暴露了225个同名C++符号，其中包括：

```text
AicpuExecutor::init(Runtime *)
AicpuExecutor::deinit(Runtime *)
Runtime::Runtime()
Runtime与scheduler的大量普通C++ helper
```

HBG SO内对`AicpuExecutor::init/deinit`的调用还保留`R_AARCH64_JUMP_SLOT`重定位和PLT间接调用。CANN标准AICPU scheduler把这些inner runtime SO放进同一个global dynamic-link namespace；TRB先加载后，HBG中本应绑定自己runtime layout的调用被TRB的同名definition抢占。结果是TRB `Runtime`布局去解释HBG runtime bytes，恰好读出了默认1的affinity配置。

这个根因也解释了为什么单独启动HBG进程时可能不复现：问题不在HBG graph image内容，而在同一AICPU scheduler动态链接namespace中的先加载顺序。

#### 10.45.3 产物级隔离：HBG只导出五个CANN entry

修复没有通过改名225个C++ symbol来堆叠维护成本，而是明确缩小HBG AICPU DSO的动态符号边界：

- `RuntimeBuilder`对每个runtime target显式传入`SIMPLER_RUNTIME_NAME`，避免CMake层不知道当前是TRB还是HBG；
- A2/A3与A5 onboard AICPU CMake只对`host_build_graph`加入version script；
- `hbg_aicpu_exports.map`只保留CANN真正按名字launch的五个entry：

```text
simpler_aicpu_exec
simpler_aicpu_init
simpler_aicpu_l1_hbg_register_execution_slot
simpler_aicpu_l1_hbg_register_callable
simpler_aicpu_l1_hbg_exec
```

- `simpler_aicpu_execute_l1_hbg_platform`、`aicpu_execute_l1_hbg`、`simpler_aicpu_begin_l1_context`和所有runtime C++ implementation都必须是DSO-local；
- TRB不使用该version script，因为TRB的device orchestration SO仍需要解析历史runtime API；sim路径也保持原有`RTLD_LOCAL`机制。

重建A2/A3和A5 HBG AICPU产物后，`readelf --dyn-syms`对每个SO都只列出上述5个GLOBAL/DEFAULT function；三个HBG helper均为`LOCAL`，且不再存在`AicpuExecutor::init/deinit`或`Runtime::Runtime` PLT relocation。这是产物级证据，不是仅对CMake文本做字符串断言。

#### 10.45.4 ELF隔离后的上板结果与第二个真实根因

将新HBG runtime通过GPT工作树自己的`RuntimeBuilder` stage到自己的`runtime/build/lib`后，重新在device 1执行整个ST文件。本次结果为：

```text
2 passed, 1 failed, 3 deselected
```

这个结果非常关键：同一Host进程中先运行TRB，再运行第一个HBG eager/capture/replay已经全部通过，说明ELF symbol preemption已被实际关闭。失败发生在同一pytest进程继续创建第二个HBG L1 context时，且真正报错的device task是prepare阶段的：

```text
simpler_aicpu_l1_hbg_register_execution_slot: rejected status=7
```

`status=7`对应`HbgExecutionSlotRegistryStatus::Conflict`。两次context使用了同一HBG AICPU binary fingerprint，第一context已经按要求外部sync、reset graph并close，但第二context仍然看到第一context的static execution-slot registration。这直接推翻了10.34.2的原假设：

```text
ACL binary handle unload成功
≠
标准AICPU scheduler内部runtime DSO与static registry已经卸载
```

#### 10.45.5 resident registry的context-generation/reset协议

新协议不允许普通per-run launch reset registry，也不允许后一次dynamic build覆盖前一captured node的package。reset只能发生在明确的新L1 context边界：

1. `ChipWorker` 为每个borrowed-L1 context生成非零generation。这一generation使用主机`CLOCK_MONOTONIC`纳秒值，再用进程内atomic保证同一纳秒创建也严格递增；严格不复用保证只覆盖当前支持的单Host进程顺序context。相较每个进程都从1开始，单调时钟会降低resident DSO跨Host进程保留时的代际别名概率，但这只是best-effort，不是v1跨进程保证；
2. `InitArgs`在尾部新增`l1_context_generation`。L2/L3与所有legacy路径保持0，不进入新协议；
3. platform `simpler_aicpu_init`用weak hook调用`begin_l1_context(generation)`。TRB没有该hook，且HBG prelaunch control为0，因此保持原行为；HBG必须提供DSO-local strong hook，缺失时fail-closed；
4. generation与当前resident generation相同时幂等返回，因为同一context可能因DMA workspace配置再次发布`InitArgs`，这种re-latch不得清空已注册callable；
5. 更大的新generation只在上一context已外部quiesce、graph已reset、context已close的v1契约下接受；它先将execution-slot与所有callable entry设为`Publishing`作为fail-closed中间态，清空registration bytes，再release-publish`Empty`；
6. 新context的execution-slot/callable registration task与init task在同一caller stream上有序入队，只能在reset之后发布新trust root。

该协议不是跨进程硬件互斥锁。v1仍然要求同一device同时只有一个live L1 context，并且正式唯一性只覆盖一个Host进程内的顺序context；跨进程并发和跨进程顺序重开都不属于v1保证。如果多个Host进程违反这一契约并发init，新generation可能破坏另一context的resident registry。这一点不得被描述成“PyPTO已支持跨进程并发或可靠跨进程接管”。

#### 10.45.6 无硬件与双架构产物验证

当前generation/reset实现已完成以下无设备验证：

- `test_hbg_execution_slot_registry`新增“reset后acquire为NotReady，新generation registration可发布”反例；
- `test_hbg_callable_registry`新增“reset后相同callable id可属于新context并使用新hash”反例；
- generation判定被提取到common `hbg_context_generation.h`，新反例额外覆盖首次begin、同generation幂等re-latch、stale generation不清状态、新generation清空两个registry、null/zero失败不改owner；
- runtime C++ target全量重建通过；
- `ctest --test-dir tests/ut/cpp/build -LE requires_hardware --output-on-failure -j4`结果为`98/98 passed`；
- A2/A3与A5的TRB/HBG host、AICPU、AICore共12组onboard产物全部重建通过；
- A2/A3与A5 HBG AICPU SO都只导出5个正式entry，generation hook与HBG runtime implementation均是LOCAL；
- `test_host_runtime_abi.py::test_hbg_onboard_aicpu_exports_only_cann_entries`把五entry精确集合固化为ELF回归，`--platform=a2a3`与`--platform=a5`各`1 passed`；该测试只读staged SO，不访问设备；
- `git diff --check`通过。

`ChipWorker` generation从进程内计数器改为主机单调时钟后，又只在GPT工作树的`runtime/.venv`中重建p editable native扩展，编译成功。随后使用显式GPT `PYTHONPATH`执行两组Python反例：

```text
runtime/tests/ut/py/test_runtime_builder.py
runtime/tests/ut/py/test_l1_chip_worker.py
=> 52 passed

tests/ut/runtime/test_l1.py
tests/ut/runtime/test_run_config.py
tests/ut/ir/test_compile_pipeline.py
tests/ut/jit/test_cache.py
tests/ut/jit/test_decorator.py
tests/ut/backend/test_kernel_config_signature.py
tests/ut/codegen/distributed/test_host_orch_distributed.py
=> 373 passed, 2 skipped
```

第一组包含RuntimeBuilder的runtime-name CMake透传、L1 init/close所有权与fresh-worker反例；第二组覆盖Python L1便利API、TRB/HBG runtime选择、compile/JIT/cache key透传、distributed codegen与kernel-config signature。两组均未访问NPU。

双语L1用户页加入`mkdocs.yml`后，`tests/lint/check_docs_nav.py`结果为`137 nav entries cover all 137 pages`。当前GPT虚拟环境没有安装`mkdocs`，因此`mkdocs build --strict`未执行；这是工具缺失，不写成页面build已通过。

这些证据证明新尾字段、weak/strong hook、registry reset和version script在A2/A3、A5的编译与Host状态机上自洽，但最后的“同进程顺序创建两个HBG context”仍需再次device 1验证才能关闭。

#### 10.45.7 当前device状态、归因边界与暂停上板

2026-08-18再次只读检查时，`npu-smi info`同时显示：

```text
device 0 / Phy-ID 14: AICore 100%, HBM 3246 MiB
device 1 / Phy-ID 15: AICore 100%, HBM 2874 MiB
NPU process table: No running processes found
```

Host进程表也没有发现pytest、PyPTO或ACL测试进程；当前shell没有`fuser`命令，因此不伪造device-fd结论。“AICore 100%但没有Host PID”只能说明存在device侧残留、驱动/监控状态或不可见任务，不能证明Grok的另一session正在使用NPU。

device 1曾经运行过本GPT工作树修复前失败的HBG ST，当时AICPU registration早退、hidden AICore已入队的错误闭包尚未完整；因此device 1的当前残留也有可能来自本session自己，不能将其归因给Grok。device 0的来源同样无法凭该读数确定。

遵守“不reset任一设备、device 0不属于GPT session”的边界，本轮在此状态下暂停新的NPU task，只继续Host编译、UT、文档和产物审计。在device 1恢复可验证状态前，不把HBG第二context问题宣称为已上板闭环。

#### 10.45.8 最终Host侧回归、基线失败与并行测试边界

context-generation/reset与ELF隔离完成后，重新从GPT worktree自己的构建目录执行最终无硬件回归。仓库资源限制loader在本机Git 2.25.1上不识别`git rev-parse --path-format=absolute`，会先打印`dirname: unrecognized option '--path-format=absolute ...'`，随后按脚本的unclassified安全默认值给出`PYPTO_BUILD_JOBS=2`和`PYPTO_TEST_JOBS=2`。所有最终CMake构建都显式使用`--parallel 2`；没有根据CPU数量自行放大并发。

runtime C++ UT先重建当前diff，再执行：

```text
ctest --test-dir tests/ut/cpp/build \
  -LE requires_hardware --output-on-failure --parallel 2
=> 98/98 passed
```

随后在同一GPT runtime worktree中依次重建：

```text
a2a3 / onboard / tensormap_and_ringbuffer / host,aicpu,aicore
a2a3 / onboard / host_build_graph          / host,aicpu,aicore
a5    / onboard / tensormap_and_ringbuffer / host,aicpu,aicore
a5    / onboard / host_build_graph          / host,aicpu,aicore
=> 12/12 build passed
```

输出中的`rtStreamCreate`、`rtKernelLaunchWithHandleV2`、`rtsBinaryUnload`等deprecated warning来自仓库既有CANN API调用；本次generation、version script与registry reset没有新增编译error。两架构HBG AICPU的dynamic symbol集合继续精确等于5个CANN entry，内部generation hook不对外导出。

runtime Python UT的正式结果采用串行执行：

```text
runtime/.venv/bin/python -m pytest -q runtime/tests/ut
=> 1096 passed, 15 skipped, 1 warning
```

曾按机器`PYPTO_TEST_JOBS=2`尝试`xdist -n 2`，结果为`1092 passed, 15 skipped, 4 failed`。四个失败都位于`test_l3_l2_orch_comm.py`的sim hierarchical worker用例，child在`_bootstrap_runtime_globals`中因signal 11退出：

```text
test_sim_worker_counter_wait_timeout_does_not_poison_region_and_free_is_idempotent[a2a3sim]
test_sim_worker_counter_wait_timeout_does_not_poison_region_and_free_is_idempotent[a5sim]
test_sim_worker_region_payload_roundtrip[a2a3sim]
test_sim_worker_region_payload_roundtrip[a5sim]
```

这四项脱离xdist后立即`4 passed`，串行全量也全部通过。它们自行fork并加载sim runtime，不能把与xdist worker叠加后的native bootstrap崩溃写成本次L1/HBG回归；最终通过数字只引用上面的串行全量结果。

顶层PyPTO的L1/HBG定向集合在显式GPT `PYTHONPATH`和`xdist -n 2`下结果为：

```text
tests/ut/runtime/test_l1.py
tests/ut/runtime/test_run_config.py
tests/ut/ir/test_compile_pipeline.py
tests/ut/jit/test_cache.py
tests/ut/jit/test_decorator.py
tests/ut/backend/test_kernel_config_signature.py
tests/ut/codegen/distributed/test_host_orch_distributed.py
=> 373 passed, 2 skipped
```

此前还执行过顶层完整`tests/ut -m 'not requires_hardware'`，结果为`9724 passed, 7 skipped, 6 failed`。六项脱离完整suite后仍能稳定复现；对相关实现和测试文件执行`git diff $(git merge-base main HEAD)..HEAD`为空，`git blame`也表明它们来自本分支fork之前的基线提交，而不是L1/HBG diff：

1. `TestLoopCarryRoundtrip::test_signature_memref_base_prints_as_a_string`：当前基线IR打印结果没有以`seed`开头的参数行，测试在`next(...)`处`StopIteration`；
2. `TestAsyncDispatchHandle::test_interrupted_native_handle_publication_keeps_frame_until_close`；
3. `TestPreparedSwimlaneTwoPass::test_onboard_reuses_worker_for_graph_then_clean_timing`；
4. `TestPerCallValidation::test_accepts_device_tensor`；
5. `TestPerCallValidation::test_scalar_param_forwarded_as_is`；
6. `TestMultiProgram::test_shared_device_tensor_across_programs`。

后五项的共同原因是基线`DistributedWorker`已经以keyword传递`domain_provider=None`，但旧测试的mock side-effect只接受位置参数，均报`unexpected keyword argument 'domain_provider'`。本次任务不修改`distributed_runner.py`或这些测试，不能为制造“全量绿”而顺手改变无关基线行为；交付时必须同时给出完整suite数字与变更相关定向suite数字。

静态检查结果：

- 顶层18个本次变更Python文件以Ruff 0.15.14的isolated等价配置执行lint与format check，全部通过；
- runtime 3个变更Python文件lint通过；本机Ruff 0.15.14会重排`test_runtime_builder.py`中多处基线lambda括号，而仓库钉的是0.14.8，因此没有接受这份无关整文件format churn；
- runtime 15个本次变更C++/header文件全部通过`clang-format --dry-run --Werror`；
- `tests/lint/check_docs_nav.py`结果为`137 nav entries cover all 137 pages`；
- `git diff --check`在top与nested runtime均通过；
- 新增英文L1页147行、中文页134行，均低于500行文档上限；用户明确要求的过程记录属于本任务特例，继续保留完整上下文，不按普通用户文档压缩。

最后再次只读核对设备时，`npu-smi`仍显示physical id 14与15的AICore均为100%，HBM分别约3245 MiB与2874 MiB，但NPU process table为空。遍历所有可见`/proc/<pid>/fd`也没有进程持有`/dev/davinci14`、`/dev/davinci15`、`/dev/davinci_manager`或`/dev/devmm_svm`。Host上存在长期`grok`会话进程，但没有PyPTO/pytest子进程或device fd；因此只能确认“没有可见Host使用者”，不能把device侧残留归因给Grok。当前仍不reset device 0或device 1，也不在AICore 100%的残留状态上提交新的上板任务。

#### 10.45.9 TRB/HBG产物目录的第二层隔离与最新回归

在准备最终提交时，独立集成审查发现了一条与“runtime已经进入JIT cache key”不同的文件系统别名链。原实现的内存cache确实会把TRB和HBG分成两个`CacheKey`，但cache miss默认交给`ir.compile()`创建的目录名只精确到秒：

```text
build_output/<program_name>_<YYYYmmdd_HHMMSS>
```

同一个JIT specialization在一秒内先编译TRB、再编译HBG时，两个`CompiledProgram`可能仍指向同一路径。显式复用`RunConfig.save_kernels_dir`时则必然同路径。`CompiledProgram`只保存输出目录，`chip_callable/runtime_name/runtime_config`在首次访问时才延迟执行`compile_and_assemble`；所以第二次编译覆盖`kernel_config.py`和二进制后，先返回的TRB对象也可能被加载成HBG。这不是“cache key少了runtime”，而是两个正确的cache entry共享了错误的artifact owner。

最终采用三层边界，避免改变单次显式保存目录的历史布局：

1. `ir.compile(output_dir=None)`把时间戳扩展到微秒，并通过`os.mkdir`原子重试领取目录；即使多个Host进程采样到相同clock tick，也只有一个能取得无后缀路径，其余使用递增collision suffix。没有采用`tempfile.mkdtemp`，因为它会把目录固定成0700，改变原有umask与同组读取行为；
2. 每个输出目录在pass/codegen写入前用`O_CREAT|O_EXCL`原子创建`.pypto_runtime_owner`，内容是规范runtime名。任何`@pl.jit`对象、直接`ir.compile`或其他Host进程，只要请求另一runtime，就会在改写artifact前fail-fast。同runtime显式重编译继续保持历史caller-owned覆盖语义；
3. 单个`JITFunction`额外保存“显式绝对输出路径→CacheKey”的成功编译映射。同一JIT函数即使runtime相同，只要shape、compile option或其他key维度不同，也不能覆盖仍被cache中`CompiledProgram`延迟拥有的目录。失败的compile不会发布这一内存owner，允许同一key修复后重试。

对应新增反例覆盖：固定同一时间戳的两次默认compile得到两个不同目录；直接TRB compile后以相同显式目录请求HBG在第二次`generate`前失败，而同runtime重编译仍允许；同一`JITFunction`用一个显式目录先产生TRB key、再产生HBG key时只发生第一次compile。三项定向结果为`3 passed`。

加入这一隔离后重新执行当前变更相关集合：

```text
test_compile_pipeline.py
test_compiled_program.py
test_jit_compile_extraction.py
test_cache.py
test_decorator.py
test_run_config.py
test_l1.py
=> 420 passed
```

随后重新执行完整顶层无硬件集合：

```text
runtime/.venv/bin/python -m pytest -q -n 2 \
  -m 'not requires_hardware' tests/ut
=> 9731 passed, 3 skipped, 6 failed
```

六个失败与10.45.8逐项列出的基线失败完全相同：一个IR printer
`StopIteration`和五个旧`DistributedWorker` mock不接受
`domain_provider=None`关键字参数；本轮新增的runtime owner、JIT cache、L1和HBG
用例全部通过。与10.45.8较早的`9724 passed, 7 skipped`相比，通过数增加来自本轮
新增回归，skip数变化来自当前`-m 'not requires_hardware'`收集结果；不能把完整suite
写成全绿，也没有为了消除无关红项去修改基线测试。

runtime Python串行全量重新执行为：

```text
runtime/tests/ut
=> 1103 passed, 8 skipped, 14 warnings
```

runtime C++当前树全量无硬件结果仍为`98/98 passed`；其中L1/HBG定向19项全部通过。A2/A3与A5的HBG staged AICPU SO再次用`readelf --dyn-syms`核对，GLOBAL/DEFAULT定义精确等于五个CANN entry，内部`AicpuExecutor`、`Runtime`、generation hook和HBG trampoline没有动态重定位泄漏。十二套CMake cache的`CMAKE_HOME_DIRECTORY`全部指向`pto/gpt_pypto/runtime`，且`SIMPLER_RUNTIME_NAME`分别精确为TRB或HBG；这同时证明当前产物没有从Grok工作树取源码。

本节也收紧generation的事实边界：进程内atomic是v1支持范围内的严格唯一性来源；`CLOCK_MONOTONIC`只让同一Host上跨进程顺序重开时的generation更不容易与resident DSO旧值别名，不构成跨进程lease。v1不保证跨进程并发，也不保证可靠跨进程顺序接管。

最后一次只读设备检查仍显示device 0/physical 14与device 1/physical 15的AICore均为100%，HBM约3246 MiB与2874 MiB，NPU process table为空；Host仅有长期Grok会话本体，没有PyPTO/pytest子进程。因而第二个HBG context与双callable replay的generation修复仍没有最终device 1复验，本节不能把HBG第二阶段状态升级为“全部完成”。

#### 10.45.10 GPT工作树隔离与runtime阶段提交

本轮所有实现、构建和测试都在独立工作树
`/mnt/workspace/inductor/pto/gpt_pypto`中完成；Grok会话使用的是
`/mnt/workspace/inductor/pto/pypto`。两套工作树拥有各自的Git index、顶层分支、
nested runtime分支、虚拟环境editable指向和CMake cache。GPT验证进程实际导入的是：

```text
/mnt/workspace/inductor/pto/gpt_pypto/python/pypto/__init__.py
/mnt/workspace/inductor/pto/gpt_pypto/runtime/python/simpler/__init__.py
```

十二套runtime CMake cache的`CMAKE_HOME_DIRECTORY`也全部指向GPT工作树；因此
PyPTO、simpler源码和构建产物不会因为Grok在另一会话修改同名文件而串用。两边仍共享
Git object database、系统CANN安装和物理NPU，这也是设备使用必须单独协调、但文件修改
不需要互相抢锁的边界。

runtime/simpler的本阶段改动已经形成独立提交：

```text
f8b9056a Fix: 收口HBG L1运行时代际与ELF隔离
```

该提交包含HBG AICPU五入口version script、单Host进程内非零递增context
generation、resident execution-slot/callable registry的新代reset、A2/A3与A5
capability启用及对应C++/Python/产物ABI反例。提交前验证范围为runtime Python
`1103 passed, 8 skipped`、C++非硬件`98/98 passed`，以及两架构两runtime三类
host/AICPU/AICore共十二套构建。提交信息同样明确说明：这些结果不等价于device 1
上的第二个HBG context已经复验；该项与双callable capture/replay仍是下一次上板的
硬门槛。

顶层PyPTO的runtime选择、artifact owner、L1/HBG Python入口、ST与用户文档形成
后续独立提交：

```text
beb06bd4 Add: 接通HBG L1编译选择与ACLGraph入口
```

该提交将nested runtime指针固定到`f8b9056a`，所以顶层提交不能在未同步对应runtime
提交的环境中单独验证。提交后再次只读执行`npu-smi info -m`与`npu-smi info`：
device 0/physical 14仍为AICore 100%、HBM 3246 MiB，device 1/physical 15仍为
AICore 100%、HBM 2874 MiB，process table仍为空。该证据只能排除`npu-smi`可见的
活跃Host进程，不能证明device已经quiescent；因此没有在此状态下执行第二个HBG
context、双callable ACLGraph replay或任何reset。

### 10.46 完成度审计后的扩展上板矩阵

#### 10.46.1 为什么已有两个ST仍不足以宣称完成

按设计文档第13、15节、附录I和N.10逐项反查当前测试后，原有
`test_l1_aclgraph.py`只能直接证明以下事实：单个add callable的eager/warmup、
PyTorch pre-op -> L1 -> post-op顺序、固定地址三次replay，以及一个graph内两个
HBG callable的package/函数表隔离。它没有形成同等级device证据的项目包括：

- runtime scalar与tensor地址在多次Host异步调用中的独立snapshot；
- 多output方向、同一program内多个child kernel和`pl.create_tensor`内部workspace；
- 两个独立captured graph长期同时存活时的地址/scalar package隔离；
- graph reset/close后，在同一Host进程中创建第二个不同HBG context并重新使用
  `callable_id=0`/`func_id=0`；
- N.10中的large HostArgs、allocator环绕、cache多线、memory accounting和no-reset
  fault injection。

因此本轮没有把“Host UT很多”换算成“上板矩阵已完成”，而是新增独立文件
`tests/st/runtime/l1/test_l1_extended_matrix.py`，把无需production test hook的四组
场景先变成可直接执行的正式ST；剩余专用probe继续显式保留。

#### 10.46.2 新增四组正式ST

1. `test_l1_async_tensor_and_scalar_snapshots_do_not_alias`同时参数化TRB/HBG。先完成
   prepare/warmup，再连续enqueue四组不同input/output地址和FP32 scalar，中间不做
   任何同步；所有临时queue-call/HostArgs容器离开作用域后才统一device synchronize并
   逐项验值。这对应ST-E-006/ST-E-007，不通过保留Python list假装参数仍存活。
2. `test_l1_multi_output_multi_child_workspace_aclgraph`在同一个context注册两个program：
   第一个child同时写sum/diff两个output，第二个program由两个`@pl.jit.incore` child和
   一块`pl.create_tensor` intermediate组成。graph捕获两个连续L1 node，多次replay后
   验证`(lhs + rhs + lhs - rhs) * 2 == 4 * lhs`。它覆盖ST-E-003/004/005与
   ST-G-005/006/007，并同时验证多output返回值仍是调用方原tensor身份。
3. `test_hbg_two_graphs_retain_distinct_addresses_and_scalars`让graph A/B同时存活，使用
   同一callable但不同input/output地址和scalar，按A/B/A/B交替replay。该case直接针对
   “所有captured node退化为最后一份Host graph blob”的错误实现，对应N.10.3和
   N.10.4中可由普通数值ST覆盖的部分。
4. `test_hbg_sequential_context_generation_resets_resident_registries`在一个pytest进程内
   先用scalar callable创建、warmup、capture/replay、reset并close第一context，再用
   不同multi-child callable创建第二context并完成同一流程。两份program在各自context
   都从callable/func id 0开始，专门复现10.45首次上板发现的resident registry冲突。

所有test的finally路径都保持统一顺序：调用方device synchronize，逆序reset所有graph，
最后`context.close()`；graph-bound tensor局部变量在该顺序完成前始终保持强引用。

#### 10.46.3 Host侧验证结果

新增文件没有在当前残留device状态下执行测试体，只完成以下无NPU验证：

- GPT editable `_task_interface`最初仍带`80615b1e` build stamp，Python按设计主动拒绝
  与`f8b9056a`源码混用；仅在`gpt_pypto/runtime/.venv`内以并发2重新editable build后，
  extension的`__build_commit__`精确为`f8b9056ae769...`；
- isolated Ruff lint与format均通过，文件379行，`git diff --check`通过；
- `--collect-only --platform=a2a3 --device=1`与A5对应命令各选择6项、deselect 6项，
  两平台合计形成12个待上板case；collect只导入模块，没有创建NPU tensor或context；
- A2/A3与A5分别在独立Host进程lower三个program：scalar和multi-output均得到
  orchestration + 1个child，multi-child得到orchestration + 2个child；两平台全部通过；
- 显式使用仓库PTOAS后，A2/A3与A5各自对TRB/HBG编译scalar、multi-output、
  multi-child，共12份完整Host artifact全部成功。每份都生成`kernel_config.py`并写入
  请求的runtime；orchestration均导出`pypto_orchestration_requirements_v1`，
  multi-child artifact同时含`_child_add`与`_child_scale`两份内核产物。

这组证据证明新增矩阵已经通过Parser、passes、双架构codegen、PTOAS和runtime manifest
边界，但不证明任何CANN launch/capture/replay行为。最后一次检查device 1仍为AICore
100%、HBM 2874 MiB，`npu-smi` process table与device fd扫描均为空；本轮没有执行
ACL/NPU task或reset。

#### 10.46.4 仍不能被普通数值ST替代的门槛

扩展矩阵完成后，第二阶段仍至少需要独立证明：64 KiB到真实最大image的HostArgs完整
copy与失败边界、capture后args allocator压力/环绕、placeholder实际device地址、canonical
plan不被原地patch、restore首中尾cache-line可见性、graph destroy前后memory accounting，
以及N.10.8列出的slot/callable/blob/affinity/KernelArgs/physical-core/scheduler阶段故障注入。
这些case需要runtime测试入口、trace或fault hook；不能因为普通eager/replay数值正确就勾选。

### 10.47 N.10专用WithHostArgs/placeholder探针

#### 10.47.1 为什么继续增加独立探针，而不是扩大普通PyTorch ST

10.46新增的正式ST可以从最终数值反推“某个captured node大概率保留了自己的地址、
scalar和HBG package”，但它仍然无法回答CANN参数层的几个关键事实：

1. `aclrtPlaceHolderInfo`修补后的pointer是否真的等于本次runtime-owned task-args基址加
   `dataOffset`，还是恰好指向了另一份仍存活的Host/device buffer；
2. `aclrtLaunchKernelWithHostArgs`返回时是否已经完成Host bytes snapshot，还是到
   `CaptureEnd`、graph instantiate或第一次replay才延迟读取调用方scratch；
3. variable args经过regular allocator和large/max allocator时是否完整复制到tail，还是
   只复制了header后仍能让简单数值case看起来正确；
4. captured graph持有的args allocation是否会被后续大量WithHostArgs task环绕复用；
5. graph A/B同时存活时，runtime-owned source是否分别保留，而不是二者都指向最后一次
   Host scratch；
6. capture、args allocator压力、replay和graph destroy前后的HBM变化是什么，是否存在
   随replay单调增长的明显泄漏。

这些问题如果直接让production HBG parser回答，会形成“实现用自己的validator证明自己
正确”的循环证据。因此runtime提交`3575f60b`新增
`tests/st/l1/host_args_probe`，探针的Host blob、AICPU parser和result ABI都独立于
`HbgGraphPlan/HbgLaunchBlob/restore_hbg_launch_blob`；它只复用CANN真实
`aclrtLaunchKernelWithHostArgs`和AICPU动态加载通道。

并行Grok工作树已有的Phase-0 probe被作为只读参考：它已经证明一套custom AICPU inner
SO bootstrap方式可用，并覆盖caller/hidden双stream event fork/join，但只传固定小参数、
零placeholder，且退出时调用`aclrtResetDevice`。GPT没有修改或构建该目录，也没有直接复制
其“reset-owning standalone程序”作为本阶段答案；新的probe只保留必要的dispatcher bootstrap
事实，专门改成variable HostArgs owner验证，强制显式device并彻底省略reset。

#### 10.47.2 冻结的probe ABI与三份payload

`common/host_args_probe_abi.h`定义两份固定ABI：

```text
HostArgsProbeHeader = 128 bytes, align 8
HostArgsProbeResult = 152 bytes, align 8
payload_count       = 3
```

header包含magic、major/minor、`header_size/total_size`、invocation id、外部result地址、
三个待patch的`payload_addr`、三个offset/size、每个payload的expected checksum以及首/中/尾
expected byte。`payload_addr[0]`固定从offset 40开始，三个8-byte pointer field连续排列；
Host为每一项分别构造一个`aclrtPlaceHolderInfo`，而不是只用production HBG当前的单pointer
布局。三个payload覆盖header后的整个args image，最后一个payload的tail精确落在
`argsSize - 1`，所以device若只复制头部或截断尾部，不能只靠header数值蒙混通过。

payload按invocation id、region id和byte index生成确定性pattern；Host把每一区域完整FNV-1a
checksum和首/中/尾样本写进header。AICPU不重新推导Host pattern，而是通过patched pointer
完整遍历实际device bytes，独立产生observed checksum与样本。这种分工同时发现pointer
patch错误、source串包和tail截断。

AICPU入口不把CANN传入的`void *args`直接cast为`HostArgsProbeHeader *`。它先用byte-copy将
固定128-byte prefix复制到`alignas(16)`局部对象，再进行typed access；payload仍以
`uint8_t *`逐字节读取。第一次真实交叉编译曾把局部对象写成`alignas(64)`，HCC明确报
“requested alignment 64 is larger than 16”；这反向证明probe不能擅自要求AICPU栈支持64
字节对齐。最终16字节局部对齐高于ABI所需8字节，同时不对CANN task-args base作任何
64字节假设。

result记录实际`args_base`及`args_base % 64`、三个observed/expected address、checksum和
样本，并用bit map区分header、region、placeholder、checksum与sample错误。AICPU在成功
取得result地址后始终返回0，把精确诊断留在device result中；否则泛化为一个AICPU非零
返回会丢掉“究竟哪个placeholder或tail失败”的原始证据。

#### 10.47.3 Host侧snapshot时点与canonical隔离

每个invocation由两层Host owner组成：

```text
Invocation::canonical  -- 生成后只读，保存完整hash
       |
       +-- deep copy --> writable scratch -- 唯一传给CANN的pointer
```

eager和capture都只把scratch交给CANN。API返回后立即执行以下顺序：

1. 读取scratch中三个pointer slot，记录CANN是否对Host memory做了可见的原地patch；
2. 将scratch全部覆盖成`0xa5`；
3. 释放其allocation；
4. 立即申请同尺寸Host vector并填`0x5a`，主动制造地址复用机会；
5. 重新hash canonical，任何被误传给CANN的第1层修改都直接失败。

capture路径有意把第2～4步放在`aclrtLaunchKernelWithHostArgs`返回之后、
`aclmdlRICaptureEnd`之前。如果CANN只是保存Host pointer并推迟到CaptureEnd读取，graph从
一开始就会得到poison或reuse bytes；probe不会因为把scratch多保留到CaptureEnd而给错误
实现额外生命周期。

result buffer在capture前使用同步memset完成，确保capture stream没有一条“外部clear task”
悬在图边界前。replay时则在同一stream上按`clear result -> execute graph -> synchronize`
排序，既不改变captured graph，又能验证每次replay真的重新执行AICPU node。

#### 10.47.4 可配置的device矩阵

Host executable没有device默认值，只有显式`--device=<id>`才会调用`aclInit/setDevice`。
默认参数为：

```text
eager sizes      = 64 KiB, 1 MiB, 16 MiB, 64 MiB
graph A/B sizes  = 1 MiB, 16 MiB
replays          = graph A和B各100次，A/B交替
pressure         = 512次、每次64 KiB的额外WithHostArgs launch
```

每个eager size都在launch返回后销毁scratch、外部stream sync后读取result，并记录launch
返回时延、task完成总时延、Host pointer slot和实际device args base alignment。graph A/B
使用不同invocation id、不同runtime-owned args allocation与不同external result buffer；捕获
完成后先插入pressure tasks，随后按A/B/A/B顺序各replay 100次。pressure循环同样每次都
构造不同payload、立即poison/free source，并在失败时报告“实际成功launch数+ACL error”，
不把2048或任何内部常量固化成产品规格。

probe在以下阶段调用`aclrtGetMemInfo(ACL_HBM_MEM)`并只记录、不做脆弱的精确delta断言：

- context建立后；
- capture前；
- 两个graph capture后；
- args allocator压力后；
- 交替replay后；
- graph destroy并外部quiescence后；
- 两个external result buffer释放后。

这组数据将用于区分runtime-owned captured args、外部result和binary/context常驻成本；CANN
allocator可能缓存内存，所以“free bytes必须精确回到某个值”不是预设判据，必须结合重复
run和增长趋势解释。

CLI允许覆盖所有size/count。README另列出64 MiB captured graph与2048次pressure的独立
命令，但明确写明2048只是一处测量点，不是CANN/PyPTO规格。真实HBG regular/max image仍
要由production HBG ST给出实际size后分别代入；generic 64 MiB通过不能替代“真实最大图”
case。

#### 10.47.5 安全与清理边界

这个程序虽然是standalone ACL probe，仍遵守本项目的设备约束：

- `Options.device_id`初始为-1，缺少`--device`直接打印usage并退出，不存在隐式device0；
- README明确本分支只在调度且空闲的device1运行，device0留给并行会话；
- 源码没有`aclrtResetDevice`、stream/device内部reset或capture query；
- 所有graph replay完成并synchronize后才destroy graph；
- model、result、AICPU binary、stream、context按反向owner顺序释放；
- `aclFinalize`只结束该standalone进程自己的ACL全局状态，不被解释成device reset。

probe本身允许显式stream synchronize，因为它是验证工具，需要读取result与划分memory
accounting阶段；这不改变production L1“launch内部禁止sync”的验收。正式trace仍要单独
证明PyPTO op body没有sync。

#### 10.47.6 无设备构建和反例自检

提交前使用GPT runtime工作树并先加载测试环境，执行：

```text
source ../.claude/skills/testing/load-env.sh
tests/st/l1/host_args_probe/build.sh /tmp/gpt_host_args_probe_build
```

构建脚本以HCC AArch64 cross compiler生成
`libhost_args_probe_aicpu.so`，再生成ACL Host executable。两边都使用C++17、
`-Wall -Wextra -Werror`。`readelf -Ws`确认SO精确导出
`simpler_host_args_probe_init/run`两个GLOBAL/DEFAULT entry。

同一build还把AICPU parser直接链接进`host_args_probe_self_test`。自检分配4097 bytes，
有意用`storage.data() + 1`作为args base，手工模拟三个runtime-patched pointer并调用同一
`simpler_host_args_probe_run`：

1. 完整三region image得到status 0，result中的args地址保持奇数；
2. 第二个pointer加1后不访问错误地址，而是得到
   `HOST_ARGS_PROBE_BAD_PLACEHOLDER_1`。

最终输出：

```text
HOST_ARGS_PROBE_SELF_TEST PASS unaligned_prefix_and_placeholder_diagnostic
built /tmp/gpt_host_args_probe_build/host_args_probe
```

编译过程还实际发现并修正两项ABI问题：最初把result误算为160 bytes，编译器证明真实冻结
布局为152 bytes；最初局部prefix要求64-byte栈对齐，HCC只支持到16并在`-Werror`下拒绝。
这些失败没有被绕过，而是按真实ABI修正static assertion与parser。

提交前hooks结果：header/license、English-only、large-file、EOF/whitespace、clang-format、
clang-tidy、cpplint和Markdown lint全部通过；`clang-format --dry-run --Werror`、`bash -n`
与`git diff --check`也通过。本机没有安装独立`shellcheck`命令，因此没有把“shellcheck未
执行”写成通过；仓库pre-commit对该脚本的已有检查均已成功。

runtime阶段提交为：

```text
3575f60b Test: 增加L1 HostArgs与placeholder上板探针
eedfdc90 Test: 将HostArgs探针纳入常规C++回归
```

第二个提交将完全相同的AICPU parser源码和Host self-test接入
`tests/ut/cpp/CMakeLists.txt`的`no_hardware`标签；不是重新实现一份mock parser。定向
`test_host_args_probe_parser`通过后，重新构建并执行完整
`ctest -LE requires_hardware`，结果从此前98项增加为`99/99 passed`。这样后续任何probe
ABI、编译属性或parser行为回归都会进入常规C++门禁，而不依赖开发者记得手工运行probe
build脚本。

#### 10.47.7 当前证据边界与下一次device1动作

本节只把“缺少工具”推进成“工具已编译、自检并可在显式device上执行”，没有产生任何
CANN device行为结论。最后检查device1时仍为AICore 100%、HBM 2874 MiB，NPU process
table为空；无法证明quiescent，所以没有运行probe，也没有运行10.46扩展ST，更没有reset。

因此N.10.1～N.10.3的device checkbox保持原样未勾选。device1恢复空闲后，执行顺序应为：

1. 先运行默认probe矩阵，保存每个size的ACL error、args base alignment、checksum、
   capture/replay时延和全部memory sample；
2. 再单独运行64 MiB captured graph与更高pressure count，不能依赖固定2048；
3. 使用实际HBG regular/max package size重复对应probe point；
4. 运行10.46的TRB/HBG正式扩展ST，特别是第二context generation与双callable；
5. 最后进入N.10.4～N.10.8的restore poison、cache多线与no-reset fault hook矩阵。

当前probe不验证HBG leader restore、working slot、scheduler completion gate或hidden AICore
CANCEL；这些仍必须由production HBG路径和专用fault injection证明，不能因为通用
WithHostArgs probe未来全绿就省略。

#### 10.47.8 HBG跨cache-line恢复与失败后重试反例

在建设device fault hook之前，先复核现有`test_hbg_launch_blob`能够证明什么。原测试已经
覆盖“每次调用restore都会复制两份region”，但其SM/arena样本较小，也没有把首、中、尾
分布到多条cache line；restore失败case只检查commit保持sentinel，没有继续证明被部分修改的
mutable slot可以由下一次完整restore修复。这两处如果没有反例，未来把restore错误地优化成
只更新header或只在首次执行复制，也可能让已有UT继续通过。

runtime提交`620f1df4`增加独立`MultiLineRestoreHarness`，使用：

```text
pristine SM size       = 5 * 64 + 13 bytes
pristine arena size    = 7 * 64 + 31 bytes
restore regions        = SM + runtime arena
连续restore次数       = 2
每个region破坏位置    = first / middle / tail
预期copy/publish次数   = 4 / 4
```

两份source由不同的逐字节确定性pattern填充，而不是只有头尾canary。第一次restore后逐字节比较
整个working SM和arena；随后分别翻转两个region的首、中、尾字节，再执行第二次restore并再次
逐字节比较。callback还记录每次publish的size，结果必须精确为
`{sm_size, arena_size, sm_size, arena_size}`。这证明Host restore算法不会因为跨越cache line、
非64-byte整数长度或第二次调用而缩短region。

失败事务case也被扩展：先分别注入copy和publish失败，确认`HbgRestoreCommit`始终保持调用前
sentinel；然后清除故障，把整个working SM/arena覆盖为新pattern，再次执行完整restore。只有
这一次成功后才能看到全部pristine bytes、`plan_generation=51`和正确`plan_hash`。这里允许
失败尝试已经部分改写mutable slot，但绝不允许把它发布为ready generation；后续恢复必须
覆盖整份source，不能依赖失败前遗留内容。

本次验证使用GPT runtime工作树、并发2执行：

```text
test_hbg_launch_blob                         1/1 passed
runtime C++ ctest -LE requires_hardware     99/99 passed
pre-commit（该C++文件）                    all passed
git diff --check                            passed
```

这组结果只证明`restore_hbg_launch_blob`的Host算法、region长度与commit事务边界。它没有让真实
AICPU leader执行copy/publish，也没有证明A2/A3或A5的cache clean/invalidate、peer acquire、
AICore descriptor读取和ACLGraph第二次replay可见性。因此设计文档N.10.4/N.10.5的device项
没有被勾选；下一步仍需test-build-only fault hook和device1的production HBG replay证据。

### 10.48 device1 HBG扩展矩阵与large HostArgs正式探针结果

#### 10.48.1 上板前先把GPT Python/native环境重新收敛

runtime提交`620f1df4`后，GPT venv最初仍加载由`f8b9056a`源码构建的`_task_interface`。
`simpler.task_interface`按设计比较build commit并拒绝导入，没有让新Python驱动旧C++布局。
先在`gpt_pypto/runtime/.venv`重建runtime editable，最终native stamp精确为：

```text
_task_interface.__build_commit__ = 620f1df4b3149e7a8a85685b942b29131f8c551b
```

第一次重建顶层adapter时没有禁用user-site，导致build阶段看到系统Torch 2.7，而pytest运行时
看到user-site Torch/Torch-NPU 2.12。`L1Context`在native init前比较adapter build version并
明确拒绝。第二次尝试修正adapter版本后，又因为shell没有固定PTOAS，`@pl.jit`自动生成
`skip_ptoas=True`的compile-only artifact；`CompiledProgram`在native init前检查缺失的
`kernel_config.py`并拒绝执行。两次都没有启动PyPTO AICPU/AICore，但test在进入context前已
创建普通NPU tensor，因此确实发生过device1的Torch allocation/fill，不能写成“完全没有NPU
行为”。

最终恢复10.43已经使用过的隔离环境：

```text
PYTHONNOUSERSITE=1
PYTHONSAFEPATH=1
PYTHONPATH=gpt_pypto/python:gpt_pypto/runtime/python:gpt_pypto/runtime:gpt_pypto/tests/st
PTOAS_ROOT unset
PATH first = PTOAS/build-v0.57-llvm21-cann9.2-clean/tools/ptoas
torch       = 2.7.1+cpu
torch_npu   = 2.7.1.post4
adapter     = built against 2.7.1+cpu / 2.7.1.post4
ptoas       = 0.57
```

该环境的`pypto`、`simpler`、`_task_interface`和`pypto._torch_npu_l1`全部来自GPT工作目录
或GPT venv；没有加载Grok editable package或build产物。

#### 10.48.2 production HBG扩展ST结果

先单独运行最关键的
`test_hbg_sequential_context_generation_resets_resident_registries[a2a3]`。第一context注册
scalar callable，在独立capture stream完成warmup、capture和replay后按“外部sync -> graph
reset -> context close”销毁；第二context在同一pytest进程里注册不同的two-child callable，
二者都从`callable_id=0`和callable-local `func_id=0`开始。结果：

```text
1 passed, 11 deselected in 9.82s
```

这直接复验了10.45首次暴露的resident registry代际问题；第二context不再把新callable视为
旧generation冲突，也没有依赖device reset。

随后一次执行全部HBG扩展case：

```text
test_l1_async_tensor_and_scalar_snapshots_do_not_alias[host_build_graph]
test_l1_multi_output_multi_child_workspace_aclgraph[host_build_graph]
test_hbg_two_graphs_retain_distinct_addresses_and_scalars
test_hbg_sequential_context_generation_resets_resident_registries

4 passed, 8 deselected in 32.21s
```

这四项共同证明：连续四次Host异步调用的tensor地址与FP32 scalar snapshot不串包；多output
方向、两个child和`pl.create_tensor`内部workspace可进入同一个captured operator序列；两张
同时存活的graph按A/B/A/B replay仍保留各自地址/scalar/package；同一进程第二context可以
重新注册不同binary。所有case结束都由调用方先quiesce，再reset graph、close context。

#### 10.48.3 探针首次507018不是large HostArgs限制

独立probe第一次运行64 KiB eager时，`aclrtLaunchKernelWithHostArgs`成功返回，但stream sync
得到507018。device日志给出更精确的根因：

```text
dispatcher wrote:
  simpler_inner_04a8002c680d06ad_1.so
descriptor requested:
  simpler_host_args_probe_04a8002c680d06ad_1.so
AICPU result:
  11002, open so failed
```

共享dispatcher的文件名协议描述被bootstrap的inner DSO内容，而不是其中导出的probe symbol；
它固定使用`simpler_inner_<fingerprint>_<device>.so`。probe错误地为descriptor创造了另一前缀，
因此设备loader根本没有进入probe parser，507018不能用来推断64 KiB参数不受支持。

runtime提交`6a5f70a9`抽出`format_dispatcher_inner_so_basename`，让descriptor复用dispatcher
真实命名；pure-Host self-test固定断言
`simpler_inner_0123456789abcdef_1.so`并拒绝负device id。修复后的build/self-test、常规
`test_host_args_probe_parser`与pre-commit全部通过。更重要的是，发生11002后没有reset设备，
下一进程的小矩阵和后续全部正式矩阵仍成功；这只证明loader错误后的device/context可重新
建立，不等价于HBG hidden AICore no-reset故障矩阵。

#### 10.48.4 默认large HostArgs与allocator压力矩阵

默认矩阵的四个eager payload全部在launch返回后立即读取scratch pointer slot、把scratch
poison为`0xa5`、释放并申请同尺寸`0x5a`复用buffer；外部stream同步后AICPU仍对runtime-owned
bytes计算出完整checksum和首/中/尾样本：

| args size | launch返回耗时 | task完成总耗时 | 实际args base | mod 64 |
| --- | ---: | ---: | --- | ---: |
| 64 KiB | 143 us | 1094 us | `0x12c0c001b000` | 0 |
| 1 MiB | 1790 us | 3134 us | `0x12c0c002b000` | 0 |
| 16 MiB | 30065 us | 49006 us | `0x12c081200000` | 0 |
| 64 MiB | 114349 us | 190851 us | `0x12c082200000` | 0 |

三个Host pointer slot在API返回时已经分别变成`args_base + payload_offset[i]`，而canonical
完整hash始终不变，证明CANN只patch本次writable scratch。随后连续发射512个64 KiB
WithHostArgs task，每个task的Host source都立即poison/free/reuse；tail invocation完整通过。

graph A为1 MiB、graph B为16 MiB，capture返回后Host scratch同样立即销毁。经过上述512次
压力后，两张graph各100次按A/B交替replay全部通过；每次观察到的args base固定为各自独立
地址：

```text
graph A args = 0x12c0c0022000, replay约2.7～3.5 ms
graph B args = 0x12c081200000, replay约42.5～43.1 ms
```

HBM采样为：

| 阶段 | used bytes |
| --- | ---: |
| context建立后 | 152674304 |
| eager完成、capture前 | 236560384 |
| 两graph capture后 | 236560384 |
| 512次压力后 | 173645824 |
| 200次交替replay后 | 177410048 |
| graph destroy后 | 177410048 |
| result释放后 | 177410048 |

小allocation由CANN allocator缓存，所以不能要求graph destroy后free bytes精确回到首样本；
关键结论是200次replay期间没有随次数单调增长，也没有因512次新task覆盖旧graph payload。

#### 10.48.5 64 MiB captured graph与2048次压力

加强矩阵把graph A扩大到64 MiB、graph B保持1 MiB，并在capture后连续发射2048个64 KiB
task。两个graph仍各100次A/B交替replay全部通过：

```text
64 MiB capture = 30966 us, replay约170.7～171.7 ms
1 MiB  capture =  1954 us, replay约  2.7～  3.3 ms
allocator pressure = 2048 successful launches
```

对应HBM为：

| 阶段 | used bytes |
| --- | ---: |
| context建立后 | 152588288 |
| 两graph capture后 | 219697152 |
| 2048次压力后 | 223891456 |
| 200次交替replay后 | 227201024 |
| graph destroy后 | 160092160 |
| result释放后 | 160092160 |

graph destroy后used精确下降`67108864` bytes，即64 MiB captured args allocation在外部quiescence
后随graph owner释放；剩余约7.5 MiB相对context首样本的差值可由runtime/allocator缓存解释，
不能假装为零。2048只是本次pressure point，不进入PyPTO产品限制。

#### 10.48.6 本轮可以勾选和仍不能勾选的边界

设计文档N.10已经按证据勾选placeholder地址、三pointer、Host scratch即时销毁、canonical
immutability、四个成功size的full checksum/tail、512/2048压力、双graph各100次replay、
64 MiB graph destroy回收、production HBG多callable/第二context和正常close。

以下项仍明确未完成：

- 所有真实CANN args base本轮都恰好`mod64=0`；unaligned parser只有纯Host反例，不能声称
  device backend给出过未对齐地址；
- 64 MiB generic payload不等于“真实HBG最大image”，也没有扫描首个失败size；
- 只有A2/A3 device1，A5没有同等级硬件时延、错误码和cache证据；
- production HBG restore还没有显式poison ready queue/wake list/completion/task state/mailbox；
- wrong slot/callable/blob/affinity/KernelArgs、restore/scheduler/shutdown/destroy阶段仍缺
  test-build-only fault hook和hidden AICore tail证据；
- graph owner在device尚未external quiescent时的拒绝/保活尚未单独注入验证。

本轮所有正式HBG与probe均显式使用device1，没有调用`aclrtResetDevice`/`aclrtResetDeviceForce`
或任何device reset。结束后`npu-smi`仍显示两个physical die的历史AICore 100%与固定HBM、
process table为空；与运行前观测一致，不能据此归因给Grok，也不能把该计数当作本轮资源泄漏。

### 10.49 HBG L1 task-local no-reset故障矩阵与device0结果

#### 10.49.1 为什么故障请求必须属于单个task package

HBG AICPU binary是resident DSO。若用一个process-global `next_fault`变量控制测试，后一个Host调用、
另一张captured graph或下一context都可能继承前一次状态，这与本阶段要证明的“每个graph node持有
自己的tiling-like package”相冲突。runtime提交`eceb3779`因此没有增加新的default-visible
AICPU导出符号，也没有把fault状态留在registry；它复用第一份`HbgLaunchRegion::reserved`并增加
私有flag，把marker直接放进每次fresh HostArgs snapshot：

```text
header.flags                     HBG_LAUNCH_TEST_FAULT_INJECTION
regions[0].reserved high 32      fixed magic
regions[0].reserved low 32       versioned HbgL1FaultStage
plan_hash                        重新覆盖identity + 全descriptor + payload
其他region.reserved              必须仍为0
```

环境变量`SIMPLER_INTERNAL_HBG_L1_TEST_FAULT`只由Host在构造本次launch blob时读取；未设置时生成
的ABI bytes与正常路径完全一致。未知字符串在任何device enqueue前稳定拒绝。validator只有在
flag与首region合法marker同时存在时才接受非零reserved；marker缺失、magic/stage非法、放在第
二region、只有marker没有flag都会返回`InvalidRegion`。对应Host UT还证明canonical GraphPlan
不被修改，只有fresh task snapshot被标记。

#### 10.49.2 多AICPU一致性审查暴露并修正的竞态

第一版实现曾让每个AICPU thread在`run()`入口自行解释marker，而完整blob invalidate/hash验证由
唯一boot leader稍后完成。这个顺序存在真实风险：若CANN复用task-args地址而不同AICPU cache看到
不同代际，部分线程可能直接走受控epilogue，其他线程仍等待`classify_ready_`；伪造marker也可能
在full hash validation之前触发soft-fault返回。

最终实现改成两层可见性协议：

1. public AICPU entry先invalidate并aligned-copy固定header与首region，固定身份解析不做未对齐
   typed dereference；
2. 只有boot leader invalidate完整blob、调用`restore_hbg_launch_blob`完成binding、identity、
   descriptor、bounds、overlap和hash校验；
3. leader把解析后的stage写入`hbg_fault_stage_`，再用既有`classify_ready_.store(release)`发布；
4. 其他participant在`classify_ready_.load(acquire)`之后读取同一stage；自然validation error不会
   设置`hbg_fault_injected_`，因此绝不被测试soft-return吞掉。

该修正由独立只读审查发现，随后重新通过A2/A3与A5 onboard AICPU交叉编译。它不是为了让测试
“更容易通过”，而是保证故障测试本身不引入一条production中不存在的多线程死锁。

#### 10.49.3 七个实际清理阶段及返回契约

当前task-local hook覆盖：

| stage | 注入位置 | 必须经过的真实收尾 |
| --- | --- | --- |
| `restore_copy` | SM已完整copy/publish后，在runtime-arena copy前失败 | peer统一跳过classify/dispatch、逐线程shutdown、completion gate、deinit |
| `restore_publish` | runtime-arena已copy，在该region cache publish前失败 | 同上；working slot允许部分改写但不发布commit |
| `after_scheduler_init` | AICore handshake/assignment完成、blob完整restore后，attach/wire前中止 | 所有已分配core由正常shutdown关闭 |
| `before_classify` | runtime attach/wire和task-count latch完成后 | 不seed ready/wake，仍destroy本轮runtime |
| `before_dispatch` | 所有peer完成classify barrier后 | 不进入scheduler dispatch，关闭core并销毁runtime |
| `shutdown` | restore/classify完成但测试刻意不dispatch；真实shutdown成功后注入 | completion gate及runtime destroy继续执行 |
| `runtime_destroy` | restore/classify完成但测试刻意不dispatch；真实runtime destroy完成后注入 | 所有线程snapshot后last-depart才deinit |

专属错误值为`-1700 - stage`。AICPU只在`hbg_fault_injected_==true`且最终共享错误精确等于本次
stage错误时，把device task结果转成0；这是为了让同一stream/context继续执行下一代的测试契约。
任何自然restore/scheduler/runtime错误、错误marker、不同共享error或runtime status仍原样失败，
不会被环境变量笼统吞掉。`shutdown`/`runtime_destroy`两个case也刻意跳过dispatch，使output
sentinel能够从Host侧证明该marker确实被device消费，而不是一次未生效的正常成功调用。

审查还发现first-wins `run_error_`可能先被合成stage错误占用，从而遮住另一participant真实的
`shutdown()`失败。最终代码增加独立`hbg_unexpected_teardown_error_`：每个participant无论当前
已有何种合成错误，都把首个真实shutdown错误汇合到该槽；所有线程完成finalize后，返回决策先
检查runtime status和unexpected teardown，再考虑controlled success。于是只有**全部真实shutdown
成功**时测试钩子才可能返回0。

#### 10.49.4 无硬件与device0实证

无硬件阶段结果：

```text
test_hbg_launch_blob + test_hbg_aicpu_invocation   2/2 passed
runtime C++ ctest -LE requires_hardware            99/99 passed
A2/A3 onboard HBG AICPU + Host                     build passed
A5 onboard HBG AICPU + Host                        build passed
A2/A3 sim HBG AICPU + Host                         build passed
A5 sim HBG AICPU + Host                            build passed
git diff --check                                    passed
```

用户随后明确device0不再需要为Grok预留。本轮使用隔离的GPT Python/runtime、Torch 2.7.1、
Torch-NPU 2.7.1.post4和PTOAS 0.57，在device0执行：

```text
pytest -q -s tests/st/runtime/l1/test_l1_hbg_fault_injection.py \
  --platform=a2a3 --device=0

1 passed, 1 deselected in 19.67s
```

设备日志依次出现stage 1～7的Host注入记录。测试先完成正常warmup；每个stage都先把output填为
`-777`，enqueue受控HBG L1调用并执行外部`torch_npu.npu.synchronize(0)`。七次同步全部返回，
output均保持sentinel；随后立即取消环境变量，在**同一个L1Context**再次调用，七次都恢复完整
working slot并得到`2 + 5 = 7`。全部fault case结束后，仍在同一context创建ACLGraph，使用输入
11与-3各replay一次，输出分别为16与2。finally严格执行external synchronize、graph reset、
context close；整个测试及实现没有调用device reset。

随后保持fault环境变量未设置，在device0重跑正常HBG选择集：普通warmup/capture/replay、两个
callable task package、异步tensor/scalar snapshot、多output/multi-child/internal workspace、
双graph独立package和同进程第二context generation共`6 passed, 12 deselected in 46.78s`。这证明
私有flag/validator和新增AICPU状态在默认路径下没有改变既有HBG数值、capture或代际行为。

#### 10.49.5 本轮证据边界

本轮可以把N.10.4的“restore失败统一verdict、不classify/dispatch、hidden AICore完成”标为通过，
也可以单独记录上述七个stage的caller-tail与same-context恢复通过。但还不能把N.10.8整组标完：

- slot registry NotReady/Publishing/Corrupt/wrong-device、callable缺失和bad fixed header尚未用device
  hook逐项触发；
- affinity非法组合、坏`KernelArgs::runtime_args`、AIC/AIV双entry及physical-core mapping故障尚未上板；
- 当前`after_scheduler_init`是在真实handshake/assign完成后受控中止，没有把assign函数内部每个失败点
  分别注入；`before_dispatch`也不是scheduler内部执行到一半的任意故障；
- A5只有交叉编译证据，没有同等级真实hardware结果；完全不report的core仍属于外部
  op-timeout/driver fault-containment边界。

因此实现加速了关键generation内部闭环，但没有用一个绿色ST替代剩余故障矩阵。

### 10.50 device0最终定向兼容性回归

#### 10.50.1 隔离与执行前提

本轮开始前，顶层与runtime子仓都是clean，且都位于
`gpt/pypto-l1-aclgraph`分支。所有命令都从
`/mnt/workspace/inductor/pto/gpt_pypto`或其`runtime`子目录执行，`PYTHONPATH`只指向
GPT工作树与其editable runtime；没有修改或导入同级Grok工作树
`/mnt/workspace/inductor/pto/pypto`。用户明确释放device0后才执行以下上板用例，
本轮没有device reset。

#### 10.50.2 TRB L1与ACLGraph

使用device0运行普通L1 ACLGraph和扩展矩阵的TRB参数分支：

```text
pytest -q \
  tests/st/runtime/l1/test_l1_aclgraph.py \
  tests/st/runtime/l1/test_l1_extended_matrix.py \
  --platform=a2a3 --device=0 -k tensormap_and_ringbuffer

3 passed, 15 deselected in 10.44s
```

`deselected`项包含HBG专属参数和非TRB case，不是TRB失败。与10.49已记录的
HBG默认选择集`6 passed`组合后，两种runtime的L1 eager/capture/replay都有当前
device0实证。

#### 10.50.3 HBG/TRB L2旧路径

HBG L2使用runtime仓现成vector graph scene，覆盖host构图、intermediate HeapRing、
AICPU/AICore调度与数值回读：

```text
pytest -q tests/st/a2a3/host_build_graph/vector_example \
  --platform a2a3 --device 0 --runtime host_build_graph --level 2

1 passed in 9.72s
```

TRB L2使用顶层显式dispatch ST，注册两个不同callable，分别执行add/mul，
复用registered handle并由`close()`回收worker-owned device tensor。为缩短本轮占卡时间，
只把性能循环调低为3次，首次注册、双callable和重用语义未缩减：

```text
PYPTO_DISPATCH_LOOP_ITERS=3 pytest -q \
  tests/st/runtime/test_explicit_dispatch_onboard.py \
  --platform=a2a3 --device=0

1 passed in 5.18s
```

#### 10.50.4 单卡L3显式dispatch与pipeline复用

首先运行单卡L3显式dispatch用例。虽然只使用device0，但仍建立完整
HOST orchestrator→CHIP worker→InCore层级，并覆盖`prepare/register/handle/run/close`、
resident weight和close后handle失效：

```text
pytest -q \
  tests/st/distributed/test_l3_explicit_dispatch_onboard.py::test_l3_explicit_dispatch_single_chip \
  --platform=a2a3 --device=0

1 passed in 3.10s
```

随后运行L3 `DeviceTensor`复用用例：一次上传resident weight，连续提交三组不同
input/output，前两次同时占用有界metadata frame，第三次等待最旧frame后复用：

```text
pytest -q tests/st/distributed/test_l3_device_tensor.py \
  --platform=a2a3 --device=0

1 passed in 3.06s
```

这两个L3用例分别给出显式接口和persistent/pipeline的真实device证据，也证明
本轮`ChipCallable` ABI、runtime选择、L1-only state和HBG改造没有破坏这两条旧路径。

#### 10.50.5 结论边界

本轮可以将“L2 one-shot/reuse与L3 persistent/pipeline定向回归”标为A2/A3 device0
通过。但这不是整个L2/L3 ST suite，也不是A5上板证据；两者继续作为更广
发布矩阵，不用这四个绿色用例过度声称“全量回归完成”。

### 10.51 真实scheduler-init失败的红绿上板闭环

#### 10.51.1 为什么原有`after_scheduler_init`还不够

10.49中的`after_scheduler_init`发生在`AicpuExecutor::init()`已经成功返回之后：AICore
handshake和core assignment确实已经完成，但故障由后续`run()`中的restore流程消费，再走
`run_epilogue`。它可以证明“初始化完成后的早退”会执行逐线程shutdown，却没有实际覆盖
`init_rc != 0`这一条独立分支。

这一差别对L1 borrowed-device语义是关键的。`post_handshake_init()`成功时，AICPU已经为各个
AICore打开register window并把core分配给不同scheduler thread；如果随后初始化失败，每个有效
AICPU participant仍必须关闭自己负责的window。仅仅让N个participant进入completion gate，只能
证明Host/AICPU状态到齐，不能让hidden stream上的AICore kernel退出。

runtime提交`50c3badd`保留原有1～7的数值，追加`SchedulerInit = 8`。Host仍只通过内部测试变量
`SIMPLER_INTERNAL_HBG_L1_TEST_FAULT=scheduler_init`在fresh task package中写marker，没有增加
production导出符号或resident全局故障状态。与run阶段故障不同，stage 8必须在调用
`AicpuExecutor::init()`之前确定，因此每个AICPU participant只在private test flag存在时：

1. invalidate本次CANN-owned完整HostArgs blob；
2. 按`DevicePatched`模式重新校验完整header、region、slot binding、identity、bounds、overlap和
   `plan_hash`；
3. 只有校验成功才把stage传入`init()`，校验失败不修改caller提供的stage output；
4. leader仅在真实`post_handshake_init()`成功后写入专属错误`-1708`并发布`init_failed_`。

正常调用没有private flag，不新增整包hash扫描；错误marker也不能借“测试成功”绕过真实package
校验。

#### 10.51.2 第一次device0运行暴露的真实缺口

第一次把stage 8加入正式ST后，device0日志依次出现stage 1～8。stage 1～7仍逐一返回，但stage 8
到达Host注入记录后，`torch_npu.npu.synchronize(0)`不再返回。此时pytest进程仍在device0，Host
线程进入不可中断等待；没有调用device reset。终止Host进程后仍等待CANN现有op-timeout/进程清理
回收，最终process table恢复为空且device0 AICore利用率回到0，再进行下一轮验证。

源码时序与这个现象严格对应：`execute_runtime_generation()`的`init_rc != 0`分支原来直接执行
`latch_run_error(-1) -> arrive_and_finalize_run()`。由于本次故障发生在成功的
`post_handshake_init()`之后，`core_trackers_`中已经有完整assignment，但这条分支从未调用
`SchedulerContext::shutdown(thread_idx)`。结果是AICPU completion gate可以完成，hidden AICore却
没有收到register EXIT，caller stream又必须等待hidden done event，因而整个算子tail不可达。

这次先红后绿的device结果说明该case不是为了增加覆盖数字：它直接找到了正常数值与原七阶段都
没有触发的no-reset错误路径。

#### 10.51.3 修正后的N路收尾

A2/A3与A5的HBG AICPU实现保持逐字一致。每个`init_rc != 0`的有效participant现在按自己的
affinity thread index执行：

```text
sched_ctx_.shutdown(thread_idx)
  -> 首个真实shutdown错误写入hbg_unexpected_teardown_error_
  -> first-wins latch保留原始init/stage错误
  -> arrive_and_finalize_run()
  -> wait_for_finalization()
  -> snapshot共享错误与runtime status
  -> depart；只有last-depart执行deinit/reset
```

`shutdown()`对已经走过`emergency_shutdown()`的初始化失败保持幂等，所以同一分支也能安全覆盖
pre/post-handshake的既有失败。若任何participant真实关闭AICore window失败，独立的
`hbg_unexpected_teardown_error_`优先于`controlled_fault`返回；只有全部真实teardown成功时，
`-1708`才会转成测试所需的task success。这样不会用合成错误掩盖硬件清理失败。

#### 10.51.4 最终验证结果

无硬件与交叉编译结果：

```text
A2/A3 onboard HBG AICPU                         build passed
A5 onboard HBG AICPU                            build passed
A2/A3 sim HBG AICPU                             build passed
A5 sim HBG AICPU                                build passed
test_hbg_launch_blob + test_hbg_aicpu_invocation 2/2 passed
runtime C++ ctest -LE requires_hardware           99/99 passed
L1 Python wrapper/simpler tests                   57/57 passed
git diff --check                                  passed
```

重新构建GPT worktree自己的editable runtime后，在device0重跑完整八阶段矩阵：

```text
pytest -q -s tests/st/runtime/l1/test_l1_hbg_fault_injection.py \
  --platform=a2a3 --device=0

1 passed, 1 deselected in 23.50s
```

stage 8这次与前七项一样到达caller tail；output保持`-777`，紧邻的同context正常generation重新
restore并得到7。八项完成后，同一context仍完成ACLGraph capture，输入11与-3的两次replay分别
得到16与2。测试finally继续遵守external synchronize、graph reset、context close顺序，全程没有
device reset。

随后在未设置fault变量的独立进程中重跑正常HBG L1选择集：

```text
pytest -q \
  tests/st/runtime/l1/test_l1_aclgraph.py \
  tests/st/runtime/l1/test_l1_extended_matrix.py \
  --platform=a2a3 --device=0 -k 'host_build_graph or hbg'

6 passed, 12 deselected in 46.70s
```

因此可以明确新增一条证据：**真实scheduler-init失败后，所有有效AICPU participant都会关闭各自
AICore window并完成arrive/finalize/snapshot/depart，caller tail可达，下一代及ACLGraph replay不
依赖reset。** N.10.8仍不能整体标完：assign函数内部、scheduler dispatch内部、generation建立前
的slot/callable/affinity/KernelArgs错误、physical-core故障和A5真实硬件仍分别缺少同等级注入证据。

### 10.52 A2/A3专属13阶段no-reset矩阵与当前范围收口

#### 10.52.1 验收范围修订与工作树隔离

用户在本轮明确：当前只管A2/A3；没有A5实机，A5 simulator也不用作为完成条件。因此从本节
开始，A5专属源码、A5交叉构建、A5 simulator和A5上板均不再是本阶段completion gate。历史记录
中已经完成的A5静态分析仍保留，因为它解释了common ABI为何如此设计，但不能再用来拖延A2/A3
收口，也不能把A2/A3实机结果外推成A5结论。

本轮继续只在GPT隔离工作树中修改：

```text
/mnt/workspace/inductor/pto/gpt_pypto
branch: gpt/pypto-l1-aclgraph
```

最终dirty文件检查中，runtime差异只包含`src/a2a3/**`、common task/host test plumbing和common
UT；没有`src/a5/**`文件。此前为了保持两架构同构而产生、但尚未提交的A5 scheduler改动已经撤回。
top层仍只有runtime gitlink、A2/A3 HBG fault ST和本设计/过程文档。所有Python命令继续显式使用
`gpt_pypto/runtime/.venv`以及GPT自己的`PYTHONPATH`，没有加载或修改同级Grok工作树
`/mnt/workspace/inductor/pto/pypto`。

运行device0前再次执行`npu-smi info`：CANN NPU process table显示`No running processes found`。
Host进程表中可见一个Grok sibling worktree的Python pytest，但其命令是纯Host L1 Python API UT，
没有NPU进程记录；用户已明确授权本轮使用device0。因此本轮device命令只指定`--device=0`，没有
访问device1，也没有执行任何device reset。

#### 10.52.2 stage 9：真实core assignment入口

原stage 8是在`post_handshake_init()`已经完成assignment以后制造`init_rc != 0`。它证明了
assignment完成后的逐线程window shutdown，却没有进入`assign_cores_to_threads()`之前的失败
边界。新增`SchedulerAssign = 9`后，A2/A3 leader按如下顺序运行：

```text
handshake全部participant到齐
  -> physical core discovery完成
  -> 统计AIC/AIV数量
  -> 命中SchedulerAssign
  -> emergency_shutdown(runtime)
  -> 返回stage专属-1709
  -> 发布init_failed最终裁决
  -> 每个有效participant执行自己的shutdown
  -> arrive/finalize/snapshot/depart
```

故障点位于`assign_cores_to_threads()`之前，不伪造assignment结果；如果是自然的handshake或
assignment失败，仍返回原始失败，不会因为test marker变成success。Host parser、task-local marker
编码/解码和完整package认证均新增对应反例。A2/A3 onboard HBG AICPU重编通过，device0九阶段
矩阵结果为：

```text
1 passed, 1 deselected in 25.46s
```

#### 10.52.3 stage 10：实际publish之后的scheduler dispatch错误

早先的`before_dispatch`只在进入scheduler loop以前中止，不能证明已经开始发布task后，多个
scheduler participant如何共同停止。新增`SchedulerDispatch = 10`把故障放进A2/A3
`SchedulerContext::resolve_and_dispatch()`：只有`dispatch_ready_tasks()`返回`try_pushed=true`，
即当前线程已经真实publish过工作后，才执行以下动作：

1. CAS写入stage专属`sched_error_code=-1710`并记录thread/bitmap；
2. 通过`completed_.exchange(true)`选出唯一执行`emergency_shutdown(runtime)`的线程；
3. 当前scheduler返回准确的stage error；
4. 全部AICPU participant仍进入既有shutdown和两阶段completion gate；
5. 因为task可能已经publish，测试不再要求这一stage的output保持`-777`，但仍要求caller tail可达，
   紧邻的正常generation必须重新restore并得到正确结果。

第一次device0十阶段运行没有hang，但stage 10由CANN报`507018`并使pytest失败。根因不是AICore
遗留，而是`execute_runtime_generation()`先无条件返回inner runtime status；即使共享error已经被
认证为本stage实际产生的`-1710`，它仍在`controlled_fault`分支之前返回。修正没有粗暴吞掉所有
runtime错误，而是计算`requested_fault_error`并使用精确三条件：

```text
test package完整认证成功
&& 本stage内部确实置位hbg_fault_injected_
&& runtime_rc/shared_error恰好等于本stage专属值
```

只有三者同时满足，完全相同的synthetic runtime status才允许进入controlled success；任何其他
runtime status、自然scheduler错误或unexpected teardown error仍优先返回。修正后十阶段device0
结果为：

```text
1 passed, 1 deselected in 27.70s
```

#### 10.52.4 stage 11～12：generation建立前的平台与affinity闭包

`PlatformBridge = 11`发生在A2/A3 runtime public HBG AICPU entry已经取得可信execution slot、
callable和fixed invocation之后。test flag存在时先invalidate完整CANN-owned HostArgs blob，再按
DevicePatched模式验证全部region、identity、placeholder、bounds、overlap与`plan_hash`。只有完整
认证的marker才能在进入platform bridge/generation以前向独立`HbgL1LaunchControl`写release
`PRELAUNCH_CANCEL`并flush；hidden AICore读取同一个prepare-time可信control后退出。缺bridge、
坏package或无control仍是自然失败，不被转换为0。十一阶段device0结果为：

```text
1 passed, 1 deselected in 32.39s
```

`AffinityInputs = 12`继续使用上述已认证invocation，在A2/A3 platform AICPU入口把代表性非法值
`allowed_cpu_count=-1`送进production `platform_aicpu_affinity_config_valid()`。校验发生在任何
线程进入affinity gate/barrier之前；命中后写同一prelaunch CANCEL，只有这一个已认证stage返回
controlled success。无硬件`test_platform_aicpu_affinity_config`继续覆盖0、负数、超过
`MAX_GATE_THREADS`、allowed大于launch以及合法配置，避免把所有输入组合都塞进device ST。
十二阶段device0结果为：

```text
1 passed, 1 deselected in 34.43s
```

#### 10.52.5 stage 13：persistent KernelArgs/Runtime binding拒绝闭包

HBG L1 prepare会持久化device `KernelArgs`，AICPU public entry必须验证其中`runtime_args`非空且
等于immutable execution-slot registration中的`outer_runtime_base`。hidden AICore又通过Host直传
trusted Runtime override读取同一Runtime/control；这条绑定是防止坏KernelArgs造成AICPU/AICore
split-brain的关键。

新增`KernelArgsRuntime = 13`时先写UT引用，再运行C++ build，得到预期红灯：

```text
error: ‘KernelArgsRuntime’ is not a member of ‘simpler::hbg::HbgL1FaultStage’
```

随后才增加enum、Host字符串映射、marker/hash认证反例和A2/A3 public entry分支。为了保证test
marker本身不能为坏package开后门，真实KernelArgs读取与binding判断被移动到完整fixed invocation
解析、且test package完整认证之后。最终逻辑同时计算：

```text
kernel_args_match = runtime_args != nullptr
                    && address(runtime_args) == slot.outer_runtime_base
inject_kernel_args_fault = authenticated stage == KernelArgsRuntime
```

自然`kernel_args_match=false`始终写prelaunch CANCEL后返回`-1`；即使blob恰好携带test flag也不会
被吞掉。只有真实binding仍正确且marker完整认证时，stage 13才走同一拒绝/CANCEL闭包并返回受控
success。这样可以安全验证hidden AICore的control来源和no-reset tail，而不在共享设备上故意写坏
persistent device pointer。本case**不等价于已经完成真实device内存破坏测试**，后者仍保留为独立
高风险验收项。

A2/A3 onboard HBG AICPU和common C++ UT重编通过。重新构建GPT worktree自己的editable runtime
后，device0完整13阶段结果为：

```text
pytest -q -s tests/st/runtime/l1/test_l1_hbg_fault_injection.py \
  --platform=a2a3 --device=0

1 passed, 1 deselected in 36.52s
```

日志严格按stage 1～13出现。每一个stage的外部`synchronize(0)`都返回；除stage 10允许已经
publish的task改变output外，其他stage保持sentinel；每个stage之后的同context正常调用都得到7。
13项完成后，同一context继续完成ACLGraph capture，输入11和-3的两次replay分别得到16和2。
finally仍是external synchronize、graph reset、context close，全程没有reset。

#### 10.52.6 最终A2/A3验证证据与剩余边界

本轮最终只把A2/A3和common路径作为gate：

```text
A2/A3 onboard HBG AICPU build                                  passed
common/A2相关C++ UT：launch blob、AICPU invocation、affinity   3/3 passed
L1 Python wrapper/simpler                                     57/57 passed
A2/A3 device0 13-stage no-reset + final ACLGraph replay        1 passed
A2/A3 device0 normal HBG L1 selection                          6 passed, 12 deselected
clang-format dry-run                                           passed
ruff check / format                                            passed
top/runtime git diff --check                                   passed
A5硬件、A5 simulator、A5专属build                              不在当前范围
```

正常HBG selection的精确命令与结果为：

```text
pytest -q \
  tests/st/runtime/l1/test_l1_aclgraph.py \
  tests/st/runtime/l1/test_l1_extended_matrix.py \
  --platform=a2a3 --device=0 -k 'host_build_graph or hbg'

6 passed, 12 deselected in 46.86s
```

由此可以把A2/A3的restore、真实scheduler init、assign、实际dispatch、shutdown、runtime destroy、
已认证platform/affinity/KernelArgs prelaunch分支的caller-tail与同context恢复标为完成。仍不能扩大为：

- slot registry NotReady/Publishing/CorruptState/wrong-device的真实device故障矩阵；
- callable缺失和bad blob/header/identity/placeholder等未认证自然损坏的逐项device注入；
- 真实篡改persistent device KernelArgs指针；
- AIC与AIV两种entry分别覆盖的physical-core id/register mapping故障；
- 完全不进入或不report的硬件core由算子内自行恢复；
- A5可用或A5与A2/A3行为一致。

这些边界不会阻止当前A2/A3主路径和13项安全可控故障矩阵收口，但在对应功能被纳入产品保证前，
必须继续保持fail-closed表述，不能用本轮绿色结果代替尚未执行的破坏性实机实验。

### 10.53 A2/A3 physical-core pre-window CANCEL的AIC/AIV实机闭环

#### 10.53.1 为什么已有静态predicate还不够

10.18已经实现了`aicore_register_mapping_invalid()`和per-core pre-window CANCEL：AICPU看到
`physical_core_id`越界或对应register address为0时，不能访问未知SPR，而是向该logical worker的
旧Handshake control line写`AICORE_PRE_WINDOW_CANCEL`并flush；AICore在等待
`DATA_MAIN_BASE`期间周期invalidate该line，看到CANCEL后直接退出。独立`L1AicoreReport`又保证
AICore report与AICPU control不再共享写所有权。

但此前证据只有predicate UT、A2/A3/A5交叉构建和源码时序，没有真实证明以下完整链条：

```text
AICore实际report
  -> AICPU识别不可用mapping
  -> 不执行platform_init_aicore_regs(0)
  -> 对该core发布GM CANCEL
  -> hidden AICore kernel中对应block退出
  -> 其他已打开core收到register EXIT
  -> hidden done event与caller tail到达
  -> 同context下一代仍可执行
```

这条链直接决定borrowed-device L1是否需要reset，不能用Host predicate为真来替代device行为。

#### 10.53.2 task-local stage 14与自然错误隔离

新增`PhysicalCoreMapping = 14`，仍沿用每份runtime-owned HostArgs中的hash认证marker。先在两组
common UT和device ST中引用新枚举，再构建得到预期红灯：

```text
error: ‘PhysicalCoreMapping’ is not a member of ‘simpler::hbg::HbgL1FaultStage’
```

实现后，A2/A3 `AicpuExecutor::init()`把认证后的stage传入每个participant的
`handshake_partition()`。每个partition照常invalidate并读取真实`L1AicoreReport`，先计算真实
`physical_core_id`、`core_type`和register address。注入不写坏report、不修改全局register表，也不
访问伪造SPR；它只对一个**原本完全有效**的report把本次局部`effective reg_addr`视为0，从而进入
production不可用mapping分支并执行真实`publish_pre_window_cancel()`。

为了避免依赖“logical worker 0恰好是哪一种core”，最终实现没有固定worker id，而是使用一个
原子type bitmask：

```text
bit 0：首个原本有效的AIC report已经走注入拒绝/CANCEL
bit 1：首个原本有效的AIV report已经走注入拒绝/CANCEL
```

同类型其余core继续走正常开window路径；不同AICPU slice通过`fetch_or`竞争各类型唯一注入者。
leader只有在以下条件全部满足时才把`-1714`识别为受控故障：

1. 完整task package与marker认证成功；
2. AIC与AIV两个bit都已经由真实有效report置位；
3. `handshake_failed_`确实到达leader；
4. `handshake_unexpected_failure_`仍为false；
5. `post_handshake_init()`返回值精确等于stage 14专属错误。

若任意core同时出现真正的id越界或真实register address为0，代码会单独置
`handshake_unexpected_failure_`；即使两个测试bit也已经命中，leader仍返回自然失败，绝不会被
test marker吞掉。leader对所有已经开window的core执行`emergency_shutdown()`；两个注入core则只
依赖各自GM CANCEL退出。之后每个AICPU participant仍执行逐线程shutdown、
arrive/finalize/snapshot/depart，last-depart才清代际状态。

#### 10.53.3 两轮device0验证及证据强度

第一版stage 14先固定选择logical worker 0，A2/A3 device0完整14阶段矩阵通过：

```text
1 passed, 1 deselected in 38.52s
```

这个结果证明了单个真实report的CANCEL链，但无法证明AIC/AIV两种kernel entry都覆盖，因此没有以
该结果结束。改为双type bitmask、重新构建GPT runtime后，再次确认NPU process table为空，并在
device0重跑同一矩阵：

```text
pytest -q -s tests/st/runtime/l1/test_l1_hbg_fault_injection.py \
  --platform=a2a3 --device=0

1 passed, 1 deselected in 38.44s
```

stage 14若没有同时命中AIC和AIV两个bit会返回非零并使该ST失败；因此这次绿色结果同时证明两类
entry都完成了report→per-core CANCEL→hidden kernel退出。该stage保持output sentinel；紧邻的
同context正常generation得到7，全部14项后同一context ACLGraph对输入11和-3的replay分别得到16
和2。全程只有外部测试侧synchronize，没有device reset。

随后在fault环境变量未设置的独立进程中重跑正常A2/A3 HBG L1选择集：

```text
6 passed, 12 deselected in 46.62s
```

无硬件与构建证据为：

```text
test_hbg_launch_blob + test_hbg_aicpu_invocation   2/2 passed
A2/A3 onboard HBG AICPU                            build passed
clang-format dry-run                               passed
```

本节仍严格遵守A2/A3-only范围，没有修改、构建或运行A5专属路径。

#### 10.53.4 仍未被本case证明的边界

本case可以勾选“范围内有效physical id但本次mapping被判为0时，AIC/AIV均通过GM CANCEL退出且
不访问未知SPR”。它不能被表述为已经证明：

- 硬件或report内存真正产生越界`physical_core_id`时的全部日志/故障传播行为；
- AICore完全未进入或从未publish report时的算子内恢复；
- 任意真实SPR/MMIO故障均可由PyPTO恢复；
- A5具有相同行为。

其中完全不report的core仍属于外部CANN op-timeout、driver fault containment或context/device
recovery边界。L1单算子既不能内部stream/device sync，也不能reset用户设备；文档继续明确这一点，
不把无法观察的硬件失联伪装成可由算子内协议修复。

### 10.54 A2/A3越界physical-id predicate的安全等价上板验证

#### 10.54.1 与zero-mapping stage的区别

10.53把实际有效report对应的effective register address置为0，证明了`reg_addr == 0`分支和
per-core CANCEL，但它没有真正让production predicate计算`physical_core_id >= max`。这两个条件
虽然最终都不能打开window，代码访问顺序却不同：越界id必须在索引`regs[physical_core_id]`以前被
拦截，否则测试可能以“成功取消”掩盖一次越界Host/AICPU内存读取。

本轮新增独立`PhysicalCoreId = 15`，不修改实际report内存，而是在读取并确认report原本有效后，
为本次局部校验构造：

```text
effective_core_id = platform_get_physical_cores_count()
effective_reg_addr = 0             // 不执行regs[effective_core_id]
```

随后调用原本的`aicore_register_mapping_invalid(effective_core_id, max, effective_reg_addr)`。因为
effective id恰好等于count，真实`physical_core_id < count`上界predicate返回false并进入现有
CANCEL分支。这个值比任意超大伪值更严格：它直接覆盖off-by-one边界，同时代码的条件表达式保证
越界时不会求值register-array索引。

#### 10.54.2 仍然要求AIC/AIV双entry与自然错误优先

stage 15复用10.53的AIC/AIV原子type bitmask，但两个stage各自属于独立launch，pre-handshake init
每代都先清零bitmask。每个stage只对每种core type的首个原本有效report注入；同类型其他core正常
开window。controlled success仍要求：

- task-local marker和完整blob hash认证通过；
- AIC bit与AIV bit均置位；
- leader确实观察到`handshake_failed_`；
- 没有任何真实report/mapping同时触发`handshake_unexpected_failure_`；
- 返回值精确等于`-1715`。

因此测试不会因为第一个logical worker属于某一种core而漏掉另一种entry；也不会在真实异常与注入
同时发生时把自然错误转换成0。两个注入core只收到旧Handshake control line上的GM CANCEL；其余
已经打开window的core由leader统一`emergency_shutdown()`。

同样采用test-first顺序：先在marker/blob UT、AICPU invocation认证UT和device ST中引用
`PhysicalCoreId`，首次target build按预期失败：

```text
error: ‘PhysicalCoreId’ is not a member of ‘simpler::hbg::HbgL1FaultStage’
```

补齐enum、Host parser、A2/A3 handshake路径和leader精确识别后，定向UT与A2/A3 onboard HBG
AICPU均构建通过。

#### 10.54.3 device0结果与当前边界

重新构建GPT worktree editable runtime、确认NPU process table为空后，device0完整15阶段矩阵为：

```text
pytest -q -s tests/st/runtime/l1/test_l1_hbg_fault_injection.py \
  --platform=a2a3 --device=0

1 passed, 1 deselected in 40.70s
```

stage 15的caller tail到达，output保持sentinel，紧邻同context正常generation验数为7；全部15项后
同一context继续完成ACLGraph capture和两次replay，全程无reset。fault变量未设置时再次运行正常
A2/A3 HBG L1 selection：

```text
6 passed, 12 deselected in 46.65s
```

本轮可以把“A2/A3 production physical-id上界predicate、AIC/AIV两entry per-core CANCEL、hidden
kernel退出、caller tail和下一代恢复”标为已由安全等价注入上板。仍未声称真实硬件/report内存
损坏的诊断日志或driver行为已经覆盖；完全不report的core也继续属于外部fault containment边界。
没有修改、构建或运行A5专属路径。

### 10.55 stage 15提交后的source guard与跨runtime兼容性复验

runtime与顶层分别提交stage 15后，runtime源码HEAD从editable extension内记录的`2c594f97`推进到
`8f238f74`。第一次启动TRB L1 ST时，测试体和NPU launch都尚未发生，
`simpler.task_interface._assert_bindings_match_source_tree()`就在conftest import阶段拒绝加载：

```text
ImportError: _task_interface was built from 2c594f979cc8,
but this source tree is at 8f238f743205.
Rebuild: pip install --no-build-isolation -e .
```

这是预期的ABI/source guard生效，不是TRB功能失败。继续只在
`/mnt/workspace/inductor/pto/gpt_pypto/runtime/.venv`执行editable rebuild，未安装到系统或Grok
工作树；重建后用显式GPT `PYTHONPATH`重跑A2/A3 TRB L1/ACLGraph选择集：

```text
3 passed, 15 deselected in 10.46s
```

随后复验共享本轮HBG scheduler代码的L2 vector scene。第一次从top仓误用了
`tests/st/a2a3/...`路径；真实scene位于runtime子仓，因目录不存在而没有加载runtime ST conftest，
pytest只报告`--platform/--device/--runtime/--level`参数未知，同样没有发NPU task。改为从
`/mnt/workspace/inductor/pto/gpt_pypto/runtime`执行正确命令：

```text
.venv/bin/python -m pytest -q \
  tests/st/a2a3/host_build_graph/vector_example \
  --platform a2a3 --device 0 --runtime host_build_graph --level 2

1 passed in 9.68s
```

因此stage 14/15对A2/A3 HBG handshake的test-only扩展没有改变TRB L1 capture/replay，也没有改变
fault marker为空时的HBG L2 scheduler行为。两次前置失败都发生在pytest collection/CLI解析阶段，
不能计作device失败或reset恢复；正式复验全程仍未调用device reset。

### 10.56 将附录L从编码前模板校准为当前A2/A3实施状态

设计计划附录L最初用于实现前评审，长期保留了大量空checkbox。随着第一阶段和HBG第二阶段已经
落地，继续保留所有空框会产生两种相反误导：一是把已由源码与上板证明的ABI/所有权/stream语义
误读为尚未实现；二是让真正缺少自动trace或延迟扰动的项目淹没在模板噪声中。

本轮没有根据“看起来应该完成”批量勾选，而是重新对照当前权威证据：

- native C ABI、mandatory caller stream和borrowed mode：`pto_runtime_c_api.h`、
  `c_api_shared.cpp`及mode/ABI/failure UT；
- resource ownership和retry close：`L1ExecutionState::Closing`、L1-only allocator/KernelArgs/
  runtime state、explicit close fault UT；
- task snapshot：AICPU WithHostArgs、persistent AICore args、连续四次异步tensor/scalar address/value
  device0 ST；
- 固定单算子fork/join：`L1LaunchSequenceOps`的可表达操作集合及exact-order/failure-close UT；
- capture语义：A2/A3 TRB/HBG独立capture stream、图内PyTorch→L1→PyTorch和连续replay；
- Python/taskQueue：独立adapter、`.stream(false)`、queue lease、default allocator `recordStream`、
  explicit `out=`/forward-only/close反例和中英文用户文档；
- 兼容性：A2/A3 device0的TRB/HBG L1、HBG/TRB L2与单卡L3定向路径。

据此，附录L.1～L.5和L.6已有直接证据的项目已改为`[x]`，并在清单顶部明确“当前只以A2/A3为
gate”。仍然保留三项未勾选：

1. 对launch禁止项提供自动化runtime trace/counter；
2. trace/counter明确证明capture query、model attach、private AICPU launch与early mode均为0；
3. eager/ACLGraph专门加入延迟predecessor和延迟AICore tail的entry/exit扰动，而不只依赖正常
   PyTorch pre-op/L1/post-op数值顺序。

当前源码的allocation-free operation table让sync、allocation、capture inspection和model attach
无法由公共launch sequence表达，且全仓搜索与静态审计未发现正式L1调用这些API；但计划要求的是
“自动trace/counter”和“延迟扰动”，普通源码审计与正常数值ST不能替代，因此三项继续为空。这个
校准不会把尚缺证据改写成完成，也不会重新引入A5 gate。

### 10.57 A2/A3 execution-slot fallback control的实机证明

#### 10.57.1 为什么slot内control还不够

正常HBG L1 public AICPU entry先从resident registry取得immutable
`HbgExecutionSlotRegistration`，再由`outer_runtime_base + prelaunch_control_offset`解析control。
这足以处理callable/blob/platform等generation前错误，但如果失败对象就是slot registry本身，代码
不能再信任从该registry读出的base/offset。为此prepare阶段的`simpler_aicpu_init`会在任何slot
registration/run task之前，把host-owned immutable slot中的control地址锁存到resident AICPU SO；
`reject_hbg_l1_without_slot()`只能使用这个独立fallback trust root。

此前源码和registry UT证明了地址选择算法，却没有让真实hidden AICore通过fallback地址收到CANCEL。
新增`SlotFallbackControl = 16`用于补这一条device证据。

#### 10.57.2 不伪造registry损坏，也不吞自然错误

stage 16仍由本次runtime-owned HostArgs携带，先通过完整blob hash/region/placeholder/identity认证。
代码随后仍读取并验证真实persistent `KernelArgs::runtime_args`；如果自然KernelArgs错误存在，先按
原逻辑返回失败，绝不进入test success。只有package、callable、slot和KernelArgs全部真实有效，且
marker精确为stage 16时，代码才刻意：

```text
不调用 hbg_l1_launch_control(valid_slot)
  -> 读取 simpler_aicpu_init 锁存的 resident fallback address
  -> 写 HBG_L1_PRELAUNCH_CANCEL
  -> cache flush 独立64-byte control line
  -> 返回controlled task success
```

这样测试的是**fallback地址来源和device可见性**，不是把registry validator改成接受坏状态。真实
`NotReady/Publishing/CorruptState/wrong-device`仍由production acquire拒绝并返回自然错误；test
flag不能绕过这些validator。

同样先增加enum引用、marker UT、完整AICPU invocation认证UT和device ST，第一次build按预期因
`SlotFallbackControl`尚不存在而失败；补齐common enum/Host parser和A2/A3 helper拆分后，两项
定向UT、clang-format及A2/A3 onboard HBG AICPU build均通过。

#### 10.57.3 device0结果

重新构建GPT worktree editable runtime、确认NPU process table为空后，A2/A3完整16阶段矩阵：

```text
pytest -q -s tests/st/runtime/l1/test_l1_hbg_fault_injection.py \
  --platform=a2a3 --device=0

1 passed, 1 deselected in 42.52s
```

stage 16若fallback地址为0、错误或未被AICore读取，hidden kernel不会完成，caller stream的done wait
也不会返回；因此绿色结果直接证明prepare-time latch→resident fallback→control flush→hidden
AICore exit这条链。output保持sentinel，紧邻同context正常generation得到7，全部16项后同一
context继续完成ACLGraph capture/replay，全程无reset。

fault环境变量未设置时再次运行A2/A3正常HBG L1 selection：

```text
6 passed, 12 deselected in 46.66s
```

计划N.10.8因此新增一条已完成的fallback trust-root证据，但仍没有把
NotReady/Publishing/CorruptState/wrong-device四种真实registry状态整体勾选。要覆盖它们，必须
保持validator自然拒绝语义，同时另行决定如何观察CANN task非零返回后的context/device状态；不能
为了让ST继续执行而把未认证registry错误改成0。本轮继续没有修改、构建或运行A5专属路径。

### 10.58 A2/A3单算子entry/exit流边界压力验证

#### 10.58.1 为什么普通pre-op/post-op还不够

此前基础ACLGraph ST已经包含`torch.add -> PyPTO L1 -> torch.mul`，能够证明最小数值顺序，但前后
各只有一个节点，L1本身也只有一个child kernel。这样的case对功能冒烟足够，却没有主动放大两类
错误窗口：hidden AICore分支若越过caller stream上的Start gate，可能在前驱数据尚未完成时读取；
caller stream若没有在单算子出口完整join hidden分支，紧邻后继可能读取尚未由最后一个child提交完成
的输出。

本轮新增A2/A3专属
`tests/st/runtime/l1/test_l1_stream_boundaries.py`，不增加capture查询、内部同步或测试专用device
控制通道。测试通过真实工作量扩大窗口：

- caller stream在L1入口前连续排入24个`torch.add(..., out=...)`节点，最终buffer是L1的真实输入；
- L1 orchestration使用7份内部workspace tensor串接8次`@pl.jit.incore` child add，最后一次才写外部
  output，从而拉长真实hidden AICore工作链；
- caller stream在L1之后立即再排入24个`torch.add(..., out=...)`节点，第一个节点直接读取L1 output；
- eager warmup与ACLGraph capture使用完全相同的前驱/L1/后继拓扑，graph对三个不同输入连续replay；
- 最终期望值同时包含48次PyTorch增量和8次L1 bias，entry早读、exit越过或任一child丢失都会导致
  数值不一致。

该case只参数化`platform="a2a3"`，runtime分别为TRB与HBG。它遵守现有生命周期契约：warmup后由
调用方外部synchronize再切换capture stream，graph reset以前保持context与全部graph tensor强引用，
最终先外部quiescence、再reset graph、最后显式close context。PyPTO内部没有新增sync或reset。

#### 10.58.2 Host门槛与device0结果

新程序先在无NPU任务的Host流程完成pytest collection，以及A2/A3 TRB/HBG两份独立lowering和
PTOAS产物生成。确认NPU process table为`No running processes found`后，在GPT隔离工作树用device0
执行：

```text
pytest -q -s tests/st/runtime/l1/test_l1_stream_boundaries.py \
  --platform=a2a3 --device=0

2 passed in 9.18s
```

两项分别证明TRB和HBG在eager及ACLGraph三次replay下都保持caller predecessor→L1 hidden branch→
caller successor的单算子边界。附录L.6与N.10.6对应边界项据此勾选。

证据边界仍保持明确：这是由真实device工作量制造窗口的顺序/数值验证，不是event时间戳或CANN
runtime trace；它不证明任意人为无限stall均可恢复，也不替代仍未完成的禁止API自动trace/counter。
本轮按最新验收范围只处理A2/A3，没有构建或运行A5实机、A5 simulator或A5专属target。

#### 10.58.3 A2/A3 L1定向矩阵复验

为避免新case只在单独进程中偶然通过，随后在device0同一pytest进程合并运行基础ACLGraph、扩展
task/package lifetime矩阵和本轮边界ST：

```text
pytest -q -s \
  tests/st/runtime/l1/test_l1_aclgraph.py \
  tests/st/runtime/l1/test_l1_extended_matrix.py \
  tests/st/runtime/l1/test_l1_stream_boundaries.py \
  --platform=a2a3 --device=0

11 passed, 9 deselected in 64.22s
```

该矩阵同时覆盖TRB/HBG基础capture/replay、两个HBG callable、异步tensor/scalar快照、多输出、
multi-child与内部workspace、两个HBG graph交替replay、同进程顺序context generation，以及本轮
entry/exit压力拓扑。20个deselection/selection中的9项来自平台参数过滤或不匹配的case，不代表失败；
本次实际选择的11项全部通过。运行前再次确认NPU process table为空，运行中与teardown均未调用
device reset。

### 10.59 A2/A3 launch禁止API的自动源码守卫与真实CANN符号trace

#### 10.59.1 为什么exact-order fake还不足以单独闭环

`L1LaunchSequenceOps`从设计上只允许wait event、memset、record event、AICPU/AICore launch和失败
CANCEL，既有C++ UT也已经检查固定调用顺序。但它只能证明这个operation table没有提供禁止操作，
不能自动发现未来有人在`DeviceRunnerBase::launch_l1_callable()`的table外直接插入一次sync、capture
query或model attach；同时，fake stream数值也不能证明真实`aclrtLaunchKernelWithHostArgs`最终收到的
就是torch_npu caller stream。

因此本轮没有把旧fake UT重新命名后冒充runtime证据，而是增加两层互补门禁。

第一层位于runtime提交`f48d7c29`：
`runtime/tests/ut/py/test_l1_launch_source_guard.py`用brace-balanced提取真实
`DeviceRunnerBase::launch_l1_callable()`函数体，并自动拒绝下列API族重新进入launch：

- ACL/RT stream或device synchronize及timeout变体；
- ACL/RT capture begin/end/query、`rtStreamAddToModel`和`rtModelBindStream`；
- stream/event create/destroy；
- device malloc/free、binary load/unload和lazy AICore registration；
- AICPU callback绕过其`stream`形参读取hidden stream，或AICore callback恢复lazy register。

测试还要求正式函数只调用一次`enqueue_l1_launch_sequence()`。它是无硬件的源码结构守卫，能在
review/CI阶段立即阻止明显回归。

#### 10.59.2 只归因PyPTO调用者的preload tracer

第二层新增
`tests/st/runtime/l1/support/l1_cann_api_trace.cpp`和
`tests/st/runtime/l1/test_l1_cann_api_trace.py`。pytest父进程用`g++ -shared -fPIC`在临时目录构建
tracer，再以`LD_PRELOAD`启动全新的子进程，保证CANN符号第一次解析前interposer已经生效。tracer
使用`dladdr(return_address)`过滤调用者，只有直接来自`libhost_runtime.so`的调用才进入snapshot；
因此测试代码/torch_npu用于warmup、graph begin/end、replay和外部synchronize的API不会被误归因给
PyPTO。

tracer的fixed-size ABI不分配内存，最多记录32个operation，并对以下禁止族独立计数：

- ACL、RT、RTS的stream/device sync及timeout变体；
- ACL/RT capture begin/end/query；
- stream-to-model attach与model bind；
- stream/event create/destroy；
- device malloc/free；
- AICPU launch stream不等于本次caller stream、AICore错误使用caller stream；
- AICPU在caller Start record以前launch的early mode。

允许调用也被真实interpose并转发。每个普通launch必须严格得到：

```text
memset(caller) -> memset(caller) -> record Start(caller)
-> wait Start(hidden) -> launch AICore(hidden) -> record Done(hidden)
-> launch AICPU(caller) -> wait Done(caller) -> record Tail(caller)
```

capture从warmup stream切换到独立stream时，序列前允许且要求出现一次非阻塞
`aclrtQueryEventStatus`，用于fail-closed确认上一次真实tail已完成；它是event completion query，不是
capture-state query，也不会向graph导入capture外event wait。tracer同时比较Start record/wait和Done
record/wait的真实event handle，避免只比较操作名称而漏掉错代event。

#### 10.59.3 device0结果与证据边界

runtime源码守卫与相邻L1 wrapper UT结果为：

```text
8 passed in 0.05s
```

确认NPU process table为空后，A2/A3 device0执行preloaded tracer。子进程内部依次创建并关闭TRB与
HBG context，各自完成warmup、一次eager trace、独立stream capture trace和一次replay验数：

```text
child trace: 1 passed in 8.65s
parent probe: 1 passed in 16.50s
```

四个真实launch窗口全部观察到一份AICPU和一份AICore launch；AICPU使用精确caller raw stream，
AICore使用同一非caller hidden stream，完整operation/event顺序匹配。PyPTO来源的stream/device
sync、capture API、model attach、resource lifecycle、device allocation、private AICPU stream、
caller-stream AICore和early launch计数全部为0。TRB与HBG replay数值均正确。

这项证据只覆盖当前A2/A3+CANN 9.2进程内实际导出的符号集合；第三方CANN在一个允许API内部执行
的实现细节不会被错误归因为PyPTO直接调用。未来若CANN增加新的同义禁止入口，应同时扩展源码token
集合和interposer列表。本轮没有构建或运行A5/A5 simulator，也没有据A2/A3结果外推A5。

### 10.60 A2/A3范围冻结与L2/L3兼容性收口

#### 10.60.1 验收范围只保留A2/A3

用户在本轮再次明确：当前没有A5实机，不需要处理A5 simulator，后续只管A2/A3。因此从本节开始，
A5源码、交叉构建、simulator和上板结果都不再作为当前L1/HBG交付的完成门槛；历史文档中保留的A5
分析仅作为未来重新开启该架构时的背景，不能把A2/A3结果外推成A5结论。

所有本轮设备命令执行前都先加载`.claude/skills/testing/load-env.sh`，并显式把`PYTHONPATH`、
`PATH`和Python解释器固定到`pto/gpt_pypto`及其`runtime/.venv`。环境脚本仍打印一条既有的
`dirname: unrecognized option '--path-format=absolute ...'`提示，但后续导入路径、native binding和
测试产物均来自GPT隔离工作树；这条提示没有改变测试选择或结果。

#### 10.60.2 A2/A3 L2全量尝试暴露一个可独立复现的长序列遗留失败

首先在device0运行A2/A3 runtime level-2选择集：

```text
pytest -q -s tests/st/a2a3 \
  --platform=a2a3 --device=0 --level=2 \
  --max-parallel=1 --pto-session-timeout=1800
```

pytest收集到65个level-2节点，另有14个节点因level/platform过滤而deselect。首个根失败位于
`TestSpmdPagedAttentionHighPerf`的普通非manual case
`b1_h32_kv8_s8192_bs128_fp16`：device分类为`sched_error_code=100`、
`S1:running-stalled`，第一次运行观察到core 22停滞以及8个core未完成deinit，随后AICPU返回
`507018/0x2a`并最终触发AICore op timeout。紧随其后的`TestSpmdStarvation`失败时复用了已经不可用的
L2 runner，因此属于根失败后的级联，而不是第二个独立根因。中断时pytest报告为：

```text
2 failed, 60 passed, 1 skipped, 14 deselected, 14 warnings in 325.42s
```

这里必须如实记录一个与L1 no-reset契约不同的事实：旧L2 runner在上述错误finalize路径中自动调用了
`aclrtResetDeviceForce(0)`。这是既有L2全资源掌控恢复逻辑由测试失败触发，并非本轮人工执行reset，
但它确实发生了，所以本次L2兼容性尝试不能被描述成“全过程无reset”。L1/HBG的16阶段fault matrix
仍是独立的无reset证据，不应与这里的legacy L2错误恢复混为一谈。

为了验证该失败是否由本分支新增的L1 pre-window CANCEL polling误入L2，曾做过一次最小诊断实验：
临时把A2/A3 TRB/HBG的AICore cancel polling限定为L1，重编对应两个onboard AICore target，并只运行
上述`s8192` case。它仍然独立复现，停滞core变为24，但错误分类、`507018`和op timeout保持一致：

```text
1 failed in 76.85s
```

该次legacy L2 finalize也再次自动force-reset device0。实验直接否定了“L1 cancel polling导致此
case回归”的假设；临时代码已用patch完整恢复，A2/A3 TRB/HBG AICore cache随后从恢复后的源码重建，
顶层与runtime仓都重新确认clean。没有把这项猜测或实验改动提交到分支。

静态对照还确认：失败case、其kernel和L2 launch路径都没有被本次L1提交直接改写；该case在当前环境
单独运行即可稳定卡在8个AIV worker的FFTS/internal wait。因此当前证据支持把它记为独立的旧L2
高性能长序列/环境问题，而不是已证明的L1回归；但在没有一份同环境基线分支成功结果前，也不能把
“A2/A3 L2全量suite绿色”写进结论。

#### 10.60.3 级联之后的7个L2场景在干净进程全部通过

为确认根失败之后未完成或受污染的尾部节点，使用全新pytest进程运行以下7组目录：

```text
tests/st/a2a3/tensormap_and_ringbuffer/spmd_starvation
tests/st/a2a3/tensormap_and_ringbuffer/spmd_sync_start
tests/st/a2a3/tensormap_and_ringbuffer/spmd_sync_start_aiv
tests/st/a2a3/tensormap_and_ringbuffer/spmd_sync_start_early_dispatch
tests/st/a2a3/tensormap_and_ringbuffer/spmd_sync_start_edge
tests/st/a2a3/tensormap_and_ringbuffer/spmd_sync_start_mix_spill
tests/st/a2a3/tensormap_and_ringbuffer/spmd_sync_start_stress
```

结果为：

```text
7 passed, 14 warnings in 36.32s
```

特别是`spmd_starvation`在干净runner中通过，证明全量尝试里的第二个失败确属级联。把首次运行中已完成
节点与这次干净尾部覆盖按节点去重后，65个A2/A3 L2节点的当前证据是：63个通过、1个skip、1个
可独立复现的`s8192`失败。这个结果足以说明没有出现一片新的L2功能回归，但不冒充65/65全绿。

#### 10.60.4 A2/A3 L3选择集7/7通过

随后运行A2/A3 runtime level-3选择集。只传`--device=0`时，6个单卡可执行节点全部通过，唯一error
来自`TestL3Group`测试自身要求两张设备而setup缺少第二个device：

```text
6 passed, 72 deselected, 1 error in 31.04s
```

检查`npu-smi info`确认NPU process table没有列出使用者后，仅对该双卡节点显式使用
`--device=0-1`重跑：

```text
1 passed in 9.66s
```

因此A2/A3 L3选择集的最终设备证据是7/7通过：6个device0节点加1个device0-1 group节点。
运行结束出现的Python resource-tracker“shared memory已unlink”告警没有改变测试结果，按既有L3
进程清理告警记录，不作为功能失败。

#### 10.60.5 legacy L2 reset之后的L1最终复验

由于L2失败恢复期间真实发生过两次device reset，文档收口前再次检查`npu-smi`：process table为
`No running processes found`。随后只在A2/A3 device0运行基础L1 ACLGraph与HBG no-reset fault ST：

```text
pytest -q -s \
  tests/st/runtime/l1/test_l1_aclgraph.py \
  tests/st/runtime/l1/test_l1_hbg_fault_injection.py \
  --platform=a2a3 --device=0

4 passed, 4 deselected, 1 warning in 59.76s
```

选中的4项覆盖TRB eager/warmup/capture/replay、HBG eager/warmup/capture/replay、同一context中两个
HBG callable的独立task package与重复replay，以及HBG 16阶段故障后caller tail、同context恢复和
最终ACLGraph replay。它证明两次legacy L2 reset之后GPT分支的A2/A3 L1主链仍可正常初始化、执行和
关闭；测试自身没有调用device reset。4个deselection仅来自A5平台参数过滤，符合当前范围。

#### 10.60.6 当前兼容性结论

截至本节，A2/A3范围内可以给出以下精确结论：

1. TRB L1与HBG L1的eager、ACLGraph、双callable、单算子stream边界、真实CANN禁止API trace和
   16阶段no-reset故障恢复已经有device0证据；
2. A2/A3 L3当前选择集7/7通过；
3. A2/A3 L2的65节点中按独立进程去重后63个通过、1个skip，唯一根失败是可单独复现的
   `b1_h32_kv8_s8192_bs128_fp16`；由它污染的`spmd_starvation`已在干净进程通过；
4. 临时L1 polling诊断改动已完全回滚，两个git仓都没有遗留源码差异；
5. 因此当前没有发现由L1实现引入的A2/A3 L2/L3功能回归，但“L2全量绿色”仍有上述一个明确例外，
   不能在交付说明中省略；A5实机与A5 simulator不属于当前结论。

### 10.61 只读对照Grok实现并补充跨方案回归矩阵

#### 10.61.1 对照范围与隔离边界

按用户要求，只读检查`/mnt/workspace/inductor/pto/pypto`及其`runtime`子仓，没有修改、checkout、
clean或提交Grok工作树。检查时Grok顶层`main`相对`origin/main`有15个本地提交，另有其自身未跟踪的
`build.pre-main-upgrade/`；runtime位于`l1-aclgraph`，从其阶段基线到HEAD有14个L1/HBG提交。
这些状态全部保持原样。

对照重点不是把两套实现机械合并，而是逐项核对Grok提交中出现的行为是否在GPT树缺失：

| Grok侧做法或发现 | GPT侧结论 | 本轮处理 |
| --- | --- | --- |
| `ctx.prepare()`、`op.warmup()`、`prepared/warmed/closed` | GPT已有正式API，并区分“成功入队”与device完成 | 不重复移植 |
| 所有`__call__`都禁止隐式prepare | 与既定“普通eager可自动prepare，ACLGraph用户必须显式prepare/warmup”契约冲突；GPT又禁止capture query，不能在内部猜capture状态 | 不采用；保留GPT契约 |
| Python全局集合拒绝同device第二context | GPT在native按device持有唯一lease，覆盖Python/direct C ABI及初始化失败回滚，边界更完整 | 不采用Python影子owner |
| close幂等且借用设备不reset | GPT已有retryable Closing状态、幂等close和no-reset fault matrix | 不重复移植 |
| launch携带per-callable `func_id`表 | GPT的TRB invocation与HBG immutable package均已有callable-local函数表、hash、capacity/identity校验；双callable同从`func_id=0`起编号已上板 | 不替换更强实现 |
| HBG graph作为WithHostArgs tiling blob，每次replay恢复 | GPT已有canonical plan、serialized scratch、runtime-owned snapshot、working slot和每代restore，且有large-args/双graph/no-reset证据 | 不退回单blob模型 |
| `RunConfig.runtime`进入kernel config/JIT key | GPT已有规范runtime名称、artifact校验、目录owner和跨runtime防覆盖 | 不重复移植 |
| Grok未接taskQueue adapter，直接读取current stream | GPT生产路径已使用`.stream(false)`的独立torch_npu adapter，并持有descriptor/tensor lease及`recordStream` | 保留GPT实现 |
| 泛化ST覆盖同一callable双节点、不同capture stream、FP16及非均匀输入 | GPT原矩阵分别覆盖双callable、双graph、scalar/multi-output/workspace，但没有把这三个组合做成独立回归 | 借鉴并新增A2/A3 ST |

因此本轮没有发现一项“Grok生产实现正确而GPT生产实现缺失”的功能性修复点；真正值得吸收的是测试场景，
尤其是同一callable在一个graph里形成两个不同captured node时，每个node必须保有独立参数快照，不能因
共享callable/working slot退化成最后一次调用。

#### 10.61.2 新增跨方案回归

新增`tests/st/runtime/l1/test_l1_cross_session_matrix.py`。它没有照抄Grok测试的teardown，而是继续执行
GPT已经确定的严格所有权顺序：外部device quiescence，逆序`graph.reset()`，最后`context.close()`；
所有graph-bound tensor也一直保活到reset之后。三个场景分别在TRB与HBG运行：

1. **同一callable在同一ACLGraph连续调用两次。** 第一节点写intermediate，第二节点读取它并写最终
   output；8轮replay使用`arange`、横向flip、逐轮bias等非均匀数据，验证两个captured node的tensor
   地址/参数快照与caller/hidden-stream串行边界没有折叠；
2. **FP16 tensor replay。** 使用test-local FP16乘法kernel，三轮正数、小数和负数输入，证明L1
   descriptor、task package和replay路径不是只在FP32下碰巧成立；
3. **两张graph使用两个不同capture stream。** graph A生成intermediate，外部等待后graph B消费它；
   三轮严格顺序replay验证外部已证明quiescent时切换stream不会把旧graph tail错误导入新capture，且
   不声称支持并发graph replay。

静态门禁结果：

```text
ruff check: All checks passed
ruff format --check: 1 file already formatted
py_compile: passed
pytest --collect-only --platform=a2a3: 6 selected, 6 deselected
git diff --check: passed
```

确认`npu-smi`进程表为`No running processes found`后，在device0运行：

```text
pytest -q -s tests/st/runtime/l1/test_l1_cross_session_matrix.py \
  --platform=a2a3 --device=0

6 passed, 6 deselected, 1 warning in 31.23s
```

6个选中项即三个场景乘TRB/HBG；6个deselection均为A5平台参数，符合当前只管A2/A3的范围。该结果
没有要求或调用device reset，也没有修改runtime生产代码。结论是：Grok的核心功能点在GPT中已有
同等或更强闭环，本轮吸收的是能增加交叉实现置信度的设备反例，而不是降低GPT的taskQueue、lifetime、
fail-closed或no-reset约束来追求表面代码一致。

### 10.62 复核Grok比较结论并补齐异构算子上板证据

#### 10.62.1 比较文字引用了GPT的过期阶段快照

用户转发的Grok评价对两边基本红线的归纳是准确的：两边都不查capture state、不拿graph handle、
不用`rtStreamAddToModel`、不在正常launch里做H2D/sync/device allocation，close不reset借用的device。
它对GPT的taskQueue、失败owner和callable-local `func_id`评价也与当前源码一致。但下列判断引用的是
10.43～10.45之间的中间状态，不是本轮HEAD：

1. **“HBG长期`supported=0`”已过期。** A2/A3 HBG L1 capability已在硬门槛通过后打开，普通eager、
   ACLGraph、双callable和故障注入均走正式高层API，不是实验私有入口；
2. **“第二个HBG context仍因`status=7 Conflict`未闭环”已过期。** 该真实问题曾经被GPT上板发现，
   之后通过context generation有序reset resident registry修复；device1已证明同一Host进程的第二context可重新
   从`callable_id=0`/`func_id=0`注册并完成capture/replay；
3. **“TRB→HBG不能共进程”已过期。** HBG orchestration/runtime DSO的ELF符号隔离已修复跨runtime
   symbol preemption；同一进程先TRB再HBG的首context和后续context已有设备证据；
4. **“CANN大HostArgs/captured-node lifetime和working-slot restore没有板上证据”已过期。** 独立探针
   已扫描64 KiB、1 MiB、16 MiB和64 MiB，包括64 MiB captured graph、2048个压力task、双graph各100次交替
   replay和graph destroy后回收；正式HBG又验证了每代full restore、双package和no-reset fault matrix；
5. **“GPT上板只有一条黄金路径”也不再成立。** 当前A2/A3证据已包括TRB/HBG、同callable双node、
   双callable、双graph、multi-output、internal workspace、scalar、FP16、非均匀输入、8次replay、stream边界压力、
   真实CANN API trace和16阶段HBG no-reset故障矩阵。

上述并不否定Grok在较早时间点更快拿到HBG简单算子验数；它只是阻止用过期快照反向得出“当前GPT
HBG仍不可用”的结论。当前对比必须分开“曾经暴露并修复的问题”和“HEAD仍未修的问题”。

#### 10.62.2 Grok HBG blob并不是比五层owner更简单的等价实现

只读核对Grok当前A2/A3 HBG实现后，其`L1HbgGraphHeader + SM + arena`确实作为
`aclrtLaunchKernelWithHostArgs`的tiling-class blob跟随task。但设备执行时并不是“不可变source每次恢复到
另一份working slot”：AICPU直接把CANN task args里的SM/arena地址挂到Runtime，执行就地修改这份blob，
下一次replay再由`reset_l1_hbg_execution_state()`手工重置选定的completion flag、slot state、watermark和ready queue。
`host_done_mask`只有64 bit，当前默认task window又为16；这个模型已能支持它实测的小图，但它对“所有被
scheduler/runtime_destroy消费的可变字段”的完整性依赖人工枚举，与full pristine restore不等价。

GPT当前保留五个独立owner，不是为了抽象层次而抽象：

1. canonical `GraphPlan`是Host不可变的构图结果和hash trust root；
2. writable serialized scratch只用来生成一次WithHostArgs launch image，允许CANN做placeholder patch，不污染plan；
3. CANN按eager task/captured node持有各自的immutable HostArgs source snapshot，因而graph A/B不会被后一次host build覆盖；
4. context内只有一份mutable device working slot，v1在外部保证无并发时允许复用；每次eager/replay由AICPU leader
   在Start之后、dispatch之前完整restore pristine SM与runtime arena；
5. Runtime/KernelArgs、workspace、binary、hidden stream/events属于context-resident resources，只在graph已reset且外部quiescent后close。

这个模型与用户对“HBG graph就是该task的AscendC tiling参数”的约束一致，又额外处理了HBG image会被
执行消费的特性。Grok的板上结果证明了“WithHostArgs可以承载这类图包”，是有价值的交叉证据；
它不证明应该把GPT的immutable source与mutable slot合并，因此本轮不迁移这一实现。

#### 10.62.3 真正值得吸收的是算子多样性反例

Grok的泛化矩阵在ReLU、SiLU、真实matmul多child和64×64小tile上比GPT早先的test-local add/mul更广。
这不是可以靠阅读代码消除的差距，因此继续扩展
`tests/st/runtime/l1/test_l1_cross_session_matrix.py`，添加两组每组同时参数化TRB/HBG的case：

1. 同一context注册`fused_add_relu`和`SiLU`，同一ACLGraph依次调用两个不同shape/不同运算链的
   callable，三轮用非均匀linspace和正负offset同时验证ReLU与SiLU；
2. 第一个callable使用真实64×64 cube matmul + vector bias，并硬断言`child_count >= 2`；第二个callable
   是独立64×64 tile add，同一graph建立`matmul+bias -> add`真实tensor依赖。三轮replay更换矩阵、
   对角rhs、bias和tail，直接检验两张差异很大的callable-local函数表、multi-child internal workspace、
   HBG package和working-slot restore，不再用add+mul作为唯一反例。

所有case继续使用严格销毁顺序：先外部device synchronize，再`graph.reset()`，最后`context.close()`，
并保持所有graph-bound tensor的强引用至graph reset完成。没有照搬Grok测试中直接`ctx.close()`且让graph
稍后随Python析构的宽松lifetime。

#### 10.62.4 环境误配拦截与device0最终结果

首次执行时虽按要求source了`.claude/skills/testing/load-env.sh`，但该脚本在当前Git上会将
`--path-format=absolute`当作普通输出，`dirname`因而报错，机器环境未真正载入。结果进程误用user-site
Torch 2.12/Torch-NPU 2.12，且PATH上没有PTOAS：

- nonlinear两case在`pypto_init`前被adapter build-version检查拒绝；
- matmul两case生成`skip_ptoas=True`的compile-only artifact，在访问`chip_callable`时被缺少
  `kernel_config.py`检查拒绝；
- 四个case都没有创建PyPTO L1 context，也没有launch PyPTO AICPU/AICore；但测试在拦截前已创建
  普通NPU tensor，因此只能精确写为“无PyPTO device task”，不写成“无任何NPU行为”。

本轮同时修正了loader的Git 2.25兼容性：不再使用该版本尚不支持的
`git rev-parse --path-format=absolute`，而是读取`--git-common-dir`后显式把相对路径归一化为绝对路径。
`bash -n`、工作树根目录source和`tests/`子目录source全部通过，两者都安静得到默认
`PYPTO_BUILD_JOBS=2`/`PYPTO_TEST_JOBS=2`。主工作树当前没有忽略的`testing.env`，所以按testing skill
保留`unclassified`的保守默认值；该loader只负责机器资源上限，不会选择Torch/PTOAS，下述显式
Python与工具隔离仍是必需的。

随后保留source脚本这一要求，另外显式固定隔离环境：

```text
PYTHONNOUSERSITE=1
PYTHONSAFEPATH=1
PYTHONPATH=gpt_pypto/python:gpt_pypto/runtime/python:gpt_pypto/runtime:gpt_pypto/tests/st
PTOAS_ROOT unset
PATH first = PTOAS/build-v0.57-llvm21-cann9.2-clean/tools/ptoas
torch       = 2.7.1+cpu
torch_npu   = 2.7.1.post4
adapter     = built against 2.7.1+cpu / 2.7.1.post4
```

静态结果为Ruff check/format、`py_compile`、`git diff --check`全部通过，collect-only得到10个A2/A3
选中项和10个A5 deselection。用户已明确授权使用device0；测试前`npu-smi`的process table为
`No running processes found`，本轮不执行device reset。先只跑新增四项：

```text
4 passed, 16 deselected, 1 warning in 30.11s
```

再在同一新pytest进程中运行整份交叉矩阵：

```text
10 passed, 10 deselected, 1 warning in 60.22s
```

10个选中项即五个场景乘TRB/HBG；除新增四项外，同进程还重新验证了同callable双node、
FP16和不同capture stream的两张graph。因此本轮“有则改之”的具体产物是四个更强的A2/A3设备
回归case，不是生产实现修补；对比和上板均没有发现一个Grok已正确解决而GPT HEAD仍缺失的A2/A3
P0/P1代码问题。

### 10.63 GPT实现转正为本地main并隔离保留Grok实现

#### 10.63.1 转正前冻结点

用户确认以GPT工作树作为后续正式实现，并要求把原Grok实现保留到独立分支和独立目录。操作前先确认
四个仓库均不存在未提交的tracked source变更，冻结点如下：

| 实现 | 仓库 | 转正前分支 | 冻结提交 |
| --- | --- | --- | --- |
| GPT | PyPTO顶层 | `gpt/pypto-l1-aclgraph` | `4426b1a7955caf28f04b93728459cefe0b038596` |
| GPT | `runtime`/simpler | `gpt/pypto-l1-aclgraph` | `f48d7c290e0979f572fa6e55a21f652f04833886` |
| Grok | PyPTO顶层 | 原本地`main` | `2adf71b1eded56e2d200fdb7b715845130fd97df` |
| Grok | `runtime`/simpler | 原`l1-aclgraph` | `35a811def1a8da944d69bbdee7207b158550fe6c` |

Grok顶层唯一额外状态是其既有的未跟踪`build.pre-main-upgrade/`，没有把它删除、提交或移动到GPT树。
本轮没有push任何远端，也没有重写、squash或丢弃任一实现的提交历史。

#### 10.63.2 分支和可见路径的最终关系

先在原Grok顶层和runtime各自建立并checkout `grok/pypto-l1-aclgraph`，再把两个本地`main`引用分别
fast-forward到GPT冻结提交。随后在同一文件系统内只用目录rename完成可见路径交换，最终关系为：

```text
/mnt/workspace/inductor/pto/pypto
  top branch:     main
  top base HEAD:  4426b1a7955caf28f04b93728459cefe0b038596
  runtime branch: main
  runtime HEAD:   f48d7c290e0979f572fa6e55a21f652f04833886

/mnt/workspace/inductor/pto/grok_pypto
  top branch:     grok/pypto-l1-aclgraph
  top HEAD:       2adf71b1eded56e2d200fdb7b715845130fd97df
  runtime branch: grok/pypto-l1-aclgraph
  runtime HEAD:   35a811def1a8da944d69bbdee7207b158550fe6c
```

因为当前Git 2.25不提供新版本的完整`worktree repair`能力，而且GPT linked worktree内初始化了
`runtime`、`libbacktrace`和`msgpack-c`三个子仓，rename后逐一修复了下列管理关系，而不是重新clone或
复制二进制：

1. GPT顶层`.git`指向移到`grok_pypto/.git/worktrees/gpt_pypto`的共享管理目录；
2. 共享管理目录的反向`gitdir`指向新的`pypto/.git`；
3. GPT runtime的`.git`和其admin `core.worktree`都改为新的`pypto/runtime`；
4. GPT的`libbacktrace`、`msgpack-c`做同样的双向修复；
5. Grok主worktree随整个目录移动，runtime和`libbacktrace`的相对指针天然仍成立；其`msgpack-c`
   原来使用绝对路径，改成与另一子仓一致的相对指针。

验证`git worktree list --porcelain`后，Git把`grok_pypto`识别为
`grok/pypto-l1-aclgraph`，把`pypto`识别为`main`；两边`git submodule status`都精确匹配各自顶层
gitlink。共享Git admin目录物理上仍位于最初的primary worktree `grok_pypto/.git`，这是linked-worktree
的管理实现细节，不会改变`/pto/pypto`作为正式main工作目录的分支、源码或构建归属。

#### 10.63.3 构建与Python editable隔离

目录rename后，旧CMake cache仍记录各自原绝对源码路径；如果继续原地使用，Grok cache甚至会把新的
`/pto/pypto`误当成自己的source。为避免两套实现串产物，没有修改或删除旧cache，而是将默认cache和
旧GPT虚拟环境移到仓库外的可恢复目录`/mnt/workspace/inductor/pto/worktree_build_backups/`：

```text
grok-2adf71b1-top-build
grok-35a811de-runtime-build
pypto-main-4426b1a7-top-build
pypto-main-f48d7c29-runtime-build
pypto-main-f48d7c29-runtime-cpp-ut-build
pypto-main-f48d7c29-venv
```

随后在正式`/pto/pypto`下新建Python 3.11 virtualenv，并从当前source分别执行simpler和PyPTO的
`--no-build-isolation -e`安装。构建过程遵守`.claude/skills/testing/load-env.sh`给出的保守
`PYPTO_BUILD_JOBS=2`限制；`nanobind`固定为旧环境已经使用的`2.13.0`。新生成的CMake cache根分别是
`/pto/pypto`和`/pto/pypto/runtime`，editable `.pth`也只包含这两个新路径。实际import来源为：

```text
pypto          /mnt/workspace/inductor/pto/pypto/python/pypto/__init__.py
simpler        /mnt/workspace/inductor/pto/pypto/runtime/python/simpler/__init__.py
pypto_core     /mnt/workspace/inductor/pto/pypto/runtime/.venv/.../pypto_core*.so
_task_interface /mnt/workspace/inductor/pto/pypto/runtime/.venv/.../_task_interface*.so
```

没有从旧GPT/Grok build目录复制任何`.so`。Grok树当前也不再有名为`build/`的默认cache；需要复核Grok
历史实现时必须在`grok_pypto`内部重新配置，因而不会无意链接正式main的source或产物。

#### 10.63.4 转正后的快速回归

在新路径、新virtualenv及显式`PYTHONNOUSERSITE=1`下运行：

```text
tests/ut/runtime/test_l1.py
runtime/tests/ut/py/test_l1_chip_worker.py

57 passed, 1 environment warning in 0.27s
```

warning仍是系统`torch_npu`安装目录中`libop_plugin_atb.so`的owner不匹配提示，与路径交换和本次源码
无关。至此正式main、Grok保留分支、两个可见目录、嵌套runtime仓、submodule、editable安装和默认build
目录均完成隔离；后续实现与A2/A3上板默认只在`/mnt/workspace/inductor/pto/pypto`进行。

### 10.64 远端发布关系

用户进一步明确：PyPTO直接发布到`nalinaly/pypto`的`main`；simpler不改写其当前远端main，必须以
PyPTO起步时gitlink固定的`3165cc89b6ea6b58a0bc01cbec2d5f72f2029c35`作为历史基线，发布到
`nalinaly/simpler`的独立分支。最终按依赖顺序完成：

1. 在simpler创建`pypto-l1-aclgraph`，确认其merge-base精确为`3165cc89`，其上保持29个原始顺序
   提交，不rebase、不merge当前upstream/main，也不force push；
2. 将该分支推送到`https://github.com/nalinaly/simpler.git`，远端HEAD为
   `f48d7c290e0979f572fa6e55a21f652f04833886`；正式工作目录中的runtime也checkout该分支并跟踪
   `nalinaly/pypto-l1-aclgraph`；
3. simpler gitlink远端可获取后，把PyPTO本地main从远端`8ead22af`普通fast-forward推到
   `52d2d62f`，没有force push。

因此最终发布拓扑是`nalinaly/pypto:main`引用`nalinaly/simpler:pypto-l1-aclgraph`中的
`f48d7c29`。simpler的`origin`仍保留为`hw-native-sys/simpler`用于对照，新增`nalinaly` remote只承载
本次PyPTO L1分支。

### 10.65 以最新simpler/main为基线完成正式发布迁移

本节记录2026-08-19的最终迁移状态，并明确取代10.64中“simpler保留在基于`3165cc89`的独立
`pypto-l1-aclgraph`分支”这一过渡发布关系。用户最终要求是：

1. `https://github.com/nalinaly/simpler`的`main`先强制对齐主仓
   `https://github.com/hw-native-sys/simpler`的最新`main`；
2. L1/ACLGraph修改直接提交在该最新`main`之上，并推送到fork的`main`；
3. PyPTO同步迁移到最新simpler的Buffer/ChipTensor ABI，submodule直接固定fork的L1提交；
4. 当前只把A2/A3列为实现和验收范围，不以A5或A5sim构建结果阻塞发布。

#### 10.65.1 simpler fork main的强制对齐与安全备份

操作前读取两个远端引用并冻结旧fork状态；没有把旧`main`直接丢弃，而是先把它保存在远端
`main-backup-20260819`。随后将fork的`main`用带lease保护的force push精确对齐主仓：

```text
upstream/origin main: 93a0fde014f2206ad3b71b31109698aefb7064ea
fork backup branch:   main-backup-20260819
fork aligned main:    93a0fde014f2206ad3b71b31109698aefb7064ea
```

L1实现迁移、冲突处理、A2/A3构建与测试完成后，在这个新基线上形成一个可审阅的原子提交：

```text
repository: https://github.com/nalinaly/simpler.git
branch:     main
commit:     4922d5933e2937790aa5b01e737986114ac28d1d
subject:    Add: support borrowed L1 ACLGraph execution
```

本地`git ls-remote nalinaly refs/heads/main`已确认远端引用精确为`4922d593`；正式PyPTO工作树中的
`runtime`也checkout这个`main`，不再跟踪旧的`pypto-l1-aclgraph`过渡分支。

#### 10.65.2 HBG resident global迁移为Context-owned registry

这次迁移没有把旧resident DSO中的进程级可变registry继续带到最新simpler。最终所有权被收敛为
`DeviceContextHandle`/L1 context拥有的`HbgContextRegistry`，其核心约束如下：

1. callable registration、callable-local `func_id -> kernel address`表、graph package目录、mutable working
   slot和generation都属于一个L1 context；不同context不再共享可写registry；
2. resident orchestration SO只保留执行代码和显式传入的context入口，不再作为callable或graph package
   的隐式owner；context close后不会留下可与下一context发生`status=7 Conflict`的注册表状态；
3. generation用于同一context内部识别每次execution image，不再承担跨context清理进程级global的职责；
4. registry销毁仍遵守L1 borrowed-resource契约：调用方必须先让caller/hidden stream quiescent，并销毁或
   reset所有引用该operator的ACLGraph，然后才允许显式`context.close()`释放slot、Runtime和binary；
5. close中任一资源释放失败，context保持Closing/可重试owner，不能释放registry所有权后让第二context
   进入半拆状态。

这个变化不是单纯把一张map从全局变量搬到成员变量。它把“注册身份、graph package、working state、
generation和资源回收”放进同一个生命周期事务，从结构上消除resident DSO跨context污染；同时保留
同一context内多个`@pl.program`都从`func_id=0`开始编号的合法性，因为函数表按callable隔离。

#### 10.65.3 当前HBG graph package的生命周期

HBG graph被当作本次L1 task的tiling-class入参管理，但必须区分不可变source与执行时会被scheduler修改的
working state。当前正式路径的owner关系是：

1. **Canonical GraphPlan**：Host build得到不可变拓扑、初始shared memory、runtime arena、function
   binding、tensor/scalar布局和完整性元数据；它是构图结果与校验的trust root。
2. **Writable serialization scratch**：每次形成launch image时，从GraphPlan生成一份可写序列化副本。
   `aclrtLaunchKernelWithHostArgs`允许runtime对placeholder做patch，因此不能把canonical plan本体直接
   交给runtime修改。
3. **Runtime-owned task snapshot**：成功launch后，完整HostArgs由CANN跟随该eager task或captured node
   保存，等价于AscendC launch中由runtime管理的tiling参数。下一次Host调用可以立即复用自己的临时
   serialization scratch，不会覆盖前一个node的图包；PyPTO不依赖固定kernel-launch数量，也不按Host
   launch返回时机回收device task args。
4. **Context-owned mutable working slot**：graph image不能原地重复执行，因为scheduler会消费ready queue、
   completion flags、watermark和Runtime指针。每次eager执行或ACLGraph replay时，AICPU leader都在Start
   event之后、scheduler dispatch之前，把该node snapshot中的pristine SM与arena完整恢复到context的
   fixed working slot。v1不允许同一context并发，因此slot与workspace可以按context单份复用。
5. **Context-resident execution resources**：Runtime、KernelArgs、workspace、AICore report区、hidden stream、
   events和binary都由context持有。PyPTO不查询capture状态、不获取graph handle、不调用
   `rtStreamAddToModel`、不在launch中H2D/分配/synchronize/reset；graph-bound tensor由torch adapter和
   allocator stream recording覆盖device消费期，context则必须由用户持有到graph reset之后。

因此一次capture中的两个HBG node即使来自同一callable，也各自拥有独立的CANN task snapshot；两个
callable还各自拥有独立function table与GraphPlan。它们可以串行复用context working slot，但不会退化成
“所有captured node都指向最后一次Host build图包”。若未来允许L1并发，必须把working slot/workspace提升为
per-inflight execution lease；不能只移除当前的并发检查。

#### 10.65.4 PyPTO对最新simpler ABI的迁移

PyPTO没有在旧`3165cc89` ABI上继续打补丁，而是吸收主仓已经完成的soft SYNCALL与Buffer ABI迁移
`8662deb9`，并按当前L1分支解决冲突。本地正式迁移提交为：

```text
PyPTO base before migration: d920dd37
upstream migration source:   8662deb9
local integrated commit:     245cadb1
simpler gitlink:              4922d5933e2937790aa5b01e737986114ac28d1d
```

迁移后的关键事实包括：

1. L1参数打包使用最新`ChipTensor.make_strided`，不再引用旧`Tensor` wire对象；
2. L2/L3走worker-owned Buffer/ChipTensor语义，PyPTO不再把旧address-owning Tensor ABI混入新simpler；
3. HBG requirements仍由orchestration codegen、Python binding、类型桩、artifact和simpler validator完整贯通；
4. distributed dispatch只有存在`domain_provider`时才传该可选关键字，保持旧mock/第三方dispatch callable
   的源兼容；为避免`runtime.__init__`的既有导入环，`RunConfig`继续显式使用有说明的lazy import；
5. cherry-pick中来自另一基线的rank-reducing orchestration slice片段依赖当前分支不存在的`drop_dims`
   前置改动，已经按本分支原有feature边界剔除；这不是回退当前main已有功能，而是避免把半套无定义实现
   带入正式提交。

#### 10.65.5 最终验证与范围

最新simpler上的A2/A3正式路径已有device0证据：TRB golden、HBG golden、双callable独立package三项
同进程测试全部通过，HBG连续创建第二context的generation/registry反例也通过。测试前确认设备无其他
使用者，没有执行device reset。A5与A5sim按用户明确范围不运行、不作为本次结论依据。

PyPTO迁移后重新构建editable extension，并使用全新`PYTHONPYCACHEPREFIX`消除目录rename遗留字节码，
完整执行runtime与codegen无硬件回归：

```text
tests/ut/runtime + tests/ut/codegen
1555 passed, 63 warnings in 223.26s
```

此前普通cache下的34个golden/path失败已证明来自`.pyc`中残留的旧`/pto/gpt_pypto`路径；新cache下全部
消失。另5个distributed mock兼容失败由上述可选`domain_provider`调用修复。最终pre-commit的header、
英文、文档一致性、YAML、EOF/空白、clang-format、cpplint、ruff和pyright全部通过，`git diff --check`
也通过。

最终发布拓扑因此变为：

```text
nalinaly/pypto:main
  -> runtime gitlink 4922d593
     -> nalinaly/simpler:main
        -> based directly on hw-native-sys/simpler:main 93a0fde0
```

旧`nalinaly/simpler:pypto-l1-aclgraph`仍可用于历史追溯，但不再是正式PyPTO main的依赖或后续开发
基线；旧fork main则由`main-backup-20260819`提供可恢复引用。

#### 10.66 Triton风格L1 JIT、无固定callable上限与binary pin收口

##### 10.66.1 最终决策冻结

本轮以下决策被明确固定，实现不再保留相反的隐式路径：

1. 普通用户使用 `@pl.jit(execution="l1", runtime=...)` 和直接Python call，不接触
   `pypto_init/context/operator/prepare/warmup/close`。manual API仅作advanced/debug控制面保留。
2. eager可省略pure Out，由PyTorch wrapper调用 `torch.empty` 和torch allocator。ACLGraph capture要求
   输出已在图外分配并显式传入。
3. 不提供公开global init或batch prepare；第一次ordinary eager自动完成init/prepare/warmup。
4. 首次调用不能在capture内；PyPTO仍不query capture，失败信息要求用户先在图外warmup。
5. scalar仅使用现有 `pl.Scalar[...]`，不新造constexpr/scalar表达。
6. 公开的64-callable限制删除，但单callable child-kernel数量和单次tensor/scalar ABI容量保留。
7. HBG launch package自包含；TRB code registry动态append，id不复用，code不覆盖。
8. 当前不尝试解决graph replay并发；可观测的host并发尽力fail-fast。
9. `pypto.l1.shutdown(device=...)` 完全可选、幂等、不做sync；失败保留owner供重试。
10. 不调用shutdown时默认pin到进程结束；GC/atexit不调runtime close。
11. CANN没有公开契约保证captured graph引用funcHandle时binary unload后继续保活，因此新L1路径
    不允许任何 `aclrtBinaryUnLoad` 或 `rtsBinaryUnload`。
12. 本阶段只以A2/A3为实现与验收门槛，不要求A5或A5sim。

##### 10.66.2 Python公开调用面落地

`python/pypto/jit/decorator.py` 现在将execution选择与现有specialization cache直接结合：

- `_JITDecorator.__call__` 接受 `execution="default"|"l1"` 和L1-only `runtime`。
- `JITFunction` 使用一个internal `execution_config` tuple保存模式和runtime，避免扩大已经较长的
  constructor参数表。
- L1 `compile/lower` 默认到A2/A3 onboard和decorator固定runtime；dispatch之前拒绝platform、runtime、
  distributed、device和codegen-only冲突。
- pure Out全省略时解析静态annotation并用第一个tensor的device调 `torch.empty`；显式Out原样返回。
- 部分Out省略直接fail，不制造部分内部所有的输出集。

新文件 `python/pypto/runtime/l1_jit.py` 提供模块级strong device owner registry。owner绑定device、
platform、runtime、L1Config和owner thread；第一个specialization建context，后续specialization通过
`L1Context.add_program()` append。host dispatch/shutdown用non-blocking invoke lock检测可观测并发。

`python/pypto/runtime/l1.py` 将manual和JIT路径共享的事务收敛到同一个实现：

- `prepared` 从context一次性状态改为per-callable状态。
- `add_program()` 允许第一个callable已launch后追加第二个。
- canonical `ChipCallable` bytes以SHA-256去重，身份id单调不复用。
- 张量shape/dtype/stride/device、scalar bit pattern、output和allocator lease的已有强校验不被绕开。

##### 10.66.3 native append admission与64-cap移除

`L1ExecutionState::seal()` 保留旧ABI函数名，但成功launch后的live phase保持 `ReadyEnqueued`。
prepare/launch仍受同一operation mutex、execution mode和close/poison状态保护，但“已launch”不再等价于
“禁止新callable”。

Host `DeviceRunnerBase` 对borrowed L1只要求non-negative `int32_t callable_id`；同一函数中legacy L2/L3仍保留
`MAX_REGISTERED_CALLABLE_IDS`检查。Python simpler wrapper的L1 id校验也改为 `[0, INT32_MAX]`，而旧
`ChipWorker.register_callable()` 仍使用L2/L3的64-slot allocator。

这保证了“删除L1公开上限”不会偷偷改变L2/L3 wire和资源行为。

##### 10.66.4 HBG package自包含

HBG路径已删除context中的fixed callable registry和prepare-time resident callable registration。
`HbgAicpuInvocationView` 仅依赖：

1. CANN-owned launch blob中已seal的identity、argument snapshot、function-binding hash/table和pristine regions。
2. Context-owned execution-slot registration中的working destination、Runtime/KernelArgs地址和generation。

`HbgContextRegistry` ABI minor升级且只保留execution slot。旧
`simpler_aicpu_l1_hbg_register_callable` 导出作为无resident state的compatibility shim保留，但新runtime
symbol列表不再请求或launch它。

每个captured node因此自带一份类AscendC tiling data的graph package；新callable的prepare不会改写旧node
的callable identity或function table。可变scheduler state仍由AICPU leader每次从package的pristine image恢复到
context working slot，所以replay不是对上一轮已消费image的继续执行。

##### 10.66.5 TRB dynamic code registry

A2/A3 TRB AICPU中legacy `orch_so_table_[64]` 仅服务L2/L3。L1新增 `L1OrchSoNode` 单链表：

```cpp
struct L1OrchSoNode {
    int32_t callable_id;
    uint64_t callable_hash;
    OrchSoEntry entry;
    L1OrchSoNode *next;
};
```

`L1RegisterCallableArgs` ABI version升为2并增加非零 `callable_hash`。register task先分配未发布node、
复制callable-local kernel table、`dlopen/dlsym`，全部成功后才链入head。相同id再注册必须hash、
kernel count和全部binding都一致；否则报immutable conflict。

当前lookup是O(N)，每个node一次heap allocation，数量与映射资源没有公开固定上限。这是已认可风险：
在没有graph-aware release信号时，宁可增长也不覆盖旧graph可能在未来replay所需的code。后续应先做
byte/count观测和长时压测，再根据数据选择chunked stable index，不先发明无法安全驱逐的LRU。

##### 10.66.6 BinaryUnLoad零调用契约

`LoadAicpuOp::FinalizeL1Pinned()` 与legacy `Finalize()` 分开。L1 close的顺序是：

1. 先锁定Closing，使所有新prepare/launch fail closed。
2. 在调用者已外部quiesce的前提下释放bootstrap辅助device buffers。
3. 任一free失败立即保留后续owner，允许显式close重试。
4. 成功后只忘掉host loader中的binary/function handle记录，使destructor不会调legacy unload。
5. L1新路径不执行 `aclrtBinaryUnLoad` 或 `rtsBinaryUnload`。

源码guard test同时检查 `finalize_l1_borrowed()` 只调 `FinalizeL1Pinned()`，以及该函数body中不存在
任何BinaryUnLoad拼写。Legacy L2/L3 `Finalize()` 仍保持原有unload行为，不被这条L1契约改写。

##### 10.66.7 无硬件结果

完成的定向回归：

```text
JIT decorator + compile extraction + L1 facade: 169 passed
L1 Python/taskQueue/lifecycle/source guards:     62 passed
simpler C++ non-hardware ctest:                 120/120 passed
A2/A3 onboard TRB/HBG host/AICPU/AICore:        all six target classes built
ruff / clang-format / git diff --check:          passed for touched files
```

新测试 `tests/ut/jit/test_l1_jit_api.py` 覆盖decorator参数、默认lower、explicit/implicit Out、部分Out
拒绝、hidden owner append和shutdown retry。旧 `test_l1.py` 增加了超过legacy 64的Python admission以及
首次warmup后late append。simpler UT覆盖ABI v2 hash、HBG无resident callable view、context registry ABI和pinned
close source guard。

##### 10.66.8 device0 A2/A3上板结果与一次stale binary假失败

新增 `tests/st/runtime/l1/test_l1_jit_aclgraph.py`，通过 `PYPTO_L1_JIT_TEST_RUNTIME` 在两个全新进程分别验证
TRB和HBG。每个进程的最终矩阵是：

1. `L1 add` 首次eager，省略out并验torch allocator结果。
2. 在add已launch之后首次调 `L1 mul`，验证新specialization的late append。
3. 外部device sync。
4. 独立stream capture `torch.add -> L1 add -> L1 mul -> torch.add`。
5. 三组输入连续replay和逐次验数。
6. 先quiesce，再graph reset，最后可选shutdown。

第一次扩展为two-callable后，TRB的第二callable prepare曾在taskQueue callback返回 `-5`。源码状态
与C++ UT不一致，排查发现 `RuntimeBuilder` 实际加载的 `runtime/build/lib/.../libhost_runtime.so` 时间戳
仍是旧产物，而先前只重编了 `build/cache/...` target。用当前worktree的
`RuntimeBuilder("a2a3").get_binaries(runtime, build=True)` 把当前A2/A3产物正式stage到 `build/lib`后，
TRB late append立即通过。这次假失败证明上板前不能用“cache target编译成功”代替“测试实际
加载的staged binary已更新”。

最终当前staged binaries的实测结果：

```text
TRB: 1 passed, two late-appended L1 callables, 3 ACLGraph replays
HBG: 1 passed, two late-appended L1 callables, 3 ACLGraph replays
```

HBG因此已不再是“只有单callable黄金路径”；本次公开JIT API下的late append、同图两个
L1 node、前后torch op和连续replay都有device0 A2/A3证据。

提交前又在当前最终源码和当前staged binary上各用一个全新进程重跑TRB/HBG。这次复验先后遇到
三个纯Host环境问题：未选中PTOAS时生成了`skip_ptoas` artifact；`ptoas-bin`中的打包可执行文件
需要当前Host不具备的更新GLIBC/GLIBCXX；editable环境里原有`_torch_npu_l1.so`又是旧Torch ABI
产物，导入时报undefined symbol。这三次都在PyPTO native L1 init/launch前结束，不是TRB/HBG设备路径
失败。明确使用`PTOAS/build-v0.57-llvm21-cann9.2-clean/tools/ptoas` 0.57，并在Torch
2.7.1/Torch-NPU 2.7.1.post4环境重建当前PyPTO editable扩展后，最终复验结果仍为：

```text
TRB: 1 passed in 5.75s
HBG: 1 passed in 12.35s
```

两个最终进程都使用device0、A2/A3 onboard、当前工作树Python/runtime，并执行了全部三次
ACLGraph replay验数和严格quiesce→graph reset→optional shutdown顺序。

##### 10.66.9 仍保留的明示风险

1. TRB linked registry是O(N)查找，且code mapping、node和kernel table随specialization持续增长。
2. 暂无byte accounting、软阈值和long-running压测；不将“去掉64限制”误述为“资源无限”。
3. binary、AICore registered handle和TRB resident code按进程pin，显式shutdown也不卸载它们。
4. 高层registry把成功shutdown的owner保留为retired，当前不承诺同进程同device重新init。
5. 无graph-aware release/concurrency协议，所以不做LRU、id复用、working-slot并发或跨graph自动回收。
6. capture内必须显式out，首次specialization必须在ordinary eager完成。
7. A5/A5sim不在本次用户指定的验收范围内，对其不做通过声明。

##### 10.66.10 提交与推送

在两个远端main均与本地基线一致后，严格按submodule先行顺序发布：

1. simpler修改提交为`e58d54c0 Add: support dynamic L1 JIT callables`，已推送到
   `https://github.com/nalinaly/simpler.git`的`main`。
2. PyPTO公开JIT API、文档、ST/UT和上述runtime gitlink提交为
   `0e61fc88 Add: expose Triton-style L1 JIT execution`，已推送到
   `https://github.com/nalinaly/pypto.git`的`main`。
3. 本次不属于GPT实现范围的未跟踪文件
   `examples/runtime/l1_tiled_add_then_mul.py`没有被暂存、提交或推送。

两个提交前均通过本仓pre-commit。simpler clang-tidy使用机器上安装完整、与resource
headers匹配的LLVM/Clang 18；系统PATH优先的独立Clang 21可执行文件没有配套的
Clang 21 resource headers，其`stddef.h`/C++ standard library失败是lint环境问题，不是代码
诊断或被跳过的检查。

#### 10.67 完整设计文档与Triton风格接口设计合并

2026-08-19按用户要求，将原来的两份设计文档收敛为唯一规范入口：

- `tests/PyPTO_L1与ACLGraph完整设计文档.md`继续作为canonical文档；
- `tests/PyPTO_Triton风格L1_JIT调用接口设计.md`的正文被完整并入canonical文档，随后删除独立文件；
- `tests/pypto_l1_aclgraph_implementation_plan.md`中的最终规范引用改指向合并后的canonical文档；
- English-only lint的例外项同步由已删除文件迁移到canonical中文设计文档。

本次合并没有把接口文档压缩成摘要。canonical文档前篇保留native执行协议、stream/event时序、
taskQueue、TRB/HBG、错误闭包、pto2对比和实机证据；后篇完整保留Triton风格公开API的设计推导、
hidden owner、specialization late append、输出分配、可选shutdown、拒绝方案、实现步骤和测试矩阵。
两篇之间有意保留少量从不同视角重复建立的推导链，以便下游分别按native协议或产品API进行评审。

合并时同时消除了两份文档之间的最终态冲突：

1. 普通用户入口统一为`@pl.jit(execution="l1", runtime=...)`；
   `pypto_init/L1Context/L1Operator`只保留为advanced/debug控制面。
2. 首次ordinary eager隐式完成specialize/init/prepare/真实warmup；首次调用发生在capture内时明确失败，
   capture前仍由caller做外部sync并预分配输出。
3. HBG callable identity、参数快照、callable-local function table与pristine graph由每个CANN-owned
   launch package自包含；ContextRegistry只拥有mutable execution slot。
4. TRB L1 registry改为动态append-only、旧entry不复用，不再公开64-callable上限；其O(N)和无界增长风险
   继续明确记录，不伪装成无限资源。
5. 新L1路径中的binary与graph-visible code handle按进程pin，init rollback、prepare rollback、
   可选shutdown和析构均不调用`BinaryUnLoad`。
6. `pypto.l1.shutdown(device=...)`完全可选。GC和`atexit`不调用runtime close；不调用shutdown时资源
   安静保留到进程结束。销毁一张ACLGraph不意味着device上不存在其他graph，也不会触发设备级shutdown。
7. 当前验收范围只声明A2/A3 onboard；A5和A5sim不作为本次完成条件。

本节只记录文档合并与规范收口，没有改动PyPTO或Simpler运行时代码，也没有触碰未跟踪的
`examples/runtime/l1_tiled_add_then_mul.py`。

#### 10.68 Inductor HBG L1 profiling根因与第一轮性能收口

2026-08-19检查 `inductor_pto/build_output/profiles/inductor_pto_hbg_aclgraph_summary.json`时，
`torch.add -> HBG L1 add -> torch.mul` 的功能、TaskQueue顺序和ACLGraph replay验数都正确，但一个
64x128 FP32 add的device chain竟长达约304ms。前后Torch kernel各约2us，Host submit仅约100us；
HBG AICPU task和PyPTO AICore task重叠占据了几乎全部时间。独立的no-profiler replay也约305ms，
所以这不是profiler打点或flow-arrow解析产生的假象。

##### 10.68.1 被排除的第一个假设

最初怀疑HBG scheduler shutdown对每个AICore串行关闭寄存器窗口，因此曾将normal/emergency
shutdown改为先广播EXIT，再在共享deadline下join。这个改动的实机A/B完全无改善：

```text
before: eager median 843.895 ms, replay median 305.132 ms
trial:  eager median 843.984 ms, replay median 305.300 ms
```

因此该试验代码已完整撤回，没有把无效复杂度留在runtime中。

##### 10.68.2 真实根因：把L2最大容量整份当作每次L1 tiling data

CANN runtime INFO log给出了直接证据：每个 `aclrtLaunchKernelWithHostArgs` HBG task都携带
`92,936,560` bytes。当时的HBG L1 package包含：

1. 默认 `PTO2_TASK_WINDOW_SIZE=16384` 对应的整份shared-memory graph image；
2. HBG L2规格的 `PTO2_READY_QUEUE_SIZE=8192` 多组scheduler queue；
3. HBG L2规格的65536-entry / 4096-bucket TensorMap和其他runtime-arena状态。

CANN正确地把这份HostArgs当作每个task/captured node自有的tiling snapshot管理；问题不在
CANN ownership，而在PyPTO把“working slot最大容量”误当成了“这个小算子的实际tiling数据量”。
AICPU leader每次eager/replay要把全部pristine image恢复到context-owned working slot，其他AICPU线程
还要invalidate整段SM和arena，因而时间与package size直接相关。

只设 `PTO2_RING_TASK_WINDOW=16`、尚未改arena时已经能将HostArgs缩到 `11,276,656` bytes，并将
replay从约305ms降到约37.20ms。这个受控实验确认了size-scaling根因，但也说明仅调task
window不够：剩下的11MB几乎全是沿用L2规格的runtime arena。

##### 10.68.3 实现：显式L1 compact arena profile

A2/A3 HBG runtime新增了值类型 `PTO2RuntimeArenaSizing`，并保留两条彻底分离的路径：

- 原有 `runtime_reserve_layout(...)` 继续使用8192 ready slots、65536 TensorMap entries和4096 buckets，
  HBG L2的布局与容量不变。
- 仅 `prepare_l1_runtime_impl` 和 `build_l1_hbg_graph_plan_impl` 显式传入
  `pto2_hbg_l1_runtime_arena_sizing(task_window)`。prepare和每callable graph build都从同一个纯值函数得到
  完全相同的frozen layout，不依赖进程全局开关。

compact profile的规则是：

1. Ready queue容量表示“某一队列的峰值并发occupancy”，不是graph总节点数。首版使用
   `max(task_window, 64)`，但不超过历史8192上限。不寻常的宽并行nested graph如果真超过容量，
   现有runtime会以 `READY_QUEUE_OVERFLOW` fail closed，用户可显式放大task window；不会静默丢task。
2. TensorMap只服务Host建图的top-level task依赖发现，每个live task最多注册
   `CORE_MAX_TENSOR_ARGS=32` 个producer entries。pool因此使用
   `clamp(task_window * 32, 256, 65536)`，bucket数按4:1目标load factor取幂级，并保留64/4096下上限。
3. 容量全部写入 `PTO2RuntimeArenaLayout`，AICPU仍按package中的布局恢复和wire pointer；
   launch阶段没有新增H2D、allocation、stream sync或capture query。

Grok历史分支中也有过“64 ready slots + 256 TensorMap entries”的compact尝试，它用的是
`g_l1_hbg_graph_blob != nullptr` 加weak symbol的进程全局切换。本次吸收了其“L1不应携带L2最大池”的
经验，但改成显式layout value，并使TensorMap容量随task window有可证明的上界，避免L1/L2同进程
或多callable prepare时被全局时序污染。

##### 10.68.4 Inductor默认容量策略

PyPTO原生API仍允许用 `RunConfig.ring_task_window` 或 `PTO2_RING_TASK_WINDOW` 选择容量。
Inductor PTO对A2/A3 onboard + HBG L1设置64的性能默认值，并把它写入 `RunConfig`，所以也进入
compile/artifact cache identity。如果环境显式给出 `PTO2_RING_TASK_WINDOW`，则使用该power-of-two整数。
TRB、simulator和A5路径不套用这个Inductor默认值。

64不是callable registry限制，也不是CANN可以保有的captured graph数量限制；它仅是一个HBG单次
host-built graph的resident task-slot容量。超出时warmup/prepare明确失败，调大该值后重新specialize即可。

##### 10.68.5 device0 A2/A3结果

同一个64x128 FP32 add，同一套 `runner -> external synchronize` 与ACLGraph replay量测得到：

| 阶段 | HBG HostArgs | eager median | ACLGraph replay median |
|---|---:|---:|---:|
| 原始默认（16384 window + L2 arena） | 92,936,560 B | 843.895 ms | 305.132 ms |
| 只设window=16，未compact arena | 11,276,656 B | 102.232 ms | 37.202 ms |
| compact arena首轮（1024 ready floor）+ window=16 | 未单独记录 | 7.963 ms | 3.059 ms |
| 最终compact arena（64 ready floor）+ window=16 | 未单独记录 | 未单独记录 | 2.371 ms |
| 最终Inductor默认：compact arena + window=64 | 1,097,392 B | 10.047 ms | 3.790 ms |
| compact arena + window=256对照 | 未单独记录 | 未单独记录 | 10.503 ms |

最终默认相对原始默认：HostArgs减少约84.7倍，eager约84.0倍，ACLGraph replay约80.5倍。
数值正确性在每次量测后都用 `torch.testing.assert_close` 检查。

正式的device0验收 `tests/inductor/static/test_l1_hbg_aclgraph.py` 也在最终默认下2/2通过，覆盖：

1. TaskQueue中 `torch op -> HBG L1 -> torch op` 的eager顺序和后续调用新tensor地址；
2. capture前warmup，独立capture stream，图再串接Torch predecessor/successor；
3. 三组输入replay、逐次验数，最后external quiesce -> graph reset -> optional shutdown。

无硬件验证为：

```text
new HBG arena sizing UT:       3/3 passed
simpler C++ non-hardware:      121/121 passed
Inductor runtime config tests: 13/13 passed
A2/A3 HBG staged runtime:      RuntimeBuilder build passed
clang-format / diff-check:     passed
```

##### 10.68.6 仍需继续的性能问题

约3.8ms对一个2us级别pointwise kernel仍然偏大，本轮只解决了最大且证据最硬的数量级问题。
最终package仍有1,097,392B，且非leader AICPU线程在leader restore后仍对整个working SM/runtime arena
执行cache invalidate。下一轮优化应在保持CANN task-owned snapshot的前提下，研究：

1. 按 `host_total_tasks` 将SM的descriptor/payload/slot-state/completion-flag前缀序列化为sparse regions，
   working slot仍可保留较大容量；
2. 不把Host-only TensorMap/scope scratch放入device restore package；
3. 让follower仅invalidate本次restore且它实际会读的region/cache line，而不是粗粒度扫完整capacity。

这三项要同时保证每次replay必然重置所有会被scheduler原地修改的状态，不能为了缩包而让第二次
replay继承上一轮的queue/completion/runtime pointer。因此本轮没有在无完整mutable-region证明时直接
放宽 `hbg_launch_blob` 的“full SM + full runtime arena”校验。

本轮按用户要求只修改和验收A2/A3，没有修改A5/A5sim实现。只做阶段性本地commit，不默认push。

#### 10.69 A2/A3同构无依赖HBG的direct-AIV调度路径

2026-08-19继续分析上一节剩余的毫秒级开销。compact arena已经把HBG replay从约305ms降到约
3.8ms，但对一个2us级pointwise child kernel来说，AICPU restore、scheduler启动、AICore握手和
shutdown仍远大于真实计算。用户指出`nalinaly/fdwic-swimlane-deps`分支已经尝试用AICore的Scalar
控制路径完成分布式多task调度，并要求研究它对“同构、无依赖task”子集的借鉴价值。

##### 10.69.1 首先收紧平台边界：不能把A5实现移植到A3

该参考分支的当前tip为`6a0378b2`，它不是一份跨平台scheduler库，而是一整套A5
`fully_distributed_within_core`实验：

- 固定拓扑是96个worker，即32 AIC + 64 AIV；
- 使用A5专属Scalar/SIMT执行模型、跨核atomic、claim tournament、shared TensorMap、heap frontier、
  completion和fatal协议；
- state、cache、intrinsic、地址空间、worker编号和PMU/swimlane schema都与A5 ABI耦合；
- 它解决的是通用跨核建图/依赖/输出分配问题，不是一个可直接在A3编译的轻量work queue。

因此本次没有复制该分支的源码、A5 atomic primitive、96-worker常量或wire ABI，也没有修改或构建
A5/A5sim。真正吸收的只有一个架构思想：当Host已经证明任务是同一child kernel、无依赖且task数静态
可知时，不必为了分配这些work而启动AICPU scheduler；可以让AICore kernel中的Scalar控制流自行选择
逻辑work。

A2/A3首版采用比全局atomic cursor更窄、更可证明的确定性grid-stride：

```text
physical_lane = get_block_idx()
physical_lane_count = get_block_num()       # 当前A3为48个AIV block

for work_id = physical_lane;
    work_id < work_count;
    work_id += physical_lane_count:
        task_id = work_id / logical_block_num
        block_idx = work_id % logical_block_num
        call same_child(task[task_id].args, block_idx)
```

它不要求task数恰好等于核数。对于50个逻辑work和48个AIV block，block 0、1各执行两个work，
其余block各执行一个；96、97、145等数量也按同一规则自然展开。因为首版只接受同构且无依赖的work，
确定性映射不需要跨核atomic、claim loser、完成队列或AICPU仲裁。若未来task代价明显不均匀，再单独设计
A2/A3可证明的dynamic cursor；不能以A5分支存在为由直接引入其协议。

##### 10.69.2 严格eligibility与普通HBG fallback

Host orchestration仍先执行一次，用生成后的真实HBG ring image判断是否可以降级成direct package。
只有全部task同时满足下列条件才选择fast path：

1. 全部task使用同一个AIV0 child kernel；AIC与AIV1为空。
2. 全部task的tensor/scalar数量相同。
3. `fanin_count == 0`，没有dispatch predicate。
4. 没有task attrs、dump metadata、PMU、scope stats或chip swimlane。
5. `logical_block_num`为正且所有task相同，`total_required_subtasks == logical_block_num`。
6. callable-local `func_id`存在，且能从device `CoreCallable`基址按稳定wire offset推导child code地址。

任何普通但不满足该子集的graph返回`NotEligible`并继续原有AICPU HBG scheduler，不能因为优化识别失败
改变功能。结构损坏、地址/容量溢出或Host分配失败才作为真实错误返回。内部测试环境变量
`SIMPLER_INTERNAL_HBG_L1_REQUIRE_DIRECT_AIV=1`可以把fallback变成明确失败，只用于证明ST确实走了
direct path，不作为用户API。

##### 10.69.3 direct package与ACLGraph生命周期

每次Host build生成一个immutable `HbgL1DirectAivPackage`：

- 128-byte versioned header记录task/work/lane数量、record stride、child code地址和scratch地址；
- 每个HBG task保存一份`ChipTensor[] + scalar[]`参数快照；
- package不包含会被执行过程消费的scheduler queue/SM/runtime image；
- `HbgGraphPlan`在Host侧深拷贝canonical package，每次launch再产生独立可写serialization。

专用入口`hbg_l1_direct_aiv_kernel_1_mix_aiv`只声明一个native pointer参数。Host通过
`aclrtLaunchKernelWithHostArgs`传入`[64B prefix | inline package]`，placeholder把pointer patch到CANN
为该task/captured node管理的device args blob。这样tensor地址、scalar和逻辑work表的生命周期与普通
AscendC tiling_data相同：临时Host vector在launch返回后可销毁，graph replay继续使用CANN为该node保留的
snapshot。不同captured node不会共享一份Host package。

每个AIV block仍需要可写的`ChipTensor`副本、child args、`LocalContext/GlobalContext`。首版从已经pin住的
HBG execution-slot runtime arena尾部划出`48 * 8192` bytes、64B对齐的per-lane scratch；package只保存
该context-owned地址。每次child返回后执行`pipe_barrier(PIPE_ALL)`，保证同一lane执行下一个grid-stride
work前不会复用仍在pipeline中的UB和args。当前单context、禁止并发契约下该scratch安全；后续若开放并发，
必须把它提升为显式slot reservation，不能继续让两个graph同时写同一span。

direct entry从已经生成的HBG AICore binary中用`aclrtBinaryGetFunction`解析。为避免captured graph引用
失效，新`aclrtBinaryLoadFromData` handle与已有L1 binary一样按进程pin，任何新代码路径都不调用
`aclrtBinaryUnLoad`。direct launch只在caller stream提交一个48-block AIV kernel并record完整operator tail；
没有AICPU scheduler、hidden AICore stream、内部stream sync、capture query或model attach。

##### 10.69.4 实现中排除的错误方向

从独立A3 probe迁入production时依次排除了四个容易误判的方案：

1. **normal mixed entry中的weak hook。** 同一源码同时编译AIC/AIV，最终relocatable link可能为两边都选择
   AIC版本的weak实现，导致AIV入口实际不执行direct逻辑。最终改成独立AIV-only function entry。
2. **64-bit magic pointer tag。** 最初试图用`"HBGDIREC"`样式高位tag复用普通kernel参数，但A3 GM地址
   只有有效低位范围，入口解引用前已触发地址异常。独立entry不再需要tag或入口复用。
3. **猜测普通hidden kernel的私有launch ABI。** 实际ELF显示普通entry为16-byte native args，新的direct
   entry为8-byte单pointer args。最终使用公开`aclrtLaunchKernelWithHostArgs`和独立function handle，
   不复刻内部ffts/workspace ABI。
4. **把`CoreCallable*`当作code PC。** callable-local table保存的是device `CoreCallable`对象地址；generic
   scheduler会读取其中已修正的`resolved_addr_`。direct Host builder无法解引用device对象，最终按上传时
   invariant使用`object_base + CoreCallable::binary_data_offset()`得到相同child binary地址。

独立probe完成这些平台实验后没有保留进正式提交；最终ST直接通过PyPTO HBG L1 API验证产品路径，避免维护
一套使用private runtime launch API的重复测试框架。

##### 10.69.5 单测与device0 A3证据

新增无硬件测试覆盖：

- 2个HBG task × 25 logical block = 50 work，在48 lane header中保持精确计数；
- per-task `ChipTensor`与scalar逐byte进入immutable package；
- child code地址等于callable device base加`CoreCallable::binary_data_offset()`；
- fanin、不同kernel、predicate、task attrs、零lane和未对齐scratch都拒绝，失败不覆盖已有output owner；
- `HbgGraphPlan`深拷贝direct package，每次serialization独立，修改一个CANN-facing snapshot不会污染下一次。

结果：`test_hbg_launch_blob` 21/21通过，`test_hbg_l1_direct_aiv_package` 3/3通过；L1 Python定向测试
59/59通过。A2/A3 onboard HBG host、AICPU、AICore目标均构建通过。

device0使用当前正式源码、Torch `2.12.0+cpu`、Torch-NPU `2.12.0+git5462a1b`和PTOAS 0.57执行
`tests/st/runtime/l1/test_l1_hbg_direct_aiv.py`：

```text
one @pl.jit HBG task
logical_block_num = 50
physical AIV launch blocks = 48
eager result: passed
ACLGraph: direct L1 -> torch.mul
three replay values: passed
first replay run latency(us): 971.1, 229.4, 72.2
post-format/rebuild rerun latency(us): 985.6, 266.1, 61.4
```

最后一次约61.4us包含`graph.replay()`、图内`torch.mul`后继和capture-stream synchronize，不能直接解释为
纯child kernel时间，但已经从普通HBG scheduler的毫秒级链路降到普通单算子量级。原有A2/A3 L1回归也在
同一2.12环境下3/3通过：TRB基本图、普通HBG基本图、HBG两个callable各自package/capture/replay均未回归。

复验初期曾机械沿用过程记录中的旧`PYTHONNOUSERSITE=1`命令，导致运行环境从当前user-site Torch 2.12
退回venv/system Torch 2.7；第一次因此得到compile-only artifact，第二次因旧adapter ABI在import阶段失败。
两次都在native L1 init/launch前终止，没有执行2.7版本的L1 kernel。随后终止错误构建进程，明确以正常
user-site 2.12重建editable adapter，并逐项核对adapter build/runtime版本完全一致后才得到上述真机结果。

##### 10.69.6 当前边界与后续优化

1. 这不是A5 FDWIC Scalar scheduler的A3移植，也不支持有依赖、异构child、predicate、动态建图或输出分配。
2. grid-stride适合task代价相近的严格子集；task代价高度不均匀时会有tail imbalance。
3. scratch借用frozen runtime arena尾部依赖当前no-concurrency契约；开放多graph并发前必须显式reserve并按
   execution slot隔离。
4. Host每次仍要执行orchestration来证明eligibility并构造package；可在specialization identity稳定后研究
   template化，但不能跳过tensor/scalar地址快照。
5. binary与function handle继续append/pin，不做unload；这是ACLGraph graph-aware release缺失下的正确取舍。
6. A5/A5sim没有被修改、编译或宣称支持；未来若做A5，必须在A5平台上独立选择复用原FDWIC协议还是实现
   更窄的direct路径，不能从本次A3结果外推。
