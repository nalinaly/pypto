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

每次后续提交继续补充顶层/runtime SHA、中文提交主题、完整变更范围和对应验证；不 push。

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
