# PyPTO 正式调用接口：修改文件范围与目的

更新日期：2026-09-14

状态：设计范围清单，尚未实施。本文汇总本轮讨论及最新六项决定，按要求放在 `tests/` 下。
文中的现有路径均相对 PyPTO 仓库根目录；表格中只有文件名的后续项沿用同格前项的目录。
“拟新增”表示建议落点，不表示文件已经存在。
旧 L1 文档及 demo 只作实现经验参考；涉及本次接口的冲突约定，以本文列出的已确认决定为准。

## 1. 已确认的设计边界

| 分类 | 决定 |
| --- | --- |
| 模式入口 | 在 `pl.jit` 增加 `mode`，区分 `kernel` 与 `program`。不保留 `L1Context/L1Operator` 和 `execution="l1"` 两套 demo 接口。 |
| 普通用户体验 | 定义算子后直接调用；编译、初始化、加载及 callable prepare 按需完成。用户不管理 context、注册句柄或执行槽位。 |
| 显式编译 | 可以保留显式编译并返回可调用对象；编译不是设备初始化，也不是执行算子进行 warmup。 |
| prepare | 不公开独立 prepare API，放入内部 lazy init/load 流程。不预先增加“capture 内一律禁止首次调用”的规则；遇到真实 ACLGraph capture 问题再具体解决。 |
| close | 不公开新的 close/shutdown API。参考 Triton/CuTe DSL 的对象、缓存及内部资源所有权管理方式，不要求用户配对 init/close。 |
| 参数 | Tensor 与 Scalar 都保留；Scalar 是按值传递的运行时变量，不因数值改变而自动变成新的编译特化。显式编译期常量与其分开。 |
| 输入输出 | 输入、输出及 InOut cache 均由外部传入。正式调用入口不代替调用者分配业务输出。 |
| 动态 shape | 沿用现有前端表达，不另造一套。ACLGraph 捕获后的 shape、地址和参数变化由使用者考虑，不自动重新捕获。 |
| 编译产物 | 算子 IR、生成源码、AICore binary、orchestration SO、签名元数据及编译缓存归 PyPTO 管理。simpler 管理加载后的运行时资源。 |
| workspace | 不公开按算子的 workspace 查询/allocate API，也不提供外部 allocator 回调。动态调度所需中间内存由 runtime 内部申请、回收和复用。 |
| 异步与串行 | 调用异步提交；共享执行资源的设备侧串行由 simpler 保证。不能只依赖 Python 锁，也不能只锁住 Host 提交区间。 |
| program pipeline | 保留现有两个槽位的 Host/Device 流水；这一步与 kernel 路径并列，不共用 `kernel_launch`。 |
| 进程隔离 | 同一进程不支持 program/kernel runtime 共存；需要覆盖所有初始化入口，而不是只限制 `pl.jit`。 |
| 支持范围 | A2/A3、A5 × HBG、TRB × eager、ACLGraph 全组合，不以单一 demo 跑通代替正式支持。 |
| PyTorch 接入 | 集中到独立的 `python/pypto/torch/` 目录，通用编译及 runtime ABI 不依赖 torch_npu 私有接口。 |

`mode` 的建议取值为 `"kernel"`、`"program"`。默认值尚未由用户指定；建议暂取 `"program"`，
保持已有普通 program 调用的选择，实施时再确认。下文示例均显式指定，不依赖该默认值。

## 2. 对外调用形态及内部责任

### 2.1 接口变化示意

现有 demo 从 `@pl.jit(execution="l1", ...)` 分支或手动创建 `L1Context/L1Operator` 进入。
正式入口收敛为以下形态；这是接口示意，不是已实现、已测试的完整算子：

```python
import pypto.language as pl

@pl.jit(mode="kernel")
def csa(
    hidden: pl.Tensor,
    out: pl.Out[pl.Tensor],
    cache: pl.InOut[pl.Tensor],
    step: pl.Scalar[pl.INT32],
) -> None:
    ...

# All tensors are supplied by the caller; step is per-call data.
csa(hidden, out, cache, step)

# Optional compilation; this does not execute the operator body.
compiled = csa.compile(hidden, out, cache, step)
compiled(hidden, out, cache, step)
```

Program 入口使用 `@pl.jit(mode="program")`，内部仍走独立的 program 提交及两槽位 pipeline。
两种入口可以被定义、分析或编译，但不能在同一进程初始化并运行两种 runtime 模式。
不把本次“不公开新的 close”扩大为删除现有 program Worker 的所有底层管理接口。

### 2.2 三条边界不能混在一起

```text
普通调用 / 显式编译 → PyPTO 特化与编译 → 可调用对象及其编译产物
kernel 调用         → 内部 lazy init / load / prepare → kernel launch
program 调用        → 两槽位准入 → program prepare / launch / 完成回收
```

- 产物与已加载状态在内部区分，但不要求用户分别维护“产物、context、executor”三个对象。
- 普通 eager 调用每次获取当前设备/stream；已编译对象不永久绑定第一次调用的用户 stream。
- `kernel_init`、`kernel_prepare_callable` 不接收用户 stream；单次 launch 接收本次执行 stream。
- lazy prepare 不偷偷执行一次业务算子，否则 CSA 等算子的 InOut cache 会被多修改一次。
- 首次初始化并发必须防重入、失败可正确回滚；热路径不重复解析签名、编译、prepare 或同步。
- 不将“`.compile()` 成功”等同于“任意冷启动 capture 已验证可用”。capture 失败保留具体原因，
  后续针对真实限制修复内部流程，而不是先增加公开 prepare/warmup 仪式。
- 正式入口缺少输出参数时报错，不自动补输出；算子内部 IR 的 return/alias 表达不因此被删除。
- HBG 中若 Scalar 影响 Host 构建的任务结构，应按本次值生成/绑定对应执行内容，不能复用首次值；
  这与重编译算子、重新捕获外层 ACLGraph 是不同的事情。

## 3. PyPTO 必改文件与目的

### 3.1 JIT、参数语义与模式选择

| 现有文件 / 主要锚点 | 修改目的 |
| --- | --- |
| `python/pypto/jit/decorator.py`：`JITFunction`、`_JITDecorator.__call__`（约 3014 行）、`__call__`（2342 行）、`compile`（2394 行） | 新增 `mode` 并贯穿直接调用、显式编译和编译后调用；移除 demo 的 `execution="l1"` 路由。两种 mode 分别派发，不静默切换。 |
| 同文件：`_param_runtime_scalar_dtypes`（346 行）、`_resolve_compiled`（2077 行） | 统一样本参数编译、仅签名编译与实际调用的 Scalar 语义。现在依赖 `= pl.RUNTIME` 的特殊路径不能成为普通 Scalar 才能保持动态的额外门槛。 |
| 同文件：`_l1_allocate_outputs`（2204 行）及参数绑定路径 | 去除正式入口隐式分配业务输出的行为，完整绑定外部输入、Out、InOut；预解析稳定签名，减少每次调用的 Python 开销。 |
| `python/pypto/jit/cache.py`：`make_cache_key`、`ScalarCacheInfo` | 区分目标平台、调度 runtime、执行 mode、静态签名及编译选项；运行时 Scalar 数值、Tensor 地址不成为编译特化条件。编译缓存与设备加载缓存分开。 |
| `python/pypto/jit/specializer.py`：`SpecializeContext`、`_classify_params`、`_build_params` | Scalar 保持 IR 参数，不替换成首次调用字面量；显式常量走单独的特化分类。沿用现有动态维度及 Out/InOut 信息。 |
| `python/pypto/language/typing/scalar.py`：`Scalar`、`RuntimeScalarMarker` | 对齐新的默认运行时 Scalar 契约；明确 `pl.RUNTIME` 作为编译占位符的作用，不再与“普通 Scalar 默认冻结”绑定。 |
| `python/pypto/_runtime_names.py` | 保持 HBG/TRB 名称及冲突检查唯一来源，区分 `runtime` 与 `mode`，不以 `mode` 替代硬件或调度后端选择。 |
| `python/pypto/runtime/runner.py`：`RunConfig` | 梳理装饰器、编译对象及运行配置的来源和优先级；冲突明确报错，不能运行时用另一种 mode/runtime 执行已有产物。现有 program 调试配置与普通 kernel 调用分开。 |

这里的缓存改动只服务正确的编译复用，不新增独立的数据质量检查或产物审计流程。
常量的具体前端拼写仍需单独定稿；不能为了暂未定名而继续把普通 Scalar 当常量。

### 3.2 编译链、产物与可调用对象

| 现有文件 / 主要锚点 | 修改目的 |
| --- | --- |
| `python/pypto/ir/compile.py`：`compile`（220 行） | 编译结果记录目标、runtime、mode 及参数契约；保持编译与设备初始化分离。明确 IR/源码生成与完整可执行产物的阶段，不把未生成 binary 的对象误报为设备可执行。 |
| `python/pypto/ir/compiled_program.py`：`_RuntimeFacade`（526 行）、`CompiledProgram`（591 行） | 对外维持一个可调用对象，内部分离编译产物和按设备加载状态。保留 mode，避免 `.compile()` 后调用绕回默认 program 路径。 |
| 同文件：`_coerce_args`（364 行）、`_invoke_compiled`（433 行）、`__call__`（918 行） | 正式调用统一要求外部输出；移出 torch 专属转换、自动分配和提交逻辑。参数类型、方向、返回别名保持同一份元数据，不另推断一套。 |
| `python/pypto/backend/pto_backend.py`：`_generate_config_file`（793 行）、`_generate_arg_unpacking`（418 行） | 生成与正式 ABI 一致的编译产物描述、参数元数据及包装代码；明确 Tensor 顺序、Scalar 顺序/类型、Out/InOut、动态维度。HBG/TRB 与 A2/A3、A5 使用正确目标配置。 |
| `python/pypto/runtime/kernel_compiler.py`：`KernelCompiler`（31 行） | PyPTO 主导自己算子的 AICore binary 和 orchestration SO 编译。可消费 simpler SDK 的头文件/工具链信息，但不再将算子项目目录与编译生命周期委托给 simpler 管理。 |
| `python/pypto/runtime/device_runner.py`：`compile_single_kernel`、`compile_single_orchestration`、`compile_and_assemble`（698 行） | 拆清编译、组装与执行职责，组装正式 callable package。保留 program 适配入口；kernel 适配不能调用会初始化 owned-device Worker 的旧执行捷径。 |
| `python/pypto/runtime/_binary_cache.py` | 沿用并调整现有编译缓存上下文，正确区分兼容目标与 ABI；其中不存可跨进程复用的 device 指针/句柄。无需另造一套缓存系统。 |
| `python/pypto/runtime/execute_artifact.py` | 离线产物重建及执行读取同一份 mode/runtime/签名信息，不能丢失模式后默认走 program；缺少必要元数据应明确说明，不猜测。 |

`orchestration SO` 与 AICore binary 都是算子编译产物。把 SO 加载进 AICPU 进程属于 simpler
运行时的加载行为，不改变 SO 的编译与缓存归 PyPTO 的边界。当前编译文件位于 `runtime/` 下，
不代表必须先整体搬目录；先落实职责分离，避免把目录重组当成独立目标。

### 3.3 拟新增的内部 kernel 适配目录

以下是建议文件划分，可以在实现时合并小文件，但职责不能重新塞回一个 demo 大文件。

| 拟新增文件 | 目的 |
| --- | --- |
| `python/pypto/runtime/_execution_mode.py` | PyPTO 侧统一 mode 校验和初始化入口防护；仅在进入实际 runtime 初始化时认领模式。simpler 仍是防止底层入口绕过限制的权威层。 |
| `python/pypto/runtime/kernel/__init__.py` | 内部调度入口；不向普通用户导出 init、prepare、close、workspace 分配接口。 |
| `python/pypto/runtime/kernel/context.py` | lazy 初始化、并发防重入、设备共享执行域、配置一致性、初始化失败回滚与内部资源释放。区分自有和借用资源。 |
| `python/pypto/runtime/kernel/callable.py` | 编译对象到已加载 callable 的内部映射、一次性加载/prepare、单次异步提交及参数快照存活期。不向用户暴露第二个必须维护的 executor。 |
| `python/pypto/runtime/kernel/abi.py` | 集中封装 simpler 正式接口及 descriptor/参数编码，检查 ABI 长度、类型、方向和返回错误。隔离 Python 业务对象与底层注册/launch 结构。 |

生命周期借鉴 Triton/CuTe DSL 的“可调用对象持有加载状态，内部按需建立执行资源”方式：

- import、装饰函数及仅编译不无谓初始化设备。
- 普通用户不调用 close；对象与内部缓存持有资源引用，不把 `del compiled` 宣称为立即卸载。
- 内部释放只处理自己的资源，不 reset 外部设备，不销毁调用方 stream，不释放业务 Tensor。
- 解释器退出时避免调用已拆除的模块；不为了自动清理增加隐式全设备同步。
- 不新增一套追踪所有外部 graph 生命周期、证明外部已无在途工作的机制。
- Kernel 模式每次执行是有界任务，不能把 AICPU task 改成长驻服务；代码/资源驻留与 task 常驻不同。

### 3.4 program 路径与两槽位 pipeline

| 现有文件 | 修改目的 |
| --- | --- |
| `python/pypto/runtime/worker.py`：`ChipWorker`、`RegistrationHandle` | 接入统一 mode 防护和产物契约，保留 program Worker 提交路径；不得内部转调 `kernel_launch`。初始化入口与 kernel 的互斥一致。 |
| `python/pypto/runtime/runtime_base.py` | 仅调整共享初始化、资源所有权和调用契约所需部分；现有 Worker 管理业务 Tensor 的能力不是新的 per-op workspace API，不混淆或无关删除。 |
| `python/pypto/runtime/runner.py`、`device_runner.py` | 单次执行/复用 Worker 路径保持同一模式约束。异步接口与等待/收集结果分开，不通过提前准备大量 program 获取表面并发。 |

两槽位必须满足：

1. 最多一个运行中的 program 加一个已准入的后继准备任务；第三个不能先准备、分配，再等槽位。
2. 槽位在真实完成及对应 finalize 后回收，不能以 Host enqueue 返回作为复用条件。
3. HBG 当前 per-run arena 与 TRB 共享 scratch 的资源拓扑分别遵守 simpler 契约，不能假定两者内存复制数相同。
4. 槽位准入约束跨 program 的积压；单个 program 内动态 task/中间 Tensor 的积压仍由 runtime 的任务窗口、heap 容量及回收机制控制。
5. 异步不等于允许无限积压；必要背压应在资源继续增长前发生。PyPTO 不复制 simpler 的槽位调度器。

这里说的是 Host/Device 的两槽位执行流水，**不是 DSL 的 tile pipeline**。
不能因此把 `lower_pipeline_to_slots`、`lower_pipeline_loops` 等编译 pass 列入必改项。

### 3.5 PyTorch 独立适配

| 文件 | 修改目的 |
| --- | --- |
| `python/pypto/torch/__init__.py`（拟新增） | PyTorch 接入目录入口；普通 tensor 调用自动使用适配，不要求用户先注册 context。框架高级集成可按需使用。 |
| `python/pypto/torch/interop.py`（拟新增） | 集中 dtype/shape/stride/device/Tensor 指针转换及当前 stream 获取，处理 Tensor 存活期；不分配算子的业务输出或 runtime workspace。 |
| `python/pypto/torch/registration.py`（拟新增） | 独立容纳 `torch.library`、FakeTensor/meta、mutation/alias 描述及必要的注册辅助；不是所有 eager 调用的前置步骤，不默认承诺 autograd。 |
| `python/bindings/torch_npu_l1_adapter.cpp`（重构，建议更名为 `torch_npu_adapter.cpp`） | 将现有 taskQueue 顺序、延迟提交参数保活和 storage `recordStream` 机制迁到正式 ABI，去掉对 demo queue-call 类型的依赖。保持独立扩展，不能裸 ACL 提交绕过框架队列。 |
| `python/pypto/runtime/task_interface.py` | 通用 simpler 类型/ABI 与 torch helper 分离，不再让通用接口承担 PyTorch 专属转换入口。 |
| `python/pypto/runtime/tensor_arg.py` | 按参数来源调用独立 torch 适配；保留 program 的 Worker-aware、address-free Tensor 路径，不把它误改成 kernel 的裸指针路径。 |
| `python/pypto/ir/compiled_program.py` | 业务框架相关转换委托给该目录，可调用对象本身不继续同时负责编译、输出分配、框架提交。 |

不因“参考 Triton/CuTe DSL”就新增 TVM-FFI 依赖或通用多框架插件系统。
框架 adapter 的 `recordStream` 用来维护**外部 Tensor**的 allocator 生命周期，
不等于向外开放 PyPTO 动态 workspace allocator。

### 3.6 公共导出、构建与 demo 退场

| 现有文件 | 修改目的 |
| --- | --- |
| `python/pypto/__init__.py`、`python/pypto/runtime/__init__.py` | 移除 demo 公共入口，组织 lazy 导入；不公开新的 prepare/close，不因 import 加载设备 runtime。 |
| `python/pypto/jit/__init__.py`、`python/pypto/language/__init__.py`、`python/pypto/language/typing/__init__.py` | 同步 JIT/Scalar 的导出、类型说明和签名；只在实际符号变化时调整导出，不为内部 context 扩大公共 API。 |
| `python/pypto/l1.py` | 正式路径落地后删除 demo 公共门面，包括 `pypto_init`、`L1Context`、`L1Operator`、`shutdown` 导出，不做长期兼容层。 |
| `python/pypto/runtime/l1.py`、`python/pypto/runtime/l1_jit.py` | 可复用内部经验迁入正式目录，随后移除旧实现，避免同时维护两套初始化、注册、参数绑定和退出状态。 |
| `python/bindings/CMakeLists.txt` | 调整独立 torch_npu 扩展的源文件、目标名、正式 simpler ABI 依赖及安装位置；编译器核心不链接 torch_npu 私有 ABI。 |
| `cmake/detect_torch_npu.py`、`pyproject.toml` | 随 adapter/包结构变化更新探测及打包；没有 torch_npu 的编译/IR 环境仍能使用不依赖设备的能力。 |

不要求先删除 demo 再开始开发。先完成新路径及必要回归，最后退场；退场不等于无关删除
`ChipWorker`、分布式 Worker、IR 构建接口或历史调试记录。

## 4. 按 ABI 差异联动的文件，不预设全部重写

| 文件范围 | 触发条件与目的 |
| --- | --- |
| `src/codegen/orchestration/orchestration_codegen.cpp`、`include/pypto/codegen/orchestration/orchestration_codegen.h` | 正式 callable ABI 改变 orchestration entry/config、Scalar 解包或参数元数据时同步；保留现有动态调度及外部 Out/InOut 语义。 |
| `python/bindings/modules/codegen.cpp`、`python/pypto/pypto_core/codegen.pyi` | 上述 C++ codegen 公共签名变化时同步绑定和类型存根；不能只改其中一层。 |
| `python/pypto/runtime/device_tensor.py`、`tensor_spec.py` | 若通用编译对象去除 torch 强依赖涉及这些类型，则迁移框架专属转换；不趁机重做 program 的 Tensor 所有权系统。 |
| `python/pypto/ir/distributed_compiled_program.py`、`python/pypto/runtime/distributed_runner.py` | 共享编译对象、导出或模式防护变化带来的联动与回归；本次不新增分布式 kernel mode 功能。 |
| `include/pypto/backend/`、`src/backend/`、`python/pypto/backend/` 中 A2/A3、A5 对应配置 | 仅在目标选择、ABI 或实际平台能力差异需要时改动；不能把另一平台的 demo 常量照搬。 |

现有 Tensor/Scalar IR 类型不因这次接入而重定义。算子数学逻辑、tile、SPMD、依赖生成与
内存规划 pass 不属于此次接口改造的默认修改范围；发现真实适配问题时再精确扩大落点。

## 5. simpler 侧前提与职责，不在 PyPTO 重复实现

以下是正式接入所需的跨仓契约，不代表本次文档已经确认所有 simpler 实现均完成：

| 依赖项 | 要求 |
| --- | --- |
| 初始化/准备/提交 | 提供清晰的 kernel init、callable prepare、launch 内部 API；init/prepare 不要求传用户 stream，launch 才接收单次 stream。 |
| 所有入口的 mode 互斥 | program 与 kernel 同进程不共存；包括绕过 PyPTO 直接进入 simpler 的初始化路径，避免 ACL 所有权旁路及误 reset。 |
| 共享设备执行资源 | 由 simpler 保证异步调用和 ACLGraph replay 的执行顺序；Host Python 锁不能覆盖绕过 Host 的 replay。 |
| 动态内存 | runtime 内部管理 heap、task window、临时 Tensor 及回收；没有外部 per-op allocate API。 |
| callable 驻留规格 | 沿用此前要求：8192 个条目、总 device code 容量上限 2 GiB、按需以 2 MiB 粒度申请，不预留 512 MiB 大 arena。这个上限不是执行 heap 大小。 |
| program 两槽位 | 复用已有准入、资源拓扑及完成回收协议，保留独立 launch 路径；异步准备不能越过容量限制。 |
| 资源释放 | 仅关闭/释放自身创建的资源；借用设备和 stream 的归属不因初始化而改变。错误不能被吞掉后伪装成成功。 |

PyPTO 负责提交正确包、参数和单次 stream，并保留必要 Host 引用；不在 Python 层另建 device
注册器、执行 heap 分配器、两槽位调度器或依赖事件调度系统。

## 6. 配套测试文件范围与验收目标

以下是后续实现需要的测试范围，**本次仅写文档，不新建或运行测试**。

| 文件 / 拟新增范围 | 覆盖内容 |
| --- | --- |
| `tests/ut/jit/test_decorator.py`、`test_cache.py`、`test_specializer.py`、`test_jit_compile_extraction.py` | mode 参数及非法值；直接调用/显式编译一致；Scalar 连续变值但不按值重新编译；显式常量仍可特化；动态 shape 既有语义不变；缺少输出时报错。 |
| `tests/ut/ir/test_compiled_program.py`、`test_compile_pipeline.py` | 编译不启动设备、不执行算子；compiled 调用保留 mode；全部参数由外部传入，Out/InOut 和别名一致；产物元数据重建不丢失。 |
| `tests/ut/runtime/test_run_config.py`、`test_execute_artifact.py`、`test_external_kernel_cache.py`、`test_binary_cache_context.py` | 配置来源一致；产物重载正确；不把已加载 device 状态当作可复用编译产物。 |
| `tests/ut/runtime/test_worker_reuse.py`、`test_worker_memory.py`、`test_chip_worker_explicit_dispatch.py` | program 路径保留；所有入口的 mode 互斥；接入两槽位时无提前分配第三份执行资源。设备协议本身在 simpler 侧验证。 |
| `tests/ut/runtime/test_kernel_context.py`、`test_kernel_callable.py`（拟新增） | 并发首次初始化、失败回滚、重复加载、内部资源所有权；prepare 不执行业务；每次参数快照独立；无公开 prepare/close 要求。 |
| `tests/ut/torch/test_interop.py`、`test_registration.py`（拟新增） | 当前设备/stream、Tensor 类型与布局、外部输出、mutation/alias、FakeTensor；正常 eager 调用不强制注册 custom op。 |
| `tests/ut/backend/test_kernel_config_signature.py`、`tests/ut/codegen/test_orchestration_codegen.py`、`test_orchestration_returned_param_map.py` | 完整 Tensor/Scalar 签名、方向、参数顺序与 wrapper/ABI 一致；有 codegen 联动时覆盖对应变化。 |
| `tests/st/runtime/kernel/`（拟新增，迁移有效 L1 用例） | 全组合 eager/capture/replay、重复异步调用、多个 callable、stream 切换、运行时 Scalar、外部 Out/InOut、torch_npu taskQueue 开关两条路径。 |
| `tests/st/runtime/framework_and_models/test_jit.py`、`test_compiled_program.py` 及相关 scheduling 用例 | program 的既有调用与两槽位路径回归；不能用 kernel 测试结果替代 program 路径结果。 |

支持矩阵如下；A2/A3 按同一个后端族列出，硬件验证仍需分别说明 A2、A3 的覆盖情况：

| 平台族 | Runtime | eager | ACLGraph |
| --- | --- | --- | --- |
| A2/A3 | HBG（`host_build_graph`） | 必须覆盖 | 必须覆盖 |
| A2/A3 | TRB（`tensormap_and_ringbuffer`） | 必须覆盖 | 必须覆盖 |
| A5 | HBG（`host_build_graph`） | 必须覆盖 | 必须覆盖 |
| A5 | TRB（`tensormap_and_ringbuffer`） | 必须覆盖 | 必须覆盖 |

这是 8 个平台族/runtime/执行方式组合的目标，不是当前通过清单。`mode` 另行区分：
kernel 与 program 分别记录覆盖与实现情况；保留 program pipeline 不代表自动支持 capture，
也不能偷偷切换到 kernel 路径填补 program 的缺口。两种模式的验证使用不同进程。
冷启动 capture 的具体行为需要如实记录；发生失败时定位真实限制，不预先用手动 prepare 规避。

## 7. 文档、示例与旧测试迁移范围

| 文件范围 | 目的 |
| --- | --- |
| `docs/en/user/02-quickstart.md`、`docs/en/user/language/00-types.md`、`01-functions.md` 及 `docs/zh/` 对应文件 | 更新 `pl.jit(mode=...)`、运行时 Scalar、外部输入输出、lazy 行为；移除要求用户创建 demo context/prepare/close 的指导。 |
| `docs/en/dev/00-ecosystem.md`、`docs/en/dev/codegen/01-orchestration_codegen.md` 及中文对应文件 | 写清 PyPTO 编译产物所有权、simpler ABI、Scalar 编码与输出别名。 |
| `docs/en/dev/05-runtime-ring-sizing.md` 及中文对应文件 | 区分内部动态 heap/task window 与业务 Tensor；区分 program 两槽位和 kernel 执行资源，不导出新的 per-op workspace 接口。 |
| `docs/en/dev/runtime/kernel-mode.md`、`docs/zh/dev/runtime/kernel-mode.md`（拟新增） | 归档正式接口、内部生命周期、两种 mode 的关系、支持矩阵及错误边界；本范围清单不是其实现完成说明。 |
| `examples/runtime/l1_tiled_add_then_mul.py` | 迁为正式 mode 调用示例，外部传入输出；不保留 demo 的手动初始化仪式。 |
| `tests/ut/jit/test_l1_jit_api.py`、`tests/ut/runtime/test_l1.py`、`tests/st/runtime/l1/` | 筛选并迁移仍有效的精度、异步、stream、capture 和资源回归；主动替换已改变的接口契约，不机械保留 demo 行为断言。 |
| `tests/` 下已有 L1 设计/调试 Markdown | 保留历史记录，后续标注其 demo 身份并指向正式设计；不批量覆盖旧结论或改写定位历史。 |

参考过的代码落点（分别相对 Triton、CUTLASS 仓库根目录）：
Triton 的 `python/triton/runtime/jit.py`、`python/triton/compiler/compiler.py`；
CuTe DSL 的 `python/CuTeDSL/cutlass/base_dsl/compiler.py`、
`python/CuTeDSL/cutlass/base_dsl/jit_executor.py`、`python/CuTeDSL/cutlass/torch.py`。
借鉴的是可调用编译对象、lazy 加载、框架适配分层及内部生命周期，
不是逐项复制它们的 Scalar 特化策略、workspace API 或全部依赖。

## 8. 建议实施顺序与尚需定稿的小项

1. 先定 mode 默认值、显式常量表达及内部 ABI 参数契约，锁定外部输入输出语义。
2. 改 JIT/编译对象，落实 mode、Scalar、产物所有权与编译/执行分离。
3. 接正式 kernel 内部目录及独立 PyTorch adapter，完成 lazy 生命周期和异步调用链。
4. 接 program 共用上层契约，保留独立两槽位 launch；同步落实 simpler 的底层前提。
5. 按完整矩阵补齐实现和回归，再退场 demo，更新正式中英文文档与示例。

实施前还需精确定义的只是细节，不重新打开已决定的方向：

- `mode` 默认值；建议 `program`，不把建议写成用户已确认。
- 显式编译期常量的语法，以及 `pl.RUNTIME` 编译占位符的保留方式。
- 正式产物/ABI 元数据的具体字段及内部错误类型；用户仍只持有可调用对象。
- PyTorch 注册辅助的具体签名和无 backward 时的错误表现，不默认扩大到训练支持。

本次不修改算子数学逻辑，不做性能自动调优，不公开 allocator/prepare/close，不统一两条
launch，不新增无限深 pipeline，也不把模型仓、vLLM/recipes 接入修改混进 PyPTO 文件清单。
