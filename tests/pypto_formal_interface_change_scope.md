# PyPTO 正式调用接口：修改文件范围与目的

更新日期：2026-09-14

状态：设计范围清单，尚未实施。本文汇总本轮讨论、六项决定及后续确认事项，按要求放在 `tests/` 下。
文中的现有路径均相对 PyPTO 仓库根目录；表格中只有文件名的后续项沿用同格前项的目录。
“拟新增”表示建议落点，不表示文件已经存在。
旧 L1 文档及 demo 只作实现经验参考；涉及本次接口的冲突约定，以本文列出的已确认决定为准。

## 1. 已确认的设计边界

| 分类 | 决定 |
| --- | --- |
| 模式入口 | 在 `pl.jit` 增加 `mode`，区分 `kernel` 与 `program`。不保留 `L1Context/L1Operator` 和 `execution="l1"` 两套 demo 接口。 |
| 普通用户体验 | 定义算子后直接调用；编译、初始化、加载及 callable prepare 按需完成。用户不管理 context、注册句柄或执行槽位。 |
| 编译入口 | 不对外提供显式编译 API。首次调用遇到未缓存的特化时自动编译，后续调用复用缓存；用户不调用 `.compile()`，也不单独获取和维护编译结果对象。 |
| prepare | 不公开独立 prepare API，放入内部 lazy init/load 流程。不预先增加“capture 内一律禁止首次调用”的规则；遇到真实 ACLGraph capture 问题再具体解决。 |
| close | 不公开新的 close/shutdown API。参考 Triton/CuTe DSL 的对象、缓存及内部资源所有权管理方式，不要求用户配对 init/close。 |
| 参数 | Tensor 与 Scalar 都保留；Scalar 是按值传递的运行时变量，不因数值改变而自动变成新的编译特化。显式编译期常量与其分开。 |
| 输入输出 | 输入、输出及 InOut cache 均由外部传入。正式调用入口不代替调用者分配业务输出。 |
| Tensor 接入 | 参考 CuTe DSL，采用 DLPack 作为通用 Tensor 数据交换协议。普通用户直接传 Tensor，内部自动接入，不要求手动 `from_dlpack()`；DLPack 不替代独立的框架执行适配层。 |
| 动态 shape | 沿用现有前端表达，不另造一套。ACLGraph 捕获后的 shape、地址和参数变化由使用者考虑，不自动重新捕获。 |
| 编译产物 | 算子 IR、生成源码、AICore binary、orchestration SO、签名元数据及编译缓存归 PyPTO 管理。simpler 管理加载后的运行时资源。 |
| workspace | 不公开按算子的 workspace 查询/allocate API，也不提供外部 allocator 回调。动态调度所需中间内存由 runtime 内部申请、回收和复用。 |
| 异步与串行 | 正常调用异步提交；共享执行资源的设备侧串行由 simpler 保证。仅在第 2.4 节定义的可恢复大 args 分配失败路径允许内部同步回收及一次重试。不能只依赖 Python 锁。 |
| 整网大 args 恢复 | 整网 decode 接入 vLLM/PyTorch 继续采用 kernel 模式；RTS 管理的 HBG 大 args 积压由 simpler 内部在失败后 timeline event 同步回收、成功后重试一次，不改框架执行层。此方案待实施及整网验证。 |
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

# The first call compiles lazily; all tensors are supplied by the caller.
# Tensor interop through DLPack is handled internally.
# step is per-call data, not a compilation constant.
csa(hidden, out, cache, step)
```

Program 入口使用 `@pl.jit(mode="program")`，内部仍走独立的 program 提交及两槽位 pipeline。
两种入口可以同时定义；定义阶段不触发编译或 runtime 初始化。实际调用时，不能在同一进程
初始化并运行两种 runtime 模式。
不把本次“不公开新的 close”扩大为删除现有 program Worker 的所有底层管理接口。
底层编译函数及产物类型仍可供 JIT 内部、编译器开发和测试使用，不构成正式算子用户的显式编译 API。

### 2.2 三条边界不能混在一起

```text
普通调用            → 内部缓存查找 / 按需特化编译 → 内部编译产物
kernel 调用         → 内部 lazy init / load / prepare → kernel launch
program 调用        → 两槽位准入 → program prepare / launch / 完成回收
```

- 用户只调用 `pl.jit` 装饰后的算子；产物与已加载状态在内部区分，不公开独立的编译结果调用流程。
- 普通 eager 调用每次获取当前设备/stream；已编译对象不永久绑定第一次调用的用户 stream。
- `kernel_init`、`kernel_prepare_callable` 不接收用户 stream；单次 launch 接收本次执行 stream。
- lazy prepare 不偷偷执行一次业务算子，否则 CSA 等算子的 InOut cache 会被多修改一次。
- 首次初始化并发必须防重入、失败可正确回滚；热路径不重复解析签名、编译、prepare 或同步。
- 首次调用进入 capture 时，也由内部 lazy 流程处理编译和准备；不要求用户先调用显式编译 API。
  capture 失败保留具体原因，后续针对真实限制修复内部流程，不预先增加公开 compile/prepare/warmup 仪式。
- 正式入口缺少输出参数时报错，不自动补输出；算子内部 IR 的 return/alias 表达不因此被删除。
- HBG 中若 Scalar 影响 Host 构建的任务结构，应按本次值生成/绑定对应执行内容，不能复用首次值；
  这与重编译算子、重新捕获外层 ACLGraph 是不同的事情。

### 2.3 DLPack Tensor 接入契约

采用的是 DLPack 数据交换协议，不是直接复用 CuTe 的 CUDA Tensor 包装类。
正式调用仍是 `csa(hidden, out, cache, step)`，不增加显式编译或手动转换的必经步骤。

| 层次 | 责任 |
| --- | --- |
| 通用 DLPack 接入 | 导入数据地址、dtype、shape、stride、device，持有共享内存引用；归一化为内部 Tensor 视图。 |
| PyPTO 前端 | 定义 Out/InOut、动态维度、Scalar 与特化规则；这些语义不从 DLPack 元数据猜测。 |
| 框架执行适配 | 获取当前 stream，维护 taskQueue 顺序和异步使用期间的 Tensor 存活期；PyTorch 专属部分放在 `python/pypto/torch/`。 |

具体要求：

- 对同设备、可表达布局的外部 Tensor 使用零拷贝接入；Out/InOut 必须可写且保持原始别名关系。
  接入不得偷偷 `clone()`、`contiguous()`、转换格式或跨设备复制，使写入落到临时 buffer。
- 正确处理 `data + byte_offset`、以元素为单位的 stride、dtype 位宽及设备映射；
  按生产者实际协议版本解析 legacy/versioned capsule，不把两种结构直接混用。
- 消费 capsule 后由内部对象持有共享引用；若生产者提供 deleter，按协议仅调用一次。不能只保存裸地址后立即释放包装对象。
  归还共享引用不等于 PyPTO 直接释放外部设备内存，也不要求用户增加 close API。
- DLPack 的 stream 协商不能替代 torch_npu 的 taskQueue 顺序。不得直接照搬 CuTe 的 `stream=-1`
  策略假定 NPU 已同步；当前 stream、延迟提交保活及 `recordStream` 等职责仍留在框架适配层。
- DLPack 的 deleter 不会自动感知 NPU 任务完成；共享引用及框架 allocator 的使用记录必须覆盖异步访问，
  但不新增追踪所有外部 ACLGraph 生命周期的系统。
- 普通 strided Tensor 的接入不等于支持任意 NPU 私有物理格式。NZ 等格式需要明确的布局契约，
  不得按逻辑 ND shape 直接解释；无法安全表达的 dtype/布局应明确报错，不静默转换。
- 零拷贝不等于零 Host 开销。descriptor 解析尽量放在原生层，复用稳定的参数布局；
  每次调用仍正确更新指针及动态元数据，不能仅按旧地址缓存而忽略换 Tensor、view 或形状变化。
- DLPack 用于框架 Tensor 的进程内接入，不取代 program 的 Worker-aware/address-free Tensor 协议；
  已有显式 Host→Device 路径保持原有语义，不能把 CPU 指针直接当成 NPU 地址。
- FakeTensor/meta 没有真实存储，其注册与形状推导走元数据路径，不调用真实 DLPack 导出或初始化设备。

参考及当前实现差异见第 7.1 节；这些是设计要求和源码观察，不是设备测试通过结论。

### 2.4 整网 kernel 模式：HBG 大 args 回收与一次重试

本节为 2026-09-14 用户确认的处理方向，**属于设计要求，尚未实施到正式 runtime**。
整网每轮 decode 只进行一次 PyPTO `kernel_launch`，覆盖模型执行；只考虑单模型多 step 串行。
不因 HBG 大参数在 eager 队列中积压，就要求退回 program 模式或修改 vLLM/PyTorch 执行层。

首先区分两类 device 内存，**8192 和 2 GiB 不能套用到大 args**：

| 对象 | 生命周期管理者 | 规格及本次恢复方案的作用 |
| --- | --- | --- |
| 已 prepare 的 callable 注册条目及驻留代码 | simpler 自管，不由 CANN RTS 随每次 task 完成自动回收 | 此前确定的 8192 个条目、device code 总容量上限 2 GiB、按需以 2 MiB 粒度申请仅约束这一类资源。timeline sync 不卸载 callable，也不解除这些上限。 |
| HBG `build_graph` 结果序列化后经 launch args 下发的 device 副本 | CANN RTS，随 task 或 captured graph node 管理 | 同一 callable 的每次 eager 提交可能各持有一份大 args；本次失败回收处理的是先前 task 的这些副本，不占新的 callable 注册条目，不受上述 2 GiB 配额约束。 |

两者都会占用物理 HBM，但管理者、生命周期及规格独立。simpler 通过 RTS 的完成/回收机制
推动后者释放，不取得 args device 指针后自行 free，也不将其迁入 callable arena。
可复用的执行 heap/slot 是另外一类资源；复用 heap 不会消除每次提交的大 args 副本。
当前 Qwen3-14B B1/B16 HBG 图包为 83,400,640 bytes，本机每个待执行副本增加约 80 MiB HBM；
同一个 callable 连续调用就能产生积压，无须先用完 8192 个 callable 条目。

**已确认的恢复流程：在 simpler 的一次对外 launch 调用内部完成，最多尝试提交两次。**

```text
首次 kernel_launch
  ├─ 成功 → 返回成功，保持异步
  └─ 可恢复的大 args 分配失败
       → 清理/取消并汇合本次已提交的 AICore 分支
       → 在 caller stream 记录 ACL_EVENT_TIME_LINE event，并同步等待
            ├─ 清理、record 或 sync 失败 → 返回失败
            └─ sync 成功 → 用同一份调用参数再次 kernel_launch
                              ├─ 成功 → 返回成功
                              └─ 失败 → 必要清理后返回失败，不再重试
```

这里的可恢复失败指本轮讨论的、确认业务 AICPU 尚未提交的大 args 分配失败。
不能只按一个通用非零返回值重放：业务已经提交、执行异常、参数非法或 callable 容量超限，
不因同步而变成安全可重试错误。要保留失败阶段，避免重复更新 Out/InOut cache。
第二次提交复用相同的 Tensor、Scalar 和 HBG 参数快照，不重新读取可能变化的调用者状态。

- 回收 event 在 simpler 内部 lazy 初始化，使用 `ACL_EVENT_TIME_LINE`，保持自有资源生命周期；
  回收时调用 `aclrtRecordEvent`、`aclrtSynchronizeEventWithTimeout`，不公开新的用户 API。
- 普通 stream sync 或仅 `ACL_EVENT_SYNC` 不能替代这个回收等待。现有跨流 event 保持原职责，
  不要求每次正常 decode 都同步或把所有 event 改成 timeline。
- 恢复应位于 simpler 实际提交及错误处理层，不在 PyPTO Python 或框架外层捕获异常后补做。
  首次可恢复失败不能先 poison context、终止 taskQueue；恢复成功对上层仍是一次成功调用。
  整个恢复区间遵守 simpler 的串行约束；sync 或第二次 launch 失败才按最终失败处理。
- 本阶段不新增在途 args 的固定字节配额、外部 workspace allocate API 或无限重试循环；
  尤其不把 callable 的 2 GiB 上限重新解释为 args 积压阈值。
- 这是 eager 失败慢路径的同步例外，不改变正常路径异步，也不沿用旧 demo 的“任何路径绝不同步”限制。
  capture 中不能执行该 host 等待；graph node 持有的参数也不会因 replay sync 自动释放。
  保留 ACLGraph 支持目标，但不能把 eager 的失败回收流程直接用于 capture 失败。

该机制解决的是“旧的在途大 args 尚未回收，挤占下一次提交空间”，并不承诺无限模型规模：
单次图包本身无法容纳、固定 heap 不足、权重/KV 等长期占用过大时，回收后仍应返回失败。
因此整网 kernel 接入方向已确定，但“可以放心使用”的交付前提是完成实现及目标组合验证。
本机 A3/CANN 探针已验证 timeline 同步后旧 args 全部释放、立即重试成功；
未以该探针代替真实 PyPTO AICore 取消恢复、vLLM 整网、A5 或 ACLGraph 验证。
源码依据、Qwen 图包尺寸及原始实验记录见[HBG 参数内存调查](../docs/zh/dev/debug/kernel-mode-hbg-args-memory.md)。

## 3. PyPTO 必改文件与目的

### 3.1 JIT、参数语义与模式选择

| 现有文件 / 主要锚点 | 修改目的 |
| --- | --- |
| `python/pypto/jit/decorator.py`：`JITFunction`、`_JITDecorator.__call__`（约 3014 行）、`__call__`（2342 行）、现有 `compile`（2394 行） | 新增 `mode` 并贯穿直接调用、内部 lazy 编译及缓存命中后的执行；现有 `.compile()` 的编译能力转为内部实现，不保留正式显式编译入口。移除 demo 的 `execution="l1"` 路由，两种 mode 分别派发。 |
| 同文件：`_param_runtime_scalar_dtypes`（346 行）、`_resolve_compiled`（2077 行） | 统一首次调用触发编译与缓存命中时的 Scalar 语义；不再设计独立的用户样本编译/仅签名编译入口。现在依赖 `= pl.RUNTIME` 的特殊路径不能成为普通 Scalar 保持动态的额外门槛。 |
| 同文件：`_l1_allocate_outputs`（2204 行）及参数绑定路径 | 去除正式入口隐式分配业务输出的行为，完整绑定外部输入、Out、InOut；预解析稳定签名，减少每次调用的 Python 开销。 |
| 同文件：Tensor 元数据提取及 `_resolve_compiled` | 消费通用 DLPack 接入层归一化后的 Tensor 元数据，不在 JIT 内分散读取 torch 专属字段；运行时 Tensor 与 FakeTensor/meta 的入口分开。 |
| `python/pypto/jit/cache.py`：`make_cache_key`、`ScalarCacheInfo` | 区分目标平台、调度 runtime、执行 mode、静态签名及编译选项；运行时 Scalar 数值、Tensor 地址不成为编译特化条件。编译缓存与设备加载缓存分开。 |
| `python/pypto/jit/specializer.py`：`SpecializeContext`、`_classify_params`、`_build_params` | Scalar 保持 IR 参数，不替换成首次调用字面量；显式常量走单独的特化分类。沿用现有动态维度及 Out/InOut 信息。 |
| `python/pypto/language/typing/scalar.py`：`Scalar`、`RuntimeScalarMarker` | 对齐默认运行时 Scalar 契约，调用者直接传实际数值，不需要 `pl.RUNTIME` 编译占位符。清理依赖显式编译入口的用户说明；内部标记按实现需要处理，不再作为正式调用要求。 |
| `python/pypto/_runtime_names.py` | 保持 HBG/TRB 名称及冲突检查唯一来源，区分 `runtime` 与 `mode`，不以 `mode` 替代硬件或调度后端选择。 |
| `python/pypto/runtime/runner.py`：`RunConfig` | 梳理装饰器、编译对象及运行配置的来源和优先级；冲突明确报错，不能运行时用另一种 mode/runtime 执行已有产物。现有 program 调试配置与普通 kernel 调用分开。 |

这里的缓存改动只服务正确的编译复用，不新增独立的数据质量检查或产物审计流程。
常量的具体前端拼写仍需单独定稿；不能为了暂未定名而继续把普通 Scalar 当常量。

### 3.2 内部编译链、产物与可调用对象

| 现有文件 / 主要锚点 | 修改目的 |
| --- | --- |
| `python/pypto/ir/compile.py`：`compile`（220 行） | 作为 JIT 内部编译入口，结果记录目标、runtime、mode 及参数契约；不对算子用户提供显式编译 API。保持编译与设备初始化分离，明确 IR/源码生成与完整可执行产物的阶段。 |
| `python/pypto/ir/compiled_program.py`：`_RuntimeFacade`（526 行）、`CompiledProgram`（591 行） | 对外只保留 JIT 算子的直接调用体验；该对象承载内部编译产物及按设备加载状态，不要求用户单独取得。保留 mode，避免缓存命中后绕回默认 program 路径。 |
| 同文件：`_coerce_args`（364 行）、`_invoke_compiled`（433 行）、`__call__`（918 行） | 正式调用统一要求外部输出；消费通用 Tensor 视图，移出 torch 专属转换、自动分配和提交逻辑。参数类型、方向、返回别名保持同一份元数据，不另推断一套。 |
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
| `python/pypto/runtime/kernel/callable.py` | 编译对象到已加载 callable 的内部映射、一次性加载/prepare、单次异步提交及参数快照存活期；保活覆盖 simpler 内部回收及重试，不在 Python 重复实现恢复循环。不向用户暴露第二个必须维护的 executor。 |
| `python/pypto/runtime/kernel/abi.py` | 集中封装 simpler 正式接口及 descriptor/参数编码，检查 ABI 长度、类型、方向和返回错误。隔离 Python 业务对象与底层注册/launch 结构。 |

生命周期借鉴 Triton/CuTe DSL 的“可调用对象持有加载状态，内部按需建立执行资源”方式：

- import、装饰函数不触发编译或设备初始化；内部编译阶段本身不负责初始化 runtime。
- 普通用户不调用 close；JIT 对象与内部缓存持有资源引用，不把删除用户侧算子引用宣称为立即卸载。
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

### 3.5 通用 DLPack 接入与 PyTorch 独立适配

| 文件 | 修改目的 |
| --- | --- |
| `python/pypto/runtime/dlpack.py`（拟新增） | 通用内部协议入口，组织 `__dlpack__`/`__dlpack_device__` 调用及版本协商，形成带共享引用的 Tensor 视图；不依赖 torch_npu 私有接口，不作为新的用户必调 API。 |
| `python/bindings/dlpack_adapter.cpp`（拟新增） | 原生 capsule/descriptor 解析及共享引用管理，正确处理偏移、stride、dtype、版本和 deleter，降低重复 Python 解包开销。保持与 torch_npu 私有 C++ ABI 独立。 |
| `python/pypto/torch/__init__.py`（拟新增） | PyTorch 接入目录入口；普通 tensor 调用自动使用适配，不要求用户先注册 context。框架高级集成可按需使用。 |
| `python/pypto/torch/interop.py`（拟新增） | 对接 torch_npu 的 DLPack 导出及必要版本兼容，归一化 NPU 设备编码，保留原框架 Tensor 引用；负责当前 stream、私有布局识别及框架存活期适配。通用元数据解析委托给 DLPack 层，不分配业务输出或 runtime workspace。 |
| `python/pypto/torch/registration.py`（拟新增） | 独立容纳 `torch.library`、FakeTensor/meta、mutation/alias 描述及必要的注册辅助；不是所有 eager 调用的前置步骤，不默认承诺 autograd。 |
| `python/bindings/torch_npu_l1_adapter.cpp`（重构，建议更名为 `torch_npu_adapter.cpp`） | 将现有 taskQueue 顺序、延迟提交参数保活和 storage `recordStream` 机制迁到正式 ABI，去掉对 demo queue-call 类型的依赖。保持独立扩展，不能裸 ACL 提交绕过框架队列。 |
| `python/pypto/runtime/task_interface.py` | 通用 simpler 类型/ABI 与 torch helper 分离；接收由通用 Tensor 视图组装的参数，不要求 simpler 理解 Python capsule 或 torch 对象。 |
| `python/pypto/runtime/tensor_arg.py` | 将 DLPack Tensor 视图及现有 DeviceTensor 分别适配到对应执行路径；保留 program 的 Worker-aware、address-free Tensor 路径，不把它误改成 kernel 的裸指针路径。 |
| `python/pypto/ir/compiled_program.py` | 业务框架相关转换委托给该目录，可调用对象本身不继续同时负责编译、输出分配、框架提交。 |

不因“参考 Triton/CuTe DSL”就新增 TVM-FFI 依赖或通用多框架插件系统。
框架 adapter 的 `recordStream` 用来维护**外部 Tensor**的 allocator 生命周期，
不等于向外开放 PyPTO 动态 workspace allocator。

### 3.6 公共导出、构建与 demo 退场

| 现有文件 | 修改目的 |
| --- | --- |
| `python/pypto/__init__.py`、`python/pypto/runtime/__init__.py` | 移除 demo 公共入口，组织 lazy 导入；不公开新的 prepare/close，不因 import 加载设备 runtime。 |
| `python/pypto/ir/__init__.py` | 梳理现有 `compile`、`CompiledProgram` 导出；正式算子用户接口不再提供显式编译及获取编译结果对象的入口。编译器内部和测试改用内部实现，不删除底层编译能力。 |
| `python/pypto/jit/__init__.py`、`python/pypto/language/__init__.py`、`python/pypto/language/typing/__init__.py` | 同步 JIT/Scalar 的导出、类型说明和签名；只在实际符号变化时调整导出，不为内部 context 扩大公共 API。 |
| `python/pypto/l1.py` | 正式路径落地后删除 demo 公共门面，包括 `pypto_init`、`L1Context`、`L1Operator`、`shutdown` 导出，不做长期兼容层。 |
| `python/pypto/runtime/l1.py`、`python/pypto/runtime/l1_jit.py` | 可复用内部经验迁入正式目录，随后移除旧实现，避免同时维护两套初始化、注册、参数绑定和退出状态。 |
| `python/bindings/CMakeLists.txt` | 接入通用 DLPack 原生解析模块及公共 DLPack 头文件依赖，并调整独立 torch_npu 扩展的源文件、目标名、正式 simpler ABI 依赖及安装位置；通用模块与编译器核心不链接 torch_npu 私有 ABI。 |
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
| callable 驻留规格 | simpler 自管的 callable 注册/驻留代码：8192 个条目、总 device code 容量上限 2 GiB、按需以 2 MiB 粒度申请，不预留 512 MiB 大 arena。该规格既不是执行 heap 大小，也不是 RTS 管理的 HBG 大 args 总量/配额。 |
| RTS 大 args 失败恢复 | 按第 2.4 节在 simpler 内部实现“首次提交失败 → timeline event 回收同步 → 成功后重试一次”；不按 callable 配额管理 args，不向框架转嫁回收逻辑。区分失败阶段，保留部分提交清理，最终失败才向上返回。 |
| program 两槽位 | 复用已有准入、资源拓扑及完成回收协议，保留独立 launch 路径；异步准备不能越过容量限制。 |
| 资源释放 | 仅关闭/释放自身创建的资源；借用设备和 stream 的归属不因初始化而改变。错误不能被吞掉后伪装成成功。 |

PyPTO 负责提交正确包、参数和单次 stream，并保留必要 Host 引用；不在 Python 层另建 device
注册器、执行 heap 分配器、两槽位调度器或依赖事件调度系统。

## 6. 配套测试文件范围与验收目标

以下是后续实现需要的测试范围，**本次仅写文档，不新建或运行测试**。

| 文件 / 拟新增范围 | 覆盖内容 |
| --- | --- |
| `tests/ut/jit/test_decorator.py`、`test_cache.py`、`test_specializer.py`、`test_jit_compile_extraction.py` | mode 参数及非法值；定义时不编译、首次调用自动编译、后续复用缓存；正式接口不暴露 `.compile()`，相关旧测试迁为内部编译测试；Scalar 连续变值但不按值重新编译；显式常量仍可特化；动态 shape 不变；缺少输出时报错。 |
| `tests/ut/ir/test_compiled_program.py`、`test_compile_pipeline.py` | 内部编译阶段不启动设备、不执行算子；内部产物执行保留 mode；全部参数由外部传入，Out/InOut 和别名一致；产物元数据重建不丢失。 |
| `tests/ut/runtime/test_run_config.py`、`test_execute_artifact.py`、`test_external_kernel_cache.py`、`test_binary_cache_context.py` | 配置来源一致；产物重载正确；不把已加载 device 状态当作可复用编译产物。 |
| `tests/ut/runtime/test_worker_reuse.py`、`test_worker_memory.py`、`test_chip_worker_explicit_dispatch.py` | program 路径保留；所有入口的 mode 互斥；接入两槽位时无提前分配第三份执行资源。设备协议本身在 simpler 侧验证。 |
| `tests/ut/runtime/test_kernel_context.py`、`test_kernel_callable.py`（拟新增） | 并发首次初始化、失败回滚、重复加载、内部资源所有权；prepare 不执行业务；参数快照保活覆盖 simpler 恢复区间；内部恢复成功不被 Python 二次重试，最终失败正确上抛；无公开 prepare/close 要求。 |
| `tests/ut/runtime/test_dlpack.py`（拟新增） | legacy/versioned capsule、设备编码、dtype、非连续 stride、byte offset、空 Tensor、共享引用与单次 deleter；Out/InOut 不允许只读或隐式复制；新 Tensor/view 的元数据不被旧缓存覆盖。 |
| `tests/ut/torch/test_interop.py`、`test_registration.py`（拟新增） | 自动 DLPack 接入及 torch_npu 兼容导出、当前设备/stream、Tensor 类型与布局、外部输出、mutation/alias；FakeTensor 不做真实导出，正常 eager 调用不强制注册 custom op。 |
| `tests/ut/backend/test_kernel_config_signature.py`、`tests/ut/codegen/test_orchestration_codegen.py`、`test_orchestration_returned_param_map.py` | 完整 Tensor/Scalar 签名、方向、参数顺序与 wrapper/ABI 一致；有 codegen 联动时覆盖对应变化。 |
| `tests/st/runtime/kernel/`（拟新增，迁移有效 L1 用例） | 全组合 eager/capture/replay、重复异步调用、多个 callable、stream 切换、运行时 Scalar、DLPack 零拷贝外部 Out/InOut、异步引用存活及 torch_npu taskQueue 开关两条路径；公开未覆盖的 dtype/布局，不用已知失败掩盖缺口。 |
| `tests/st/runtime/framework_and_models/test_jit.py`、`test_compiled_program.py` 及相关 scheduling 用例 | program 的既有调用与两槽位路径回归；不能用 kernel 测试结果替代 program 路径结果。 |

大 args 恢复协议在 simpler 侧验证：首次成功不做额外同步；OOM 后回收成功且第二次提交成功；
清理/record/sync 失败；第二次提交失败且无第三次重试；非可恢复错误不重放；capture 不进入 host 等待。
整网集成还需覆盖连续大 args 积压、部分提交后的 AICore 退出、Out/InOut 不重复更新及 taskQueue 错误传播；
重复调用同一 callable 应仅增加 RTS 在途 args，不能误消耗 callable 条目或其 2 GiB 配额。

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
| `docs/en/user/02-quickstart.md`、`docs/en/user/language/00-types.md`、`01-functions.md` 及 `docs/zh/` 对应文件 | 更新 `pl.jit(mode=...)`、运行时 Scalar、外部输入输出、内部自动 DLPack 接入与首次调用 lazy 编译；说明 dtype/布局边界，移除显式编译及手动 context/prepare/close 指导。 |
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

### 7.1 DLPack 要求的参考依据与当前差异

协议依据：

- [DLPack Python 协议](https://dmlc.github.io/dlpack/latest/python_spec.html)：稠密 strided 数组的数据交换、
  stream 协商、capsule 消费与共享引用管理；这些机制不等于 torch_npu taskQueue 集成。
- [DLPack C 结构定义](https://github.com/dmlc/dlpack/blob/main/include/dlpack/dlpack.h)：
  `DLTensor`、legacy/versioned managed tensor、设备及 dtype 编码；本轮查阅的上游已包含 `kDLAscend`。
- [Array API 的 from_dlpack 契约](https://data-apis.org/array-api/latest/API_specification/generated/array_api.from_dlpack.html)：
  区分共享视图与复制；`copy=False` 表达不允许复制。PyPTO 的外部 Out/InOut 不能依赖可能产生副本的导入行为。

下表的代码结论来自本轮已读取的本地源码，路径分别相对所列仓库根目录；符号用于定位，
不把本地源码观察等同于当前部署环境或所有版本的实测结果。

| 仓库与文件 | 已观察到的行为 / 借鉴点 |
| --- | --- |
| CUTLASS：`python/CuTeDSL/cutlass/cute/runtime.py`，`from_dlpack`、`_Tensor`、`load_dltensor`、`__c_pointers__` | 导入后创建包装对象，按需解析 Tensor descriptor 并缓存调用描述；借鉴共享数据与惰性元数据处理，不照搬 CUDA 设备假设。其 `__dlpack__(stream=-1)` 路径不能直接作为 NPU 无需同步的证明。 |
| torch_npu：`torch_npu/csrc/aten/common/DLConvertor.cpp`，`toDLPack`、`getDLDevice`、`getDLDataType` | 导出持有原 Tensor 的 view，使用其地址/shape/stride。当前 NPU 编码为 `kDLExtDev`，且明确拒绝 FP8；需与标准设备编码及目标 dtype 做适配，不能宣称已全类型打通。 |
| torch_npu：`torch_npu/utils/dlpack.py`，`_apply_dlpack_patch`；`torch_npu/csrc/npu/Module.cpp`，`THPModule_toDLPack` | 已有 NPU capsule 导入/导出及 torch 工具函数适配；可作为兼容导出路径参考，不代表 Python Tensor 协议在所有版本完全一致，也不替代提交顺序处理。 |
| PyPTO：[现有 torch_npu adapter](../python/bindings/torch_npu_l1_adapter.cpp)，`enqueue`、`DeferredQueueCall` | 当前 stream、延迟提交保活、`recordStream`、`RunOpApiV2` 各自承担框架集成职责；采用 DLPack 后这些职责仍需迁入正式适配层。 |

当前差异的处理原则：

- `kDLExtDev` 是扩展设备类别，不是 NPU 专用编号；仅在已确认的 torch_npu 生产者路径中映射到 NPU。
  对提供 `kDLAscend` 的生产者按对应标准编码处理，不把未知扩展设备默认为 Ascend。
- 当前导出端拒绝 FP8 属于实现兼容性缺口，不是协议原理不支持。补齐导出/适配前明确报错，
  不用无类型字节包装偷偷丢失 dtype 语义；NZ 等私有布局也必须有明确契约。
- 通用接入层保留协议共享引用，框架层保证异步使用顺序和存活期；不引入外部 workspace allocator。
- DLPack 主要整理 Host Tensor 接入，不改变设备 task 划分、SPMD 调度或 tile；不能将采用协议本身宣称为 CSA 设备性能优化。

## 8. 建议实施顺序与尚需定稿的小项

1. 先定 mode 默认值、显式常量表达及内部 ABI 参数契约，锁定外部输入输出语义。
2. 改 JIT/内部编译对象，落实 mode、Scalar、产物所有权及首次调用 lazy 编译，不公开显式编译 API。
3. 接通用 DLPack Tensor 层、正式 kernel 内部目录及独立 PyTorch adapter，完成 lazy 生命周期和异步调用链。
4. 接 program 共用上层契约，保留独立两槽位 launch；同步落实 simpler 的底层前提及第 2.4 节大 args 回收重试。
5. 按完整矩阵补齐实现和回归，再退场 demo，更新正式中英文文档与示例。

实施前还需精确定义的只是细节，不重新打开已决定的方向：

- `mode` 默认值；建议 `program`，不把建议写成用户已确认。
- 显式编译期常量的语法；这是参数特化规则，不是显式编译 API，普通 Scalar 不要求编译占位符。
- 正式产物/ABI 元数据的具体字段及内部错误类型；用户只持有 JIT 算子，不单独管理编译结果对象。
- PyTorch 注册辅助的具体签名和无 backward 时的错误表现，不默认扩大到训练支持。

本次不修改算子数学逻辑，不做性能自动调优，不公开显式编译及 allocator/prepare/close API，不统一两条
launch，不新增无限深 pipeline，也不把模型仓、vLLM/recipes 接入修改混进 PyPTO 文件清单。
