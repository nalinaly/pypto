# NPU / GPU 九种编程与编译执行路径：分层解剖、用户表达与技术相似性

核对日期：2026-09-07。本文以编程模型、编译执行分层、megaKernel 性能机会及工程代价的分析为主体；A5 实测用于给这些分析提供具体的验证对象与边界。源码来源固定到 GitHub/GitCode commit，CANNBot DSL 引用本地代码。GPU profiling 来自历史实验；A5 环境、PA 结果和复跑入口见第 12 章。

本文面向项目经理、技术主管、开发者、客户和测试人员，目的在于建立可用于技术分析及后续管理讨论的共同事实基础，而不是选择“获胜团队”。关注顺序为：技术相似性、megaKernel 可达成性、用户感知、表达能力、优化空间、动态能力、开发成本。

**比较范围：**A—I 九条路线按相同的编程、编译和执行层次比较，各自的源码及实验适用范围分别标注。

- **A（PyPTO2-tensor版）** 对应 `pypto2/python/pypto`；**B（PyPTO2-block版）** 对应同仓的 `python/pypto_pro`。
- **PyPTO3（Simpler）** 对应 `pypto3/pypto`；`pypto-lib` 是其算子 / 模型库，不单列路线。
- **graph-autofusion 仅比较 AutoFuse，排除 SuperKernel**；SuperKernel 的静态多 kernel 重组能力不计入本文任何能力判断。
- **H（CATLASS DSL）** 特指 CATLASS 的 Python TLA DSL，不能将整个 CATLASS C++ 库的能力计入。本机验证见第 12 章。
- **I（CuTe DSL）** 特指 NVIDIA CuTe DSL，不能与 E 的 TensorIR / CUDA Tile IR 路径混同。其分析依据源码，未在 NVIDIA GPU 上执行验证。

阅读导航：[主结论](#conclusions) → [softmax / paged decode 用户示例](#examples) → [Python 前端与 API 能力边界](#frontend-capabilities) → [分层架构](#layers) → [两组重点架构对比](#focused-comparisons) → [MLIR 专章](#mlir) → [megaKernel：性能、代价与差距](#megakernel) → [动态 scheduler 的真实收益边界](#scheduler-performance) → [动态 shape：谁写 tiling、具体怎么写](#dynamic-tiling) → [用户怎样实现核内融合与片上复用](#intra-kernel-fusion) → [GPU launch](#gpu-launch) → [替代及共用边界](#replacement) → 验证与源码 → [最后的逐项“最相似者”](#nearest)。

<a id="conclusions"></a>
## 1. 主结论：九个名称，并不对应九套互斥的完整技术栈

### 1.1 比较对象及架构归类

| 标识（名称） | 具体范围 | 核心定位 | 主要执行组织 |
| --- | --- | --- | --- |
| A（PyPTO2-tensor版） | `pypto2/python/pypto` + `framework` | Tensor 程序编译、自有 IR、TileFwk 任务体系 | 设备侧控制 / 依赖解析 / 任务派发 |
| B（PyPTO2-block版） | 同仓 `python/pypto_pro` | 显式 Tile / 核内编程，自有 IR → CCE | Host 直接 launch 多核 kernel；kernel 内划分工作 |
| C（PyPTO3（Simpler）） | `pypto3/pypto` + `runtime`；用户例子优先 `pypto3/pypto-lib` | 多层程序 / InCore / Orchestration + PTOAS + Simpler | 任务图；也能把混合核 SPMD kernel 作为一个多 block task |
| D（CANNBot DSL） | `cannbot_dsl/cannbot-dsl`，补充 `cannbot_dsl/cannbot-arena` | Python 分阶段 Host / Device 编程，MLIR CANNIR → AscendC | Host launch；用户显式内存、Channel 和核内流水 |
| E（PyPTO on GPU） | `PyPTO-LOVE-TensorIR` 集成层及 source lock 固定的 PyPTO 源码 bundle | PyPTO 前端 / 自有 IR → 受支持模式 → NVIDIA TensorIR / CUDA Tile IR | PyPTO GPU runtime 直接 CUDA launch；硬件安排 CTA |
| F（Triton-Ascend） | `triton-ascend/triton-ascend` + `triton-ascend/triton-ascend-kernels`；Inductor 入口 `torch_npu/torch_npu/_inductor` | Triton kernel DSL，也可接受 Inductor 生成的 kernel | 编译器映射逻辑 program 到 NPU；CANN launch |
| G（AutoFuse + Inductor） | `graph-autofusion/autofuse` + `torchair/experimental/_inductor_npu_ext` | 融合图编译 / 自动 tiling / kernel 生成组件；客户通过 PyTorch 使用 | Inductor 分组及 Host wrapper；生成的多核融合 kernel |
| H（CATLASS DSL） | `catlass/python/tla_dsl/catlass/catlass_dsl`，公共 API 为 `catlass.tla`；示例 `python/tla_dsl/examples` | Python 显式 tile / 物理 layout / Cube / Vector 编程；TLA MLIR → AscendNPU-IR → CANN | 编译 kernel 后由 Host 调用产物；支持 AIC/AIV mixed kernel，例子自行安排工作与同步 |
| I（CuTe DSL） | `cutlass/python/CuTeDSL`；示例 `examples/python/CuTeDSL` | Python layout 代数、线程/值分区、Copy/MMA atom、异步流水；CuTe MLIR → NVVM / cubin | Host JIT / CUDA launch；CTA/cluster 内合作，另有 persistent tile scheduler 与实验性 Task Scheduling |

前三个“PyPTO”必须按上述代码位置识别：A（PyPTO2-tensor版）/B（PyPTO2-block版） 同仓共享基础设施；C（PyPTO3（Simpler）） 不等于 A（PyPTO2-tensor版）；E（PyPTO on GPU） 使用的是另一份 PyPTO checkout，不能把 C（PyPTO3（Simpler）） 的所有新能力直接投射到 E（PyPTO on GPU）。

### 1.2 十项最重要的判断

以下十项按相同责任层比较九条路线。局部相似组不排除其他路线；H/I 的具体定位见第 1.5 节，各路线按整体架构选择的最相似者见第 14 章。

1. **A（PyPTO2-tensor版） 与 C（PyPTO3（Simpler）） 最接近的是“程序 / 任务系统”层。** 两者可用设备侧执行体系组织多个计算阶段，但 A（PyPTO2-tensor版） 的 TileFwk runtime 不是 C（PyPTO3（Simpler）） 的 Simpler。
2. **B（PyPTO2-block版） 与 D（CANNBot DSL） 最接近的是“显式核内工程”层。** 用户较直接地承担物理 tile、核间分工、尾块、流水及局部资源选择；二者并不共享同一个 IR 或后端。H（CATLASS DSL）也直接落在这一责任层，且与 D 都较早进入 MLIR；I（CuTe DSL）则把相近责任落实到 GPU layout/atom/warp，物理协议需另外比较。
3. **F（Triton-Ascend） 与 G（AutoFuse + Inductor） 最接近的是“PyTorch → Inductor → 自生成 NPU kernel”的集成位置。** F（Triton-Ascend） 还拥有明确的独立 kernel DSL；G（AutoFuse + Inductor） 没有同等定位的终端用户 kernel DSL，但不是“没有用户入口”。H 的 Python TLA DSL 与 torch_npu 中名为 CATLASS 的 C++ 模板接入应分开，I 的 tensor bridge 也不自动成为通用 Inductor backend。
4. **E（PyPTO on GPU） 的前端血缘接近 C（PyPTO3（Simpler）），当前执行与核内编译分工更接近 F（Triton-Ascend）。** E（PyPTO on GPU） 没有把 Simpler scheduler 移植为 GPU runtime；多 SM 主要来自一个 launch 内的多个 CTA。I 同样采用 CUDA launch，但作者显式安排 layout/atom/线程分区；这使其核内控制面与 E 不同，也不共享 E 的历史实测。
5. **PTOAS、AscendNPU-IR、NVIDIA TensorIR、AscendC 不能与上述九条上层路径当作平级替代品。** 前三者是不同层次的编译基础设施；AscendC 主要是设备编程 API / 库及配套编译接口，不是任务调度框架。
6. **MLIR 和 SSA 不是对立选项；MLIR 的表示能力、已实现 pass 和运行时能力也不是同一件事。** A（PyPTO2-tensor版）/B（PyPTO2-block版）/C（PyPTO3（Simpler））/E（PyPTO on GPU） 的前端可以使用自有 SSA IR；C（PyPTO3（Simpler）） 在后端交 PTO dialect MLIR，E（PyPTO on GPU） 交 TensorIR；D（CANNBot DSL）/F（Triton-Ascend） 在更长的编译区间使用 MLIR；G（AutoFuse + Inductor） 的 ASCIR 图不是因为名字有 IR 就成为 MLIR。H 的 TLA→HIVM/AVE 与 I 的 CuTe→NVVM 都有具体 MLIR 链，前者共享部分 NPU 下游，后者仍有独立的 CUDA 目标与配套编译组件。
7. **block版 以 SPMD kernel 为主，PyPTO3 的外层则支持 MPMD 任务编排；两者不是互斥标签。** C（PyPTO3（Simpler）） 的 `pypto-lib` 已有多 block、混合 Cube/Vector、带显式任务依赖的 SPMD attention；框架与其中一个 kernel 的执行方式属于不同层。第 4.10 节展开用户分工、设备派发及组合边界。H 的 AIC/AIV 分工、I 的 warp 专门化也能在单个合作 kernel 内异构执行；I 的 TS 资源任务与 C 的模型任务属于不同粒度。
8. **动态 shape 不必要求客户另写 tiling 函数。** B（PyPTO2-block版）/D（CANNBot DSL）/F（Triton-Ascend） 可以在固定物理 tile 上循环并处理有效长度；G（AutoFuse + Inductor） 会生成 Host tiling；A（PyPTO2-tensor版）/C（PyPTO3（Simpler）） 也仍然存在 tiling 决策，甚至有显式 tiling task。H/I 同样不普遍要求注册式 Host tiler，但作者仍需区分动态逻辑范围、编译期物理资源和设备工作分配；H 的 MMAD 动态复用与 FA 常量特化不能合并记账。
9. **“一层 Transformer 一个算子”必须同时验收组织方式、性能与代价。** 一个入口、一次提交、设备任务图、一个物理 kernel、消除中间 GM 不能互相替代证明。极致性能取决于权重/KV 访问、跨阶段重分片、局部效率、同步及负载均衡；第 6 章逐路线拆解已有实现、性能机会、用户责任与工程差距。
10. **当前没有足以给九条路线做性能名次的同口径实验。** 本文可以判断已有实现、可行的组合边界和可能代价，不能据源码长度、MLIR 使用量、launch 数或核心利用率推导谁更快。

依据见各章及末尾源码索引。上面的“最接近”属于架构判断，不是测量出的相似度分数。

### 1.3 从组织视角真正重叠的工作

| 重叠区域 | 主要对象 | 可以讨论的共用物 | 不能据此推出 |
| --- | --- | --- | --- |
| 设备任务图、依赖、逻辑任务 ABI、运行时诊断 | A（PyPTO2-tensor版）、C（PyPTO3（Simpler）） | 任务描述契约、依赖测试集、诊断事件模型 | 两套 scheduler 直接替换 |
| Tile / Vector / Cube 语义、流水与片上资源 | B（PyPTO2-block版）、C（PyPTO3（Simpler）） 的 InCore、D（CANNBot DSL）、F（Triton-Ascend） 的后端、G（AutoFuse + Inductor） 的 kernel 生成 | 算子语义、资源描述、效应 / 生命周期模型、正确性用例 | 现有内存规划 pass 可不改就共用 |
| Inductor 接入、分组、外部算子边界、符号 shape | F（Triton-Ascend）、G（AutoFuse + Inductor） | FX/Inductor 测试、wrapper / profiling 约定、fallback 分类 | 两者生成相同 kernel 或共享全部 lowering |
| 形状专门化、缓存、tile 选择和直接 launch | B（PyPTO2-block版）、D（CANNBot DSL）、E（PyPTO on GPU）、F（Triton-Ascend）、G（AutoFuse + Inductor） | 编译产物 manifest、缓存键要求、launch 记录 | 编译缓存或二进制 ABI 已兼容 |
| 最底层目标代码生成 | PTOAS 的末端、AscendNPU-IR 的末端、CCE / AscendC 工具链、GPU tile 编译器 | 硬件语义和指令级验证 | NPU / GPU 指令或内存层次可统一成一份实现 |
| Python layout、显式搬运/矩阵与局部资源协议 | H（CATLASS DSL）、I（CuTe DSL）；与 B/D 的核内责任交叉 | tile/资源/异步效应描述、尾块与生命周期用例 | H 的物理 tag 等于 CuTe layout 代数，或 NPU CV 等于 GPU warp/TMA。[H-api-layout] [I-layout] [I-task] |
| MLIR 下游、编译产物与直接 launch | H 与 F 的部分 AscendNPU-IR 基础设施；I 与 E 的 CUDA artifact/launch 分工 | 固定版本、IR 输入契约、manifest/cache 与 launch 诊断约定 | H/F 使用同一完整 pipeline，或 I 已集成为 E 的后端。[H-passes] [H-execution] [I-compiler] [I-executor] |

这比按仓库名判断“重复建设”更准确：有的重叠是合理的上下游分工，有的是同层不同策略，有的才是可以通过公共契约减少重复投入的区域。

### 1.4 关键疑问与专题入口

- **设备 scheduler 是否使 megaKernel 更容易？** 有助于组织和推进整层，但 PyPTO2-tensor版也有设备任务体系；性能还取决于分解、核内质量、GM 和调度成本，见第 6.13 节。
- **tensor版是否把 tile 交给用户？** 是：Vector/Cube tile、split-K、view/loop/valid_shape 和 `sg_set_scope` 均可控制；与 PyPTO3 显式 InCore/TaskId/SPMD 边界的区别见第 4.8 节。
- **block版、PyPTO3 与 CANNBot 哪里相近？** block版 与 CANNBot 更接近显式核内工程；与 PyPTO3 相近的是核内子域，不是完整任务体系。TileGroup/Channel、前端及后端差异见第 4.9—4.10 节。
- **Python 前端与 API 是否完整？** 九条路线都有 Python 使用入口，但 DSL/构图层次不同；不能据此宣称覆盖全部 Vector/Cube/Tensor Core 指令组合。第 2.9—2.12 节按 API、lowering、目标支持和验证分层。
- **怎样让连续 Vector 计算不经中间 GM？** 九条路线在各自支持域内均有源码路径，用户写法见第 8 章；同一核内区域、消除显式中间 GM、进一步减少 UB/片上读写须分别验收。

- **CATLASS / CuTe 与既有路线怎样同层比较？** H 与 B/D 对照显式 NPU 资源责任，与 F 对照下游编译；I 与 H 对照 layout/流水，与 E 对照 CUDA 编译/launch。动态 layout、persistent/CLC、warp TS 各自的边界见第 4.13、6.14、7.15、8.12 节。

### 1.5 CATLASS DSL / CuTe DSL 的定位与能力边界

1. **H/I 都强调显式核内工程，硬件控制对象不同。** H 与 B/D 在 NPU 的物理 tile、搬运、混合核与同步责任上相近；H 与 I 则在 Python 元编程、显式 layout/tensor 和编译产物调用的思路上相近。H 的 layout tag / `origin_shape` 与 I 的通用 layout 组合、线程—值分区不能按 API 名字一一替换。[H-api-layout] [I-layout]
2. **“有设备 scheduler”必须注明调度粒度。** H 的 FA 用 `block_idx/block_num` 做固定步长任务循环，StreamK 另有工作切分及归并；I 有静态 persistent、CLC 动态 tile 分配以及 warp 级 Task Scheduling。它们为第 6 章提供了有用的中间形态，但都不能直接当成 A/C 的整层任务 runtime。[H-fa] [H-streamk] [I-static] [I-dynamic] [I-task]
3. **H/I 采用不同的 MLIR 编译链。** H 有可定位的 TLA passes，并依赖 CATLASS 固定的 AscendNPU-IR 子模块；I 的 Python 源码明确调用 `cute-to-nvvm`，使用独立配套编译组件。H 不走 PTOAS，I 不走 E 的 TensorIR emitter；同用 MLIR 不说明后端、ABI 或 pass 已兼容。[H-passes] [H-ir-build] [I-dsl] [I-requirements]
4. **Attention 示例分别证明不同组合。** H 是连续 KV 的 online-softmax FlashAttention；I 找到分页 MLA decode，以及另一套连续 GQA decode。后者的 simple 版本在同一个 JIT 入口中 launch decode 和 reduction 两个 kernel。标准 GQA PA、分页 MLA、连续 FA 的能力不能互相冒充。[H-fa] [I-mla] [I-gqa]
5. **局部实现与整层性能需要分别评估。** H/I 的源码为片上预算、全局归约、重分片、负载均衡和调度开销提供具体实现参照，分析见第 6—8 章。目前没有九条路线的同口径性能实验，也没有整层单物理 kernel 已完成的证明。

<a id="examples"></a>
## 2. 用户表达：先用相同计算问题比较

### 2.1 示例口径与完整性

本章提供数学参考、原生 kernel、Host 调用及源码入口，按下列四类标注完整性。**本章的代码示例用于展示表达方式与责任划分；已执行的 A5 PA 对象逐项列于第 12 章，不能把某个用例通过扩展为本章所有代码、shape 或融合模式都已验证。**“完整代码”不等于“已实测”，注明“摘录”的代码仍依赖所列上下文。版权与许可证以原文件为准。

源码摘录、整理后的用户写法和数学参考分别注明上下文。源码链接指向核对版本，历史 GPU 脚本与记录指向实验归档；第 13 章分别列出 GPU 与 A5 环境版本，避免把不同 checkout 的实验混为一谈。

- **完整语义参考**：普通 Python / PyTorch，可用于构造 golden；不代表会自动融合成一个 kernel。
- **源码中的完整实现**：提供原文件、入口和适用边界。
- **源码推导的写法**：用于说明责任划分，不作为编译成功或性能证据。
- **缺少对应例子**：明确说明，不把 FlashAttention、PagedAttention 设计文档、外部库调用自动当作“本路线原生 paged decode kernel”。

统一形状：

| 场景 | 张量 / 参数 | 要检验的区别 |
| --- | --- | --- |
| 尾轴 softmax | `X[M,N]`，FP32；例如 `M=777,N=300` | reduce 在最后一维；同时存在列不对齐与行 tile 尾块 |
| 超长 reduce softmax | 例如 `X[8,131075]`；逻辑整行及中间值超出选定片上预算 | 单核分段遍历、跨核拆分、全局归并是不同选择 |
| Paged decode attention | `Q[B,Hq,D]`；`K/V[P,Hkv,S,D]`；`BT[B,T]`；`L[B]` | `L` 为每请求实际 KV 长度；`P/T/B` 为 shape |
| 代表性 PA 几何 | `Hq=40,Hkv=8,D=128,S=128`，GQA=5 | 与 PyPTO3 Qwen 例子便于对应；不要求其他路线原生 ABI 相同 |
| 静态 / 动态矩阵 | `L` 取 1/127/128/129/511/512/513/4096；`B` 取 1/8/16；另改变 `P/T` | 内容变化与张量元数据变化分开验证 |

这里只固定 rank、数据类型及上述约束；不将动态维度自动扩展成动态 rank。零长度序列不是以下参考的定义域，需要单独约定输出语义。

### 2.2 完整 softmax 参考：超长行不必整体驻留片上

```python
import torch

def softmax_reference(x):
    z = x.float()
    e = torch.exp(z - z.amax(dim=-1, keepdim=True))
    return (e / e.sum(dim=-1, keepdim=True)).to(x.dtype)

def softmax_blocked_reference(x, chunk=1024):
    # Assumptions: rank 2, finite input, N >= 1, chunk >= 1.
    assert x.ndim == 2 and x.shape[1] > 0 and chunk > 0
    rows, cols = x.shape
    m = torch.full((rows, 1), -torch.inf, device=x.device, dtype=torch.float32)
    l = torch.zeros_like(m)

    # First traversal: combine per-chunk online max and exp-sum.
    for start in range(0, cols, chunk):
        z = x[:, start:min(start + chunk, cols)].float()
        next_m = torch.maximum(m, z.amax(dim=1, keepdim=True))
        l = l * torch.exp(m - next_m) + torch.exp(z - next_m).sum(1, keepdim=True)
        m = next_m

    # Second traversal: write normalized output without retaining a whole row.
    y = torch.empty_like(x)
    for start in range(0, cols, chunk):
        end = min(start + chunk, cols)
        y[:, start:end] = (torch.exp(x[:, start:end].float() - m) / l).to(x.dtype)
    return y
```

这是完整算法，不是高性能实现承诺：PyTorch eager 仍可能逐语句 launch。将其写入一个核内 kernel 时，每个执行实例可以只保留 `chunk` 数据、`m/l` 和必要暂存。

如果为小 batch 增加跨核并行，可让多个核生成每段 `(m_i,l_i)`，再归并：

```text
m = max_i(m_i)
l = Σ_i exp(m_i - m) * l_i
y_j = exp(x_j - m) / l
```

归并需要依赖与数据可见性：可以是多 kernel，也可以是任务图，也可以是满足进展条件的单 kernel 协作。**“整行放不下”不等于“必然跨核”，也不等于“必然多次 Host launch”。**

### 2.3 九条路径写 softmax 时，用户到底承担什么

| 路径 | 实际用户表达 | 尾轴 / 尾块 | 超长 reduce 的工程动作 |
| --- | --- | --- | --- |
| A（PyPTO2-tensor版） | Tensor 的 max/sub/exp/sum/div，设置 vec tile，写 `pypto.loop` | 动态标记、view 的 valid_shape、编译器 tiling | 调整 tile / reduce 分解；检查生成的子图和 task，不能只数 Python 运算 |
| B（PyPTO2-block版） | TileType、物理存储、load/store、row reduce、显式 core-stride | `DYNAMIC` + `set_validshape`；物理 TileType 上限仍固定 | 在同一 kernel 写分段循环或设计多核归并；可加 Host 策略但不强制另写 TilingFunc |
| C（PyPTO3（Simpler）） | `pl.parallel` + `CORE_GROUP` 中的 tensor/tile 运算；也可显式多 InCore | bind 动态维、slice/load 有效窗口、编译器核内 lowering | 选择核内循环、多个 task 或 SPMD task；任务依赖与局部算法分别设计 |
| D（CANNBot DSL） | UB Buffer / Channel、reduce/expand、显式搬运，Host `@jit` 调 kernel | 动态 TensorSpec 与有效访问 / padding；Channel 物理容量仍需约束 | 分段 UB 流水，必要时 GM partial 与多 kernel / 显式同步 |
| E（PyPTO on GPU） | PyPTO 图经当前支持的 pattern 编译；已有五阶段实验为五个 executable | 当前高层集成主要是静态 shape / stride 专门化 | 必须确认 emitter 支持该模式；NPU 写法不能直接保证 GPU 可编译 |
| F（Triton-Ascend） | `tl.program_id`、`tl.arange`、mask、reduce；Host 决定 grid / constexpr | runtime `n_cols` 与 mask；`BLOCK_SIZE` 是编译期常量 | 固定 chunk 的核内循环或 split-reduce；后端负责 local layout、buffer 和同步 |
| G（AutoFuse + Inductor） | 客户写 PyTorch max/sub/exp/sum/div，`torch.compile` 选择 AscendC 后端 | Inductor 符号维 + AutoFuse 生成的 tiling / 尾块模板 | 编译器选择融合、分段与 workspace；须检查是否拆分或落外部算子 |
| H（CATLASS DSL） | 现有 FA 子过程显式分配 UB 状态，在 `vec.func` 中写 max/exp/sum 与在线更新 | 固定物理块配合有效范围、寄存器 mask；本例全零输入 mask，不是完整 mask 支持验收 | 作者组织分块统计、重读/状态合并、CV 交接；不能把连续 FA 当独立通用 softmax 或 PA 已验证。[H-fa] |
| I（CuTe DSL） | 教程八种 softmax，作者选择线程/值分区、shuffle/shared 归约及输出策略 | tile、线程数、mask 与编译常量由示例约定；本机只核对源码 | kernel 8 在线统计后重读输入；`range_constexpr` 展开也会带来编译/代码体积成本，不是任意长行的自动最优策略。[I-softmax] |

#### A（PyPTO2-tensor版）：完整用户入口的典型写法

以下依据tensor版 softmax 示例整理，展示动态 batch 与五个数学步骤。完整原文件包含数据生成、golden、运行参数：[A12（PyPTO2-tensor版）]。

```python
import pypto

@pypto.frontend.jit
def softmax_pypto2(
    x: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    y: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
):
    batch, seq, heads, width = x.shape
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    for b in pypto.loop(0, batch, 1, name="batch_loop", idx_name="b"):
        z = x[b:b + 1, :seq, :heads, :width]
        m = pypto.amax(z, dim=-1, keepdim=True)
        e = pypto.exp(z - m)
        y[b:b + 1, ...] = e / pypto.sum(e, dim=-1, keepdim=True)
```

这里 `DYNAMIC` 只明确标在第一轴，`...` 不是“其他轴都动态”的声明。将普通参数取值改掉、将轴声明为动态、改物理 tile，是三种不同修改。

#### B（PyPTO2-block版）：双动态 softmax 已有完整测试，但“动态”有物理上限

原文件 [B14（PyPTO2-block版）] 含 kernel、参数生成、多核 launch 和六组 golden case。下面保留常量、双缓冲地址与**完整 kernel**，补齐导入：

```python
import pypto_pro.language as pl

# ================================================================
MAX_N = 512  # max supported columns == compile-time UB tile width
TILE_ROWS = 16  # rows processed per tile-group slot (row count is dynamic)

SLOT_BYTES = TILE_ROWS * MAX_N * 4  # fp32 [TILE_ROWS, MAX_N]
RED_BYTES = 512  # fp32 [TILE_ROWS, 1] reduction result (padded/aligned)

# UB addresses: double-buffered input / output / workspace groups + reduction result.
VA_IN0 = 0
VA_IN1 = VA_IN0 + SLOT_BYTES
VA_OUT0 = VA_IN1 + SLOT_BYTES
VA_OUT1 = VA_OUT0 + SLOT_BYTES
VA_TMP0 = VA_OUT1 + SLOT_BYTES
VA_TMP1 = VA_TMP0 + SLOT_BYTES
VA_RED0 = VA_TMP1 + SLOT_BYTES
VA_RED1 = VA_RED0 + RED_BYTES


@pl.jit(auto_mutex=True)
def softmax_tile_group_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    # valid_shape=[-1, -1] makes the per-tile valid window dynamic (set at runtime via
    # set_validshape): the tail row-tile carries fewer rows and N narrows the columns.
    tile_type = pl.TileType(
        shape=[TILE_ROWS, MAX_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1]
    )
    # Row reductions write a [TILE_ROWS, 1] column vector (layout=pl.DN).
    red_type = pl.TileType(
        shape=[TILE_ROWS, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.DN, valid_shape=[-1, -1]
    )
    in_group = pl.make_tile_group(type=tile_type, addrs=[VA_IN0, VA_IN1], mutex_ids=[0, 1])
    out_group = pl.make_tile_group(type=tile_type, addrs=[VA_OUT0, VA_OUT1], mutex_ids=[2, 3])
    tmp_group = pl.make_tile_group(type=tile_type, addrs=[VA_TMP0, VA_TMP1], mutex_ids=[4, 5])
    red_group = pl.make_tile_group(type=red_type, addrs=[VA_RED0, VA_RED1], mutex_ids=[6, 7])

    with pl.section_vector():
        rows = x.shape[0]
        cols = x.shape[1]
        num_cores = pl.get_block_num()
        core_id = pl.get_block_idx()

        num_tiles = (rows + TILE_ROWS - 1) // TILE_ROWS

        # Each core strides over the row-tile grid; the tail row-tile is partial.
        for tile_id in pl.range(core_id, num_tiles, num_cores):
            row_off = tile_id * TILE_ROWS
            valid_rows = pl.min(TILE_ROWS, rows - row_off)

            in_slot = in_group.next()
            pl.set_validshape(in_slot, [valid_rows, cols])
            pl.load(in_slot, x, [row_off, 0])

            out_slot = out_group.next()
            tmp_slot = tmp_group.next()
            red_slot = red_group.next()
            pl.set_validshape(out_slot, [valid_rows, cols])
            pl.set_validshape(tmp_slot, [valid_rows, cols])
            pl.set_validshape(red_slot, [valid_rows, 1])

            # ---- pass 1: row max ----
            pl.row_max(red_slot, in_slot, tmp_slot)  # red = max over N valid cols
            pl.row_expand_sub(out_slot, in_slot, red_slot)  # out = x - max (row broadcast)

            # ---- pass 2: exp then row sum ----
            pl.exp(out_slot, out_slot)  # out = exp(x - max)
            pl.row_sum(red_slot, out_slot, tmp_slot)  # red = sum over N valid cols

            # ---- pass 3: normalize ----
            pl.row_expand_div(out_slot, out_slot, red_slot)  # out = exp(x - max) / sum

            pl.store(y, out_slot, [row_off, 0])

    return
```

原 `_run_case` 中影响多核及动态范围的调用如下（省略日志；需原测试 `enable_slice=False` 配置）：

```python
import torch

rows, cols = 777, 300
assert cols <= MAX_N
x = torch.rand((rows, cols), device="npu", dtype=torch.float32) * 8 - 4
y = torch.empty_like(x)
num_tiles = (rows + TILE_ROWS - 1) // TILE_ROWS
num_cores = min(32, num_tiles)
softmax_tile_group_kernel[None, num_cores](x, y)
torch.npu.synchronize()
torch.testing.assert_close(y.cpu(), torch.softmax(x.cpu(), dim=-1),
                           rtol=1e-3, atol=1e-3)
```

`DYNAMIC` 决定逻辑形状；`MAX_N/TILE_ROWS` 决定容量；`set_validshape` 决定当前有效窗口；block ID/数量决定行块分工。四者不能混称一个 tiling 参数。

测试包括 `(1000,200)`、`(777,300)`、`(2049,100)`，且 Host 明确断言 `cols <= 512`。因此该例证明的是“固定容量内两维动态”的源码契约，不是任意长行或无限动态 UB。测试标记为 `soc("950")`，不能把它写成 A2/A3 上已验证。

超长行需要用前述分段算法重写 kernel，不是把 `cols` 改成 131075 而保持 `MAX_N=512`。同样，不必为每一个 200/300/512 的取值新增 tiling 函数。

#### C（PyPTO3（Simpler））：优先使用 pypto-lib 的真实用户写法

下面完整 kernel 来自 `pypto-lib/examples/intermediate/softmax.py` 的同等写法；完整文件还提供 golden harness：[C15（PyPTO3（Simpler））]。

```python
import pypto.language as pl

ROWS, COLS, ROW_TILE = 512, 256, 64

@pl.jit
def softmax_pypto3(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    for r in pl.parallel(0, ROWS, ROW_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="softmax_rows"):
            z = x[r:r + ROW_TILE, :]
            m = pl.row_max(z)
            e = pl.exp(pl.row_expand_sub(z, m))
            y[r:r + ROW_TILE, :] = pl.row_expand_div(e, pl.row_sum(e))
    return y
```

这五个数学步骤可以写在 **一个 `pl.jit`** 中。`pl.parallel` 表达行块的并行工作；不能把 max/sub/exp/sum/div 的五次调用直接解释成五次 runtime task submit，更不能解释成五次 Host launch。应看 scope lowering 和生成的 InCore / Orchestration。

该源码是固定 `512×256` 的教学例子，不能直接当作 `777×300` 或超长行的完整动态实现。动态有效窗口与超长 reduce 的分段仍须分别表达。完整运行入口：

```bash
cd npu/pypto-lib
python examples/intermediate/softmax.py -p a2a3 -d 0
```

这是旧 workspace 的 A2/A3 调用示例，保留用于解释入口。当前算子库目录为 `pypto3/pypto-lib`；本 session 已用 `python examples/intermediate/softmax.py -p a5 -d 0` 验证该固定 `512×256` FP32 例子。它与通用 PA、Qwen 融合 PA 是不同的验证对象，详见第 12 章。[RUN-smoke]

#### D（CANNBot DSL）：源码已经区分“所有权队列”与“普通局部暂存”

完整例子：[D10（CANNBot DSL）]（Buffer 版 kernel）、[D11（CANNBot DSL）]（真实 tensor 输入和 NPU golden）。Buffer 版核心不是将每个临时值都包装成 Channel：

```text
GM → ch_x.acquire/commit/wait → UB x
    → reduce_max(Buffer m)
    → expand(Buffer m_full)
    → sub(Buffer shifted)
    → exp(Buffer e)
    → reduce_sum(Buffer sum)
    → expand(Buffer m_full，复用)
    → div(ch_y 的生产槽位)
    → ch_y.commit/wait → GM
```

下面嵌入 [D11（CANNBot DSL）] 的**完整 `SoftmaxKernel` 类**及真实 tensor 调用。它是 Channel 版本；上面的 Buffer 版本是另一份实现。

```python
from cannbotdsl import dtypes
from cannbotdsl.channel import Channel
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.math import div, exp, expand, reduce_max, reduce_sum, sub
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.tensor import tile_view
from cannbotdsl.types import MemLoc, Tensor

class SoftmaxKernel:


    def __init__(self, m, n):
        self.m = m
        self.n = n

    @kernel
    def softmax_kernel(self, gm_x: Tensor, gm_y: Tensor):
        self.ch_x = Channel(MemLoc.UB, shape=(self.m, self.n), dtype=dtypes.float32, depth=1)
        self.ch_y = Channel(MemLoc.UB, shape=(self.m, self.n), dtype=dtypes.float32, depth=1)
        self.ch_m = Channel(MemLoc.UB, shape=(self.m, 1), dtype=dtypes.float32, depth=1)
        self.ch_m_full = Channel(MemLoc.UB, shape=(self.m, self.n), dtype=dtypes.float32, depth=1)
        self.ch_x_shift = Channel(MemLoc.UB, shape=(self.m, self.n), dtype=dtypes.float32, depth=1)
        self.ch_e = Channel(MemLoc.UB, shape=(self.m, self.n), dtype=dtypes.float32, depth=1)
        self.ch_s = Channel(MemLoc.UB, shape=(self.m, 1), dtype=dtypes.float32, depth=1)
        self.ch_tmp = Channel(MemLoc.UB, shape=(self.m, self.n), dtype=dtypes.float32, depth=1)
        for i in range(4):
            gm_x_tile = tile_view(gm_x, (32, 32), (i, 0))
            x = self.ch_x.acquire()
            mem_copy(x, gm_x_tile)
            self.ch_x.commit(x)

            x_r = self.ch_x.wait()
            ub_m = self.ch_m.acquire()
            reduce_max(ub_m, x_r, axis=1)
            self.ch_m.commit(ub_m)
            m_r = self.ch_m.wait()
            m_full = self.ch_m_full.acquire()
            expand(m_full, m_r, axis=1)
            self.ch_m_full.commit(m_full)
            self.ch_m.release(m_r)
            m_full_r = self.ch_m_full.wait()
            x_shift = self.ch_x_shift.acquire()
            sub(x_shift, x_r, m_full_r)
            self.ch_x_shift.commit(x_shift)
            self.ch_m_full.release(m_full_r)
            self.ch_x.release(x_r)

            x_shift_r = self.ch_x_shift.wait()
            e = self.ch_e.acquire()
            exp(e, x_shift_r)
            self.ch_e.commit(e)
            self.ch_x_shift.release(x_shift_r)

            e_r = self.ch_e.wait()
            s = self.ch_s.acquire()
            reduce_sum(s, e_r, axis=1)
            self.ch_s.commit(s)
            s_r = self.ch_s.wait()
            m_full2 = self.ch_m_full.acquire()
            expand(m_full2, s_r, axis=1)
            self.ch_m_full.commit(m_full2)
            self.ch_s.release(s_r)

            m_full2_r = self.ch_m_full.wait()
            y = self.ch_y.acquire()
            div(y, e_r, m_full2_r)
            self.ch_y.commit(y)
            self.ch_m_full.release(m_full2_r)
            self.ch_e.release(e_r)
            gm_y_tile = tile_view(gm_y, (32, 32), (i, 0))

            y_r = self.ch_y.wait()
            mem_copy(gm_y_tile, y_r)
            self.ch_y.release(y_r)

    @jit
    def softmax_host(self, x: Tensor, y: Tensor):
        self.softmax_kernel[1](x, y)


@pytest.mark.npu
def test_softmax_npu_accuracy():
    pytest.importorskip("torch_npu")
```

原 NPU 测试的调用（去除 pytest 外壳）：

```python
import torch
import torch_npu

x = torch.randn(128, 32, dtype=torch.float32, device="npu")
y = torch.zeros_like(x)
op = SoftmaxKernel(32, 32)
op.softmax_host(x, y)
torch.npu.synchronize()
torch.testing.assert_close(y.cpu(), torch.softmax(x.cpu(), dim=-1),
                           atol=1e-4, rtol=1e-3)
```

`kernel[1]` 只发一个 block，遍历四个 `32×32` tile。不能只增大 `[1]` 就获得正确的多核版本：当前 kernel 没有按 block ID 划分输出，多个 block 将重复处理同一区域。多核、尾块、长 reduce 都要相应修改工作划分。

Channel 的 acquire/commit/wait/release 是数据生产消费与缓冲槽位协议，不是 Simpler 的全局任务 ready queue。`Buffer` 是局部暂存，不是 Host GM allocator。

动态入口的已实现独立例子是 [D7（CANNBot DSL）]：客户用 `Dim("M", multiple_of=32)` 构造 `TensorSpec((M,32),...)`，调用 `host_fn.compile(...)`；同一个产物用于 M=32/64/96。把这套入口用于 softmax，需要相应 kernel 使用 runtime shape 并正确处理 reduce / mask；不能仅通过动态 TensorSpec 自动修复固定循环和越界访问。

#### E（PyPTO on GPU）：当前真实 softmax 实验的完整性边界

以下代码、诊断和数值保留自旧 Ada SM89 环境的实验记录。本次 workspace 没有迁入原 profiler 脚本和 trace/summary；[E7（PyPTO on GPU）]—[E9（PyPTO on GPU）] 因而指向旧文档的固定 GitHub 归档，作为历史记录来源，不声称它们是公开上游测试或本次重新核验的原始 trace。当前 GPU 源码入口及 bundle 另见第 13 章。

完整实验脚本：[E8（PyPTO on GPU）]；summary 和原始 trace：[E7（PyPTO on GPU）]、[E9（PyPTO on GPU）]。它分别构建 max、sub、exp、sum、div 五个 GPU executable，再在同一 stream 上依次 launch，golden 为 torch softmax。

下面直接列出历史实验中的**五个完整 JIT kernel**：

```python
import pypto.language as pl

ROWS, COLS = 4096, 128



@pl.jit
def row_max_kernel(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    maximum: pl.Out[pl.Tensor[[ROWS, 1], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        for row in pl.range(ROWS):
            x_tile = pl.load(x, [row, 0], [1, COLS])
            scratch = pl.create_tile(
                [1, COLS], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
            )
            maximum_tile = pl.row_max(x_tile, scratch)
            pl.store(maximum_tile, [row, 0], maximum)
    return maximum


@pl.jit
def row_expand_sub_kernel(
    x: pl.Tensor[[ROWS, COLS], pl.FP32],
    maximum: pl.Tensor[[ROWS, 1], pl.FP32],
    shifted: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        for row in pl.range(ROWS):
            x_tile = pl.load(x, [row, 0], [1, COLS])
            maximum_tile = pl.load(maximum, [row, 0], [1, 1])
            shifted_tile = pl.row_expand_sub(x_tile, maximum_tile)
            pl.store(shifted_tile, [row, 0], shifted)
    return shifted


@pl.jit
def exp_kernel(
    shifted: pl.Tensor[[ROWS, COLS], pl.FP32],
    exponent: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        for row in pl.range(ROWS):
            shifted_tile = pl.load(shifted, [row, 0], [1, COLS])
            exponent_tile = pl.exp(shifted_tile)
            pl.store(exponent_tile, [row, 0], exponent)
    return exponent


@pl.jit
def row_sum_kernel(
    exponent: pl.Tensor[[ROWS, COLS], pl.FP32],
    denominator: pl.Out[pl.Tensor[[ROWS, 1], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        for row in pl.range(ROWS):
            exponent_tile = pl.load(exponent, [row, 0], [1, COLS])
            scratch = pl.create_tile(
                [1, COLS], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
            )
            denominator_tile = pl.row_sum(exponent_tile, scratch)
            pl.store(denominator_tile, [row, 0], denominator)
    return denominator


@pl.jit
def row_expand_div_kernel(
    exponent: pl.Tensor[[ROWS, COLS], pl.FP32],
    denominator: pl.Tensor[[ROWS, 1], pl.FP32],
    output: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        for row in pl.range(ROWS):
            exponent_tile = pl.load(exponent, [row, 0], [1, COLS])
            denominator_tile = pl.load(denominator, [row, 0], [1, 1])
            output_tile = pl.row_expand_div(exponent_tile, denominator_tile)
            pl.store(output_tile, [row, 0], output)
    return output
```

`_compile_stages()` 对五个函数分别调用 `compile_jit_kernel`：reduce tile 为 `[128]`，broadcast pointwise 为 `[1,128]`，dense exp 为 `[128]`。实际提交函数如下；`keys/STAGE_NAMES/record_function/launch_graph` 由原脚本提供，因此这段 Host 函数需在该脚本上下文中使用：

```python
def _launch_pipeline(
    keys: list[str],
    tensors: tuple[torch.Tensor, ...],
    raw_stream: int,
    *,
    annotate: bool,
) -> None:
    x, maximum, shifted, exponent, denominator, output = tensors
    arguments = (
        (x, maximum),
        (x, maximum, shifted),
        (shifted, exponent),
        (exponent, denominator),
        (exponent, denominator, output),
    )
    for stage_name, key, stage_arguments in zip(STAGE_NAMES, keys, arguments):
        if annotate:
            with record_function(stage_name):
                launch_graph(key, stage_arguments, raw_stream)
        else:
            launch_graph(key, stage_arguments, raw_stream)
```

**该实验已经支持“五阶段可以运行”，但没有支持“五个 NPU InCore + Orchestration 可以原样在 GPU 编成一个程序”。** 后者当时明确报错 `program must contain exactly one function`；单 scope 的那个 FP32 softmax 尝试也被模式检查拒绝。详见第 9 章。

静态形状桶、单图支持范围、是否拆成多个 executable，都是 E（PyPTO on GPU） 当前用户必须能感知的约束；不是 CUDA 本身无法做单 kernel softmax。

#### F（Triton-Ascend）：原生 kernel + Host launch 的完整短例

下面按本地教程 API 整理为完整尾轴计算与 launch 示例：[F5（Triton-Ascend）]。

```python
import torch
import triton
import triton.language as tl

@triton.jit
def softmax_triton_kernel(X, Y, M, N, SX, SY, BLOCK: tl.constexpr):
    col = tl.arange(0, BLOCK)
    for row in tl.range(tl.program_id(0), M, tl.num_programs(0)):
        z = tl.load(X + row * SX + col, col < N, other=-float("inf"))
        e = tl.exp(z - tl.max(z, axis=0))
        y = e / tl.sum(e, axis=0)
        tl.store(Y + row * SY + col, y, col < N)

def softmax_triton(x):
    assert x.ndim == 2 and x.shape[0] > 0 and x.shape[1] > 0
    m, n = x.shape
    y = torch.empty_like(x)
    programs = min(m, 32)  # Example policy, not a universal NPU core count.
    softmax_triton_kernel[(programs, 1, 1)](
        x, y, m, n, x.stride(0), y.stride(0),
        BLOCK=triton.next_power_of_2(n),
    )
    return y
```

`N` 是 kernel 参数，`BLOCK` 是 constexpr。同一个 `BLOCK` 桶内可以利用 mask 支持多个 N；是否另生缓存变体还取决于 JIT 的标量专门化、对齐及其他选项，不能只看函数签名保证只编一次。

对 `N=131075` 直接取 next_power_of_2 得到一个巨型 Tile，并非推荐的长行实现。应使用固定 chunk 的循环，或显式 split-reduce。Triton 的逻辑 program grid 也不等于物理 NPU block，后面会解释 AutoBlockify。

#### G（AutoFuse + Inductor）：已有真实 softmax 客户测试，后端名称应显式写出

[G10（AutoFuse + Inductor）] 的 `test_softmax` 在文件级选择 AscendC 后端。下例改用显式 options，去除清理调试目录的测试外壳：

```python
import torch
import torch_npu

@torch.compile(options={"npu_backend": "ascendc"})
def softmax_autofuse(x):
    return torch.softmax(x, dim=-1)

x = torch.randn(8, 64, device="npu")
y = softmax_autofuse(x)
torch.npu.synchronize()
torch.testing.assert_close(y.cpu(), torch.softmax(x.cpu(), dim=-1),
                           atol=1e-3, rtol=1e-3)
```

客户不写 TileType、Channel 或 Host TilingFunc；这些职责进入 Inductor/AutoFuse。这个源码测试证明存在这样的用户入口与测试意图，不能在未运行生成物的情况下保证一个物理 kernel。

#### F（Triton-Ascend） / G（AutoFuse + Inductor）：同一份完整客户程序，可以选择不同 Inductor 后端

此程序展示“用户不写 kernel、不写 tiling”的入口；运行需本地对应构建正确安装，不是仅把源码目录放在磁盘上即可。

```python
import torch
import torch_npu

def softmax_torch(x):
    z = x.float()
    e = torch.exp(z - torch.amax(z, dim=-1, keepdim=True))
    return (e / torch.sum(e, dim=-1, keepdim=True)).to(x.dtype)

def make_compiled_softmax(path, dynamic):
    # Current torch_npu dispatch: default -> Triton; ascendc -> AutoFuse ingress.
    backend_name = {"Triton-Ascend": "default", "AutoFuse": "ascendc"}[path]
    return torch.compile(
        softmax_torch,
        backend="inductor",
        fullgraph=True,
        dynamic=dynamic,
        options={"npu_backend": backend_name},
    )

def verify_softmax(path):
    fn = make_compiled_softmax(path, dynamic=True)
    for m, n in [(777, 300), (1000, 200), (8, 131075)]:
        x = torch.randn((m, n), device="npu", dtype=torch.float32)
        actual = fn(x)
        torch.testing.assert_close(actual, softmax_torch(x), rtol=1e-3, atol=1e-3)
```

`fullgraph=True` 约束图捕获，不承诺一个物理 kernel；`dynamic=True` 不承诺没有 guard、specialization、fallback 或编译失败。切换后端建议独立进程验证，避免全局注册和缓存干扰。

#### H（CATLASS DSL）：从真实 FA 的 softmax 子过程看用户责任

所选 `examples` 中未找到与第 2.2 节同 ABI 的独立、完整尾轴 softmax 用例。可以具体核对的是 `flash_attention_infer.py` 中已经写出的 softmax：Cube 产出 QK，Vector 从 UB 加载，显式建立 full/tail mask，做 MAX、减最大值、exp、ADD 归约、精度转换，并维护跨 KV tile 的最大值、分母和输出修正。它是 **FA 内的 online-softmax 实现证据**，不能单列为 `X[777,300]` 或超长行测试通过。[H-fa]

与 B/D 相比，H 同样把局部容量、buffer 槽、跨 Cube/Vector 通知交给作者；更具体的表达是 `tla.make_tensor` / `tile_view` / `allocate`，以及 `with tla.vec.func(mode="simd")` 内的寄存器 load/store、`ReductionOp.MAX/ADD`、`create_mask/update_mask`。数学上仍使用第 2.2/2.5 节的在线归约公式，语言变化不会免去跨分段状态合并。[H-api-layout] [H-api-allocate] [H-fa]

例如下列归约核心摘录发生在 `vec.func` 内；两个寄存器片段此前已经完成 load、缩放和尾部处理，mask 与 UB 状态的声明仍在完整源文件中。这不是一个可独立调用的 softmax：[H-fa]

```python
tmp_reg_p1 = tla.max(ub_s_reg0_p1, ub_s_reg1_p1, mask=pregFull)
max_reg_p1 = tmp_reg_p1.reduce(tla.ReductionOp.MAX, mask=pregFull)
```

若要补独立 softmax，下一步应抽取已有 Vector 原语，单独给出输入输出、分段遍历和 golden；此次保留现有 kernel 的验证边界，不用临时新实现填成仓库原生用例。

#### I（CuTe DSL）：同一份 softmax 教程给出了八种工作分解

`experimental/primitives/tutorial/06_softmax.py` 包含逐线程串行、一行一 CTA、warp shuffle、warp + shared memory、online 等八种写法，以及 Host launcher。它使第 2.2 节关于“长行不必整体驻留片上”的理论有了另一组完整源文件，但本次未运行这些 CUDA kernel。[I-softmax]

| 源码中的策略 | 用户要安排什么 | 对应的分析维度 |
| --- | --- | --- |
| 一线程处理一行 | 线程/行索引，列遍历，max/sum/normalize | 容量可以小，行数不足时并行度可能不足 |
| 一 CTA 处理一行 | 每线程列子集、shared partial、CTA barrier | 行内并行与跨行并行分开；单 CTA 归约不需要跨 CTA 全局 barrier |
| online + warp + shared（kernel 8） | 每线程维护 `(max,sum)`，shuffle 合并，再经 shared 合并各 warp，最后再次遍历输出 | 在线合并节约暂存需求；不同 warp 的 sum 必须按共同最大值修正，不能直接相加 |

一个具体反例是：教程前面的某些实现把 exp 中间值写入输出 GM 后再读取；kernel 8 主要保留归约状态，最后重读输入产生结果。因此 **“都在一个 `@cute.kernel` 里”不等于具有相同 GM 流量**。此外，不同实现把 N/C 作为运行时元数据或 `Constexpr` 的方式不同，不能对八个变体统一宣称动态 shape。教程开头的限制性注释也只作为该教程上下文，完整语言能力仍以当前 DSL 实现为准。[I-softmax] [I-dsl]

kernel 8 具体把 C 作为 `Constexpr` 并使用 `range_constexpr` 遍历列。它说明有限归约状态的算法能够表达，不能直接证明超长行时编译成本和代码体积也理想；分段循环是否展开，是第 7 章动态与特化分析需要继续控制的另一项成本。[I-softmax]

对应的 warp 内合并源码片段如下，`maxval/sumval` 是每线程此前累积的状态；跨 warp 的 shared-memory 归并与最后写回仍见完整函数。对比 H 的寄存器向量 API，这里直接暴露 warp shuffle 和每线程标量状态。[I-softmax]

```python
for offset in [16, 8, 4, 2, 1]:
    other_max = cute.arch.shuffle_sync_down(maxval, offset)
    other_sum = cute.arch.shuffle_sync_down(sumval, offset)
    if other_max > maxval:
        scale = cute.math.exp(maxval - other_max, fastmath=True)
        sumval = other_sum + sumval * scale
        maxval = other_max
    else:
        scale = cute.math.exp(other_max - maxval, fastmath=True)
        sumval = sumval + other_sum * scale
```

### 2.4 完整 paged decode 语义参考：actual_seq_len 与 shape 分开

以下采用 `K/V[P,Hkv,S,D]`，仅做只读 cache 的 decode attention；不包含 QK norm、RoPE、KV append，避免把不同算子边界的性能混比。

```python
import torch

def paged_decode_reference(q, k, v, block_table, actual_seq_len):
    # q: [B,Hq,D]; k/v: [P,Hkv,S,D]; block_table: [B,T]; lengths: [B].
    b, hq, d = q.shape
    p, hkv, page, kd = k.shape
    assert v.shape == k.shape and kd == d and hq % hkv == 0
    assert block_table.shape[0] == b and actual_seq_len.shape == (b,)
    group = hq // hkv
    out = torch.empty_like(q)

    for batch in range(b):
        # Host extraction belongs to this golden only, not a desired device ABI.
        length = int(actual_seq_len[batch].item())
        assert 1 <= length <= block_table.shape[1] * page
        pos = torch.arange(length, device=q.device)
        physical = block_table[batch, pos // page].long()
        assert bool(((physical >= 0) & (physical < p)).all())
        offset = pos % page
        for head in range(hq):
            kv_head = head // group
            keys = k[physical, kv_head, offset, :].float()
            vals = v[physical, kv_head, offset, :].float()
            score = (keys @ q[batch, head].float()) * (d ** -0.5)
            prob = torch.softmax(score, dim=0)
            out[batch, head] = (prob @ vals).to(q.dtype)
    return out

def make_decode_case(device="cpu"):
    torch.manual_seed(0)
    b, hq, hkv, d, page, max_pages, pool = 3, 40, 8, 128, 128, 5, 19
    q = torch.randn((b, hq, d), device=device, dtype=torch.bfloat16)
    k = torch.randn((pool, hkv, page, d), device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    table = torch.randint(pool, (b, max_pages), device=device, dtype=torch.int32)
    lengths = torch.tensor([1, 129, 513], device=device, dtype=torch.int32)
    return q, k, v, table, lengths
```

`lengths=[1,129,513]` 改成 `[127,128,512]` 不改变任何 tensor shape；改变 `B`、cache pool `P` 或 table 宽度 `T` 才是元数据变化。page 大小 `S`、head dimension `D` 或 GQA 比例变化，可能进一步改变物理 tile、layout、指令及编译变体。

下面是供 F（Triton-Ascend）/G（AutoFuse + Inductor） 的 PyTorch 图入口使用的另一份完整表达。它不用 `.item()`，将长度作为设备 mask；但显式展开 padded cache，会产生很大的逻辑中间值，不能当作高性能 paged attention：

```python
def paged_decode_padded(q, k, v, block_table, actual_seq_len):
    b, hq, d = q.shape
    hkv, page = k.shape[1], k.shape[2]
    max_pages = block_table.shape[1]
    capacity = max_pages * page
    group = hq // hkv

    # All table entries, including padded ones, must point to valid pages.
    ids = block_table.reshape(-1).long()
    keys = k.index_select(0, ids).reshape(b, max_pages, hkv, page, d)
    vals = v.index_select(0, ids).reshape(b, max_pages, hkv, page, d)
    keys = keys.permute(0, 2, 1, 3, 4).reshape(b, hkv, capacity, d)
    vals = vals.permute(0, 2, 1, 3, 4).reshape(b, hkv, capacity, d)
    keys = keys.repeat_interleave(group, dim=1).float()
    vals = vals.repeat_interleave(group, dim=1).float()

    score = torch.matmul(q.float().unsqueeze(-2), keys.transpose(-2, -1))
    score = score * (d ** -0.5)
    valid = torch.arange(capacity, device=q.device)[None, :] < actual_seq_len[:, None]
    score = score.masked_fill(~valid[:, None, None, :], -torch.inf)
    result = torch.matmul(torch.softmax(score, dim=-1), vals).squeeze(-2)
    return result.to(q.dtype)

def make_compiled_decode(path, dynamic=True):
    return torch.compile(
        paged_decode_padded,
        backend="inductor",
        fullgraph=True,
        dynamic=dynamic,
        options={"npu_backend": {"Triton-Ascend": "default", "AutoFuse": "ascendc"}[path]},
    )
```

该表达可以完整定义输入和输出，却不能证明 gather → QK → softmax → PV 会被 G（AutoFuse + Inductor） 或 F（Triton-Ascend） 的 Inductor 路径融合。需要核对生成物；遇到外部 attention 算子时，必须把归属记为外部算子，而不是这两个后端生成的 kernel。

### 2.5 高性能 PA 的共有算法与实际分歧

对一个请求及其 GQA head 组，逐 page / stack 计算，维护 `m,l,o`：

```text
page_id = block_table[batch, logical_page]
valid = min(page_size, actual_seq_len[batch] - logical_page*page_size)
S_i = Q @ K[page_id]^T * scale
S_i[invalid] = -∞
m_new = max(m, row_max(S_i))
alpha = exp(m - m_new)
P_i = exp(S_i - m_new)
l_new = alpha*l + row_sum(P_i)
o_new = alpha*o + P_i @ V[page_id]
最后 O = o/l
```

第一次迭代可单独初始化，避免对空块计算 `-∞ - -∞`。后续 page 计算与流水尾部同样需要有效长度、首尾标志和依赖。

相同数学算法可能有四种完全不同的表达位置：

| 决策 | Tensor / task 路线 | 显式核内路线 | Triton 路线 | AutoFuse 客户路线 | H（CATLASS DSL，现有连续 FA） | I（CuTe DSL，分页 MLA） |
| --- | --- | --- | --- | --- | --- | --- |
| 遍历多少 page | A（PyPTO2-tensor版） 的程序控制流，C（PyPTO3（Simpler）） 的 Orchestration 或 SPMD 内循环 | B（PyPTO2-block版）/D（CANNBot DSL） kernel 从 GM 读取长度后循环 | `tl.load(seq_lens)` → runtime loop | padded mask 或受支持图算子；能否降为紧凑动态循环取决于 lowering | 当前按连续 KV 块循环；没有 page table，传入的实际长度参数未在设备体读取 | page-table/TMA 路径定位 KV 页；变量长度/split 还受具体入口约束 |
| QK / PV | Tensor matmul 或独立 InCore | 显式 Cube / memory stage / pipeline | `tl.dot`，后端安排 Cube / Vector 交互 | `torch.matmul` / 模板 / 外部库，须查生成归属 | 作者安排 MMAD、L1/L0 与 UB 的 CV 数据路径 | 作者选 MMA atom、tile、warp 角色与 TMA/pipeline |
| m/l/o 放哪里 | 编译器规划的 tensor 或 task 间 GM | UB / registers / GM transfer，由用户与 passes 分工 | 编译器 bufferization / workspace / pipeline | AutoFuse schedule、buffer 分配和 workspace | 显式 UB 状态、寄存器运算与跨 CV 交接 | 寄存器/SMEM/TMEM 及 split 输出策略，依具体实现 |
| 请求分配给核 | 任务派发，或用户选择 SPMD 映射 | core-stride、work_ranges 或自定义协议 | program grid + 编译器映射 | 自动 tiling 决定 blockDim 和 work split | 逻辑 task 编号与 block-stride；不是 ready-task 队列 | tile scheduler、CTA/cluster 与 split 分工；不是模型任务 runtime |
| 性能相关长度策略 | 可动态生成更多 task；不保证自动负载均衡最优 | 可传 Host 计算的 work_ranges，也可设备侧计算 | Host 上界 grid + kernel 读取真实长度 | Host generated tiling 通常处理元数据；tensor 内容不能自动当成 Host shape | Host 改 Q/KV/TOTAL_TASKS 全局常量后编译；语言的动态 layout 能力另看 MMAD | Host 策略、动态序列元数据与 kernel 专门化分别处理；MLA latent/rope ABI 不能直接替标准 GQA |

H/I 两列分别依据 [H-fa]、[I-mla] [I-gqa]。H 列刻意记录现有连续 FA 尚缺哪些分页责任；I 列以已找到的分页 MLA 为对象。它们用于对照算法与资源分工，不把连续 KV、MLA latent/rope 和标准 GQA 的数据契约合并成一个已通过的 PA 用例。

### 2.6 每条路径的完整 PA 证据与动态写法

#### A（PyPTO2-tensor版）已有动态长度 + paged cache 的完整程序

[A13（PyPTO2-tensor版）] 的 `ctrl_perf_kernel` 包含残差/RMS 预处理、投影、cache 更新和 attention。下列摘录完整保留查页、QK、online max/sum、PV 与归一化循环，仅去除外围缩进；`q_2d/k_cache_2d/v_cache_2d`、常量及参数依赖函数前段，不是独立入口。

```python
for b_idx in pypto.loop(b, name="LOOP_B", idx_name="b_idx"):
    cur_seq = act_seq[b_idx]
    s2_loop = (cur_seq + _S2_TILE - 1) // _S2_TILE
    for g_idx in pypto.loop(g_loop, name="LOOP_G", idx_name="g_idx"):
        oi_update = pypto.tensor([_G_TILE, _D], pypto.DT_FP32, "oi_update")
        sum_update = pypto.tensor([_G_TILE, 1], pypto.DT_FP32, "sum_update")
        max_update = pypto.tensor([_G_TILE, 1], pypto.DT_FP32, "max_update")
        for s2_idx in pypto.loop(s2_loop, name="LOOP_S2", idx_name="s2_idx", unroll_list=unroll_list):
            idx = s2_idx * _BLOCK_NUM
            n1g_ofs = g_idx * _G_TILE
            actual_s2 = (cur_seq - s2_idx * _S2_TILE).min(_S2_TILE)
            pypto.set_vec_tile_shapes(_G_TILE, _D)
            qi = pypto.view(q_2d, [_G_TILE, _D], [b_idx * n1 + n1g_ofs, 0])
            kj_assemble = pypto.tensor([_S2_TILE, _D], pypto.DT_FP16, "kj_assemble")
            vj_assemble = pypto.tensor([_S2_TILE, _D], pypto.DT_FP16, "vj_assemble")
            for i in range(_BLOCK_NUM):
                block_idx = block_table[b_idx, idx + i].max(0)
                kj_assemble[i * _BLOCK:(i + 1) * _BLOCK, 0:] = pypto.view(
                    k_cache_2d, [_BLOCK, _D], [block_idx * _BLOCK, 0]
                )
                vj_assemble[i * _BLOCK:(i + 1) * _BLOCK, 0:] = pypto.view(
                    v_cache_2d, [_BLOCK, _D], [block_idx * _BLOCK, 0]
                )
            kj_assemble = pypto.view(kj_assemble, [_S2_TILE, _D], [0, 0],
                                     valid_shape=[actual_s2, _D])
            vj_assemble = pypto.view(vj_assemble, [_S2_TILE, _D], [0, 0],
                                     valid_shape=[actual_s2, _D])
            pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
            sij = pypto.view(
                pypto.matmul(qi, kj_assemble, pypto.DT_FP32, a_trans=False, b_trans=True),
                [_G_TILE, _S2_TILE], [0, 0], valid_shape=[_G_TILE, actual_s2],
            )
            pypto.set_vec_tile_shapes(_G_TILE, _S2_TILE)
            if pypto.is_loop_begin(s2_idx):
                pypto.set_pass_options(sg_set_scope=3)
                sij_scale = pypto.mul(sij, softmax_scale)
                tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)
                tilda_pij = pypto.exp(pypto.sub(sij_scale, tilda_mij))
                sum_update[:] = pypto.sum(tilda_pij, dim=-1, keepdim=True)
                max_update[:] = tilda_mij
                pypto.set_pass_options(sg_set_scope=-1)
                pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
                oi_update[:] = pypto.matmul(pypto.cast(tilda_pij, pypto.DT_FP16), vj_assemble, pypto.DT_FP32)
            else:
                pypto.set_pass_options(sg_set_scope=1)
                sij_scale = pypto.mul(sij, softmax_scale)
                tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)
                max_new = pypto.maximum(max_update, tilda_mij)
                tilda_pij = pypto.exp(pypto.sub(sij_scale, max_new))
                sum_local = pypto.sum(tilda_pij, dim=-1, keepdim=True)
                pypto.set_pass_options(sg_set_scope=-1)
                pypto.set_pass_options(sg_set_scope=2)
                update_mul = pypto.exp(pypto.sub(max_update, max_new))
                max_update[:] = max_new
                sum_update[:] = sum_update * update_mul + sum_local
                pypto.set_pass_options(sg_set_scope=-1)
                pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
                oi_tmp = pypto.matmul(pypto.cast(tilda_pij, pypto.DT_FP16), vj_assemble, pypto.DT_FP32)
                pypto.set_vec_tile_shapes(_G_TILE, _D)
                oi_update[:] = oi_update * update_mul + oi_tmp
        pypto.set_vec_tile_shapes(_G_TILE, _D)
        o_norm = pypto.div(oi_update, sum_update)
        pypto.assemble(pypto.cast(o_norm, pypto.DT_FP16), [b_idx * n1 + g_idx * _G_TILE, 0], atten_out)
```

`block_table` 两轴动态，cache 的 page 数轴动态，head / page 几何有静态约束。客户使用 Tensor API 和 tile 设置，而不是另注册通用 TilingFunc。实际 task 数取决于 graph / loop / tiling 编译，不能把源码段落数当 task 数。

该例不是完整 Transformer 层性能证明；它是 A（PyPTO2-tensor版） 能组织多阶段程序及数据相关 page 循环的实现证据。

**A5 实测补充：程序组织能力已有运行对象，但原版 KV 更新语义有数值失败。** 在 FP16、B=4、Hq/Hkv=8/1、D=128、page=128、L=[127,128,129,513] 下，新增包含前处理和 KV append 的完整 golden 后，原版板端最大绝对误差为 `0.7950679659843445`；板端 PA 与更新前 cache 的参考结果接近。前端解释器对更新后 cache 的 golden 通过。在 `LOOP_PRE` 后、PA 前重新建立 K/V 的二维 view，本地修正版通过，误差为 `0.0002454519271850586`。两行补丁及原版/修正版日志见第 12.1.1 节和[实测快照][RUN-results]；这为“view/别名、任务依赖及数据可见性须联合验证”的分析提供具体反例，尚未定位具体编译 Pass 或 runtime 根因。原始上游代码未被本地驱动覆盖。

#### B（PyPTO2-block版）：完整 paged prefill 实现，以及同一实现的 A5 decode 补充验证

以下分析 prefill 的完整算法、Host 分工与动态 tiling。A5 验证包括原文件的页大小测试（128/256/512），以及通过同一 `flex_attention_bf16` / `_run_case` 入口执行的 Q长度=1、L=[127,128,129,513] BF16 decode，均通过。原文件以 prefill 为主组织代码；同一实现已有 decode 正确性结果，但不能代替 decode 专项性能和负载均衡测量。[页大小测试][B-pa-test] [RUN-results]

[B15（PyPTO2-block版）] 的 `test_flex_attention_prefill.py` 含完整 kernel、Host 工作分配、page cache 构造和 golden。实际 ABI：

```text
K/V：[num_blocks, block_size, n_head_kv, dim]，NHD page layout
cu_seqlens_q：[B+1]，第 0 项必须为 0
seqlens_kvcache：[B]，每请求绝对长度，不是前缀和
block_ids：[B,max_blocks]
work_ranges：[num_cores,4]
```

kernel 用 `pl.getval(seqlens_kvcache,b)` 读取长度、`pl.getval(block_ids,...)` 查询页；Host `build_work_ranges(...)` 选择 core 工作分配，再 `kernel[None,actual_num_cores](...)`。这就是**Python 化 tiling / work partition 的具体例子**：普通 Python 函数计算一个 tensor，而不一定是特殊注册的 TilingFunc。

下面是 Host 调用链及 Cube section 的实际连续循环。第二段省略外围 TileGroup 定义与后面的 Vector section，依赖原文件的 `compute_qk/compute_pv/scalar_bound`，**不是独立 kernel**。

```python
work_ranges = build_work_ranges(
    seq_q_list, seq_kv_list, spans_per_batch, n, num_cores)
cu_seqlens_q = make_cu_seqlens_q(seq_q_list, device)
seqlens_kvcache = torch.tensor(seq_kv_list, device=device, dtype=torch.int32)
flex_attention_bf16[None, min(num_cores, total_work)](
    q.to(device), kcache.to(device), vcache.to(device), o.to(device),
    cu_seqlens_q, block_ids.to(device), seqlens_kvcache,
    spans_dev.to(device), zero_mask.to(device), work_ranges.to(device))
```

```python
work_start = pl.getval(work_ranges, core_id * 4)
n_items = pl.getval(work_ranges, core_id * 4 + 1)
split_mode = pl.getval(work_ranges, core_id * 4 + 2)
two_c = pl.getval(work_ranges, core_id * 4 + 3)
task_id = 0
b_idx = 0
actual_s1 = 0
actual_s2 = 0
s1o_acc = 0
ctx_arr = pl.struct_array(4, "CubeCtx", n_idx=0, kv_n_idx=0, qi=0, ki=0,
                          task_id=0, kv_page=0, kv_slot=0, s1_size=0, s2_size=0, loop_count=0)
s1o_size = 0 # tmp-val
for idx in pl.range(batch):
    actual_s1 = pl.getval(cu_seqlens_q, idx + 1) - pl.getval(cu_seqlens_q, idx)
    actual_s2 = pl.getval(seqlens_kvcache, idx)
    s1o_size = s1o_size + (actual_s1 + TS - 1) // TS * n_dim
    if (work_start >= s1o_size):
        s1o_acc = s1o_size
        b_idx = b_idx + 1
        continue
    break
for i in pl.range(0, n_items):
    # Two core-split modes. Contiguous (mode 0) hands each core one
    # [start, start+n_items) run; strided-mirrored (mode 1) hands it
    # indices t*2C+c and t*2C+2C-1-c. Mode 1 is what keeps L2 warm once
    # B*N exceeds the core count -- see build_work_ranges.
    work_id = work_start + i
    if split_mode == 1:
        work_id = (i // 2) * two_c + core_id
        if i % 2 == 1:
            work_id = (i // 2) * two_c + two_c - 1 - core_id
    for _ in pl.range(b_idx, batch):
        actual_s1 = pl.getval(cu_seqlens_q, b_idx + 1) - pl.getval(cu_seqlens_q, b_idx)
        actual_s2 = pl.getval(seqlens_kvcache, b_idx)
        s1o_size = s1o_acc + (actual_s1 + TS - 1) // TS * n_dim
        if (work_id >= s1o_size):
            s1o_acc = s1o_size
            b_idx = b_idx + 1
            continue
        break
    cur_b_s1o = (actual_s1 + TS - 1) // TS
    s1o_size = work_id - s1o_acc
    n_idx = s1o_size // cur_b_s1o
    s1_idx = s1o_size % cur_b_s1o
    s1_size = pl.min(TS, actual_s1 - s1_idx * TS)

    # cu_seqlens_q[b] is already this batch's Q offset -- no accumulator.
    sq_off = pl.getval(cu_seqlens_q, b_idx) + s1_idx * TS
    cur_q_slot = q_l1_db.next()
    # Right-down alignment: this Q tile's rows occupy absolute KV
    # positions [q_abs0, q_abs0 + s1_size). bound is monotone in the row,
    # so the tile's last row gives the furthest KV column anyone needs.
    q_abs0 = actual_s2 - actual_s1 + s1_idx * TS
    kv_end = pl.min(scalar_bound(mm_prefix_range, b_idx, span_num, q_abs0 + s1_size - 1),
                    actual_s2)
    kv_loop = (kv_end + TKV - 1) // TKV
    drain = 0
    if i == n_items - 1:
        drain = 2
    for ki in pl.range(0, kv_loop + drain):
        if ki < kv_loop:
            # A prefix mask leaves no gaps: blocks 0..kv_loop-1 are all
            # computed, so there is no per-block skip bitmap to consult.
            ctx_curr = ctx_arr[task_id % 4]
            ctx_curr.task_id = task_id
            ctx_curr.n_idx = n_idx
            ctx_curr.kv_n_idx = n_idx // group
            ctx_curr.ki = ki
            # Resolve page and in-page offset here, in the producer:
            # compute_pv runs two pipeline steps later and must see the
            # values for its own ki.
            ctx_curr.kv_page = pl.getval(block_ids,
                                         b_idx * max_blocks + ki // page_tiles)
            ctx_curr.kv_slot = (ki % page_tiles) * TKV
            ctx_curr.s1_size = s1_size
            ctx_curr.s2_size = pl.min(TKV, kv_end - ki * TKV)
            ctx_curr.loop_count = ki  # ki == 0 -> load Q

            # ========== compute_qk (current step) ==========
            compute_qk(ctx_curr, ki, sq_off, q, k, cur_q_slot, k_l1_db,
                       left_db, right_db, acc_db, qk_vec_db, task_id,
                       left_db2, right_db2, acc_db2)

        # ========== compute_pv (delayed 1 step: uses ctx from task_id-1) ==========
        if task_id > 1:
            ctx_pre2 = ctx_arr[(task_id + 2) % 4]
            compute_pv(ctx_pre2, v_l1_db, p_mat_db, left_db2, right_db2, acc_db2, pv_vec_db, v,
                       left_db, right_db, acc_db)
```

这里 `task_id` 是核内流水计数，`work_ranges` 是算子参数 tensor；二者都不是tensor版或 Simpler 的任务队列。这是“Python Host 分工策略”不等于“AICPU scheduler”的具体代码。

但是，这个文件定位是 prefill；其 `TS/TKV/TD`、mask、Q tile 利用率不应当自动算作 `q_len=1` decode 的优化实现。另有 [B16（PyPTO2-block版）] 的动态 TND attention，可证明设备读取 actual_seq 的方式，不能替代 decode 专项验证。

对于客户，改变 actual_seq 内容可能只改变 kernel 循环；若继续使用 Host cost-balanced 的 `work_ranges`，还要保证该元数据与新长度一致。改变 `B/P/T` 可由动态轴及 runtime descriptor 承接；改变 `D/page` 组织可能要求新变体。**是否“需要新 tiling”应区分：重新调用既有策略、修改策略、重新编译 kernel。**

#### C（PyPTO3（Simpler））：pypto-lib 的 native decode 是本文最具体的 SPMD-in-task 例子

**验证对象要分开：** 本次 A5 已通过的是 `pypto/examples/models/04_paged_attention.py` 的通用 PA，PTOAS 测试覆盖 L=8192/8100（2例），使用 `scale=1.0`。以下 Qwen3-14B native PA、三阶段融合、TaskId 及 GM ring 的分析完整保留；其测试 CLI 仍限定 `a2a3/a2a3sim`，本次未在 A5 运行该融合实现。通用 PA 的通过给出了 PyPTO3→PTOAS→A5 runtime 的具体运行对象，不能替代本节融合 SPMD kernel 的验证。[通用实现][C-pa] [通用测试][C-pa-test] [融合测试平台][C-qwen-platform]

完整实现：[C16（PyPTO3（Simpler））]；完整动态测试驱动：[C17（PyPTO3（Simpler））]；讲解文档：[C18（PyPTO3（Simpler））]。该例不是“五个各自独立提交的 InCore”：

```text
@pl.jit entry
  → 按 active_batch 分配 q_tnd_flat
  → 分配 score / probability / PV transfer 和同步 workspace
  → @pl.jit.inline(auto_scope=False) paged_attention_pypto_swpipe
      → pl.spmd(24, sync_start=True, deps=[...])
      → Phase 0：Q/K norm + RoPE + KV append
      → publish / fence / mixed-core synchronization
      → for task in range(core, active_batch*8,24):
           batch = task//8
           kv_head = task%8
           seq_len = pl.read(seq_lens,[batch])
           page_count = ceildiv(seq_len,128)
           stack_count = ceildiv(seq_len,512)
           AIC：QK / PV
           AIV：online softmax / output update
      → 返回一个 attn TaskId，供外层依赖
```

首先直接给出 [C17（PyPTO3（Simpler））] 的**完整编译入口**。引用同目录 helper 和常量；应在原模型目录使用，主体由 `paged_attention_pypto_swpipe` 提供：

```python
import pypto.language as pl
from paged_attention_pypto import (
    NUM_HEADS, HEAD_DIM, TRANSFER_ROWS, STACK_TOKENS,
    FFTS_WORKSPACE_ELEMENTS, paged_attention_pypto_swpipe,
)

@pl.jit
def paged_attention_pypto_dynamic(
    key_cache: pl.InOut[pl.Tensor],
    value_cache: pl.InOut[pl.Tensor],
    block_table: pl.Tensor,
    seq_lens: pl.Tensor,
    inv_rms_states: pl.Tensor,
    slot_mapping: pl.Tensor,
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    q_proj: pl.Tensor,
    k_proj: pl.Tensor,
    v_proj: pl.Tensor,
    q_norm_w: pl.Tensor,
    k_norm_w: pl.Tensor,
    layer_cache_base_token_rows: pl.Scalar[pl.INDEX],
    out: pl.Out[pl.Tensor],
) -> pl.Tensor:
    """Allocate fused Phase-0/PA scratch and launch the PyPTO helper."""
    active_batch = pl.tensor.dim(seq_lens, 0)
    q_tnd_flat = pl.create_tensor([active_batch * NUM_HEADS, HEAD_DIM], dtype=pl.BF16)
    score_transfer = pl.create_tensor([TRANSFER_ROWS, STACK_TOKENS], dtype=pl.FP32)
    probability_transfer = pl.create_tensor([TRANSFER_ROWS, STACK_TOKENS], dtype=pl.BF16)
    pv_transfer = pl.create_tensor([TRANSFER_ROWS, HEAD_DIM], dtype=pl.FP32)
    ffts_workspace = pl.create_tensor([FFTS_WORKSPACE_ELEMENTS], dtype=pl.INT64)
    q_proj_tid = pl.system.task_dummy(deps=[])
    k_proj_tid = pl.system.task_dummy(deps=[])
    v_proj_tid = pl.system.task_dummy(deps=[])
    rms_tid = pl.system.task_dummy(deps=[])
    attn_out_seed_tid = pl.system.task_dummy(deps=[])
    mlp_out_seed_tid = pl.system.task_dummy(deps=[])
    scratch_ready_tid = pl.system.task_dummy(deps=[])
    paged_attention_pypto_swpipe(
        q_tnd_flat,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        inv_rms_states,
        slot_mapping,
        rope_cos,
        rope_sin,
        q_proj,
        k_proj,
        v_proj,
        q_norm_w,
        k_norm_w,
        layer_cache_base_token_rows,
        out,
        score_transfer,
        probability_transfer,
        pv_transfer,
        ffts_workspace,
        q_proj_tid,
        k_proj_tid,
        v_proj_tid,
        rms_tid,
        attn_out_seed_tid,
        mlp_out_seed_tid,
        scratch_ready_tid,
    )
    return out
```

helper 中纳入 Simpler 依赖图的实际 scope 是下面这一段；这里只展示进入 scope 的代码，并未省略成 `deps=[...]` 占位：

```python
with pl.spmd(
    ATTN_SPMD_BLOCKS,
    name_hint="attn_swpipe_spmd",
    sync_start=True,
    allow_early_resolve=True,
    deps=[
        q_proj_tid,
        k_proj_tid,
        v_proj_tid,
        rms_tid,
        attn_out_seed_tid,
        mlp_out_seed_tid,
        scratch_ready_tid,
    ],
) as attn_tid:
    core = pl.tile.get_block_idx()
    pl.system.set_ffts(ffts_workspace)
```

再看 helper 内**同一 SPMD task 的实际 page/QK 段**。下面止于完整 page 分支；尾 page、PV 和 AIV online softmax 在原文件中，不将它伪装成另一个完整函数：

```python
# SYNCALL is arrival-only. Publish the Phase-0 GM writes before the
# barrier, then invalidate each consumer's GM cache before PA reads
# Q/K/V.
pl.system.cacheinvalid()
pl.system.fence()
pl.system.syncall(core_type=pl.KernelType.MIX)
pl.system.cacheinvalid()

for task in pl.range(core, num_tasks, ATTN_SPMD_BLOCKS):
    batch = task // NUM_KV_HEADS
    kv_head = task % NUM_KV_HEADS
    seq_len = pl.read(seq_lens, [batch])
    page_count = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    stack_count = (seq_len + STACK_TOKENS - 1) // STACK_TOKENS
    col = kv_head * HEAD_DIM
    transfer_base = core * TRANSFER_SLOTS * ROW_TILE
    qp_row = batch * NUM_HEADS + kv_head * GROUP
    q_tile = pl.load(
        q2d,
        [qp_row, 0],
        [ROW_TILE, HEAD_DIM],
        valid_shape=[GROUP, HEAD_DIM],
        target_memory=pl.MemorySpace.Mat,
    )

    # Keep 128-token cache pages inside each 512-token softmax/update stack.
    for tick in pl.range(stack_count + PRE_LAUNCH):
        if tick < stack_count:
            produce_stack = tick
            produce_page = produce_stack * STACK_PAGES
            produce_row = transfer_base + (produce_stack % TRANSFER_SLOTS) * ROW_TILE
            if produce_page + STACK_PAGES <= page_count:
                for qk_page_offset in pl.pipeline(STACK_PAGES, stage=2):
                    qk_ph = pl.cast(
                        pl.tensor.read(block_table_2d, [batch, produce_page + qk_page_offset]),
                        pl.INDEX,
                    )
                    qk_k_page = pl.load(
                        key_cache_bsnd,
                        [cache_base + qk_ph * BLOCK_SIZE, col],
                        [BLOCK_SIZE, HEAD_DIM],
                        target_memory=pl.MemorySpace.Mat,
                    )
                    qk_score_page = pl.matmul(
                        q_tile,
                        pl.tile.transpose_view(qk_k_page),
                        out_dtype=pl.FP32,
                    )
                    pl.store(
                        qk_score_page,
                        [produce_row, qk_page_offset * BLOCK_SIZE],
                        score_transfer,
                    )
            else:
                qk_tail_ph0 = pl.cast(
                    pl.tensor.read(block_table_2d, [batch, produce_page]),
                    pl.INDEX,
                )
```

同一 helper 的 AIV 侧也有明确代码，而不是抽象的“由后端处理 softmax”。以下是该流水分支的连续摘录，依赖原函数中的循环状态和 event 定义：

```python
pl.system.sync_wait(
    QK_READY_EVENT,
    pipe=pl.PipeType.MTE2,
    core_type=pl.KernelType.AIV,
)
score_aiv = pl.load(
    score_transfer,
    [produce_row + lane_row, 0],
    [AIV_ROW_TILE, STACK_TOKENS],
    valid_shape=[lane_rows, STACK_TOKENS],
    target_memory=pl.MemorySpace.Vec,
)
score_scaled = pl.tile.muls(score_aiv, SCALE)
valid_cols = pl.min(
    STACK_TOKENS,
    seq_len - produce_stack * STACK_TOKENS,
)
score_valid = pl.set_validshape(score_scaled, lane_rows, valid_cols)
score_filled = pl.fillpad(score_valid, pad_value=pl.PadValue.min)
score_masked = pl.set_validshape(
    score_filled,
    lane_rows,
    STACK_TOKENS,
)
local_m = pl.row_max(score_masked, tmp)
next_m = pl.maximum(local_m, m_iter)
rescale = pl.exp(pl.sub(m_iter, next_m))
probability_aiv = pl.exp(pl.row_expand_sub(score_masked, next_m))
probability_reduce = pl.tile.move(
    probability_aiv,
    target_memory=pl.MemorySpace.Vec,
    slayout=pl.TileLayout.none_box,
)
next_l = pl.add(
    pl.mul(rescale, l_iter),
    pl.row_sum(probability_reduce, tmp),
)
probability_bf16 = pl.cast(
    probability_aiv,
    target_type=pl.BF16,
    mode="rint",
)
probability_valid = pl.set_validshape(
    probability_bf16,
    lane_rows,
    STACK_TOKENS,
)
pl.store(
    probability_valid,
    [produce_row + lane_row, 0],
    probability_transfer,
)
pl.system.sync_set(
    SOFTMAX_READY_EVENT,
    pipe=pl.PipeType.MTE3,
    ffts_mode=2,
    core_type=pl.KernelType.AIV,
)
m_after, l_after, rescale_after = pl.yield_(next_m, next_l, rescale)
```

读法：先等待 QK 发布，再从 GM 读 score；尾列用有效窗口和 `fillpad(min)` 屏蔽；更新 online `m/l`；将 BF16 probability 写入 GM，并向 AIC 发出 ready event。可直接看到计算融合与 GM 通信同时存在。

这里循环变量 `task` 是用户定义的“一个 batch + 一个 KV head”工作项，**不等于每次循环都往 Simpler 提交一个新 task**。`pl.spmd` 外壳本身作为多 block task 纳入外层任务依赖；内部的 24 核按 core-stride 做工作。

物理 tile 和 GM transfer：

| 项 | 当前例子的安排 |
| --- | --- |
| Q heads / KV heads | 40 / 8，每个 KV head 对应 5 个 Q head |
| Q tile | 物理 `[16,128]`，有效 `[5,128]` |
| page / stack | 128 token/page；4 page/stack，即 512 token |
| SPMD block | `ATTN_SPMD_BLOCKS=24`，属于此算子策略而非所有芯片统一核数 |
| transfer ring | 3 slots/core；`24×3×16=1152` 行 |
| score transfer | `[1152,512]` FP32，GM |
| probability transfer | `[1152,512]` BF16，GM |
| PV transfer | `[1152,128]` FP32，GM |
| 同步 workspace | 专门的 GM 控制状态，不是数值 tensor |

因此，它可以把多个阶段放进同一个 SPMD task，却仍保留跨 AIC/AIV 的 GM 中间缓冲。**task 融合 ≠ 自动消除 GM。**

动态测试驱动从 `seq_lens.dim(0)` 获取 active batch，支持 page 乱序、共享前缀、容量与物理页数量变化的用例；其 `_compile_signature` 仍包含 batch/capacity/cache_layers/physical_pages/table_width 等字段，说明测试会按这些几何分组。不能把“动态算法”进一步写成“所有不同 shape 已证明共用一个编译产物”。该驱动限定 A2/A3，A5 支持需单独验证。

另一个重要反例在 [C19（PyPTO3（Simpler））]：保留的 CCE attention 先把 `paged_attention_tiling_cce` 声明为 `core_type="aiv"` 的 extern，用 `pl.spmd(1)` 生成 metadata，然后以 `deps=[tiling_tid]` 提交 attention。**tiling 在一个 AIV task 上执行，不是在 Host，也不是 AICPU tiler。**

#### D（CANNBot DSL）：显式流水与 paged attention 实现

[D12（CANNBot DSL）] 是设计过程和阶段性记录；`examples/flash_attn_noquant` 提供完整的 `FlashAttnNoquant(paged=True)` 及其测试。QK/PV 阶段读取 `block_table[batch_idx,n_idx]` 得到物理页，K/V 布局为 `[P,Hkv,S,D]`，真实长度通过设备端 `seqused_kv` 输入。[分页实现][D-pa] [分页输入构造与测试][D-pa-test]

下面用已有连续、无 mask FlashAttention 的**完整 Cube helper 类**对比 `matmul/tl.dot`。它只负责 QK/PV 和搬运，不是 PA 实现；Source/Vector、online softmax、流水与 launch 在 [D14（CANNBot DSL）]，本摘录不能单独运行。

```python
class Matmul:
    def __init__(self, tile_cube_m, tile_n, tile_d):
        self.tile_cube_m = tile_cube_m
        self.tile_n = tile_n
        self.tile_d = tile_d

        self.nd2nz = make_copy_engine(format_transform="nd2nz", dtype=dtypes.float16, pad_value=0.0)
        self.fixpipe = make_copy_engine(dtype=dtypes.float32, dual_dst_ctl=1)

        tmp_n = max(tile_n, tile_d)
        self.q_l1 = Channel(MemLoc.L1, shape=(tile_cube_m, tile_d), dtype=dtypes.float16, depth=2)
        self.k_l1 = Channel(MemLoc.L1, shape=(tile_n, tile_d), dtype=dtypes.float16, depth=2)
        self.v_l1 = Channel(MemLoc.L1, shape=(tile_n, tile_d), dtype=dtypes.float16, depth=2)
        self.l0a = Channel(MemLoc.L0A, shape=(tile_cube_m, tmp_n), dtype=dtypes.float16, depth=2)
        self.l0b = Channel(MemLoc.L0B, shape=(tile_d, tile_n), dtype=dtypes.float16, depth=2)
        self.l0c = Channel(MemLoc.L0C, shape=(tile_cube_m, tmp_n), dtype=dtypes.float32, depth=2)

    def load_q(self, gm_tensor):
        """GM -> L1 (MTE2), nd2nz. Channel-first produce."""
        mem_copy(self.q_l1, gm_tensor, engine=self.nd2nz)

    def load_k(self, gm_tensor):
        """GM -> L1 (MTE2), nd2nz. Channel-first produce."""
        mem_copy(self.k_l1, gm_tensor, engine=self.nd2nz)

    def load_v(self, gm_tensor):
        """GM -> L1 (MTE2), nd2nz. Channel-first produce."""
        mem_copy(self.v_l1, gm_tensor, engine=self.nd2nz)

    def compute_qk(self):
        """S = Q @ K^T -> L0C. q_l1 read-many, k_l1 per-op consume."""
        mem_copy(self.l0a, self.q_l1)        # Q L1->L0A
        mem_copy(self.l0b, self.k_l1)        # K L1->L0B
        matmul(self.l0c, self.l0a, self.l0b, init=True)

    def compute_pv(self, p_l1_ch):
        """O_tile = P @ V -> L0C. p_l1 cross-core consume."""
        mem_copy(self.l0b, self.v_l1, transpose=True)   # V L1->L0B^T
        mem_copy(self.l0a, p_l1_ch)                    # P L1->L0A
        matmul(self.l0c, self.l0a, self.l0b, init=True)

    def store_s(self, ub_ch, partition):
        """S = Q @ K^T: L0C -> UB (FIXPIPE, split-M dual_dst_ctl). Cross-core produce."""
        mem_copy(ub_ch, self.l0c, engine=self.fixpipe, partition=partition)

    def store_o(self, ub_ch, partition):
        """O = P @ V: L0C -> UB (FIXPIPE, split-M dual_dst_ctl). Cross-core produce."""
        mem_copy(ub_ch, self.l0c, engine=self.fixpipe, partition=partition)
```

该类依赖原文件导入的 `Channel/MemLoc/dtypes/make_copy_engine/mem_copy/matmul`，显式选择 L1/L0A/L0B/L0C、depth=2、ND→NZ 和 FIXPIPE；Triton 例子主要由后端承担这些决策。完整 FlashAttention 蓝本 [D13（CANNBot DSL）] 还展示 Source/Matmul/Vector、Channel 与 Cube/Vector 流水。

从上述连续 helper 推演到 paged decode，需要把连续 KV 寻址改为设备查表，并评估实际长度尾块、q_len=1 的 GQA 小 M 打包及 page_size/tile_n 关系。当前 paged 实现已提供查表与 decode 的具体对象；这些算法及资源分析仍然适用，尤其是本次尚未覆盖的尾页和异长负载。

动态 metadata 能通过动态 TensorSpec、GM 整数 tensor 或 Host 结构传入。是否需要策略函数由选择的划分决定，而不是“CANNIR”强制客户写一份 C++ TilingFunc。

**A5 对应实测：** 两个原始 paged 用例通过：B=4、D=128、page=128，分别为 Hq/Hkv=36/36、Q=256、KV=1024，以及 Hq/Hkv=9/1、Q=1024、KV=1024。补充 Hq/Hkv=9/1、Q=1、KV=512 的 decode 也通过，使用随机打乱的物理页。以上多 query 用例是无 mask full attention；本次未覆盖 KV 尾页和同一 batch 内 KV 长度不同的情况。原来的连续 Cube helper 仍用于解释 L1/L0、Channel 和搬运职责，实际分页分支见前述源码。arena 的连续 FlashAttention 样例也仍有比较价值，但本次 PA 来源是 DSL 仓 examples。[RUN-D-xml] [RUN-results]

#### E（PyPTO on GPU）：有实际 paged decode API，但布局和 launch 数不能简化

[E10（PyPTO on GPU）] 的完整 `paged_attention_decode(...)` 接收：

```text
query / key_cache / value_cache
req_to_token：device INT32 请求到逻辑 token 的映射
request_index：device INT64，每 batch 一个值
valid_tokens：device INT64，每 batch 一个真实长度
virtual_to_physical：第二层映射
kv_heads、bucket_tokens：Host 几何参数
stream：CUDA stream
```

这是 token-index indirection / flat cache ABI，不是本文参考的 `[P,Hkv,S,D] + block_table` 原样输入；需要布局 / 元数据适配。完整演示在 [E11（PyPTO on GPU）]。

下面是 [E12（PyPTO on GPU）] 的**完整 DSL kernel**。公共 API 会将 flat cache/query 和长度 tensor 适配到这些逻辑视图，并选择受支持的编译几何；不能把公共比较布局不加转换直接传入：

```python
import pypto.language as pl

@pl.jit
def paged_attention_decode_kernel(
    query: pl.Tensor,
    key_cache: pl.Tensor,
    value_cache: pl.Tensor,
    req_to_token: pl.Tensor,
    request_index: pl.Tensor,
    valid_tokens: pl.Tensor,
    virtual_to_physical: pl.Tensor,
    scale: pl.FP32,
    out: pl.Out[pl.Tensor],
):
    """Gather each request's paged KV rows and mask its static KV bucket."""

    with pl.at(level=pl.Level.CORE_GROUP):
        for batch_row in pl.range(query.shape[0]):
            for q_head in pl.range(query.shape[1]):
                kv_heads = key_cache.shape[1] // query.shape[2]
                queries_per_kv = query.shape[1] // kv_heads
                kv_head = q_head // queries_per_kv
                request_id = pl.read(request_index, [batch_row, 0])
                valid_token_count_i64 = pl.read(valid_tokens, [batch_row, 0])
                valid_token_count = pl.cast(valid_token_count_i64, pl.INT32)

                keys = pl.tile.create(
                    [query.shape[2], request_index.shape[1]],
                    dtype=pl.BF16,
                    target_memory=pl.MemorySpace.Mat,
                    transpose=True,
                )
                values = pl.tile.create(
                    [request_index.shape[1], query.shape[2]],
                    dtype=pl.BF16,
                    target_memory=pl.MemorySpace.Mat,
                )
                for slot in pl.range(request_index.shape[1]):
                    virtual = pl.read(req_to_token, [request_id, slot])
                    physical_i64 = pl.read(virtual_to_physical, [virtual, 0])
                    physical = pl.cast(physical_i64, pl.INT32)
                    cache_column = kv_head * query.shape[2]
                    keys = pl.tile.gather_row(
                        keys,
                        key_cache,
                        [0, slot],
                        [physical, cache_column],
                        [1, query.shape[2]],
                        transpose=True,
                    )
                    values = pl.tile.gather_row(
                        values,
                        value_cache,
                        [slot, 0],
                        [physical, cache_column],
                        [1, query.shape[2]],
                    )

                query_box = pl.load(
                    query,
                    [batch_row, q_head, 0],
                    [1, 1, query.shape[2]],
                    target_memory=pl.MemorySpace.Mat,
                )
                query_tile = pl.reshape(query_box, [1, query.shape[2]])
                score = pl.matmul(query_tile, keys, out_dtype=pl.FP32)
                scaled = pl.mul(score, scale)
                positions = pl.tile.ci(
                    0,
                    [1, request_index.shape[1]],
                    dtype=pl.INT32,
                )
                valid_mask = pl.cmps(positions, valid_token_count, cmp_type=2)
                mask_scratch = pl.tile.create(
                    [1, 32], dtype=pl.UINT8, target_memory=pl.MemorySpace.Vec
                )
                masked_scaled = pl.sels(
                    valid_mask,
                    scaled,
                    mask_scratch,
                    -3.4028234663852886e38,
                )
                max_scratch = pl.create_tile(
                    [1, request_index.shape[1]],
                    dtype=pl.FP32,
                    target_memory=pl.MemorySpace.Vec,
                )
                row_max = pl.row_max(masked_scaled, max_scratch)
                centered = pl.row_expand_sub(masked_scaled, row_max)
                exponent = pl.exp(centered)
                sum_scratch = pl.create_tile(
                    [1, request_index.shape[1]],
                    dtype=pl.FP32,
                    target_memory=pl.MemorySpace.Vec,
                )
                row_sum = pl.row_sum(exponent, sum_scratch)
                probability = pl.row_expand_div(exponent, row_sum)
                probability_bf16 = pl.cast(probability, target_type=pl.BF16)
                mixed = pl.matmul(probability_bf16, values, out_dtype=pl.FP32)
                result = pl.cast(mixed, target_type=pl.BF16)
                result_box = pl.reshape(result, [1, 1, query.shape[2]])
                pl.store(result_box, [batch_row, q_head, 0], out)
    return out
```

原客户调用形态如下，输入构造见 [E11（PyPTO on GPU）]：

```python
output = attention.paged_attention_decode(
    query, key_cache, value_cache,
    req_to_token, request_index, valid_tokens, virtual_to_physical,
    kv_heads=kv_heads, bucket_tokens=bucket_tokens, stream=stream,
)
stream.synchronize()
```

`request_index.shape[1]` 是适配视图携带的**静态 bucket 长度**，编译和 K/V tile 构造按 bucket 进行，而非实际 page_count 的 online 循环。`valid_tokens` 留在设备 tensor 中控制 mask，不为每个长度做 Host 同步；emitter 可专门 lowering，实际工作量须看生成物。当前函数体有：

- `batch_size==1` 时，尝试所有 Q head 合并的一次 launch，并可探测更宽 bucket；
- 合并不可用或其他分支，逐 Q head 构造零拷贝视图并多次 launch；
- 最后遍历 `pending_launches` 执行。

函数早期 docstring 仍偏重逐 head 描述，本文以函数体为准。**有一个 `paged_attention_decode` Python 调用，不代表恒为一次 kernel launch。**

改变 `valid_tokens` 内容与改变 bucket、batch、shape/stride 的代价不同；后者可能选择 / 编译新图。该路径证明的是受支持模式下的 paged decode，不是通用动态图编译。

#### F（Triton-Ascend）：完整统一 attention 中已有原生 paged decode case

**源码示例与实测对象：** 本节使用 Triton-Ascend 编译器仓的完整 unified attention 代码，分析 runtime loop、online softmax、program 映射和编译器职责。A5 实测对象是 `triton-ascend-kernels` 仓的 `paged_attention_fwd`，其原测试 8例通过，覆盖 MHA/GQA、causal prefill/decode、KV 尾页及 QK/V 维度不同；该结果不代表下面 unified attention 全参数矩阵的执行结果。[实测 kernel][F-pa] [实测测试][F-pa-test] [RUN-F-xml]

完整 kernel、Host 和 reference：[F6（Triton-Ascend）]。测试参数包括 `[(1,523),(1,37),(1,2011)]`，每请求一个 query 的 decode 形态就在测试源码中。

下面保留原文件的**两个 helper、完整 kernel 和 Host wrapper**及可选 ALiBi/softcap，补齐导入：

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import triton
import triton.language as tl

@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def apply_softcap(S, x):
    Sdiv = S / x
    p1 = tl.exp(Sdiv)
    p2 = tl.exp(-Sdiv)
    return x * (p1 - p2) / (p1 + p2)


@triton.jit
def kernel_unified_attention_2d(output_ptr,  # [num_tokens, num_query_heads, head_size]
                                query_ptr,  # [num_tokens, num_query_heads, head_size]
                                key_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
                                value_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
                                block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
                                seq_lens_ptr,  # [num_seqs]
                                alibi_slopes_ptr,  # [num_query_heads]
                                scale,  # float32
                                k_scale,  # float32
                                v_scale,  # float32
                                softcap,  # float32
                                num_query_heads: tl.constexpr,  # int
                                num_queries_per_kv: tl.constexpr,  # int
                                block_table_stride: tl.int64,  # int
                                query_stride_0: tl.int64,  # int
                                query_stride_1: tl.int64,  # int, should be equal to head_size
                                output_stride_0: tl.int64,  # int
                                output_stride_1: tl.int64,  # int, should be equal to head_size
                                BLOCK_SIZE: tl.constexpr,  # int
                                HEAD_SIZE: tl.constexpr,  # int
                                HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
                                USE_ALIBI_SLOPES: tl.constexpr,  # bool
                                USE_SOFTCAP: tl.constexpr,  # bool
                                SLIDING_WINDOW: tl.constexpr,  # int
                                stride_k_cache_0: tl.int64,  # int
                                stride_k_cache_1: tl.int64,  # int
                                stride_k_cache_2: tl.int64,  # int
                                stride_k_cache_3: tl.constexpr,  # int
                                stride_v_cache_0: tl.int64,  # int
                                stride_v_cache_1: tl.int64,  # int
                                stride_v_cache_2: tl.int64,  # int
                                stride_v_cache_3: tl.constexpr,  # int
                                query_start_len_ptr,  # [num_seqs+1]
                                BLOCK_Q: tl.constexpr,  # int
                                num_seqs: tl.int32, BLOCK_M: tl.constexpr,  # int
                                ):

    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        mid_val = tl.load(query_start_len_ptr + mid) // BLOCK_Q + mid
        if mid_val <= q_block_global_idx:
            left = mid + 1
        else:
            right = mid

    seq_idx = left - 1
    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx

    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)

    cur_batch_query_len = cur_batch_in_all_stop_index \
        - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + \
        offs_m % num_queries_per_kv
    query_offset = (query_offset_0[:, None] * query_stride_0 + query_offset_1[:, None] * query_stride_1 +
                    offs_d[None, :])

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # context length for this particular sequences
    context_len = seq_len - cur_batch_query_len

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0)

    num_blocks = cdiv_fn(seq_len, BLOCK_SIZE)

    # iterate through tiles
    for j in range(0, num_blocks):

        physical_block_idx = tl.load(block_tables_ptr + block_table_offset + j)

        offs_n = tl.arange(0, BLOCK_SIZE)

        v_offset = (physical_block_idx * stride_v_cache_0 + kv_head_idx * stride_v_cache_2 +
                    offs_d[None, :] * stride_v_cache_3 + offs_n[:, None] * stride_v_cache_1)

        k_offset = (physical_block_idx * stride_k_cache_0 + kv_head_idx * stride_k_cache_2 +
                    offs_d[:, None] * stride_k_cache_3 + offs_n[None, :] * stride_k_cache_1)

        K_load = tl.load(key_cache_ptr + k_offset, mask=dim_mask[:, None], other=0.0)

        if K_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                K = K_load
            else:
                K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
        else:
            K = K_load

        V_load = tl.load(value_cache_ptr + v_offset, mask=dim_mask[None, :], other=0.0)

        if V_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                V = V_load
            else:
                V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
        else:
            V = V_load

        seq_offset = j * BLOCK_SIZE + offs_n

        seq_mask = seq_offset[None, :] < context_len + query_pos[:, None] + 1

        S = tl.zeros(shape=(BLOCK_M, BLOCK_SIZE), dtype=tl.float32)

        S += scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        S = tl.where(query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf"))

        if SLIDING_WINDOW > 0:
            S = tl.where((context_len + query_pos[:, None] - seq_offset) < SLIDING_WINDOW, S, float("-inf"))

        if USE_ALIBI_SLOPES:
            S += alibi_slope[:, None] * (seq_offset - context_len)

        # compute running maximum
        m_j = tl.maximum(M, tl.max(S, axis=1))
        # For sliding window there's a chance the max is -inf due to masking of
        # the entire row. In this case we need to set m_j 0 to avoid NaN
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

        P = tl.exp(S - m_j[:, None])

        l_j = tl.sum(P, axis=1)

        alpha = tl.exp(M - m_j)

        acc = acc * alpha[:, None]

        # update constants
        L = L * alpha + l_j
        M = m_j

        acc += tl.dot(P.to(V.dtype), V)

    # epilogue
    acc = acc / L[:, None]

    output_offset = (query_offset_0[:, None] * output_stride_0 + query_offset_1[:, None] * output_stride_1 +
                     offs_d[None, :])

    tl.store(
        output_ptr + output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )


def unified_attention(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    alibi_slopes=None,
):
    assert causal, "Only causal attention is supported"
    assert q_descale is None, "Q scales not supported"

    block_size = v.shape[1]
    assert q.element_size() >= 2 or block_size >= 32, \
        "Block size must be at least 32 for fp8"

    use_alibi_slopes = alibi_slopes is not None

    block_size = v.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]

    BLOCK_M = 16
    BLOCK_Q = BLOCK_M // num_queries_per_kv

    # Ideally we would launch with kernel with:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)] blocks.
    # However, it is slow to realize the query_lens on cpu.
    # Instead we use upper-bound:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)]
    #   <= \sum_i[floor(query_len[i] / BLOCK_Q) + 1]
    #    = \sum_i[floor(query_len[i] / BLOCK_Q)] + num_seqs
    #   <= floor(\sum_i(query_len[i]) / BLOCK_Q) + num_seqs
    #    = floor(q.shape[0] / BLOCK_Q) + num_seqs
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs

    kernel_unified_attention_2d[(
        total_num_q_blocks,
        num_kv_heads,
    )](
        output_ptr=out,
        query_ptr=q,
        key_cache_ptr=k,
        value_cache_ptr=v,
        block_tables_ptr=block_table,
        seq_lens_ptr=seqused_k,
        alibi_slopes_ptr=alibi_slopes,
        scale=softmax_scale,
        k_scale=k_descale,
        v_scale=v_descale,
        softcap=softcap,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        BLOCK_SIZE=block_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
        USE_ALIBI_SLOPES=use_alibi_slopes,
        USE_SOFTCAP=(softcap > 0),
        SLIDING_WINDOW=(1 + window_size[0]),
        stride_k_cache_0=k.stride(0),
        stride_k_cache_1=k.stride(1),
        stride_k_cache_2=k.stride(2),
        stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0),
        stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2),
        stride_v_cache_3=v.stride(3),
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=BLOCK_Q,
        num_seqs=num_seqs,
        BLOCK_M=BLOCK_M,
    )
```

上述 decode case 对应 `cu_seqlens_q=[0,1,2,3]` 与 `seqused_k=[523,37,2011]`。该文件当前测试常量为 FP16、`HEAD_SIZE=128`、page/block size=32，head 组合 `(8,2)` / `(16,2)`；公共比较几何 page=128、GQA=5 **不是该测试已覆盖的配置**。[F6（Triton-Ascend）]

关键代码真实使用 `tl.load(seq_lens_ptr + seq_idx)`、`ceildiv(seq_len,BLOCK_SIZE)`、运行时 page 循环和 `tl.load(block_tables_ptr + ...)`；QK / PV 为两次 `tl.dot`，online softmax 在中间。`BLOCK_SIZE/HEAD_SIZE` 等仍为 constexpr。

Host 为避免将每请求 query length 拷回 CPU，使用 `q.shape[0] // BLOCK_Q + num_seqs` 的上界 grid，第二轴为 KV heads；真实长度在 kernel 内读取。这个例子直接说明：**SPMD 的动态长度并不要求新增一个独立 tiling 函数。** 代价是上界空工作、负载均衡和后端资源规划都仍需关注。

#### G（AutoFuse + Inductor）：完整用户输入表达有，原生单 kernel PA 能力不能据此补齐

第 2.4 节 `paged_decode_padded` 是完整 PyTorch 客户表达；G（AutoFuse + Inductor） 的接入层有 gather、matmul、reduction 等 lowering / 模板相关代码，但“部件存在”不等于该整段 graph 被单 kernel 接住。

特别是 [G9（AutoFuse + Inductor）] 使用 `torch.ops.npu.npu_fusion_attention_v3`，测试实际长度变化及 ACLGraph 更新。这是 **外部 attention 算子与 Inductor runtime 集成的证据**，不是 AutoFuse 原生生成 paged decode kernel 的证据，也不是 megaKernel 证据。

为明确“外部 attention 调用”到底长什么样，[G9（AutoFuse + Inductor）] 的编译函数如下（从测试方法内去除缩进，仍引用 `self.N` 和测试设定的 `scale`，不是独立 PA kernel）：

```python
@torch.compile(options={"npu_backend": "ascendc", "triton.cudagraphs": True})
def fa_compiled(q, k, v, qlen, kvlen):
    return torch.ops.npu.npu_fusion_attention_v3(
        q, k, v, head_num=self.N, input_layout="TND",
        scale=scale, keep_prob=1.0,
        actual_seq_qlen=qlen, actual_seq_kvlen=kvlen,
    )[0]
```

原测试 `qlen/kvlen` 是 **CPU int64 累积长度**，不是本文 PA ABI 的设备端绝对 `L[B]`，也没有 paged cache 输入。本次未找到与 PyPTO3/Triton-Ascend 同等级的 AutoFuse 原生 paged decode 完整例子；客户入口存在与原生 kernel 覆盖仍须分开，具体归属和融合边界看 lowering/generated code/trace。

**A5 补充给出了这条边界的实际对象：** 本文 `paged_decode_padded` 图表达经源码版 `inductor_npu_ext`、`torch.compile(fullgraph=True, dynamic=False)` 执行，固定 shape、两组 L 内容均通过。生成 wrapper 内有7个不同 AutoFuse 函数、9个调用点，同时保留 `aten.index_select`、`aten.bmm`、`repeat_interleave` 等外部算子。因此“客户可以保留完整 PyTorch 表达”已有执行证据，“全部 PA 被降成一个原生融合 kernel”仍未证明。静态调用点计数不是设备 launch 计数，具体误差、外部调用及证据见第 12.1.1 节。[RUN-G-lowering] [RUN-results]

#### H（CATLASS DSL）：连续 FlashAttention 已有实现，分页和设备异长尚未在这个例子落地

完整入口是 `examples/end_to_end/flash_attention_infer/flash_attention_infer.py`，辅助 tiler 在同目录 `fa_tiling.py`。输入为 `Q[B,Sq,Hq,D]`、连续 `K/V[B,Skv,Hkv,D]`，输出与 Q 同布局；默认 `B=1,Sq=117,Skv=512,Hq=8,Hkv=1,D=128`，输入 f16/bf16，使用 online softmax。QK、softmax、PV、rescale 之间有显式跨核 flag 和多缓冲传递。源码及 README 均将此限定为连续 KV，mask 当前只能全零，**没有 page table 查询和 paged cache ABI**。[H-fa] [H-fa-readme] [H-fa-tiling]

必须继续看参数是否被消费：kernel 签名有 `tiling_data`、`actual_q_seqlen` 和 `actual_kv_seqlen`，但当前函数体没有读取这三个 tensor 的内容。实际 task 数、Q/KV 上界和 batch 偏移来自 `HEAD_NUM/Q_SEQ/KV_SEQ/TOTAL_TASKS` 等 Python 全局量；Host `apply_shape_args()` 在编译前改写这些量。`fa_tiling.py` 能构造长度数组，并不足以证明这个 kernel 支持同一产物下每请求异长。[H-fa]

它展示了 A5 上 **Cube↔Vector 交接、KV 分段、在线状态与物理 layout 转换** 的完整用户实现，可用于核内资源与流水的理论比较；缺分页是这个算子的实现差距。若扩成第 2.4 节定义的 PA，至少还需页表间接寻址、每请求有效长度真正下沉设备、尾页 mask、页内跨界搬运及对应验证。把 Q 长度改为 1 只能得到连续 decode 场景，不能因此改名为 PagedAttention。本轮连续多 query、BF16 及 KV 尾块用例通过；Q=1 用例实际精度失败，且复跑仍失败，详见第 12.1.3 节。

#### I（CuTe DSL）：找到的是分页 MLA decode，另有连续 GQA decode

| 源文件 | 直接可确认的能力 | 与第 2.4 节统一 PA 口径的差别 |
| --- | --- | --- |
| `cute/blackwell/kernel/attention/mla/mla_decode_fp16.py` | 真正接收 page table；paged TMA、page-table pipeline、online softmax、split-KV、可选 persistent / var-seq | MLA latent/rope 表达；并非独立 K/V head cache 的标准 GQA PA |
| 同目录 `mla_decode_fp8.py` | 另一条含 page table 的 FP8 MLA 实现 | dtype、量化及 tile 约束须独立核对，不能沿用 FP16 的全部边界 |
| `cute_ext/blackwell/attention/gqa_decode_simple.py` | 连续 GQA；decode 内 QK→softmax→PV→partial，第二个 kernel 合并 split | 直接索引连续 K/V，没有 page table；一个 Host JIT 入口含两处 `.launch()` |

来源：[I-mla] [I-mla-fp8] [I-gqa]。搜索范围为这份固定 CUTLASS checkout 的 `examples/python/CuTeDSL` 与 `python/CuTeDSL`；不是对外部整个 CuTe 生态作“只有这些 PA”的断言。

FP16 MLA 的 `can_implement()` 给出具体约束：latent 维 512、rope 维 64，输入/输出 f16，累加/LSE f32；Q 长度 1—4，head 数不超过 128；head 数小于 128 时 split-KV 必须为 1；page size 不能为 1 且须整除 QK 的 N tile；可变 split-KV 依赖 var-seq。上述是源码检查，仍不等价于所有通过检查的组合均已验证。[I-mla]

它的测试页表先构造成 `[B,page_count]`，填入 `b+j*B` 使不同请求的物理页交错，再转为设备视图 `[page_count,B]` 并标记动态 layout；参考计算按页表 gather latent/rope 数据。因此它确实能为“间接页访问怎样进入 TMA、流水和在线计算”提供代码参照。若与本文约定的 `Hq=40,Hkv=8,D=128,S=128` 比性能，应另找或适配标准 GQA PA；不能靠转置页表消除 MLA 与 GQA 的数学差异。[I-mla]

### 2.7 实际长度变化，对 tiling 的四种可能影响

| 变化 | 不一定需要做的事 | 可能确实需要做的事 |
| --- | --- | --- |
| `L` 内容变化，`B/P/T/D/S` 不变 | 不一定重编译；不一定 Host 读回；不一定新策略函数 | 更新 mask / 循环；若 Host work_ranges 按旧 L 生成，则重算或改为上界 / 设备侧分配 |
| `B` 或 table 宽度 `T` 变化，rank 不变 | 不一定重写 kernel 算法 | 更新动态描述符、grid/workspace；若该轴静态则新缓存变体 |
| `D/S/GQA` 几何变化 | 不一定每个值手写新算法 | 改 constexpr / tile / layout / key；已有策略的支持域可能要扩展 |
| L 的分布极不均匀 | 正确性不一定要求动态 scheduler | 为性能重新选择 task 粒度、split-KV、work_ranges 或动态派发 |
| L 依赖前一设备计算 | 不必一定同步到 Host | 设备读取、设备 tiling task、动态编排，或明确付出 Host 同步代价 |

“是否需要 tiling 函数”应问：由谁运行、运行在哪、输入是 shape 还是 tensor 内容、产生的是块大小还是任务划分、是否触发 kernel 专门化。这五个问题比“有 / 没有”更有信息量。

### 2.8 怎样实际改“静态/动态”：不要只给变量加一个 dynamic 标签

下面是按前面具体实现得到的修改清单；“需要重算策略”“需要重编译”“需要重写算法”分开记录。

| 具体修改 | 直接对应的代码动作 | 不应误解为 |
| --- | --- | --- |
| block版 softmax `(777,300)→(2049,100)` | 两轴已经 DYNAMIC；原 Host 重新算 num_tiles/num_cores，kernel 用 set_validshape | 每个新取值必须新增一个 tiling 函数 |
| block版 softmax `N=300→131075` | 超出 MAX_N=512；需分段算法/新 tile 策略，不能沿用原 kernel | 只改 shape 元数据就能突破片上容量 |
| PyPTO3 教学 softmax 改 ROWS/COLS | 修改静态常量后专门化；想运行时动态应换成动态声明/有效窗口的实现 | 固定切片循环自动获得任意 shape 安全性 |
| PyPTO3 native PA 只改 seq_lens 内容 | 设备 pl.read 决定 page/stack 次数和尾列；符合已有容量等契约时无需新增 Host TilingFunc | 所有 shape 一定共用同一个编译产物 |
| block版 paged prefill 只改实际长度 | kernel getval 读新值；Host work_ranges 若依赖旧长度必须更新 | 设备循环动态意味着 Host 分配元数据永远不用更新 |
| GPU PA 只改 valid_tokens 内容 | 保持 bucket/geometry 合法，改变 mask；保留有效 padded 映射 | mask 一变就会改变静态 tile 容量，或工作量必然线性下降 |
| Triton PA 改长度与改 page/D | 长度是 tl.load 的数据；page/D 是 constexpr/资源几何，可能走新编译变体 | 二者都是“改输入”，所以代价一样 |
| AutoFuse 客户 PA 改 L 与改 B/P/T | L 可作为设备 mask；B/P/T 进入符号 shape/guards/tiling，由 compiler 接受范围决定 | dynamic=True 等于无 guards、无 fallback、零编译成本 |

客户可用第 2.4 节函数直接建立两类对照，下面是完整的 **NPU 调用驱动**，依赖该节已经给出的 `make_decode_case/paged_decode_reference/make_compiled_decode`；它是待运行的能力探针，不是报告“AutoFuse 已生成一个 PA kernel”：

```python
import torch
import torch_npu

# Choose one backend per process to make cache/registration effects clear.
compiled = make_compiled_decode("AutoFuse", dynamic=True)
q, k, v, table, lengths = make_decode_case("npu")

# Content dynamics: identical tensor metadata, changed device lengths.
for values in ([1, 129, 513], [127, 128, 512]):
    lengths.copy_(torch.tensor(values, dtype=torch.int32, device="npu"))
    actual = compiled(q, k, v, table, lengths)
    expected = paged_decode_reference(
        q.cpu(), k.cpu(), v.cpu(), table.cpu(), lengths.cpu())
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-2, rtol=2e-2)

# Metadata dynamics: B changes from 3 to 1; record guards and compile count.
actual = compiled(q[:1], k, v, table[:1], lengths[:1])
expected = paged_decode_reference(
    q[:1].cpu(), k.cpu(), v.cpu(), table[:1].cpu(), lengths[:1].cpu())
torch.npu.synchronize()
torch.testing.assert_close(actual.cpu(), expected, atol=2e-2, rtol=2e-2)
```

`.cpu()` 和同步只出现在 golden/验证路径，不是性能测量路径。若编译失败，应保留失败与 generated code，而不是换成外部 attention 后仍宣称原生融合成功。把后端参数改为 `"Triton-Ascend"` 可测其 **Inductor 路径**；这不是第 2.6 节手写 Triton native PA kernel 的同一测试。

以上例子还揭示表达代价的一条边界：完整的数学表达可以很短，完整的低层高性能实现可能很长；用其中特定 head/page/dtype 的测试，不能证明另一套布局或精度的实现也已覆盖。

H/I 的实际改法也应落到参数来源：H basic MMAD 的 Host helper 将 GM layout 标成动态，kernel 读取 `origin_shape`；现有 FA 则由 Host 改 Python 全局 Q/KV 常量后编译，单改传入长度 tensor 尚不会改变设备循环。I 要区分 runtime tensor 的动态元数据、`Constexpr` tile/角色，以及 kernel 自己读取的序列内容；把 tile 从常量改成标量并不会自动产生合法的动态 SMEM/TMEM 计划。对应完整契约和产物复用证据见第 7.15 节。[H-common-utils] [H-mmad-example] [H-fa] [I-tensor-runtime] [I-gqa]

<a id="frontend-capabilities"></a>
### 2.9 前端表达能力的上下界：能描述多大的程序，能控制多细的硬件动作

这里把三个容易混用的问题分开：**上界**是能组织单个计算、融合 kernel，还是带动态依赖的多阶段程序；**下探粒度**是客户能控制到 Tensor、Tile/存储、寄存器/事件还是具体指令选项；**覆盖面**是其中哪些操作和参数组合真的贯通后端。表达范围大不等于硬件控制更细，控制更细也不等于操作覆盖完整。

#### Python 是用户语言、元编程语言，还是构图工具？

| 路线 | 客户实际写什么 | Python 如何进入编译器 | 不能据此宣称的能力 |
| --- | --- | --- | --- |
| PyPTO2-tensor版 | `@pypto.frontend.jit`、Tensor 运算、切片、动态循环、tile/pass 配置 | 当前默认新前端生成 PIL/自有 IR，再进入 TileFwk；另有直接 Tensor/IR 构建接口 | 不是只能手工连图；也不是任意 Python 程序或任意 PyTorch 调用都能进入设备程序 |
| PyPTO2-block版 | `@pl.jit`、`TileType`、out-first 核内操作、Cube/Vector section、VF；Host 用方括号指定 launch | ASTParser 识别 DSL，生成共享基础自有 IR 上的 block版 操作；设备入口直接编译/launch | 有 Python kernel DSL；并不因此具备tensor版或 Simpler 的通用外层任务系统 |
| PyPTO3（Simpler） | `@pl.jit`、Tensor/Tile、层次 scope、InCore/Orchestration，必要时显式 TaskId/SPMD/extern | Python DSL parser → 自有程序 IR；核内与编排分路编译 | 不只是构图 API；但某个语言构件只在规定层级合法，不能任意跨 Host/Orchestration/InCore 使用 |
| CANNBot DSL | Python 类/方法、`@jit` Host、`@kernel` Device、Tensor/Buffer/Channel、寄存器函数 | 捕获源码，执行 `analyze → lower → materialize`，再以分阶段执行/IR builder 生成 MLIR | **不是“没有 AST 的纯 tracing”**；Python helper 的 trace-time 执行也不是设备上运行任意 Python |
| PyPTO on GPU | 集成 operator API 或受支持的 PyPTO DSL kernel | PyPTO 自有 IR → 当前 emitter/pattern → TensorIR | 前端能解析的表达，不保证该 GPU 后端能编；不能沿用 PyPTO3 的全部任务语义 |
| Triton-Ascend | 独立 `@triton.jit` + `tl.*`；或者 PyTorch 交由 Inductor 生成 Triton | Triton Python kernel DSL → TTIR；编译常量与运行时 tensor/control flow 分开 | 不是只有构图接口；但 Triton 公共名字存在，不代表 NVIDIA 参数语义及所有 NPU 目标都支持 |
| AutoFuse + Inductor | 客户主要写 PyTorch；接入开发者使用 ASCGraph/ASCIR 操作接口 | Inductor 把 loop/index/size/compute 组织为 ASCIR 图，AutoFuse 生成 kernel/tiling | 有 Python 客户入口、有 Python 构图接口；本次路径未提供与 block版/CANNBot/Triton 同定位的独立客户 kernel DSL |
| H（CATLASS DSL） | `catlass.tla`、`@tla.kernel`、layout/allocate/copy/MMAD/Vector，Host 编译后调用产物 | Python staging 建 TLA MLIR；`@tla.jit` 在 kernel 编译中内联设备 helper，Host 直接调用它仍是普通 Python | 不能把 H 的 `jit` 等同 I 的编译 Host 多 launch，也不能把 C++ CATLASS 全部抽象算作本 Python DSL 已覆盖。[H-dsl] [H-readme] |
| I（CuTe DSL） | `@cute.jit` Host、`@cute.kernel` Device、layout/atom、线程分区及 pipeline | 分阶段生成 MLIR；kernel 调用先形成 launcher，再 `.launch()`；Host JIT 可以编排多个 launch | 编译期 Python 不等于设备执行任意 Python；一个 JIT 不是一个物理 kernel，TS 不是跨模型任务 runtime。[I-dsl] [I-gqa] [I-task] |

依据：[A1（PyPTO2-tensor版）]—[A3（PyPTO2-tensor版）]、[B17（PyPTO2-block版）] [B20（PyPTO2-block版）]、[C21（PyPTO3（Simpler））]、[D1（CANNBot DSL）] [D17（CANNBot DSL）]、[E1（PyPTO on GPU）] [E2（PyPTO on GPU）]、[F1（Triton-Ascend）] [F5（Triton-Ascend）]、[G11（AutoFuse + Inductor）]。

因此，“有 Python 前端”不是有区分度的终点。更有区分度的是：**用户写的是被编译的程序语义，还是构造编译器 IR 的 Python 工具代码；哪部分 Python 只在编译期执行；运行时动态值在哪个处理器上被读取。** 三者可以在一个系统中共存，不能仅靠 `@jit` 这个名字归类。

### 2.10 API 丰富度：按能力族核对，不按导出符号数量排名

下表列出当前源码里可找到的代表性表面，不是完整 API 手册，也不是这些 API 全部组合均已在三类硬件上验证。

| 路线 | Vector / 索引 / 数值 API 示例 | Cube / 矩阵 API 示例 | 更低层与更高层的表达范围 |
| --- | --- | --- | --- |
| PyPTO2-tensor版 | `exp/amax/sum/argmax/topk/gather/scatter_update`、cast、quantize/dequantize | `matmul/scaled_mm`、conv/conv_backward_input；Cube tile 和 split-K 配置 | 以 Tensor 值及 view/loop/有效形状组织程序；能设融合/复用策略，不等于逐条安排 VF 寄存器或固定物理核 |
| PyPTO2-block版 | Tile 算术、row/column reduce/expand、gather/scatter/sort、quant；`pl.Vf` 寄存器、mask、cast 等 | `matmul/matmul_acc/matmul_mx` 及相应累加形式 | Tile 地址/布局/槽、mutex、pipeline、Cube/Vector section、system；还有 SIMT API 表面，目标约束必须另查；通常以一个显式 kernel 为直接编译对象 |
| PyPTO3（Simpler） | Tensor/Tile 统一分派的算术/归约/广播/转换；paged_gather、sort、scatter；显式 load/store/move | Tensor/Tile matmul、acc、gemv、MX 相关操作 | 片上 MemRef/slot、system 同步与跨核传递，以及 TaskId/SPMD/Orchestration；不同层级的操作并非全部互通 |
| CANNBot DSL | Tensor math/reduce/expand/cast；`reg.*` 算术、mask、转换、gather/scatter、load/store | `matmul`、Cube 控制和 conv2d 数据通路 helper | 显式逻辑/物理 layout、Buffer、Channel、搬运引擎、VF 和同步；Host 也能编译，但不是设备任务 DAG runtime |
| PyPTO on GPU | 已接通的逐元素、行归约、norm/激活等模式；算子包装层及特定融合 builder | structured matmul、attention 等 emitter 路径 | 主要以 tile/张量语义与 schedule 选项进入 TensorIR；不是面向每种 GPU 指令的 Python 封装，也不是任意自有 IR 图都能通过 |
| Triton-Ascend | `tl.load/store/exp/max/sum/where`、索引/mask 等；还有本地 Ascend 扩展 | `tl.dot` 与本地扩展/后端选择 | 常规写法控制 program/tile/算法；本地 `al/bl` 扩展还暴露 buffer、地址空间、copy、fixpipe、CV 同步等，不宜概括成“始终不能碰局部存储” |
| AutoFuse + Inductor | 客户 PyTorch 运算经 decomposition/lowering；内部 `Add/Exp/Reduce/Gather` 等 ASCIR 构图 | ASCIR matmul/带 bias 形式、模板及受限融合；某些形态 fallback | 客户通常不编 UB 槽/指令；这些责任进入 compiler/template。最终进入自生成 kernel 的集合小于“PyTorch 可表达”的集合 |
| H（CATLASS DSL） | `vec.func` 内寄存器 load/store、mask、exp/reduce 等；tensor/view、逻辑有效范围 | MMAD、L1→L0A/B、L0C→GM/UB 等显式搬运 | `allocate`、地址空间、物理 layout tag、flag/mutex 与跨 CV 协议；部分局部同步可自动生成，不提供默认模型任务图。[H-api-copy] [H-api-mmad] [H-fa] [H-auto-sync] |
| I（CuTe DSL） | layout 的 composition/product/divide/coalesce、local tile、线程/值 fragment、warp 归约 | Copy/MMA atom，cp.async/TMA、架构特定 tcgen05 形式 | 线程/warp/CTA/cluster、SMEM/TMEM、pipeline、persistent/CLC，experimental TS 声明资源协议；具体 SM/dtype/layout 组合须逐项核对。[I-layout] [I-copy] [I-mma] [I-task] |

API 入口与实现索引：[A17（PyPTO2-tensor版）] [A19（PyPTO2-tensor版）]、[B17（PyPTO2-block版）] [B18（PyPTO2-block版）]、[C21（PyPTO3（Simpler））]、[D18（CANNBot DSL）]、[E2（PyPTO on GPU）] [E12（PyPTO on GPU）]、[F5（Triton-Ascend）] [F6（Triton-Ascend）] [F16（Triton-Ascend）]、[G11（AutoFuse + Inductor）]。这些属于**接口/实现存在性**证据；相应 softmax/PA 的验证边界仍按第 2.3/2.6 节。

#### “支持所有 Vector 算子 / Cube / Tensor Core 指令”为什么不能直接打勾

需要逐层回答四个不同问题：

1. **数学语义**：能否写出这个计算？一个高层操作可以分解成许多原语；没有同名 API 不必然意味着数学计算无法表达。
2. **硬件原语**：能否表达指定 dtype、布局、舍入、饱和、量化 scale、累加和同步选项？有 `matmul` 不代表能选择每一代 Cube/Tensor Core 的每种形式。
3. **lowering 路径**：当前前端/IR/pass/codegen 是否保留这些选项并交给目标？下游 PTOAS、AscendC、AscendNPU-IR、TensorIR 有某个操作，上层也不自动获得它。
4. **实际产物**：是否真的用了所需单元/指令，数值和性能是否达标？API 名、dtype 枚举、生成的 C++ 调用甚至一个 MLIR op，都不能单独替代目标编译产物和硬件验证。

当前报告没有“完整目标指令清单 × 所有参数组合”的分母，也没有覆盖该集合的端到端测试。因此不提供 API/ISA 覆盖百分比，不写“全部支持”。这不表示九条路线同样受限：**block版/CANNBot 更直接暴露低层选项，tensor版/AutoFuse 更多交给编译器，PyPTO3 同时提供 Tensor/Tile 核内计算与带依赖的任务编排接口，GPU 集成还明显受 emitter 覆盖约束**；这是控制面的差异，不是未经测量的完整性名次。

#### PyPTO3（Simpler）的“多级表达”：核内抽象与核外任务组织是两个维度

这里的“多级”不是指 MLIR dialect 层级，也不是“已经暴露全部硬件指令”，而是指**同一套编程体系既能选择核内计算写到多细，也能组织多个核内计算如何协作**。两者不能排成一条简单的高低层阶梯；应按下面四种用户责任理解。

1. **Tensor 计算表达：写数学链和切片，不必逐个声明片上临时量。**例如 `pypto-lib` softmax 用 `x[r : r + ROW_TILE, :]`、`pl.row_max`、`pl.exp`、`pl.row_sum` 等表达计算。这里的 `tile_x` 虽然名字带 tile，源码仍是 Tensor 切片写法；划入 InCore 后，`ConvertTensorToTileOps` 将相应 Tensor 操作降为 Tile 操作，并按需要补 load/store、归约临时空间。用户仍指定 `ROW_TILE`、切片范围和循环，并非把任意大 Tensor 交进去就自动获得合适的全局分块。一个直接的责任差异是：Tensor 输入写 `pl.row_max(tensor)`，scratch 由 lowering 生成；Tile 输入必须写 `pl.row_max(tile, tmp_tile)`，作者提供满足约束的临时 Tile。[C15（PyPTO3（Simpler））] [C28（PyPTO3（Simpler））] [C30（PyPTO3（Simpler））]

2. **Tile / 存储 / 流水表达：在需要优化的核内区域接管更多资源决策。**作者可以显式写 `pl.Tile`、`pl.load/store/move`、目标存储空间、`pl.MemRef(..., slots=N)[slot]`，以及支持的同步操作。例如 native PA 把 V 页放入 `pl.Mem.Mat` 对应的 L1，通过 `pl.MemRef("pv_v_l1", slots=STACK_PAGES)[0]` 等区分槽，再用 `pl.tile.move(..., target_memory=pl.MemorySpace.Right)` 准备矩阵操作数；其中还有 `pl.system.sync_wait` 协调阶段。用户从“这块数据做什么”进一步控制“放哪类存储、何时搬、用哪个槽、等待什么”。但声明命名 MemRef 不等于手工指定最终物理地址；省略大小/地址时仍由编译器推导。显式 slots 与编译器 `pl.pipeline(stage=...)` 多缓冲也不能随意叠加，同一资源的手工轮转须由作者保证正确；当前 Tile 物理容量仍要求静态，动态有效范围另表达。[C16（PyPTO3（Simpler））] [C29（PyPTO3（Simpler））] [C10（PyPTO3（Simpler））]

3. **核外程序 / 任务表达：指定计算区域、工作分解和依赖，而不只编一个核内函数。**`@pl.jit` 是程序编译入口，不是“只会产生一个物理 kernel”的承诺；也可用 `@pl.program` / `@pl.function` 显式组织函数。`pl.at(level=pl.Level.CORE_GROUP)` 标记 InCore 计算区域，outlining 将其抽成核内函数；外层循环及调用留在 Orchestration。作者可用 `pl.parallel` 组织工作项、用 `pl.spmd` 表达同一任务的多逻辑块分工，并按需通过 TaskId、`deps=`、`pl.submit`、`pl.manual_scope()` 控制任务依赖。默认自动依赖并不要求用户逐条写边；选择手工模式才承担相应责任。`decode_fwd.py` 已将投影、attention、residual/MLP 等这样连接，并管理中间 Tensor/scratch。Simpler 负责按依赖及可用资源派发，不会替用户决定所有算法分块，也不会把一个 SPMD 任务内部的每次 page 循环自动变成独立可调度任务。[C20（PyPTO3（Simpler））] [C6（PyPTO3（Simpler））] [C25（PyPTO3（Simpler））] 分工边界见第 6.13 节，动态 tiling 的不同位置见第 7.9 节。

4. **外部核内实现接入：保留任务编排，局部换成手写底层代码。**`@pl.jit.extern` / `external_source` 可以声明手写 CCE/C++ 核内入口，再由同一任务体系提交。该核内实现跳过 PTOAS，但必须满足 `kernel_entry(args)`、参数方向、核类型等 runtime ABI；这说明“核内实现”与“外层任务组织”可以分开选择，**不说明 Python DSL 自身已经能表达该外部代码里的全部指令**。当前 Group 还要求成员全是 extern 或全是 DSL，不能任意混搭。[C13（PyPTO3（Simpler））]

以 softmax 串起来看：用户可以选择 `ROW_TILE=64`，在一个 `pl.at(CORE_GROUP)` 内用 Tensor 写完 max → sub → exp → sum → div；编译器处理该区域的 Tensor→Tile lowering，外层行块任务由 Simpler 调度。需要更细控制时，再将核内局部改成显式 Tile/存储/同步写法。**“能编排五个 InCore”与“把五项计算融合在一个 InCore”是两种能力**：前者不自动获得后者的 UB/L1 中间值复用，`pl.scope()` 也不是核内融合 scope。完整代码及反例见第 8.3.3 节。

**PyPTO3 把 Tensor 写法、显式 Tile 写法和设备侧任务编排接在同一编程体系里，用户可按区域承担不同深度的实现责任**，但各层并非完全手工可控或完全自动。block版/CANNBot 同样有 Tensor/Tile 与低层选项；这里有区分度的是它们是否进一步接入这类通用设备侧任务依赖与调度体系，不能仅凭“也能写 Tensor 或一个完整 kernel”判断路线相同。

H/I 进一步说明“下探粒度”与“程序上界”并不成正比：H 暴露物理 layout tag、搬运与 MMAD/Vector，I 暴露 layout 代数、Copy/MMA atom 与线程/warp 资源；这些接口可以减少手写高性能 kernel 时表达硬件细节的障碍，但不自动提供 C 的跨计算入口任务编排。H 的 API 封装范围与 I 的架构/experimental 层仍应逐项核对，不能借整个 C++ 模板库或一个顶层命名空间推断覆盖率。[H-readme] [H-api-copy] [I-layout] [I-mma] [I-task]

### 2.11 已确认的具体边界：用拒绝/约束路径限定能力表

| 路线 | 本地实现中的具体证据 | 正确解释 |
| --- | --- | --- |
| PyPTO2-tensor版 | `conv` 对 transpose 分支明确报 `Conv transpose true is not supported yet.` | 有 conv API 不等于卷积的所有形态已贯通；不排除用其他分解表达同一数学结果。[A18（PyPTO2-tensor版）] |
| PyPTO2-block版 | `EmitVFExp` 检查源/目标类型一致、只收 FP16/FP32，并走 ZEROING-only 检查；precision 分支还展开额外指令 | `Vf.exp` 不等于“任意 dtype/mask 模式的一条 exp 指令”；显式低层 API 也有参数域。[B21（PyPTO2-block版）] |
| PyPTO3（Simpler） | A2/A3 backend 注册明确排除 `tile.matmul_mx*`、相关 scale/quant 操作；片上物理分配要求静态容量 | 语言中出现 MX API 不代表 A2/A3 可用；动态 GM/有效窗口与动态物理 Tile 不是同一能力。[C22（PyPTO3（Simpler））] [C10（PyPTO3（Simpler））] |
| CANNBot DSL | `reg.vstore_pack` 的 `b64_to_b32` 分支抛出 NotImplementedError，指明 arch3510 的 mask 约束 | 具体硬件/形式受限；不能把 raw-register 命名空间当完整 ISA 映射。[D19（CANNBot DSL）] |
| PyPTO on GPU | native tile 程序入口检查只有一个函数；静态 tensor 读取路径检查完全静态 shape；已有五 InCore + Orchestration 历史失败 | 这是当前集成支持域，不是 TensorIR/MLIR/GPU 在原理上不能承载多阶段程序。[E15（PyPTO on GPU）] [E2（PyPTO on GPU）] [E7（PyPTO on GPU）] |
| Triton-Ascend | `tl.dot(max_num_imprecise_acc=...)` 在 Ascend patch 中警告并忽略；`copy_from_ub_to_l1` 限 910_95、要求 buffer/地址空间/shape/dtype 匹配 | 同名 Triton API 的可用选项不必与 NVIDIA 相同；扩展能力也有 A2/A3/A5 分界。[F14（Triton-Ascend）] [F15（Triton-Ascend）] |
| AutoFuse + Inductor | 所选接入的 `_can_fuse_vertical_impl` 拒绝多处 indirect indexing 合并、matmul prologue；带 indirect indexing 的归约组合也有限制 | 这是当前融合路径的限制，不等于这些 PyTorch 运算不能分 kernel 执行，也不能推为 AutoFuse 所有其他入口都同样受限。[G12（AutoFuse + Inductor）] |
| H（CATLASS DSL） | `auto_sync="v0"` 限局部流水，不支持与显式 local flag/mutex 混用，extern/raw on-chip pointer 等也有约束；FA 的实际长度及 mask 参数未在设备计算体读取 | 自动同步不包含跨核或 `vec.func` 线程同步；参数名存在不证明本例动态长度/mask 生效。Q=1 的精度失败另见第 12.1.3 节。[H-auto-sync] [H-fa] [H-run-diagnostics] |
| I（CuTe DSL） | FP16 MLA `can_implement` 要求 latent=512、rope=64，Q=1..4、H≤128；H<128 时 split=1，page_size≠1 且整除 QK 的 N tile | 找到的是受约束分页 MLA，不能替任意 head dimension 的标准 GQA PA 验收；语言/其他示例的能力也不能被这一个 kernel 的条件一并限定。[I-mla] |

尤其是 paged attention：**gather、matmul、softmax 各自有 API，不足以证明 gather→QK→online softmax→PV 能一起生成一个高性能 kernel。** 要检查组合域、循环携带状态、矩阵/向量传递以及最终融合分组。第 2.6 节的完整原生代码与“仅外部算子/仅设计说明”才是更强的组合证据。

### 2.12 对使用客户与测试团队：把“表达得出”变成可验收的能力契约

建议将每项能力记为 `(语义, 表达层级, dtype/layout, 目标硬件, 动态契约, 实际执行路径)`，再分别记录 API、lowering、目标编译和运行验证状态。不要把“没有专项验证”填成“不支持”，也不要把“测试文件存在”填成“本次已通过”。

| 用户需求 | 至少应验收的组合 | 哪类接口差异会实际影响用户 |
| --- | --- | --- |
| 尾轴 softmax | 动态 M/N、非对齐尾块、reduce/广播、数值稳定性 | Tensor 形状/策略配置，或物理 tile 与 valid_shape/mask；不是五个算术名字齐全就完成 |
| 超长 reduce softmax | 有限局部容量下分段；小 M 时可选跨核 partial/merge/normalize | 谁表达阶段依赖、谁设 task/block 粒度、谁保证全局进展；会超出普通单 tile 例子 |
| Paged decode | 设备 `actual_seq_len`、间接 page 索引、GQA、QK/PV、online 状态、尾页 | 数学 API 可组合性、数据相关循环、Host 分工 metadata 更新、跨 CV 数据路径 |
| 新矩阵精度/新硬件形式 | 输入/accumulator/output dtype、scale 布局、舍入、目标指令 | 新增一个 dtype 枚举远远不够；可能需要前端→IR→codegen→底层工具链贯通 |
| 一层 Transformer | 上述组合加重分片、workspace/状态生命周期、阶段调度 | kernel DSL 的上界与程序 runtime 的上界必须分开；客户调用成本与库实现成本分别计算 |

从客户感知看，封装后的九条路线都可能只需一行函数调用。真正区分它们的是：**新增不支持的组合、新 shape 策略或硬件特性时，客户/算子作者/编译器团队分别需要改哪一层**。第 4.8—4.10 节把这一点落实到两组重点路线，第 10 章再讨论是否能替换或共用对应层。

---

在这个验收格式下，H 应分别记录“MMAD 动态逻辑尺寸”“连续 FA 的特化配置”“尚需实现的分页/异长”，I 应分别记录“通用 softmax”“受约束分页 MLA”“连续 GQA 两阶段”。客户同样只调用 attention，并不意味着这几项有相同输入 ABI、launch 结构或验证结果；H 的已测边界与 I 的源码边界见第 12 章。[H-run-results] [I-softmax] [I-mla] [I-gqa]

### 2.13 H/I 的前端与 API：layout 相似性要具体到映射对象

| 能力族 | H：CATLASS DSL | I：CuTe DSL |
| --- | --- | --- |
| 编程入口 | `@tla.kernel` → `tla.compile` → 调用产物；`@tla.jit` 是编译时内联的设备 helper | `@cute.kernel` 定义设备计算，`@cute.jit` 可组织 Host 调用/launch，`cute.compile` 生成可重复调用对象 |
| tensor / layout | `make_shape/make_stride/make_layout`、`origin_shape`、`make_tensor_like/tile_view`；RowMajor/ColumnMajor/zN/nZ/L0C 等 tag | shape/stride 嵌套 layout，`composition/logical_divide/logical_product/coalesce/local_tile`；还表达 thread/value partition |
| 搬运与矩阵 | `copy` 按源/目标地址空间和 tag 选择路径，`mmad`；MX scale 在 L1→L0 load 关联 | Copy/MMA atom、TiledCopy/TiledMma、线程切片；架构特定 cp.async/TMA、MMA/TCGen05 等 |
| 向量与归约 | `vec.func` 中寄存器 load/store、mask、算术、reduce/cast；显式 SIMD/SIMT 区域能力按具体 API/目标核对 | 线程/warp 索引、寄存器片段、shuffle、reduce、shared/CTA 同步；另有实验性 primitives / cute extension |
| 缓冲与同步 | `allocate`、flag、cross_flag、mutex；`auto_sync="v0"` 有明确局部约束 | shared/TMEM 分配、mbarrier、pipeline；实验性 TS 声明资源与 schedule |
| 程序组织上界的直接证据 | 完整核内 FA、mixed GEMM、StreamK；未见与 A/C 同定位的通用整层任务 runtime | Host 多 launch、persistent tile 和 warp 专门化 TS；未据这些例子证明整层通用任务 runtime |

来源：[H-dsl] [H-compile] [H-api-layout] [H-api-copy] [H-api-mmad] [I-dsl] [I-layout] [I-mma] [I-copy] [I-task]。这张表是能力族及其责任位置，沿用第 2.10 节的四层验收方法，不表示 ISA 覆盖完整。

H 的 `shape/stride` 描述物理存放，`origin_shape` 描述逻辑有效范围；例如 zN 的嵌套物理 shape 不能随意写成普通二维 shape 后仅改 tag。I 则进一步把 CuTe layout 代数用于线程/值与 MMA/copy 分区。两者都让 layout 成为程序的一部分，但 **“有 layout 对象”并不意味着拥有同样的变换代数或自动线程分配能力**。[H-api-layout] [I-layout]

H 的 `allocate` 当前只接受静态容量和片上地址空间；`auto_sync="v0"` 只生成单 AIC 或 AIV 内的流水同步，跨核 flag、`vec.func` 内线程同步仍须显式写。I 的架构特定 API 及例子也有 target、对齐、dtype 和 CTA/cluster 限制；Blackwell MLA 的约束见第 2.6 节。本次没有因 API 可导入或源文件存在而把它们全部标为“可运行”。[H-api-allocate] [H-dsl] [I-mla]

F 的 Inductor 组合调度中的 CATLASS template 路径生成 C++ 模板代码，与 H 的 `catlass.tla` Python DSL 是两个具体入口。不能从 F 能选择 CATLASS 模板，直接推断 H 已接入同一套 Inductor 自动融合。[F10（Triton-Ascend）] [H-inductor-template]

最后看 Host 写法。H 的 basic mixed 示例先桥接好四个 tensor，再显式编译和执行；以下为其 Host 调用摘录，完整输入/golden 在源文件中。[H-mixed]

```python
artifact = tla.compile(
    basic_mixed, a_tensor, b_tensor, c_tensor, d_tensor,
    options="--npu-arch 3510",
)
artifact(a_tensor, b_tensor, c_tensor, d_tensor, block_num=block_num)
torch.npu.synchronize()
```

I 的 softmax 则可将 kernel 和 block size 作为编译期参数传给 Host JIT launcher。下列保留实际签名与 launch 的整理片段依赖原文件的 imports 和 kernel 定义；一次这个 launcher 调用只启动其所选 kernel，不能推及其他多 launch 的 JIT 函数。[I-softmax]

```python
@cute.jit
def softmax_block_per_row(
    inp_tensor: cute.Tensor,
    out_tensor: cute.Tensor,
    N: cutlass.Constexpr,
    C: cutlass.Constexpr,
    Kernel: cutlass.Constexpr[Callable],
    BlockSize: cutlass.Constexpr,
):
    Kernel(inp_tensor, out_tensor, N, C).launch(
        grid=(N, 1, 1), block=(BlockSize, 1, 1),
    )
```

这两种写法都可封装成客户的一行调用；差异在编译/调用分界及 kernel 作者能控制的对象，不能只按装饰器和调用行数评价易用性。

<a id="layers"></a>
## 3. 把所有层次拆开：名称相似不等于承担相同责任

### 3.1 全栈分层图

```text
L0  客户数学 / 模型表达：Python Tensor、kernel DSL、PyTorch
 ↓
L1  框架接入：Dynamo / FX / Inductor、算子注册、外部库边界
 ↓
L2  程序划分：算子融合、task / InCore 拆分、scope、控制流
 ↓
L3  逻辑 IR：Tensor / tile / SSA / graph / symbolic shape / effects
 ↓
L4  算法与执行空间：tiling、loop、reduction 分解、program / core mapping
 ↓
L5  局部资源：bufferization、layout、UB/L1/L0/shared/register 分配与复用
 ↓
L6  局部时序：搬运流水、double/multi-buffer、事件、屏障、指令调度
 ↓
L7  目标代码：PTO/AscendC/CCE 或 HIVM/LLVM、CUDA Tile IR、二进制
 ↓
L8  kernel ABI 与 launch：参数、shape/stride、block/grid、stream
 ↓
L9  跨 kernel / task：依赖、就绪队列、设备派发、完成通知、资源组
 ↓
L10 GM / workspace 全生命周期：框架张量、跨 task buffer、kernel scratch

L10 不是最后才执行：它横跨 L1/L2/L4/L5/L8/L9。
静态编译时 schedule 与 L9 的运行时派发也不是同一件事。
```

这张图是责任分层，不是所有系统都必须经过的统一 pass 顺序。A（PyPTO2-tensor版）/C（PyPTO3（Simpler）） 的 L2/L9 紧密耦合；B（PyPTO2-block版）/D（CANNBot DSL）/F（Triton-Ascend） 可把很多控制和合作直接写进 L4/L6；G（AutoFuse + Inductor） 则把大量 L2/L4 决策交给编译器。

### 3.2 用户与编译层责任矩阵

| 层 | A（PyPTO2-tensor版） | B（PyPTO2-block版） | C（PyPTO3（Simpler）） | D（CANNBot DSL） | E（PyPTO on GPU） | F（Triton-Ascend） | G（AutoFuse + Inductor） | H（CATLASS DSL） | I（CuTe DSL） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L0 客户入口 | Tensor DSL | Tile / VF / Cube DSL | Tensor / Tile、多级 scope | Python Host / kernel / Channel | PyPTO DSL + operator API | Triton DSL；或 PyTorch | 主要是 PyTorch；开发者 ASCIR 接口 | Python TLA、layout/存储/Vector/MMAD | Python CuTe、layout/atom/线程分区 |
| L1 框架入口 | 自有 JIT/运行接口；不据此认定 Inductor 原生后端 | 直接 kernel 接口 | 自有 JIT / model zoo | JIT/AOT Host 接口 | 集成包 `compile_graph` | 指定 torch_npu Inductor 路径 | 指定 TorchAir experimental Inductor 路径 | `tla.compile` + tensor bridge/直接调用；不是已验证的 Python DSL Inductor 后端 | CuTe JIT/compile + tensor bridge；本次无框架集成实测 |
| L2 分组与程序 | TileFwk 图 / 函数 / task 编译 | 用户 kernel 边界及核内 passes | InCore / Orchestration / Group / SPMD | Host 与 Device 分阶段 | 当前支持的单图模式；包装层可多 launch | Inductor 融合或用户独立 kernel | Inductor 分组 + AutoFuse schedule | 显式 kernel + 内联设备 helper；mixed 函数再拆核种 | Host JIT 与 kernel launcher；一个 Host 可多 launch |
| L3 主 IR | PIL / 自有 IR → TileFwk | 共享基础自有 IR + block版 操作 | 自有 IR → PTO MLIR | MLIR CANNIR / Asc / vector 等 | 自有 IR → NVIDIA TensorIR | TTIR → 适配 MLIR → HFusion/HIVM 等 | ASCIR / FusedScheduledResult | TLA MLIR → HIVM/AVE/标准 dialect | CuTe/相关 MLIR → NVVM/CUDA 编译链 |
| L4 tiling / 映射 | tile 设置 + 编译分解 | 物理 tile、block-stride、Host 策略 | 编译 tile + runtime task；可手写 SPMD | tile 与 Host/block 分工显式 | schedule tile / pattern → grid | constexpr tile + grid + 后端再映射 | 自动模板 / cost model / ATT tiling | Host 选 tile/常量，kernel block-stride/有效范围 | tile/atom/thread-value layout；静态 persistent 或 CLC 等 |
| L5 片上规划 | 图编译及核内 codegen | 显式地址/TileGroup + passes | PyPTO memory/layout passes + PTOAS | Buffer / Channel + MLIR passes | TensorIR / Tile 编译器 | AscendNPU-IR PlanMemory/layout | buffer/queue allocator、reuse、代码生成 | 静态 allocate + 按空间对齐递增偏移；非此 pass 自动 last-use 复用 | 显式 SMEM/TMEM/fragment；TS 可声明 phase alias |
| L6 流水与同步 | 编译生成及用户策略 | section、mutex、pipeline、显式同步 | InCore passes / PTOAS；可显式混合核事件 | Channel、软件流水、同步推断 | GPU lowering / tile compiler | CV pipeline、MultiBuffer、InjectSync/GSS | schedule、queue、生成的同步与模板 | flag/mutex/CV 事件；受限局部 AutoSync | pipeline/mbarrier/warp 角色；TS 生成已声明协议 |

“用户显式”不等于“编译器不优化”，“自动”也不等于“客户无需理解资源限制”。这些词表达的是默认责任分工。

H/I 的分层依据是各自前端、pipeline 与局部分配实现：[H-dsl] [H-passes] [H-scratch]、[I-dsl] [I-layout] [I-ts-memory]。其中“作者选策略”与“编译器生成指令/协议”可以同时成立，不宜压缩成全手工或全自动两档。

### 3.3 二进制、启动与全局运行层矩阵

| 层 | A（PyPTO2-tensor版） | B（PyPTO2-block版） | C（PyPTO3（Simpler）） | D（CANNBot DSL） | E（PyPTO on GPU） | F（Triton-Ascend） | G（AutoFuse + Inductor） | H（CATLASS DSL） | I（CuTe DSL） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L7 常见末端 | CCE codegen + 工具链 | CCECodegen + 工具链 | PTOAS 默认 C++ 路径 + PTO tile 库 / 工具链 | AscendC source + 工具链 | TensorIR → CUDA Tile IR → Cubin | bishengir/HIVM 及目标编译器 → npubin | 生成 AscendC / 相关模板 → 工具链 | TLA lowering + CANN `hivmc-a5` → device object/产物 | CuTe 编译 pipeline/配套编译组件 → cubin |
| L8 launch | CTRL/SCHE/AICore 组合，视配置 | `kernel<<<blockDim,...,stream>>>` | runtime worker + `kernel_entry(args)` task ABI | Host C++ 的 `kernel<<<blockDim,0,stream>>>` | `cuLaunchKernelEx`，显式 grid / artifact block / stream | `cann_launch_kernel(func,blockNum,stream,...)` | wrapper 调 tiling、分配 workspace、调用 launch | artifact 参数打包、block_num/stream → AscendCL | kernel launcher `.launch(grid, block, …)` → CUDA |
| L9 跨任务主链 | TileFwk 设备任务系统 | 没有默认通用 ready-queue scheduler | Simpler，含多 block task 与依赖 | 没有默认 Simpler 式 scheduler | CUDA kernel/stream 与硬件 CTA 调度 | Host / stream；非 AICPU 任务图主链 | Inductor Host/stream；非 AICPU 任务图主链 | Host/stream；StreamK 工作分解不等于通用任务图 | Host/stream + CTA 调度；CLC/TS 是 kernel 工作/资源层 |
| L10 kernel 外 GM | 图/runtime 内存计划 + 用户 tensor | 调用者/框架分配 + 传参 | tensor planner / scope/runtime + 用户显式 GM | Host/调用者；AOT contract | torch 集成分配，shape 固化 | 调用者或 Inductor；wrapper 分配内核 scratch | Inductor buffer + wrapper workspace | 调用者/框架 tensor；算子显式 workspace/partial | 调用者/框架；例如 GQA split partial 后归约 |
| 跨核协调 | runtime依赖 + kernel内部协议 | mixed kernel / 显式同步 | runtime依赖 + SPMD内部事件/屏障 | Channel / mixed-core 同步等 | CTA内同步；跨CTA必须另有合法机制 | 编译器生成CV/跨block同步等 | 生成的核内/跨核协议与workspace | mixed AIC/AIV、cross-core flag；仍需独立进展协议 | warp/CTA/cluster/pipeline；不隐含任意 grid 全局屏障 |

“没有默认 ready-queue scheduler”只描述选定主链，不宣称永远无法接入任务系统，也不等于没有 runtime。

H 的产物、参数与 AscendCL 调用见 [H-execution] [H-runtime]；I 的编译执行及 CUDA launch 见 [I-dsl] [I-executor]。两者都需将局部合作协议和 kernel 外依赖分开；H 的 mixed entry、I 的 TS/CLC 不自动填补 L9 的模型任务 runtime。[H-mixed-pass] [I-task] [I-dynamic]

### 3.4 H/I 放回同一责任分层

| 责任层级 | H（CATLASS DSL） | I（CuTe DSL） |
| --- | --- | --- |
| L0/L1 客户与框架 | 显式 Python kernel；例子由 torch_npu 分配张量，再桥接 TLA | 显式 Python kernel/Host JIT；例子用 CUDA tensor/DLPack；不能据此宣称自动编译整个 PyTorch 模型 |
| L2/L3 程序/IR | 作者选定 kernel 边界；Python lowering 生成 TLA/标准 MLIR | 作者选定 Host/device 边界；CuTe、GPU/标准 MLIR，扩展路径另有 LIR/PyIR 选项 |
| L4 工作分解 | task/block 循环、tile、StreamK 等由例子/策略编写 | CTA/warp/tile、静态或 CLC persistent，TS 安排 warp 内部任务 |
| L5/L6 局部资源/时序 | 物理空间、layout、容量、flag/mutex；TLA passes 做 lowering 和受限自动同步 | layout 分区、SMEM/TMEM、pipeline/mbarrier；TS 生成已声明资源的同步协议 |
| L7/L8 代码/launch | TLA passes → AscendNPU-IR dialects → `hivmc-a5` → `kernel.o`；PyACL load/launch | `cute-to-nvvm` → cubin；JIT executor/CUDA runtime 处理参数、装载和 launch |
| L9/L10 跨任务/全局存储 | 所查路径没有 A/C 式通用设备任务 DAG；workspace 仍由 Host/算子安排 | Host 多 kernel 顺序与片内 task graph 分开；partial/workspace 不因 persistent 自动消失 |

来源：[H-passes] [H-execution] [H-runtime] [H-streamk] [I-dsl] [I-runtime] [I-task] [I-gqa]。特别是 I 的 TS task 主要属于 L4/L6 的单 kernel 协作抽象，不能仅因名字有 task 就直接填进 A/C 的 L9。

## 4. 九条实际调用链与组件边界

### 4.1 A（PyPTO2-tensor版）：新 Python IR 入口，仍接入 TileFwk 执行体系

```text
@pypto.frontend.jit
 → JitCallableWrapper 参数 / shape / cache
 → 当前默认 new_ir=True：compile_new
 → PIL compile pipeline / IRBuilder / finalize_dynamic_function
 → FinalizeDynamicFunction → TileFwkFinalize
 → CompileTask queue / Program / graph compilation
 → GetCodeGenCCE(...)->GenCode(...)
 → KernelModule / LaunchKernelTorch / DeviceLauncher
 → AICPU CTRL、可选独立 SCHE、AICore 执行程序
 → 依赖解析 / ready queue / task dispatch
```

源码：[A1（PyPTO2-tensor版）]—[A6（PyPTO2-tensor版）]、[A9（PyPTO2-tensor版）]、[A10（PyPTO2-tensor版）]。

当前实现的 `new_ir=True` 不能忽略；也不能据“新 SSA 前端”推断其底层已更换为 MLIR/PTOAS。`IsAicoreResolveEnabled()` 会改变是否启动独立 SCHE 路径，不能把某一运行配置的 launch 拓扑当所有配置的事实。

### 4.2 B（PyPTO2-block版）：与 A（PyPTO2-tensor版） 共享基础设施，不共享主执行模型

```text
@pypto_pro.language.jit
 → AST / shape policy / 自有 IR
 → block版 核内 passes、mutex/pipeline 等
 → CCECodegen.generate_single(...)
 → kernel.cpp + Host call_kernel.cpp → .so
 → ctypes call_kernel
 → kernel<<<blockDim,nullptr,stream>>>(...)
```

源码：[B1（PyPTO2-block版）]—[B6（PyPTO2-block版）]、[B8（PyPTO2-block版）]。A（PyPTO2-tensor版）/B（PyPTO2-block版） 共享同仓 `pypto.pypto_impl`、原生 IR 及部分编译基础设施；block版 的 bootstrap / IR 导入对此是直接证据。它们的 Python 前端、编程抽象、codegen 路径、参数契约、运行时入口仍不同。

`pypto` loader 加载了某个包含 runtime 的库，不等于 block版 launch 时执行 A（PyPTO2-tensor版） 的 AICPU scheduler。B（PyPTO2-block版） 的 runtime 仍负责 JIT、缓存、参数、TilingData、stream 与错误处理，因此“完全没有 runtime”也不准确。

### 4.3 C（PyPTO3（Simpler））：任务图编译与核内编译分开，库层可以主动选择融合

```text
pypto-lib 算子 / 模型
 → @pl.jit / inline / program，parallel/pipeline/spmd，Tensor/Tile/TaskId
 → PyPTO3 自有 IR + 多层 lowering
 ├─ InCore：PTO dialect MLIR → ptoas → 默认 C++ 产物
 │          → Simpler kernel_entry(args) wrapper → 核内设备代码
 └─ Orchestration：C++，动态shape、循环、tensor创建、submit/依赖
 → SimplerWorker
 → 任务构建/依赖、AICPU调度及AIC/AIV worker执行
```

源码：[C1（PyPTO3（Simpler））]—[C8（PyPTO3（Simpler））]、[C13（PyPTO3（Simpler））]。`pypto-lib` 的 `decode_fwd.py`（含 `_decode_layer`）有跨层模型程序、显式依赖、SPMD kernel 和 scratch 管理：[C20（PyPTO3（Simpler））]。这为程序 / task 级 megaKernel 路线提供了比教学例子更具体的实现证据，但仍不能据此宣称一个物理 kernel 覆盖整个模型。

PTOAS 编的是核内函数，不会仅凭输入 PTO MLIR 就自动得到上层所有 TaskId、scope 与 Simpler GM 生命周期。任务运行协议在 C（PyPTO3（Simpler）） 的 Orchestration/runtime 层。

### 4.4 D（CANNBot DSL）：Host / Device 都可进入分阶段编译，不是只编设备函数

```text
Python 调 @jit Host 函数
 → TensorSpec / Dim / HostCallPlan / guard
 → 源码 analyze/lower/materialize（含 AST 分析与改写）
 → Python tracing / MLIR builder
 ├─ @kernel 设备代码：CANNIR / buffer / Channel / layout / vector ...
 └─ @jit Host 控制流：kernel_launch、参数及工作空间组织
 → MLIR passes → TranslateToAscendC
 → AscendC/Host C++ → .so
 → native invoke / CANN launch
```

源码：[D1（CANNBot DSL）]—[D4（CANNBot DSL）]、[D6（CANNBot DSL）]—[D9（CANNBot DSL）]。允许 Host 中出现多个 kernel launch，不代表 Device kernel 里也允许随意调用另一个全局 kernel；调用矩阵有明确边界。

D（CANNBot DSL） 的 `aicpu/kernel.py` 提供独立 AICPU kernel 的编译/launch能力 [D5（CANNBot DSL）]，但没有因此自动得到 A（PyPTO2-tensor版）/C（PyPTO3（Simpler）） 的通用任务图 scheduler。这个区分也适用于所有“支持编写 AICPU 代码”的系统。

### 4.5 E（PyPTO on GPU）：TensorIR 是核内后端，不是 Simpler 的 GPU 替身

```text
operator builder / PyPTO JIT specialize
 → compile_graph：目标、schedule、tile、layout选项
 → PyPTO 自有 IR
 → 当前支持的 NVIDIA emitter / pattern 分流
 → BuildTypedTensorIRModule：真实 MLIR Module
 → prepareToArtifact / CUDA Tile IR / assemblePreparedArtifact
 → Cubin + ABI/launch metadata
 → NvidiaExecutable / prepare launch packet
 → Driver API cuLaunchKernelEx(stream,grid,block,...)
 → GPU 硬件按资源安排 CTA 到 SM
```

源码：[E1（PyPTO on GPU）]—[E6（PyPTO on GPU）]。集成层 `launch_graph` 验证实际 shape/stride 与编译规格一致；底层 ABI 的部分动态 size / stride / grid 机制，不等于高层已经开放通用动态 tensor 程序。

`NVIDIA/tensor-ir` 与 TVM 中也称 TensorIR 的组件不是同一对象；本文指本地 vendored NVIDIA 实现及其 PyPTO bridge。

### 4.6 F（Triton-Ascend） → AscendNPU-IR，存在硬件分支与编译器桥接

```text
直接 @triton.jit
 或 torch.compile → torch_npu/_inductor → NPUTritonScheduling / codegen
 → TTIR
 → ttir_to_linalg：结构化、Triton→HIVM/HFusion/Linalg 等适配
 → ttadapter
 → triton-mlir-opt → mlirbc
 → bishengir-opt → bcmlir
 → bishengir-compile
    ├─ A2/A3 编译分支
    └─ A5/910_95 编译分支
 → npubin + task_type/workspace/sync-lock 元数据
 → CANN runtime 注册和 launch wrapper
```

另有 `is_pure_simt` 分支：TTIR 直接走对应 `ttir_to_npubin`，不能强画成同一条 SIMD / Cube-Vector lowering 链。源码：[F1（Triton-Ascend）]—[F4（Triton-Ascend）]、[F7（Triton-Ascend）]—[F10（Triton-Ascend）]。

边界要点：

- CMake 明确依赖 AscendNPU-IR；构建可用 `ASCENDNPU_IR_SRC_DIR` 指定源码。
- 中间存在 MLIR bytecode / 适配工具桥，不应因为双方都叫 MLIR 就默认二进制或 dialect 版本兼容。
- 运行时实际查找的 `bishengir-opt / bishengir-compile` 可来自 CANN 安装或环境配置；不一定由旁边 `npu/AscendNPU-IR` 的 HEAD 构建。
- `NPUCombinedScheduling` 实际可在 Triton 与 CATLASS 等节点后端间选择；不能把该 Inductor 入口的所有 matmul 或整个模型都归为 Triton 自生成 kernel。
- `taskqueue` / `async_launch` 在此主要是 Host 侧提交机制，不是 Simpler 式 AICPU ready queue。编译器的 GraphSyncSolver 也是同步插入，不是设备动态任务调度器。

### 4.7 G（AutoFuse + Inductor） 生成 kernel，也生成 Host tiling

```text
客户 PyTorch
 → torch.compile(options={"npu_backend":"ascendc"})
 → TorchAir experimental Inductor extension（安装为 torch_npu 的 ascendc 后端）
 → Inductor loop/sizevar/buffer、融合组、模板/extern 选择
 → ASCGraph / FusedASCGraph
 → Autofuser.schedule / Optimize
 → FusedScheduledResult
 → Codegen.GenerateForInductor
    ├─ tiling_data：结构定义
    ├─ tiling：Host策略/solver/分支代码
    └─ kernel：AscendC设备代码
 → autofuse.compile_adapter
 → host/device/wrapper编译与链接
 → wrapper：tiling_fn → workspace分配 → launch(block_dim,current_stream,...)
```

源码：[G1（AutoFuse + Inductor）]—[G8（AutoFuse + Inductor）]。`Autofuser::Codegen` 明确返回 `result.tiling_data / result.tiling / result.kernel` 三份代码；不是客户自己补写每个融合算子的 Host tiler。

静态条件满足时有 Top-N tiling / variant 选择；其他分支编译相应 JIT 产物。生成的动态 wrapper 会调用 `tiling_fn` 获取 tiling data、workspace_size 和 block_dim。Host taskqueue 配置可能改变这些工作在队列外还是队列内执行，但不把它们变成 AICPU 动态任务图。

G（AutoFuse + Inductor）的客户入口是 PyTorch，ASCIR Python/C++ API 面向集成开发者；它与独立 kernel DSL 的差别见第 2.9 节。

<a id="focused-comparisons"></a>
### 4.8 重点对比：PyPTO2-tensor版与 PyPTO3（Simpler）

**最接近之处在外层程序/任务执行，主要差别在“用户怎样指定分解与协作”以及具体编译/runtime 契约；不能用“自动 tiling 对手工 tiling”二分。**

#### tiling 与合图：tensor版的用户控制范围

tensor版文档给出的原生配置如下。这是嵌入 Tensor 程序/配置作用域的 API 摘录，不是独立可执行的 kernel：[A8（PyPTO2-tensor版）] [A8b（PyPTO2-tensor版）]

```python
pypto.set_vec_tile_shapes(1, 1, 8, 8)
pypto.set_cube_tile_shapes(
    [16, 16], [256, 512], [128, 128], enable_split_k=False
)
```

第一个调用指定四个 Vector 维度的 tile。第二个的三个列表依次对应 M、K、N，每个列表是 `[L0,L1]`：例如 K 的 L0 tile 是 256、L1 tile 是 512。`enable_split_k=True` 则开放多核切 K 策略，源码接口还区分 GM 累加与不启用 split-K 的累加方式。它不是“由调度器随便决定 tile”的空提示，也不能不经编译产物就把某个 tile 设置换算成固定 task 数或固定物理核分配。

而且，tensor版不只是设置尺寸。真实 attention 程序里有以下连续片段：[A13（PyPTO2-tensor版）]

```python
pypto.set_pass_options(sg_set_scope=3)
sij_scale = pypto.mul(sij, softmax_scale)
tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)
tilda_pij = pypto.exp(pypto.sub(sij_scale, tilda_mij))
sum_update[:] = pypto.sum(tilda_pij, dim=-1, keepdim=True)
max_update[:] = tilda_mij
pypto.set_pass_options(sg_set_scope=-1)
```

这里用户明确划出一个合图作用域。配置定义描述相邻、同非默认 scope 的操作合并；`Function::AddRawOperation` 将 scope 写入操作，`SuperNodeGraphBuilder` 收集/合并 scope 并检查 Cube/Vector 平台约束。[A14（PyPTO2-tensor版）]—[A16（PyPTO2-tensor版）] 因而 **“tensor版连计算子图边界都不能干预”也不成立**。不过，编译 scope、一个可执行函数、一个动态 task 实例和一个物理核并非一一对应，最终仍需检查图展开与 runtime 任务描述。

#### 用户逻辑感知：同样“切块”，控制对象并不完全相同

| 用户要控制什么 | PyPTO2-tensor版 | PyPTO3（Simpler） |
| --- | --- | --- |
| 数学计算 | Tensor 算术、matmul/reduce、切片/赋值；普通 softmax 也是五步数学运算 | Tensor/Tile API；教学 softmax 也可保持 Tensor 风格，不必手写任务提交 |
| 算法级分块 | `pypto.loop`、`view`、`valid_shape`，例如 PA 的 batch/group/sequence 分段 | `pl.parallel/range`、切片、`pl.at` 等；PA 可以自己组织请求/head/page 循环 |
| 编译 tile 几何 | `set_vec_tile_shapes`、Cube `[L0,L1]`、split-K 等，编译器落实分解 | 由 Tensor/scope 的分块与 lowering，或直接 Tile 形状、memory/layout/slot 表达落实；不是另一套同签名的 `set_vec_tile_shapes` |
| 计算区域/任务粒度 | 函数、loop 与 `sg_set_scope` 等策略控制图划分/合并；compiler/runtime 继续生成任务实例 | 显式 InCore/Orchestration/Group/SPMD；可用 `submit/spmd_submit` 为任务命名并传依赖 |
| 跨阶段依赖 | 主要由 Tensor 数据关系、程序结构和编译/runtime 生成；合图/复用配置也影响最终图 | 自动模式按数据关系生成；manual 模式下 TaskId、deps、scratch-ready 等成为用户显式责任 |
| 核内性能细节 | 借 tile/pass/reuse 等配置影响核内生成；常规 Tensor 路径不以固定地址 TileGroup 为编程单位 | 可继续下探 Tile/MemRef/slot/system；或接入 extern kernel。不能据此认为每条硬件指令都开放 |
| 多核任务形态 | compiler/runtime 根据分解与任务协议派发，用户通过分块和策略影响并行性 | 可显式将一个 SPMD dispatch 定为多个 logical blocks、一个 TaskId；仍不是永久绑定物理核 |
| 动态长度/shape | DYNAMIC、符号 shape、设备程序循环及有效窗口；仍需 tile/策略/缓存契约 | 动态 Tensor/有效窗口、设备读取和任务编排；物理片上容量仍受静态/资源约束 |
| 从“能跑”到“很快”的排错对象 | Tensor 图→tile/子图/task、核内代码、依赖/dispatch、GM 图 | 层次 scope→InCore/Orchestration→PTOAS/Simpler；专家模式还需核对自己写的依赖/传递协议 |

对应证据既有tensor版的完整组合程序 [A13（PyPTO2-tensor版）]，也有 PyPTO3 的普通 softmax、强融合 PA 和模型程序 [C15（PyPTO3（Simpler））]—[C20（PyPTO3（Simpler））]。**不能用tensor版的简短教学程序，去对比 PyPTO3 专家手写的整层代码，再把代码长度或可见细节误认为框架的全部能力。**

#### 实际架构：相同的是责任层，不同的是中间契约

```text
PyPTO2-tensor版
  Tensor/PIL + tile/scope 策略
    → TileFwk 图分解、编译、任务描述
    → CCE 核内代码 + TileFwk 控制/依赖/worker 协议

PyPTO3（Simpler）
  Tensor/Tile + 层次 scope + 可选显式 TaskId/SPMD
    → 自有 IR 的 InCore / Orchestration 分路
    → PTO MLIR/PTOAS 核内代码 + Simpler 提交/依赖/worker 协议
```

这解释了为什么二者整体相近，却不能互换某个 scheduler 库就完成迁移。任务参数 ABI、shape/stride、读写效应、完成/提前发布、GM 生命周期、异常协议都跨越编译器与 runtime；PTOAS 本身不承担这些上层协议。第 10 章的适配范围仍然适用。

**megaKernel 的区别在性能协同接口，不只是 scheduler 是否存在。** PyPTO3 库直接示范“强融合 PA + 显式依赖 + 多阶段程序”；tensor版更多通过图编译配置及任务体系优化。专家控制与维护责任同时增加，性能需按第 6.13 节的两组对照验证。

### 4.9 重点对比：PyPTO2-block版 与 CANNBot DSL

**两者相近的是“用户编写资源受约束的 kernel，并主动安排 tile、数据通路和时序”；差异不止后端格式，还包括 Python 分阶段语义和缓冲协作契约。**

#### 先看相同 softmax 中最能体现用户差别的一段

PyPTO2-block版 的完整 softmax（第 2.3 节）从 TileGroup 取得轮转 tile，再发 load。下面把声明与使用处摘在一起；`tile_type/VA_IN*/valid_rows/cols` 等均在原 kernel 中定义，`auto_mutex=True` 也是该例子的前提：[B14（PyPTO2-block版）]

```python
in_group = pl.make_tile_group(
    type=tile_type, addrs=[VA_IN0, VA_IN1], mutex_ids=[0, 1]
)
in_slot = in_group.next()
pl.set_validshape(in_slot, [valid_rows, cols])
pl.load(in_slot, x, [row_off, 0])
```

CANNBot DSL 的完整 softmax 先拿到生产槽，搬入并提交，再等待可消费槽，用完释放；以下是原例子中输入数据通路的摘录，中间计算见第 2.3 节：[D11（CANNBot DSL）]

```python
x = self.ch_x.acquire()
mem_copy(x, gm_x_tile)
self.ch_x.commit(x)
x_r = self.ch_x.wait()
```

原 kernel 在最后一次使用 `x_r` 后执行 `self.ch_x.release(x_r)`。这两种写法都在解决缓冲复用与异步访问安全，但不能逐词替换：

- **block版 TileGroup** 是带轮转 cursor 的 tile 集合，支持 `next/current/previous` 和下标；parser 将其展开为 IR handle，并保留 tile↔mutex 元数据供 `auto_mutex` 推断/插入同步。[B18（PyPTO2-block版）] [B19（PyPTO2-block版）]
- **CANNBot Channel** 是带读写双 cursor 的有界 ring：生产者 `acquire→commit`，消费者 `wait→release`；这些操作进入 CANNIR。Buffer 用于不需要 ring 协议的局部存储；Channel/Buffer 的地址可以由 pass 规划，且存在固定地址选项。[D16（CANNBot DSL）]
- block版 的 `.next()` **本身不能解释为** Channel 的“等待生产完成”；CANNBot 的 `wait()` **也不是** Simpler 的“从 task ready queue 取一个核间任务”。Channel 还可表达特定跨核通道，但与通用设备任务图调度仍属于不同协议层。

#### 用户使用逻辑与实际架构逐项对齐

| 维度 | PyPTO2-block版 | CANNBot DSL | 用户真正感知到的差别 |
| --- | --- | --- | --- |
| 编译对象/入口 | `@pl.jit` kernel，普通 Python Host 使用 `[stream, blockDim, ...]` launch | `@jit` Host 调 `@kernel[block_dim]`；Host/Device 形成分阶段编译链 | block版 示例的 Host 策略常留在 Python；CANNBot 可把 Host 控制流和 launch 一起生成 native 产物 |
| Python 语义 | ASTParser 遍历受支持 DSL，构造自有 IR；声明、常量计算和运行时值有区分 | 先分析/改写源码，再 materialize 为生成 MLIR 的可调用体；helper 可 trace-time inline | **两者都有前端分析，不是“一个有 Python，另一个只有构图”或“AST 与完全无 AST”的对立** |
| 值/存储操作 | `pl.add(out, lhs, rhs)` 等 out-first Tile 操作；TileType 指定 memory/layout，make_tile/group 绑定地址 | Tensor/view/Buffer/Channel 加 out-first math；逻辑/物理 layout、搬运格式是显式对象 | 二者都不像单纯的高层纯函数 Tensor 图；需理解写入、别名与存储 |
| 缓冲复用/同步 | 用户选择地址/槽/mutex；auto_mutex/pipeline 基于操作效应等落实局部同步 | 用户声明 Buffer/Channel 容量、depth、通道类型，并按协议使用；pass 分配/降低同步 | 控制意图相近，承载它的 IR、生命周期和验证规则不同 |
| Cube/Vector 与寄存器 | section_cube/section_vector、Tile matmul、`pl.Vf`/system 等 | Cube 路径、VF/`reg.*`、copy engine、Channel 等 | 都可下探核内硬件工程；各自 API/目标约束仍不同，见第 2.11 节 |
| 多核分工 | 核内读取 block/subblock 标识，用户编 loop；Host 选 blockDim/策略 | 核内读取 block/subblock 标识，用户编分工；Host launch 指定 block_dim | 没有 AICPU DAG 也可以用多核；硬件运行 block，不替用户分解业务工作 |
| 动态 shape | DYNAMIC 参数、静态物理 tile+valid_shape、Host 分工参数/可选 TilingData | JIT/AOT TensorSpec/Dim、动态参数和有界资源；策略可留 Host 或编入 Host 部分 | 两者都不因 shape 变化必然要求客户新增一个独立 tiling 函数 |
| 主 IR/后端 | 共享 PyPTO2 基础 native IR 上的 block版 操作 → CCECodegen | CANNIR/其他 MLIR dialect → passes → AscendC/Host C++ | 不是把最终打印格式换一下即可互用 passes 或二进制 |
| 外层设备调度 | 本次主路径是直接 kernel launch，不走tensor版 AICPU task 图 | 本次主路径是 Host launch；能编 AICPU 函数不代表有通用 scheduler | 两者若要任务式整层执行，都要额外完成外层运行协议/集成 |

源码：[B1（PyPTO2-block版）]—[B4（PyPTO2-block版）] [B17（PyPTO2-block版）]—[B20（PyPTO2-block版）]、[D1（CANNBot DSL）]—[D9（CANNBot DSL）] [D16（CANNBot DSL）]—[D18（CANNBot DSL）]。

#### 相近思路落实到 softmax、PA 和 megaKernel 的代价

尾轴 softmax 中，两者都能将五步数学计算放进一个 kernel，算子作者承担 tile 容量、尾块、局部复用和多核分工。前面 block版 示例已经写了 multicore stride，CANNBot 的所选 softmax 示例则是 `block_dim=1`；这是**例子完成度的差别**，不能当成 CANNBot 架构只能单核。

Paged attention 中，两者都可把真实长度作为数据、以固定物理 tile 循环处理；Host 均衡分工是否随长度更新，取决于算法设计。block版 的 paged prefill 代码有 `build_work_ranges`，同一实现的单 token decode 已在 A5 通过；CANNBot 已找到 `FlashAttnNoquant(paged=True)`，也通过了选定 decode。因而可以用两个实际对象比较分工与流水；但 block版 已测页大小变化及同批异长，CANNBot 此次 decode 只测 KV=512，覆盖范围和算子契约仍不相同，不能按“attention”名称直接声称等价覆盖或等价性能。[B-pa-test] [D-pa-test] [RUN-results]

走向整层时，相近的难题是：多个不同并行分解阶段如何衔接、哪些结果必须全局归并、谁拥有 scratch、哪些核必须同时进展。TileGroup 和 Channel 都有助于表达局部/指定协作域的流水，但**单靠缓冲协议不能自动补出整层 task DAG 与 scheduler**。共享 buffer/effect 测试契约、矩阵/向量语义与 profiling 口径有意义；直接共用现有内存规划 pass 或 runtime 则需要 IR/ABI 适配，不能由思路接近推出。

B/D 的比较体现同一 NPU 上 TileGroup 与 Channel 的作者责任；H 通过显式 layout/allocate/搬运和较早的 MLIR 入口，在编译分层上与 D 更直接相近。I 也有资源所有权与流水 helper，但其单位是 GPU fragment/warp/SMEM/TMEM；比较方法可共用，物理协议需单独映射。逐项差异见第 4.13 节。[H-dsl] [H-passes] [I-layout] [I-pipeline]

### 4.10 PyPTO2-block版 的 SPMD 与 PyPTO3（Simpler）的 MPMD：局部很像，整体不一样

**按当前主要编程入口归类，PyPTO2-block版 以 SPMD kernel 编程与直接 launch 为主；PyPTO3（Simpler）则提供支持 MPMD 的程序/任务执行体系，任务内部也可以采用 SPMD。**因此，“block版 主要是 SPMD、PyPTO3 的核心差异在 MPMD 任务编排”是成立的架构概括；“block版 只能 SPMD、PyPTO3 不做 SPMD”则不成立。这里比较的是已有的用户契约和运行时职责，不是理论表达能力的互斥分类，也不是性能排名。

#### 4.10.1 SPMD/MPMD 应按哪一层定义

- **SPMD（Single Program, Multiple Data，单程序多数据）**：多个逻辑工作实例运行同一份计算程序/模板，按逻辑索引处理不同数据。它们可以走不同条件分支、循环不同次数；不要求像 SIMD 指令那样锁步执行，也不意味着 shape 必须静态。在 block版 中，典型组织是一次 kernel launch 内按 block ID 划分 tile。
- **MPMD（Multiple Program, Multiple Data，多程序多数据）**：工作集合可以包含不同计算程序及不同数据，例如 norm、GEMM、attention、residual 等任务。在本文的 PyPTO3 语境中，重点是这些程序有独立的可调用入口与任务描述，可由外层依赖关系组织，再交给运行时派发；不是要求“某个物理核永远只运行某一种程序”。
- **静态/动态调度是另一条轴**：MPMD 可以静态安排，SPMD 也可以在同一程序里动态领取工作。不能从缩写本身推出“MPMD 一定有 AICPU”“SPMD 一定没有动态调度”。PyPTO3 的设备 scheduler 是当前实现提供的额外机制，不是 MPMD 一词的定义。

NPU 的 Cube/Vector 还需要区分逻辑与物理：block版 的一个 mixed kernel 可以包含不同的 Cube/Vector section，并在相应执行域展开逻辑 block；两类核显然不执行相同指令。本文称它“SPMD 为主”，指的是**重复实例化同一个逻辑合作模板**，不是所有 AIC/AIV 必须共用一份二进制。反过来，Simpler 的多个 AICore 即使先进入同一个 executor，也可以通过不同函数地址执行不同任务；公共 executor 不会把上层任务体系变成只有一种计算程序。[B13（PyPTO2-block版）] [B10（PyPTO2-block版）] [C33（PyPTO3（Simpler））]

#### 4.10.2 同层比较：差别在程序由谁组织，而非有没有 block ID

必须把比较边界对齐：**block版 的 kernel 编译能力，应先对比 PyPTO3 的 InCore/融合 SPMD kernel 子路径；随后单独比较外层程序执行能力。**

| 同层问题 | PyPTO2-block版 | PyPTO3（Simpler） | 判断 |
| --- | --- | --- | --- |
| 一个核内计算如何写 | 显式 Tile/section/VF/地址/mutex | Tensor 降到 Tile，或直接 Tile/MemRef/system/extern | 有相近的核内工程需求，但 API 粒度、操作覆盖和后端不同 |
| 多核跑同一算法 | Host 选 blockDim，kernel 自己按 block ID 分工 | SPMD task 指定 logical blocks，kernel 内同样可以 block-stride | 两者都能 SPMD；整体区别在外层是否已有多程序任务契约 |
| norm→GEMM→PA→GEMM 等异构阶段 | 普通 Host 多次 launch，或作者设计更大的合作 kernel；当前主路径未提供同等外层 DAG | 可由 Orchestration/Simpler 组织多个 InCore/SPMD/extern task | 整体能力边界不同：一个主要编 kernel，另一个还编设备侧程序与任务关系 |
| 一个 kernel 完成后谁推进下一阶段 | Host/stream 顺序，或已经写进单 kernel 的协作协议 | 外层 task 依赖、ready/dispatch/完成协议；kernel 内仍有自己的同步 | 库名含 runtime，不意味着它们在同一层做相同工作 |
| 增加一个自定义高性能核 | 直接编为 block版 kernel 并 launch | 可写对应 InCore，也可适配 extern task ABI | 存在组合可能；block版 原 Host launcher 不能原封不动变成 Simpler 的 task entry |

表中的关键区别是**谁有权、也有责任选择下一份计算程序**：block版 常见路径中，作者通过 Host 调用或 kernel 内部控制流表达；PyPTO3 则把多个可调用程序及依赖交给任务系统。后一种情况下，不同就绪任务在资源允许时可以并发，但存在依赖的阶段不会因为称为 MPMD 就自动重叠。

#### 4.10.3 PyPTO2-block版：用户先写一个逻辑 block 怎样工作，再将它展开到多核

block版 的多核指南明确采用 SPMD 的跨步切分方式。真实 softmax 中，Host 调用 `softmax_tile_group_kernel[None, num_cores](x, y)`；下面是核内工作分配的摘录，依赖原函数中的 Tensor、Tile 与 `TILE_ROWS` 定义，不是另一个完整 softmax：[B13（PyPTO2-block版）] [B14（PyPTO2-block版）]

```python
rows = x.shape[0]
cols = x.shape[1]
num_cores = pl.get_block_num()
core_id = pl.get_block_idx()

num_tiles = (rows + TILE_ROWS - 1) // TILE_ROWS
for tile_id in pl.range(core_id, num_tiles, num_cores):
    row_off = tile_id * TILE_ROWS
    valid_rows = pl.min(TILE_ROWS, rows - row_off)
```

含义是：逻辑核 `core_id` 依次处理 `core_id, core_id + num_cores, ...` 对应的行块；每个行块内部完成 max→sub→exp→sum→div。这里循环中的“工作项”不是自动提交给外层 scheduler 的独立 task。作者决定 tile 大小、尾块、每核分配和局部流水，JIT 负责把这个程序编成可启动的 kernel。

实际启动链为 `JITKernel.__getitem__ → launcher → _launch → entry/call_kernel`；`_generate_caller_cpp` 生成 `kernel<<<blockDim, nullptr, stream>>>(...)`。因此，`python/pypto_pro/runtime` 中存在 runtime 代码，并不意味着这条路径使用了tensor版的 AICPU 任务调度器。[B3（PyPTO2-block版）] [B30（PyPTO2-block版）]

这里的 block ID 是逻辑索引，不应承诺它永久绑定特定物理核；mixed kernel 的 Vector 域还需按 subblock 语义计算工作单元数。`num_cores=32` 等样例常量也不是所有芯片通用的硬件结论。这个 softmax 测试标记的目标为 `soc("950")`，本 session 已在 A5 执行其六组 FP32 输入并通过，含 `777×300` 及 `2049×100` 尾块；这为分工公式提供了设备正确性对象，仍不是所有几何或性能的证明。[B13（PyPTO2-block版）] [B14（PyPTO2-block版）] [RUN-smoke]

#### 4.10.4 PyPTO3：外层可派发不同程序，一个 task 内又可以 SPMD

PyPTO3 的运行时测试 `SPMDThreeSubmitProgram` 提供了很直接的组合例子：三个不同 InCore 分别做 add、mul、sub，Orchestration 中各自展开为四个 logical blocks。以下为原方法中的调用片段，`self.spmd_*`、参数及输出 Tensor 的定义见原文件；这里只展示执行组织，不是独立可运行程序：[C34（PyPTO3（Simpler））]

```python
with pl.spmd(4):
    t1 = self.spmd_add(a, b, t1)
with pl.spmd(4):
    t2 = self.spmd_mul(t1, a, t2)
with pl.spmd(4):
    out = self.spmd_sub(t2, b, out)
```

**外层有 add/mul/sub 三种任务程序，内层每个任务都是 SPMD。**本例的阶段顺序受数据依赖约束，并不演示三个阶段同时运行；它证明的是两层组织可以共存，而不是 MPMD 必须并发。这里也不能将三个提交直接等同于三个 Host kernel launch：Simpler 的任务派发发生在已启动的设备执行体系内，Host 启动次数须另查实际运行链。

采用显式依赖时，`pl.spmd_submit(..., core_num=N)` 返回整个 dispatch 的一个 TaskId，后继可通过 `deps=[tid]` 依赖它；普通 `pl.submit` 与 SPMD submit 共用任务依赖机制。自动 scope 则不要求用户逐条手写依赖。**不同程序入口、任务实例数、一个任务的 logical block 数，是三个不同的量。**只有一种 kernel 的任务集合也能在 Simpler 中运行，不能把每个 PyPTO3 程序都强行称为实际使用了多程序计算。[C6（PyPTO3（Simpler））]

源码上的关键不是“有 scheduler 这个文件”，而是它在派发时真的能更换计算入口。以已核对的 A2/A3 `tensormap_and_ringbuffer` 路径为例，`SchedulerContext::build_payload` 中有以下连续代码：[C32（PyPTO3（Simpler））]

```cpp
uint64_t callable_addr = get_function_bin_addr(slot_state.task->kernel_id[slot_idx]);
const CoreCallable *callable = reinterpret_cast<const CoreCallable *>(callable_addr);
dispatch_payload.function_bin_addr = callable->resolved_addr();
```

同一函数还填入该 dispatch 的 `block_idx`、`logical_block_num` 和参数。AICore 侧 `execute_task` 从 `payload->function_bin_addr` 取出函数指针，再调用 `kernel(payload->args)` 对应的统一 ABI；完成路径通知 scheduler 更新后继依赖。因此，同类物理核可以在先后任务中运行不同程序；同一任务又可以向多个核派发同一程序的不同 logical blocks。[C32（PyPTO3（Simpler））] [C33（PyPTO3（Simpler））] [C35（PyPTO3（Simpler））]

但这里**不是指令级抢占或自动重切分**：scheduler 调度已表达的 task/block，不会把一个已经进入的 kernel 内部 page 循环逐次拆出再派发。普通 SPMD task 可按可用资源分批派发尚未执行的 logical blocks；`sync_start=True` 的合作任务需要满足整组启动条件，不能当作任意空闲核都可立即承接的小任务。这也是 PyPTO3 native PA 的长尾仍取决于核内分工的原因，详见第 6.13 节。[C32（PyPTO3（Simpler））] [C6（PyPTO3（Simpler））]

#### 4.10.5 对 softmax、attention 和整层 megaKernel，用户代价具体差在哪里

1. **可在核内完成的 softmax：SPMD/MPMD 不是融合程度的分界。**block版 可以让每个逻辑核在自己的 UB Tile 上串起五项运算；PyPTO3 也可以写一个融合 InCore，并通过多个独立 tile task 或一个 SPMD task 展开。若在 PyPTO3 中将五项运算人为拆为五类 task，就额外引入了任务依赖和中间存储边界；MPMD 不会自动省掉这些 GM 往返。超长 reduce 则需要正确的分段统计/合并算法，两条路线都不能靠执行模型标签解决。具体融合写法见第 8.3 节。

2. **paged decode：外层调度与核内动态长度要分别设计。**按 block版 的 kernel 编程方式，作者需要在合作 kernel 中设计 QK、softmax、PV 及实际长度循环，或通过 Host 拆成多个 launch；第 2.6 节保留 prefill 的分工示例，并已补充同一实现的 A5 单 token decode 正确性验证；核内分工是否适合真实 decode 长度分布仍需性能测量。PyPTO3 既可将阶段拆成不同任务，也可像 `pypto-lib` native PA 一样保留为融合的多 block SPMD task。后者一旦将请求按固定 stride 分给 logical blocks，外层 Simpler 就不会自动把长请求的剩余 pages 迁移给短请求已经空闲的核。拆成更细任务可能增加调度机会，也可能增加中间 GM、依赖和调度开销；应按算法/数据生命周期决定，而不是一律拆开。

3. **整层 Transformer：PyPTO3 更直接提供程序级组织能力，block版 更集中于 kernel 内部工程。**norm、各次 GEMM、attention、MLP 的并行度、核类型及 tile 形状通常不同。PyPTO3 可让这些阶段保持不同入口、不同 block 数及依赖，复用现成任务系统推进；作者仍需设计任务粒度、局部融合及中间 Tensor。block版 作者可通过 Host/stream 组织多次 launch；若要求将整层留在一个更大的合作 kernel 中，则需要额外处理阶段转换、工作重分配、跨核同步与缓冲复用，或接入另一套任务 runtime。**前者降低的是程序级 megaKernel 的组织成本，不是对极致性能的保证。**详细性能机会与差距见第 6 章，不能用本节的源码证据代替实测。

动态 shape 也不随这一分类自动分出高下：block版 的运行时 shape/长度、固定 tile 循环、Python 策略和 TilingData，都可以承载 tiling 决策；PyPTO3 的决策可以在 Orchestration、SPMD 核内、Host Python 或专门 tiling task 中。**SPMD 不强制单独 Host tiler，MPMD/AICPU 也不消灭 tiling 逻辑。**具体何时重算参数、何时改写函数、何时重新编译，见第 7.8—7.9 节。

#### 4.10.6 能否跨过边界：可以组合，但当前不等于已经兼容

SPMD 不是理论禁令：同一个程序可以按角色分支执行不同阶段，或在适用硬件机制上实现工作队列；block版 已有 Cube/Vector section 和跨核同步配套。这说明它可以表达合作程序，但不能据此补出一套已经交付、可复用的 Simpler 式异构任务 DAG。若作者自行增加这样的调度器，负载分配、进展性、依赖、缓存一致性和资源回收也成为作者要实现和验证的内容。[B30（PyPTO2-block版）]

可组合方向是 **block版/CANNBot 专门核 → Simpler task ABI 适配 → PyPTO3 外层程序提交**，具体参数、逻辑索引、完成/依赖等契约见第 10.3 节；这是待验证的集成方向。接入后，专门核仍可采用 SPMD，只是外层由 MPMD-capable 的任务系统组织。核内子域相近，不改变 PyPTO3 与tensor版在程序/任务层更近的整体判断。

**最终表述应是：PyPTO2-block版 的主要抽象单位是“一个多核合作 kernel”；PyPTO3（Simpler）的主要架构识别点是“由不同计算程序构成、允许 task 内 SPMD 的设备任务系统”。**选用不同粒度会改变用户责任、调度空间和数据搬运边界，不能仅按是否含 SPMD/MPMD 字样判断完整能力。

H 的 mixed kernel 拆出 AIC/AIV，I 的 kernel 为 warp 指定载入、矩阵计算或归约等角色，也属于应按层命名的例子。它们可以承载一个合作计算体；若要放进本节的多程序任务体系，还需设备可调用入口、逻辑 ID、资源组及完成语义。单靠 mixed/warp 分支或 TS 的 `Task` 名称，不能推断已有整层 MPMD runtime。[H-mixed-pass] [I-gqa] [I-task]

### 4.11 H（CATLASS DSL）：TLA 前端、混合核 lowering 与 AscendCL launch

```text
Python @tla.kernel / @tla.jit helper + 编译期样本参数
  → BaseDSL lowering，TLA tensor/layout/ptr/copy/mmad/vector 等 MLIR
  → TLA 自有 passes：函数分类、受限 AutoMutex、片上分配、mixed 拆分、descriptor lowering
  → Cube/Vector/flag/mutex、HIVM/AVE 及 regbase intrinsic lowering
  → CANN hivmc-a5
  → kernel.o + manifest / ABI / workspace 元数据
  → JitCompiledFunction → PyACL 装载与 launch_kernel_with_config
```

`@tla.kernel` 本身不能直接作为 Host 函数执行；`tla.compile(kernel, *sample_args, options="--npu-arch 3510")` 的参数用于确定类型与专门化，甚至可用 fake tensor。`block_num/stream` 则传给编译产物。缓存键核对 TLA IR、编译选项、bridge 与 `hivmc-a5` 等信息；“同一个 Python 函数”不是充分的二进制复用条件。[H-dsl] [H-compile] [H-execution]

这里还要区分两个编译器来源：CATLASS 固定的 AscendNPU-IR 提供开发头文件、MLIR 与 HIVM/AVE 等静态库，供 TLA extension 使用；末端 `hivmc-a5` 由运行环境的 CANN 提供。固定子模块 SHA 不自动成为这个 CANN 二进制的源码版本声明，实测应另外记录其版本与文件哈希。[H-ir-build] [H-cmake] [H-execution]

mixed 路径在 TLA 中可以写同一个逻辑 kernel 的 `cube()` 和 `vector()` 区域；`TlaLowerFuncPass` 判别核类型，`TlaSplitMixedFuncPass` 再按角色拆分。一个源函数、AIC/AIV 的机器码实体、一次 ACL 提交属于不同计数口径；是否只有一个设备事件仍需 trace。这个机制提供核内异构合作，不自动提供多个任意算子函数间的设备就绪队列。[H-passes] [H-mixed-pass] [H-runtime]

本轮实际生成的 `basic_mixed` 与 FA `lowered.mlir` 均含 `_mix_aic`、`_mix_aiv` 及 Vector helper；对应 manifest 的 `arch_scope` 仍写 `aic.c310`。因此不能只按这一个字段把产物判断为纯 Cube kernel，具体函数声明和文件 hash 归档在第 12.1.3 节。

`tla.call_extern` 还有外部实现的编译接缝，且 AutoSync 当前明确排除它。因而 H 也有“本 DSL 表达”“嵌入外部底层代码”两种覆盖来源，不能把外部 C++ 能力全部计入 Python 原生 API。[H-dsl] [H-extern-pass]

### 4.12 I（CuTe DSL）：Host/device 分阶段编译，用户显式选择线程与 layout

```text
Python @cute.jit Host 程序 + @cute.kernel 设备函数
  → Python DSL/元编程 lowering → CuTe 与 GPU/标准 MLIR
  → 默认 cute-to-nvvm pipeline（扩展路径可先 lir-to-cute-dsl）
  → NVVM / PTX 后端工具链 → cubin，及 Host 调用产物
  → JIT executor / CUDA runtime → CUDA launch
  → CTA/cluster 执行；用户可在 kernel 内使用 persistent tile / warp 级任务流水
```

默认 pipeline 字符串、扩展 pipeline 入口、cubin 提取/装载和 CUDA launch 均能在 Python 源码定位。**本次源码证明到调用契约；未生成或反汇编 I 的 GPU 二进制。** 安装依赖文件固定 `nvidia-cutlass-dsl==4.8.0.dev0`，这份 checkout 并未提供足够材料证明整个配套 CuTe/NVVM 编译器都可从这里独立重建；不能用“CUTLASS 仓库公开”替代编译组件来源与版本核对。[I-dsl] [I-compiler] [I-executor] [I-runtime] [I-requirements]

`@cute.kernel` 调用产生待 launch 对象，`.launch(grid=...,block=...,cluster=...,smem=...,stream=...)` 确定调用；Host `@cute.jit` 内可以包含多次这样的调用。GQA simple 正是一个完整反例：Python 一次调用覆盖 decode 与 reduction，两个阶段经 GM partial 连接。它不受 E 当前“只接受某些 PyPTO 单函数/静态模式”的同一 emitter 限制，也不能把 I 的覆盖倒灌给 E。[I-dsl] [I-gqa]

### 4.13 重点对比 H/I，并与 B/D/F 对齐

| 具体问题 | H 与 I 的相似处 | 差别及相邻路线 |
| --- | --- | --- |
| 用户先写数学还是硬件分区 | 都允许 Python 元编程生成显式 tile/layout 实现 | H 组织 AIC/AIV、GM/UB/L1/L0；I 组织 thread/warp/CTA/cluster、register/SMEM/TMEM；与 F 的常规 logical program 写法不同 |
| layout 解决什么 | 都把地址/布局纳入算法 | H 的逻辑有效范围与硬件 packing tag 更突出；I 的 layout 代数和线程—值/MMA 分区更突出；D 同样值得按逻辑/物理布局比较 |
| 数据流如何流水化 | 都有多缓冲、生产/消费与异步硬件操作 | H 是 MTE/Cube/FIX/Vector 及跨核 flag/mutex；I 是 TMA/MMA/warp 与 mbarrier/pipeline/TS；不能直接翻译事件 ID |
| `jit` 在哪执行 | 都区分编译期 Python 与设备动态值 | H helper 在设备 IR 内联；I Host JIT 可发多个 kernel；D 也应按自身 Host/Device staging 判断，不能只比较装饰器 |
| 谁负责全局程序推进 | 所查 H/I 示例主要由作者组织 kernel 与 launch | A/C 有独立任务执行体系；I TS 又把单 kernel 内的异构 warp 任务提升为显式抽象，但粒度不同 |
| 谁承担新硬件适配 | kernel 作者和后端共同承担 | H 的 A5 regbase、I 的 SM100/TMA/TMEM 都是具体目标契约；不能以 CATLASS/CUTLASS C++ 全库支持范围替代 DSL 支持范围 |

据此，H/I 是新的 **layout 与显式流水** 比较组；H/B/D 则是同硬件上的 **显式核内工程** 比较组。它们的存在进一步支持原第 4.8—4.10 节的分析方式：先比较相同责任层，再讨论整体系统。相近命名与接口风格仅能支持设计相似性，不足以确认仓库之间的代码派生关系。[H-api-layout] [H-mixed] [I-layout] [I-task]

<a id="mlir"></a>
## 5. MLIR 专章：表达范围很宽，但不会自动交付 tiling、调度与极致性能

### 5.1 上界与下界：MLIR 是可扩展的编译基础设施，不是一个固定层次的图

**MLIR 的表达上界并不止于算子图，下界也不止于循环；但“能够建模”不等于“现有工具链能够优化并执行”。** 比较九条路线时，应分别回答三个问题：

1. **表示能力**：能否用 Operation、Region、类型、属性和边表达该语义？
2. **编译实现**：是否已有识别这些语义的 verifier、analysis、rewrite/lowering、目标资源模型？
3. **执行契约**：产物由谁调用，参数/内存/同步如何解释，硬件是否支持，能否保证完成？

MLIR 有实际的 C++ 对象、parser/printer、文本/内存/序列化形式及 pass 基础设施，不只是“可以有很多层 IR”的思想。Operation 可以嵌套 Region；Region 可以承载控制流或图语义。SSA 的单一定义组织不要求所有层次都变成“按文本顺序执行的单一计算 DAG”。[MLIR 语言参考](https://mlir.llvm.org/docs/LangRef/)

下面按**可建模的层次**而非“MLIR 自动附送的功能”阅读：

| 层次 | 可以承载的对象 | 需要额外实现的东西 |
| --- | --- | --- |
| 模型/程序 | 高层 tensor op、函数、控制流、符号维、子图、外部调用 | PyTorch/其他前端接入、算子语义、effect、分组和成本模型 |
| 调度/变换策略 | tile/fuse/unroll/vectorize 等策略，可由 pass 或 Transform dialect 描述 | 合法性、目标成本、搜索策略；策略可表达不等于自动找到最优策略 |
| Tile/循环/缓冲 | `linalg/scf/affine/vector/tensor/memref` 及自定义 tile/layout | 合法 tile、bufferization、物理容量、异步生命周期、bank/layout |
| 并发/任务/Host-Device | launch、token、依赖、设备模块、Host wrapper；也可定义任务 dialect | 真实 runtime、工作队列、资源分配、失败处理、同步域与进展协议 |
| 硬件相关 | address space、DMA、矩阵/向量操作、barrier、target intrinsic | ISA 映射、pipeline 约束、目标 codegen/assembler、二进制加载 |
| 更低层模型 | 框架允许继续定义更低层抽象，并无“必须止于 LLVM”的概念边界 | 不是说本地九条路线都实现了这些层，更不是 MLIR 自己执行硬件 |

例如 MLIR 的 GPU dialect 确实包含 Host/Device 和 kernel launch 相关构件；Transform dialect 可以描述变换过程。这说明“MLIR 只能编核内”是错误的**通用判断**；但当前 C（PyPTO3（Simpler））只把 InCore 交 PTOAS，是该集成的选择。[GPU dialect](https://mlir.llvm.org/docs/Dialects/GPU/)、[Transform 教程](https://mlir.llvm.org/docs/Tutorials/transform/Ch0/)

### 5.2 与 PyPTO 自有 SSA 的本质区别：容器/语义/工具生态，不是 SSA 与非 SSA

以两个极小片段说明（本文构造的语义示意，不是九条路线某次编译的 dump，也未做工具链编译）：

```mlir
// A value remains SSA whether it represents one scalar or a whole tensor.
func.func @tensor_add(%x: tensor<16x128xf32>,
                      %y: tensor<16x128xf32>) -> tensor<16x128xf32> {
  %z = arith.addf %x, %y : tensor<16x128xf32>
  return %z : tensor<16x128xf32>
}

// The buffer handle is an SSA value; the pointed-to memory can be modified.
func.func @overwrite_one(%buf: memref<?xf32>, %i: index, %v: f32) {
  memref.store %v, %buf[%i] : memref<?xf32>
  return
}
```

第一段 `%z` 是新定义的 tensor 值；第二段 `%buf` 本身没有“重新赋值”，但其指向的内存被写入。**SSA 不等于不可变内存，也不自动列出全部 read-after-write / write-after-read 依赖。** effect、alias、异步完成和可见性要有独立建模；缺失这些信息时，重排/复用就不能只看 SSA use-def。[副作用与 speculation](https://mlir.llvm.org/docs/Rationale/SideEffectsAndSpeculation/)

PyPTO 自有 IR 同样可以实现 SSA、类型、visitor、pass、控制流和 memory effect；区别在于这些对象与工具是自建的，而不是直接使用 MLIR 的 Operation/Value/Region 体系。转换时要把“自有 op 的语义”映射成“目标 dialect 的语义”，不是把 SSA 图换成“非 SSA 的 MLIR 图”。

具体到 C（PyPTO3（Simpler））：一个 `pl.jit` 先形成自有程序 IR，经过 scope/并行/内存等处理，InCore 输出 `.pto` 文本；文本中的 `func.func/arith/scf/pto.*` 是 MLIR 语法与 dialect。PTOAS 解析的是这样的 module/function/operation，而不是 PyPTO Python 对象或整份 Simpler task 数据结构。[C1（PyPTO3（Simpler））] [C2（PyPTO3（Simpler））] [C3（PyPTO3（Simpler））]

### 5.3 九条路线：MLIR 从哪里进入，在哪里退出，哪些决策已经做完

下表判断的是**本地所选主调用链**。底层商业编译器内部可能使用其他 IR，不将未审计的内部过程推测为九条前端“直接使用 MLIR”。

| 路线 | MLIR 进入位置与输入 | 到这里哪些事情通常已确定 | MLIR 区间做什么/输出什么 | 不在这个区间内的关键职责 |
| --- | --- | --- | --- | --- |
| A（PyPTO2-tensor版） | 当前审计主链不是 MLIR 主导：Python/PIL/自有 IR → TileFwk/CCE | Tensor 程序、tile 配置、图/任务构造在本栈处理 | 不应凭末端编译器名称宣称其前端是 MLIR | TileFwk 任务执行和图级 GM 管理不能由 PTOAS 代替 |
| B（PyPTO2-block版） | block版 自有原生 IR → CCECodegen；不是 PTOAS/HIVM 接口 | 用户选的 TileType、地址/槽位、core-stride、Vector/Cube body | 自有 pass 与 codegen 承担对应编译职责；没有“因为不用 MLIR 就无法优化”的结论 | Host JIT/shape ABI/launch；默认不启动tensor版任务 scheduler |
| C（PyPTO3（Simpler）） | 自有 IR 后端输出 InCore 的 PTO dialect MLIR | task/scope 边界、核内职责、部分 tile/layout/内存决策 | PTO pass、memory/sync/合法化，当前集成常用 EmitC→C++ | Orchestration、TaskId、Simpler 依赖与跨 task GM 生命周期 |
| D（CANNBot DSL） | Python tracing 较早建立 CANNIR；Host 与 kernel 可同处编译流程 | 作者显式选择的 tile/Channel/Buffer/流水结构 | layout、Channel lowering、VF grouping、mixed split、Asc 相关 dialect→AscendC | Python 专门化/AOT 契约、调用 runtime；无自动全模型动态 task scheduler |
| E（PyPTO on GPU） | 自有 IR/模式识别→typed `nv_tensor_ir` MLIR module | 高层已接受的模式、静态规格与调度配置；被拒绝的图根本到不了此处 | NVIDIA TensorIR→CUDA Tile IR→prepared/assembled artifact | PyPTO 图覆盖、shape guard、API 多 executable 分组与 CUDA launch |
| F（Triton-Ascend） | Triton 前端较早建立 TTIR；适配后进入 AscendNPU-IR | 用户/Inductor 的 kernel 边界、program 分解及 constexpr | 多层 tensor→buffer/硬件 IR、CV pipeline、local/workspace/sync、目标编译 | Inductor 原图的外部边界、JIT cache、CANN launch；不是通用 AICPU task queue |
| G（AutoFuse + Inductor） | 所选 AutoFuse 主链为 ASCGraph/ASCIR/FusedScheduledResult，不是 MLIR Module | Inductor 分组、融合图与符号 shape | 自有 schedule/optimizer/tiling/codegen 直接生成 Host/Device 代码 | 这条链的 planner/tiler 是实有能力，不能因非 MLIR 就忽略 |
| H（CATLASS DSL） | Python TLA builder 较早生成 TLA/标准 MLIR | 作者已选物理 tile/layout tag、内存空间/容量、MMAD/Vector 与搬运策略 | 分类/AutoMutex、mixed 拆分、tensor/ptr lowering、HIVM/AVE/标准转换；末端交 `hivmc-a5` | Host 编译缓存、ABI/launch 与 kernel 外 GM；不会因此获得 F 的完整高层 tiling 或模型任务 runtime。[H-passes] [H-compile] |
| I（CuTe DSL） | Python DSL 生成 CuTe/相关 MLIR，Host 与 Device 分阶段处理 | layout/atom、线程/warp 角色、tile/pipeline 和 launch 配置由作者/模板给出 | 调用 `cute-to-nvvm` 等编译 pipeline，再由配套组件形成 CUDA 产物 | Python 专门化、CUDA 上下文/stream、全局 workspace 和模型调度；仅此源码树不足以认定完整编译器可重建。[I-dsl] [I-compiler] [I-executor] |

对应证据：A（PyPTO2-tensor版）[A2（PyPTO2-tensor版）]—[A6（PyPTO2-tensor版）]；B（PyPTO2-block版）[B1（PyPTO2-block版）]；C（PyPTO3（Simpler））[C1（PyPTO3（Simpler））]—[C3（PyPTO3（Simpler））]；D（CANNBot DSL）[D2（CANNBot DSL）]—[D4（CANNBot DSL）]；E（PyPTO on GPU）[E2（PyPTO on GPU）]—[E3（PyPTO on GPU）] [E13（PyPTO on GPU）]；F（Triton-Ascend）[F1（Triton-Ascend）] [F2（Triton-Ascend）] [F8（Triton-Ascend）]；G（AutoFuse + Inductor）[G2（AutoFuse + Inductor）]—[G5（AutoFuse + Inductor）]。

**同用 MLIR 的两个后端，优化自由度也可能不同。** 输入已变成具体 buffer 和低层搬运时，后端能重排/复用这些动作，不代表还能可靠恢复跨 task 的 tensor 语义、重新分组整层或撤销前端做出的所有布局选择。

### 5.4 看实际 IR：PTO、CANNIR、HIVM 不是同一“MLIR 图”

以下为当前测试文件的真实语句摘录，省略 module 外壳与不相关计算；变量定义见完整文件。它们不是相互转换前后的同一份测试。

**PTO：tile 容量与位置已写进类型。** [P4（PTOAS）] 的 softmax-prepare 使用：

```mlir
%c128 = arith.constant 128 : index
%0 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=128, v_row=16, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>
%2 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1, v_row=16, v_col=1, blayout=col_major, slayout=none_box, fractal=512, pad=0>
```

同文件继续用 `pto.tload/tmuls/trowmax/trowexpandsub/texp/trowsum/tstore`。这里已经是带物理 tile 信息的核内程序；完整例子还发生 BF16 转换，不能当成任意 FP32 softmax 的数值等价模板。

**CANNIR：shape/stride/layout 和地址空间是显式对象。** [D15（CANNBot DSL）] 的 softmax lowering 测试使用：

```mlir
%shape = "cannir.make_shape"() : () -> !cannir.shape<(32, 32)>
%stride = "cannir.make_stride"() : () -> !cannir.stride<(32, 1)>
%ptr0 = "cannir.make_pointer"(%addr0) : (i64) -> !cannir.ptr<ub, f32>
```

该测试实际检查 `cannir.reduce_max/exp/reduce_sum/div` 被下沉到 `ascvec.reduce_max/exp/reduce_sum/div`，并经过 Channel/VF/layout/mixed-kernel 相关 pass。MLIR 提供承载机制；这些算子与策略来自 CANNBot 的具体实现。

**HIVM：标准 memref 与硬件 dialect 的 address space 组合。** [F13（Triton-Ascend）] 的 PlanMemory 测试使用：

```mlir
%copy_in_ub = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
%dst1 = memref.alloc() : memref<16x16x16xf16, #hivm.address_space<ub>>
hivm.hir.vadd ins(%copy_in_ub, %copy_in_ub : memref<16x16x16xf16, #hivm.address_space<ub>>, memref<16x16x16xf16, #hivm.address_space<ub>>)
              outs(%dst1 : memref<16x16x16xf16, #hivm.address_space<ub>>)
```

测试期待 allocation 被规划到 `hivm.hir.pointer_cast` 等具体表示。这里的 `#hivm.address_space<ub>` 是目标语义，不是通用 MLIR 会自行理解 UB 的容量、访问规则和同步域。

E（PyPTO on GPU）的名字也要说清：本地 NVIDIA TensorIR 的 dialect namespace 是 `nv_tensor_ir`，不是 MLIR 标准 `tensor` dialect，也不是 Apache TVM 的 TIR。桥接代码实际构建 MLIR module，再调用 `CudaTileCompiler::prepareToArtifact` 和 `assemblePreparedArtifact`；它不是“把 PyPTO SSA 文本换个文件扩展名”。[E13（PyPTO on GPU）] [E3（PyPTO on GPU）]

**H 的 TLA IR 与实际 lowering 产物：**`buildTlaPipeline` 先从 `tla.cube/tla.vector` 分类函数核种，再处理指针、mixed 拆分和 tensor descriptor，随后进入 HIVM/AVE 等转换。这里的 TLA tensor/layout/地址空间已经携带作者的物理决策，不等同 F 的较高层 tensor 输入。本次 mixed/FA 归档的 `lowered.mlir` 函数声明确有 `_mix_aic`、`_mix_aiv` 和 Vector helper；这是已生成产物的证据，仍不等于 profiler 的物理 launch 计数。[H-passes] [H-mixed-pass] [H-run-artifacts]

**I 的 CuTe IR 构造与编译入口：**Python 实现构造 CuTe layout/计算及 Host launch 相关 IR，并配置 `cute-to-nvvm` 编译 pipeline；它与 E 的 `nv_tensor_ir` emitter 不同。这里核对的是 builder、pipeline 配置与执行器源码，本机没有生成 CuTe 的 IR/cubin，因而不能把某段示意 IR 或 H/E 的 dump 称作 I 的编译结果。[I-dsl] [I-compiler] [I-executor]

### 5.5 动态 shape、内存规划和调度：MLIR 能表达，但要跨过不同的功能边界

| 问题 | MLIR/相关通用机制能提供的基础 | 本地路线必须补充的策略或契约 | 易混淆的说法 |
| --- | --- | --- | --- |
| 动态 shape | 动态类型维度、运行时 dim、scf 控制流和算术 | shape ABI、guards、合法容量、特化缓存与动态循环 | 类型里有 `?`，所以任意 shape 不重编译 |
| tiling | 可表示切片/循环；可实现或调用 tile transformation | tile 大小选择、长 reduce 分解、cost model、Host/device 策略执行 | 用 MLIR 就不需要 tiling 决策 |
| bufferization | tensor 值转成 buffer、分析原地更新与冲突 | UB/L1/寄存器容量、异步多缓冲、pipeline 阶段、目标 layout | bufferization 就是完整片上内存规划 |
| GM/workspace | 可以表示 allocation、offset、读写与生命周期 | 跨 task 并发、完成时刻、workspace ABI、调用者所有权 | 核内 planner 自动统一整个模型 GM |
| 同步 | 可以定义 barrier/event/token/effect | AIC/AIV 或 CTA 的具体协议、可见性、驻留与前进条件 | SSA 边就是 runtime event，编译合法就绝不死锁 |
| 调度 | 可以表示任务/launch/依赖并生成 runtime 调用 | 就绪队列实现、worker、资源组、动态分配、异常恢复 | MLIR 自动替代 Simpler/TileFwk |
| 性能搜索 | pass 组合、Transform 控制变换、候选 IR | 目标代价模型、autotuning、负载分布、编译预算 | canonicalize 就能找到最优融合 |

此处 bufferization 与物理规划的划分来自通用接口定义及本地 pipeline 的分工：One-Shot Bufferize 关注 tensor→memref/原地性；AscendNPU-IR 后面还有独立 `PlanMemory`、CV pipeline、同步与 workspace 推导。不能把这些后端工作统称为一个 MLIR 标准功能。[Bufferization](https://mlir.llvm.org/docs/Bufferization/) [F8（Triton-Ascend）]

同样，`MemoryEffectsOpInterface` 可以暴露读写等效应，却不等于按字节范围完成 alias、推导所有硬件事件或证明整个异步系统可终止。这一差距在 C（PyPTO3（Simpler））的显式 TaskId、D（CANNBot DSL）的 Channel、F（Triton-Ascend）的 CV/sync pass 上表现为不同工程层。[副作用接口说明](https://mlir.llvm.org/docs/Rationale/SideEffectsAndSpeculation/) [C20（PyPTO3（Simpler））] [D3b（CANNBot DSL）] [F8（Triton-Ascend）]

H/I 让上述边界更具体：H 的 scratch planner 只按空间对齐累加偏移，不能从“用了 MLIR”推出不同 `allocate` 已按 last-use 自动复用；I 的 TS phase alias 由作者声明可复用生命周期，再由工具排资源，也不能替模型 GM 规划。两者都需要把异步完成、别名与实际同步纳入正确性条件。[H-scratch] [I-ts-memory] [I-checker]

### 5.6 采用 MLIR 到底复用了什么，为什么仍不能随便换后端

可实质复用的是 IR 数据结构、文本/bytecode、pass 驱动、pattern rewrite、dialect conversion、部分标准 dialect/analysis/interface 和测试工具。**不能自动复用的是未映射的目标语义、性能策略与 runtime ABI。** Dialect Conversion 要明确合法 op、类型转换与转换 pattern，框架不会自动发明 PTO→HIVM 或 CANNIR→TensorIR 的语义映射。[Dialect Conversion](https://mlir.llvm.org/docs/DialectConversion/)

以 C（PyPTO3（Simpler））改接 AscendNPU-IR 为例，需要逐层决定：

1. 接在 Tensor/linalg 较高层，还是接在已选 tile/layout 的 PTO 层？前者保留优化空间但转换更多，后者更接近现接口但受既定决策限制。
2. `Mem/Mat/Vec/Left/Right/Acc`、有效 shape、别名、原子、同步等如何映射？哪些转换可能增加搬运？
3. 两套 local planner/CV pipeline 谁主导，如何避免重复多缓冲或相互破坏 liveness？
4. workspace、核心类型、mixed entry、launch/任务 ABI 和 Simpler worker 的调用方式如何适配？
5. 用哪组数值/同步/性能回归验证？非同一工具链版本的 bytecode/dialect 兼容由谁维护？

这既不是“同为 MLIR 所以简单接通”，也不是“dialect 不同所以必须推倒前端”。第 10 章列出替代边界；本章补充的结论是：**公共 IR 基础设施减少的是一类成本，不会消灭整个后端适配项目。**

EmitC 还能把 MLIR 下沉成 C/C++，因此“MLIR 的唯一终点是 LLVM IR”也不成立；当前 PTOAS/CANNBot 的 C++ 末端与 AscendNPU-IR 的目标 lowering 必须按各自实际链条分析。[EmitC dialect](https://mlir.llvm.org/docs/Dialects/EmitC/)

同样的适配问题也适用于 H/F 与 H/I：H 的 pipeline 复用部分 HIVM/AVE pass，但 TLA 输入已决定许多 layout/资源，不能原样套用 F 的全部分解/规划；H/I 虽都较早使用 MLIR，NPU CV/搬运与 GPU warp/TMA/TMEM 的效应和进展条件仍需映射。这里更现实的共用物是资源/异步契约和测试，现成 pass 或二进制是否可共用必须另做适配原型。[H-passes] [I-copy] [I-mma] [I-ts-memory]

### 5.7 PTOAS 与 AscendNPU-IR：核内编译、多 kernel 编译与 launch 生成的边界

**AscendNPU-IR 不具有“只能编一个核内函数、不能生成后续 launch”的统一限制。**其通用 HFusion/HACC 路径能够处理 Host/Device 函数、按条件拆分多个 kernel、生成 Host tiling/分支，并把 Host→Device 调用降为配置、注册和 launch stub。不过，**Triton-Ascend 当前主要使用其专用 kernel 编译分支，实际 launch 仍由 Triton driver 生成**；不能把后端本体的全部能力直接算到这一集成入口上。

同时，PTOAS 也应限定语境：**PyPTO3 当前将 PTOAS 用作核内实现编译器，不把 Simpler 的任务调用顺序交给它；但本地 PTOAS 本体并非只能容纳一个函数，VPTO/fatobj 路径也已有 Host stub 生成。**“不生成上层调用程序”与“完全没有 Host 相关产物”不是同一句话。下面依据固定版本源码区分这些边界；本次 PA 已使用 PTOAS v0.57 完成 A5 编译执行，但本节 VPTO Host stub、fatobj 及 AscendNPU-IR 通用 Host 图用例仍仅做源码分析，不能以 PA 通过替代这些路径的验证。

#### 5.7.1 先区分四种粒度，再比较职责

- **IR 函数**：一个符号和函数体；可能是设备入口、helper、Host tiler 或 Host 调用程序。
- **编译单元**：一次编译的 module，可以包含多个函数，甚至 Cube/Vector 子 module。
- **逻辑 kernel 入口**：供 runtime 启动的设备计算；一个 mixed 入口可能对应 AIC/AIV 两份实现。
- **launch 调用点与运行次数**：由谁生成启动调用、带什么参数，以及它在循环/分支中执行几次。函数数、编译单元数和源码 launch 调用点数都不能直接替代运行时次数。

下表比较的是责任边界，不是完整性或性能排名；AscendNPU-IR 的通用路径与 Triton 专用路径刻意分列。

| 维度 | PTOAS 本体及 PyPTO3 当前用法 | AscendNPU-IR 通用 HFusion/HACC 路径 | Triton-Ascend 当前使用的路径 |
| --- | --- | --- | --- |
| 输入位置 | PyPTO3 提交已经划定的核内 PTO tile/控制流；可按单函数或 Group 组织 module | 可接 Host/Device 标注的 Tensor/Linalg/HFusion/HIVM 等多层 IR | 用户/Inductor 已划定的 Triton kernel，经适配进入后端 |
| 单次输入只能一个函数？ | 否；有多入口、helper、混合核及子 module 路径 | 否；Host、多个 Device、tiling/infer-shape helper 可共存 | 以一个 JIT kernel 的编译和启动为主要契约，不等于 IR 中只能一个 helper/函数 |
| 从上层计算图拆多个 kernel | 当前 PyPTO3 的任务划分在 PTOAS 上游，不由 PTOAS 重建 Simpler 程序 | HFusion 有 `--enable-multi-kernel`，默认关闭；输入层级有明确限制 | 专用分支跳过通用 HFusion outlining/auto-schedule，不能直接继承其多 kernel 图编译能力 |
| 核内优化的类似处 | 合法化、Tile/布局、内存规划、同步、目标 lowering 的部分职责 | bufferization、布局/内存规划、CV 流水、同步及目标 lowering | 使用其中适配该 kernel 分支的 passes；不代表所有通用 passes 都运行 |
| Host 相关产物 | PyPTO3 常见路径消费 C++；PTOAS 的 VPTO/fatobj 路径另有入口 stub、设备对象打包 | 可生成 Host tiling/shape helper，以及调用设备 kernel 的 Host 包装与共享库 | 可消费后端生成的资源查询 callback 库，实际 launcher 由 Triton runtime/driver 维护 |
| 谁生成 launch 调用程序 | PyPTO3 的 Orchestration/Simpler 层负责提交；独立 PTOAS 样例仍另写 `Launch*` wrapper | HACC Host→Device call lowering 可生成配置/注册/参数打包/launch | `driver.py::make_launcher` 生成调用 CANN runtime 的代码 |
| 动态 tiling | PTOAS 不接管 PyPTO3 的上层 shape→任务/launch 策略；核内动态范围与静态容量另有约束 | AutoSchedule 可生成 tiling 函数及变体选择；Host 资源管理是可选机制 | grid/特化/核内动态循环和资源 metadata 按 Triton 契约处理；callback 不等于通用 Host tiler |
| 自动获得 AICPU 任务调度？ | 不会；Simpler 依赖、队列和派发在其他层 | 本节核对的是编译生成的 Host 调用，不是 Simpler 式设备 ready queue | 当前主链是 CANN kernel launch，不是 AICPU task DAG runtime |
| 可见末端 | 默认 `emitc`；另有 VPTO、LLVM IR、对象/fatobj 路径 | HIVM、RegBase/AVE、LLVM/目标代码；有条件地继续编译 Host | 设备二进制、资源 metadata，交给集成侧 loader/launcher |

#### 5.7.2 PTOAS：“编核内实现”成立，“输入必定只有一个函数”不成立

PyPTO3 的 `_emit_single_function_output` 编一个 InCore 输出；但 `_emit_group_output` 会把一个 Group 的 MLIR module 交给 `_compile_pto_module`，再为成员分别生成 `kernel_entry(args)` wrapper。因此，连当前集成也不能一概描述为“每次 PTOAS 只看一个函数”。相同 Group 内的联合编译有助于处理成员间接口，但并未把外层 Orchestration 编入 PTOAS。[C31（PyPTO3（Simpler））]

PTOAS 自身的 `section_sugar_multi_func.pto` 测试更直接：同一 module 有 `vec_only` 和 `cube_only` 两个 `pto.kernel`，测试期望它们进入不同 core-kind 子 module。它证明多函数/多实现的表示与编译路径存在，不证明任何上层调用顺序已生成。[P5（PTOAS）]

Host 产物还要再分两类：

- **入口 stub/打包**：`emitVPTOHostStubSource` 收集各逻辑入口，生成形如 `extern "C" __global__ AICORE void name(...) {}` 的源文件；`ObjectEmission` 调用 Bisheng 编译该 stub 并嵌入 device object，形成 Host 可链接的 fatobj。这已涉及 Host ABI/入口配套，不能称为“完全不做 Host 生成”。[P6（PTOAS）] [P7（PTOAS）]
- **调用程序/调度**：上述 stub 没有表达“先做 QK，再做 softmax，再做 PV”的程序，也不替调用者选择每次 launch 的 block 数、stream、循环或依赖。仓库独立样例仍在 `launch.cpp` 写 `kernel<<<1, nullptr, stream>>>(...)`；PyPTO3 则在自己的 Orchestration/Simpler 接口提交任务。[P8（PTOAS）] [C3（PyPTO3（Simpler））]

所以，PTOAS 与这里的问题相关的核心边界是：**编译核内计算及其入口配套，不等于接收并实现多阶段任务程序。**本地默认 `--pto-backend=emitc` 也不代表“唯一输出是 C++”；PyPTO3 当前消费 C++ 的接口约定与独立 PTOAS 的 VPTO/fatobj 能力应分别描述。[P1（PTOAS）]

#### 5.7.3 AscendNPU-IR：通用路径确实跨过了设备函数，能够生成 Host launch

**第一步：在 IR 中区分 Host 与 Device，而非看到 `func.call` 就统一当成核内调用。**HACC 使用 `hacc.function_kind = #hacc.function_kind<HOST/DEVICE>` 等属性表达函数角色。同一 module 可以携带 Host 函数、设备入口和 tiling/shape helper；Host 调用 Device 时，后续 lowering 才将其解释为启动设备计算。[F20（AscendNPU-IR）]

**第二步：通用 HFusion 路径可以决定 kernel 边界。**`HFusionOpFusionPass` 对 Host 图既有单 kernel outlining，也有多 kernel outlining；`--enable-multi-kernel` 默认 `false`，开启后可在适用图上拆多个设备函数。这个开关不只是“一次读多个函数”，而是编译器主动做图→kernel 分组。它也不是无条件 fallback：对已标为 Device 的输入，该 pass 在多 kernel 模式明确报 `enableMultiKernel not supported in Device mode`。[F21（AscendNPU-IR）] [F22（AscendNPU-IR）]

**第三步：动态 tiling 与调用控制流也可留在 Host。**AutoSchedule 能生成 tiling 函数、shape/资源查询和候选设备实现；`test-host-multiple-tiling.mlir` 的测试期望是 Host 侧出现 `scf.index_switch`，按 key 调用相应变体。`enable-manage-host-resources` 还是独立开关，默认关闭。因而“生成了两个候选 kernel”不等于一次输入必然 launch 两次，也不能因有自动 tiling 就认为调用者再无内存/参数责任。[F23（AscendNPU-IR）] [F29（AscendNPU-IR）]

**第四步：HACC→LLVM 实际生成启动代码。**`LaunchOpLowering` 虽然名字带 Launch，匹配对象实际上是“调用了 Device 的 Host `LLVMFuncOp`”。它重写 Host 签名为 `[BlockNum, l2def, stream, Args]`，保存并重映射原函数控制流，随后遍历设备调用，构造配置块、kernel stub 和注册逻辑。`emitStubBody` 生成 `rtSetupArgument` 和 `__cce_rtLaunch` 调用；初始化配套还包含 `rtFunctionRegister` / `rtLinkedDevBinaryRegister`。这些是明确的生成代码路径，不只是文档中宣称“支持 launch”。[F24（AscendNPU-IR）] [F25（AscendNPU-IR）]

本地 `hacc-to-llvm.mlir` 的 FileCheck 还检查了 Host 配置调用、`example_stub` 和注册代码；第二个 case 验证设备函数上的 `hacc.block_dim = 20` 覆盖 Host 传入值。源码 `getBlockDimension` 的规则是：优先使用该整数属性，否则保留调用者传入的 BlockNum；stream 来自 Host wrapper。**这个 pass 会生成 launch，但不凭空决定所有输入的最优 launch 配置。**此处是 Ascend block 数，不是 CUDA block 内的线程数。[F30（AscendNPU-IR）] [F24（AscendNPU-IR）]

下面是已核对源码的调用链归纳，不是本机编译产生的 IR dump；Host 中是否真的存在设备调用，决定是否生成 launch stub。

```text
通用 Host 图 / Host→Device 调用 + Device 计算
  → HFusion 分组、outlining、AutoSchedule（按所选入口/开关）
  → bufferization、HIVM/AVE、LLVM dialect
  → separateHostDeviceModule
     ├─ Device：目标 lowering → 设备对象 / mixed AIC+AIV 实现
     └─ Host：HACC→LLVM
              → 参数打包、配置、注册、kernel stub、__cce_rtLaunch
              → Host LLVM IR → Bisheng + Host bitcode/runtime → 共享库
```

**第五步：这不是仅供 CPU runner 的模拟机制。**本地 HIVMC A3 的 `runBiShengLIRCompileA3` 与 A5 的 `runBiShengLIRCompileA5` 都调用 `separateHostDeviceModule`，先编 Device，再按 Host 存在性和编译配置处理 Host；Host 路径继续向 LLVM IR 和共享库编译，存在调用设备的 Host entry 时还走 fatobj link。若只有 tiling/资源查询函数而没有 Host→Device 调用，仍可能生成 Host 库，却不会因此触发上述 launch 重写。目标工具链/bitcode 不齐、只停在中间 IR，或输入没有相应 Host 程序时，不能宣称已经生成可运行的完整 Host+Device 产物。[F26（AscendNPU-IR）] [F27（AscendNPU-IR）]

#### 5.7.4 为什么 Triton-Ascend 当前仍是“后端编 kernel，driver 来 launch”

`compiler.py` 的 A2/A3 和 A5 分支调用后端时设置 `--enable-triton-kernel-compile=true`。AscendNPU-IR 的普通及 RegBase HFusion pipeline 都以这个条件分路：通用 `inferAndOutlineOp → hfusionAutoSchedulePipeline` 位于非 Triton 分支；Triton 分支处理已经形成的 kernel，不能仅加一个多 kernel 开关就当作完整 Host 图编译入口。[F28（AscendNPU-IR）] [F31（AscendNPU-IR）] [F32（Triton-Ascend）]

Triton 接回的是设备二进制，并可从 callback 共享库查询 task type、workspace 大小、同步锁布局/初始化值。这里 `infer_task_type_function` 的“task type”是当前计算入口的执行元数据，**不是生成了跨算子的 TaskId 依赖图**；callback 库也不能被误认成承包了整层模型启动顺序的 Host 程序。[F32（Triton-Ascend）]

实际启动发生在 `driver.py::make_launcher` 生成的代码里：分配/准备所需 workspace，带 blockNum、stream、参数调用 `cann_launch_kernel`；其实现按 CANN 接口分支使用 `aclrtLaunchKernelWithHostArgs` 或 `rtKernelLaunch` / `rtKernelLaunchWithFlagV2`。后端若已生成 Host 资源 helper，与集成侧仍负责 launch 完全可以同时成立。[F3（Triton-Ascend）] [F33（Triton-Ascend）]

本地 Triton-Ascend 的 AscendNPU-IR gitlink 与独立 `npu/AscendNPU-IR` HEAD 不同（第 13.1 节已记录）。这里分别核对源码接口与后端能力，不将二者称为已在本机联编、运行过的同一套工具链。

#### 5.7.5 AscendNPU-IR 的约束到底在哪里

1. **不是“只能一个 IR 函数”，而是入口和 pipeline 有职责限制。**通用 Host 图可多 kernel；已划定的 Device 计算不是可以任意再拆出 Host launch 的程序。`--enable-multi-kernel=false` 是默认策略，`Device + multi-kernel=true` 的拒绝是具体入口约束，不是整个 IR 系统只允许一个函数。[F21（AscendNPU-IR）] [F22（AscendNPU-IR）]
2. **Host launch 不等于设备侧动态 launch/scheduler。**`SplitMixKernel::generateMixKernelDecl` 检查调用者，非 Host 调用 MIX kernel 时明确报 `Currently, MIX kernels can only be called by host functions!`。普通设备 helper call、AIC/AIV 合作和设备内部 SIMT 子程序也不能据名称视作通用动态 kernel 启动；须分别核对各自 ABI。[F34（AscendNPU-IR）]
3. **生成调用不等于拥有 Simpler 式任务系统。**上述 Host 调用顺序、条件变体、局部/跨核同步，不提供同一套 AICPU ready queue、TaskId、依赖追踪和动态派发协议。若要接入 Simpler，应选设备编译输出并适配入口、参数、workspace、mixed-core 同步及 worker 契约；不应把面向 Host CANN launch 的包装直接当作 AICPU task 入口。
4. **“一个 Host entry”的局部要求不等于“一个设备 kernel”。**`execution-engine-create-host-main` 测试包装 pass 要求恰好一个指定类型的 Host entry、输入输出为 tensor/memref，并限制 wrapper 命名。这是 runner 包装的前置条件，不能外推为 HFusion/HACC 所有 module 的函数数限制。[F35（AscendNPU-IR）]
5. **能写/能拆/能 lower 与已验证运行仍分开。**Host 控制流、tiling 变体、MIX、目标 dtype/布局和库调用需要分别合法；本文用源代码和仓库测试定义约束，未运行这些新增 NPU 案例，也没有据此证明任意 Transformer 图都能端到端生成最优多 kernel 程序。

#### 5.7.6 回到五阶段 softmax：函数数为何不能直接等于 launch 数

- **把五个设备函数交给 PTOAS**：即使它们可放在同一编译单元，也只说明这些核内实现能被编译；哪个先执行、怎么传递中间结果，仍需 PyPTO3 Orchestration/Simpler 或另一个调用方表达。PTOAS 编译次数不等于任务执行次数，更不等于 Host launch 次数。
- **把包含五次 Host→Device 调用的程序交给 AscendNPU-IR 通用路径**：若优化后五次调用都保留并各执行一次，Host lowering 可为其生成五个启动调用点；若有分支/循环，实际次数还依赖运行路径。只把五个未被调用的函数放进 module，不会自动获得调用链；若上游融合改变 kernel 边界，次数也要重新计算。
- **把五项数学操作写在同一个 Triton kernel 内**：当前集成先编这个逻辑计算入口，再由 Triton driver 启动；五项 Vector 运算不是五个 launch。能否真正保持单 kernel、是否涉及 mixed 实现、是否有额外 workspace/同步和 GM 往返，仍以对应输入的产物与运行验证为准。

因此，二者的相似处是核内编译链的一部分；关键不同是 **AscendNPU-IR 的通用编译基础设施还覆盖 Host 图→kernel 分组→tiling→Host launch，而 PyPTO3 当前没有把这些上层责任交给 PTOAS。**若仅替换 PyPTO3 的核内后端，应比较 PTOAS 与 AscendNPU-IR 的设备子流水线；若要利用后者的 Host 图编译，则涉及另一套编排/launch 契约，不能归为“只换核内 codegen”。这仍不构成任何路线的性能排名。

H 是“下游本体能力与集成入口能力分开”的第三个 NPU 例子：本次 TLA 编译使用固定子模块的 HIVM/AVE 组件及自身 pass 序列，再由 H 的 artifact/AscendCL 路径调用；这既不是 F 的 Triton driver，也不因链接 AscendNPU-IR 就继承通用 HFusion/HACC 的完整 Host 图入口。I 的 Host JIT 可以发出多个 CUDA launch，则是在另一套 CUDA 编译/执行契约中实现，不能拿来补齐任一 NPU 后端入口的能力。[H-passes] [H-ir-build] [H-execution] [I-dsl] [I-gqa]

### 5.8 AscendC 应作为下层参照，而不是额外的同等级平台

AscendC 直接暴露 LocalTensor / GlobalTensor、TBuf / TQue / TPipe、搬运、Vector/Cube API 与同步等设备编程构件，Host 侧可组织 tiling、workspace 和 launch。这使它在用户感知上最接近 B（PyPTO2-block版）/D（CANNBot DSL） 的核内工程层，在 G（AutoFuse + Inductor） 中则往往由生成器替用户使用。

本工作区可直接从 D（CANNBot DSL） 的 AscendC translator、G（AutoFuse + Inductor） 的 codegen 和 C（PyPTO3（Simpler）） 的 CCE extern 看到实际使用。[D4（CANNBot DSL）] [G5（AutoFuse + Inductor）] [C19（PyPTO3（Simpler））]

必须保留以下区别：

- “最终调用 CANN / Bisheng 工具链”不代表“所有源码都调用同一套 AscendC 高层 API”。
- PTO tile 库、AscendC API、直接 CCE intrinsic、HIVM 及 LLVM intrinsic 是不同边界；在同一个二进制工具链末端汇合，不等于上层语义相同。
- TQue / Channel 的局部生产消费队列与 Simpler task queue 不是一层。
- AscendC 可以用来编写外部计算 kernel 或 tiling kernel；是否由 Simpler 调度取决于外层包装及ABI，而不是 AscendC 自带 AICPU scheduler。

H 的选定 Python TLA 链经 HIVM/AVE lowering 与 `hivmc-a5` 形成设备产物，不能将其记为本节的“生成 AscendC source”路线；torch_npu 的 CATLASS 模板生成 C++ 是另一入口。I 的 CUDA atom/pipeline 则是 GPU 核内 API 的参照，不能与一个 NPU 设备库按整体 runtime 平级替换。[H-passes] [H-compile] [H-inductor-template] [I-mma]

### 5.9 A2/A3、A5、GPU：用“源码支持分支”代替未验证的打勾表

| 路线 | A2/A3 | A5 | GPU |
| --- | --- | --- | --- |
| A（PyPTO2-tensor版） | TileFwk执行及相应目标代码 | 同仓存在A5/950目标与测试；逐算子确认 | 本文不将E（PyPTO on GPU）视为A（PyPTO2-tensor版）的同一个GPU后端 |
| B（PyPTO2-block版） | 有架构/后端选择；不能把A5例子当A2/A3实测 | 本文双动态softmax、VF/paged prefill例子主要在此 | 选定路径非GPU |
| C（PyPTO3（Simpler）） | Simpler架构目录；pypto-lib native PA驱动在此 | PyPTO/PTOAS/runtime有对应目标；native PA该驱动未给A5入口 | E（PyPTO on GPU）为独立checkout和集成路径，不自动继承 |
| D（CANNBot DSL） | AscendC/CANNIR按目标处理 | 有arch35/VF等特化及例子 | 选定路径非GPU |
| E（PyPTO on GPU） | 非此路径目标 | 非此路径目标 | TensorIR/CUDA Tile；历史softmax是sm89，PA库另有sm120示例 |
| F（Triton-Ascend） | 明确A2/A3编译分支 | 明确910_95/RegBase和pure-SIMT等分支 | Triton语言有GPU生态；本表F（Triton-Ascend）特指Ascend后端 |
| G（AutoFuse + Inductor） | AutoFuse相关生成和模板 | v35及A5相关代码/分支 | 选定AutoFuse路径非GPU |
| H（CATLASS DSL） | 本文选定 Python TLA 构建链未建立 A2/A3 可运行证据；不借用 C++ 库覆盖作结论 | 本机 A5 已构建并运行 MMAD/mixed/连续 FA；含 Q=1 失败，详见第 12.1.3 节 | 选定 TLA/HIVM 路径非 GPU。[H-readme] [H-run-env] [H-run-results] |
| I（CuTe DSL） | 非此 CUDA 路径目标 | 非此 CUDA 路径目标 | NVIDIA GPU，按示例 SM/atom/内存原语约束；所引 Blackwell MLA/CLC 不可直接套到旧 sm89，本机未运行。[I-mma] [I-mla] [I-dynamic] |

表中“有分支/测试”不是“当前机器实测可用”，更不是所有 dtype、layout、shape 和融合形式的通用覆盖保证。

### 5.10 H/I 的 MLIR：共享点与兼容边界

H 的 `buildTlaPipeline()` 提供了明确的 pass 顺序：先区分 AIC/AIV/MIX，再处理自动 mutex、extern、pointer、mixed 函数拆分和 tensor descriptor；随后降低 Vector/Cube 区域、block 索引、flag/mutex，最后进入 HIVM/AVE、regbase intrinsic 及 SCF→CF。还有一个能联系第 8 章的局部优化例子：AVE 合并可将 `vsub` 后接 `vexp` 的序列组合为 `vexpdif`。这是**具体核内指令序列优化**，不是自动把任意两个 attention/task 融合。[H-passes]

I 的默认入口是 `cute-to-nvvm`，公共 Python 源码还可见 CuTe layout 操作和面向 NVIDIA 硬件的 atom。此处能确认 dialect 表达与后端调用关系；未公开在所选目录中的配套 pass 实现，不应凭 pipeline 名字补写内部的完整算法。二者都能复用 MLIR 的类型、SSA、region、pass/diagnostic 等基础设施；硬件地址空间、异步效应、layout 合法性和 launch ABI 仍属各自语义。[I-layout] [I-dsl] [I-compiler]

| 对比 | 已有具体共同点 | 进一步共用所需条件 |
| --- | --- | --- |
| H 与 F | 下游均涉及 AscendNPU-IR / HIVM | 固定 revision、构建选项、tensor/memref/layout/同步契约一致；H 输入已带较多物理选择，F 的 TTIR 路径还承担另一组映射工作 |
| H 与 D | 较早使用 MLIR，显式内存与流水 | TLA 与 CANNIR dialect、staging、address-space/effect 定义不同；D 选定主链生成 AscendC，不等于 H 的 `hivmc-a5` 链 |
| I 与 E | GPU 目标与 MLIR 后端基础设施 | CuTe 的 thread/value/atom 与 E 的 TensorIR tile 表达不同；必须设计 lowering 与 runtime 适配，不能换一个 compiler 路径完成 |
| H 与 I | Python DSL + 显式 layout + MLIR + 编译产物 | 先共享数学、尾块、异步所有权与资源约束的测试契约；物理映射和机器码分目标实现 |

H 依赖 CATLASS 锁定的 `AscendNPU-IR@a07821269…`，与第 13 章单列的独立 `90037fe3371c…` checkout 分开；前者再锁定自己的 LLVM/Triton 版本。**已有另一个 MLIR 19 安装不等于具备 HIVM/AVE 的开发头文件和静态库**。这正好说明第 5.6 节的工程问题：能复用编译基础设施，不代表任意同名版本的二进制或 CMake 包可直接互换。[H-ir-build] [H-cmake]

<a id="megakernel"></a>
## 6. megaKernel：一层 Transformer 变成“一个算子”究竟验收什么

### 6.1 五个独立指标，不用一个名字混盖

| 指标 | 可验证的定义 | 不代表 |
| --- | --- | --- |
| K1：单用户入口 | 一次Python/框架operator调用覆盖整层 | 一次Host launch |
| K2：提交收敛 | 记录一次请求产生的Host API、设备launch、控制/调度launch数 | 内部只有一个计算程序 |
| K3：设备侧完整程序 | 多阶段依赖在设备执行体系中推进，无需Host逐阶段提交 | 一个物理kernel；中间tensor不落GM |
| K4：单计算kernel/合作实例 | 在明确定义的设备entry / mixed-kernel边界内执行整层 | 所有数据能驻留片上、所有核可任意同步 |
| K5：跨阶段数据融合 | 中间GM分配/读写被消除或复用，给出字节与生命周期证据 | K1—K4任一项自动成立 |

NPU mixed kernel 可能有 AIC/AIV 不同机器码；应记录其运行协议、entry 和实际设备事件，不能强行按“一份同构 ISA 镜像”衡量异构 NPU。GPU 也不能把一个 CUDA graph replay API 当成图内只有一个 kernel。

用 H/I 作具体校准：H mixed/FA 的合作计算与中间片上路径，只能支持对应子算子的 K4/K5 分析，不能把它提升为整层完成；I 的 GQA Host JIT 静态包含 decode 与 reduction 两次 launch，且保留 GM partial。I 的 TS 组织同一 kernel 的 warp/资源协议，也不是自动满足“整层多阶段设备程序”的 K3。[H-mixed] [H-fa] [I-gqa] [I-task]

### 6.2 各路线当前到哪里，缺什么

| 路线 | 当前源码最强的直接证据 | 一层Transformer的路线判断 | 主要缺口或必须验证的项 |
| --- | --- | --- | --- |
| A（PyPTO2-tensor版） | 多阶段Tensor程序、设备任务runtime；[A13（PyPTO2-tensor版）]已有预处理/投影/cache/attention组合 | K1/K3方向与架构匹配；不必把整层写成一个巨大SPMD body | graph/task分解、所有算子覆盖、runtime开销、实际launch及GM图；本次无整层同口径测量 |
| B（PyPTO2-block版） | 单kernel内vector/cube/流水、动态TND及paged prefill | 可手工扩展合作kernel；不是SPMD理论上做不了整层 | 全局归约/重分片/跨阶段进展、scratch、代码体积、核利用率；未确认整层通用实现 |
| C（PyPTO3（Simpler）） | pypto-lib `decode_fwd`；显式TaskId、SPMD attention、跨阶段/层组织 | K1/K3已有具体模型程序；更强物理融合可逐子图推进 | 需清点worker/control/task entry；不能把整层程序称作一个物理kernel或零GM |
| D（CANNBot DSL） | Host/Device DSL、完整FA/paged混合流水、显式缓冲与控制；已通过选定A5 decode | 手写大合作kernel有表达基础；也可由Host组合多kernel | 尚缺整层原生运行及性能证据；PA尾页/异长范围还需扩展，不能仅凭API覆盖宣布整层完成 |
| E（PyPTO on GPU） | 受支持operator图、直接launch；PA有合并和分头分支 | 当前通用多task程序受frontend/emitter限制；先扩大图覆盖与融合 | 六函数程序历史拒绝、shape模式限制、跨CTA合作、安全进展、真实单kernel覆盖 |
| F（Triton-Ascend） | 原生paged attention + runtime loop；编译器CV pipeline和同步 | 单kernel复杂attention可表达；整层还需跨program/阶段组织 | 通用全局依赖不由普通grid自动解决；Inductor也不保证整层融合 |
| G（AutoFuse + Inductor） | 融合图→kernel+tiling、matmul/vec模板与workspace生成 | 适合编译器逐步扩大融合域；不是自动整层megaKernel已完成 | 分组/extern边界、跨阶段资源与同步、循环/归约支持、生成物及整层证明 |
| H（CATLASS DSL） | mixed MMAD+add、连续 FA，A5 有编译/数值及 mixed lowering 产物；另有 StreamK | 可沿显式 CV 合作 kernel 扩展局部融合；单入口与局部中间数据复用有对象 | Q=1 连续 decode 先补正确性；分页/设备异长、整层重分片/全局进展/GM 仍需实现验收，不能由一份 FA 推为 K1—K5 全满足。[H-run-results] [H-run-artifacts] [H-streamk] |
| I（CuTe DSL） | 分页 MLA、连续 GQA 两阶段、persistent/CLC 与 experimental warp TS 源码 | 合作 kernel 的角色、资源与调度协议有较具体工具；Host JIT 也可组织多 kernel | 本机未运行；GQA 两个 launch 和 partial GM、TS 粒度/有界检查、整层跨 CTA 进展须分别验收。[I-mla] [I-gqa] [I-task] [I-checker] |

以上不是可达成性概率排名。A（PyPTO2-tensor版）/C（PyPTO3（Simpler）） 的已有程序执行框架减少的是跨阶段组织工作的重复；B（PyPTO2-block版）/D（CANNBot DSL）/F（Triton-Ascend） 的核内控制让局部强融合更直接；G（AutoFuse + Inductor） 的优势方向是客户无需重写模型而自动融合；E（PyPTO on GPU） 的当前限制需要在具体 emitter 层判断。

H/I 也归入这个责任分析：H 为 NPU CV 合作提供更多局部可控对象，I 为 GPU warp/资源与 persistent 工作提供工具；两者能减少某些核内协议工程，但所选源码没有据此交付任意 Transformer 整层的设备任务系统。性能机会与工程缺口应继续按 K1—K5 分项判断。

### 6.3 为什么整层远比单个 attention 更难

整层各阶段的最佳分解不同：norm 的行归约、GEMM 的输出 tile/split-K、PA 的请求/head/序列分块必须衔接。主要难题及详细分析位置是：

- **数据与资源：**归约完成、cache 发布、权重预取、阶段重分片，以及最重分支对寄存器/UB/L1、缓冲槽和代码资源的约束，见第 6.5—6.7 节。
- **进展与均衡：**合作参与者能否同时推进、长请求是否拖尾、细粒度派发是否得不偿失，见第 6.8、6.13 节。
- **收益口径：**减少启动与消除 GM 是两项；常驻 worker 仍可用 GM 传数据，局部融合也可保留多次 launch，见第 6.1、6.6 节。

因此，应逐项报告 K1—K5 的现状与缺口，而不是只回答“支持 megaKernel”。

### 6.4 SPMD / MPMD 与 scheduler 的正交关系

```text
外层：MPMD / heterogeneous task DAG
  ├─ norm task：vector norm
  ├─ GEMM task：SPMD GEMM，多个logical blocks
  ├─ attention task：mixed AIC/AIV SPMD attention
  └─ residual task：vector residual

每个task内部可以是SPMD；task之间可以运行不同程序。
不同阶段也可以手工合为一个合作kernel，但需新的内存/同步协议。
```

SPMD 只要求同一逻辑程序按不同ID/数据工作，并不禁止角色分支、runtime循环或不同硬件单元执行不同部分。更重要的是，它没有禁止外层 scheduler 将其作为一个结点。

C（PyPTO3（Simpler）） 的 `pl.spmd` + `deps` 已是具体反例，证明“做SPMD就做不了MPMD”这个推论不成立。反过来，具备MPMD runtime也不会自动生成性能最优的SPMD核内流水。

H 的结构化 mixed 拆分与 I 的 warp 专门化可以放在上述图的“每个 task 内部”理解；I 的 CLC 分配工作 tile、TS 调度 warp/资源，又是内部的两个不同层次。它们不与外层 DAG 原理冲突，但组合需要第 10 章的入口/资源/完成契约，现有源码不能充当已经接入 Simpler 的证明。[H-mixed-pass] [I-dynamic] [I-task]

### 6.5 极致性能的目标不是“kernel 数最小”，而是整层关键路径最短

**单一物理 kernel 是实现手段，不是性能目标函数。** 至少要区分三种极致性能：

- **给定形状的最低单请求延迟**：允许高度专门化、固定资源分工，关注 decode 每 token / 每层关键路径。
- **给定资源的最高稳态吞吐**：多请求、长短序列和多个阶段竞争同一硬件；单请求占满全部核不一定最好。
- **动态分布下的低尾延迟**：batch/长度变化、cache miss、长请求、并发与首次特化都算入；一个固定 case 很快不足以证明这一项。

数值精度、KV cache 布局/复用、并发负载和硬件必须一致才可比较。本文没有九条路线同口径实测，下面给出的是**源码约束下的性能机会与成本模型**，不是预测的加速百分比或成熟度评分。

一个不把重叠时间重复相加的分析方法是：

```text
T_work_lower >= max(
    F_cube / P_cube,
    F_vector / P_vector,
    Bytes_HBM / BW_HBM,
    Bytes_L2 / BW_L2,
    T_dependency_critical_path
)

T_observed =
    已实现调度下的有效工作时间
    + 未被其他工作遮蔽的 launch / 调度 / 同步 / 空闲时间
```

`P/BW` 是所选精度与设备的相应峰值/有效能力；不同数据通路的字节量分别计数。第一式只是宽松资源下界，不承诺 Cube、Vector、内存可以完全重叠。第二式是测量分解要求，不允许把已藏在 GEMM 时间里的调度或 DMA 再加一次。

融合前后真正需要满足的是：

```text
节省的关键路径 launch/dispatch
+ 节省的实际内存流量耗时
+ 新增的计算/通信重叠
>
新增全局同步与重分片
+ 新增 spill/重读/格式转换
+ 专门 kernel 效率损失
+ 新增不均衡与资源占用代价
```

如果融合掉 10 个小 launch，却令权重 GEMM 的有效带宽或矩阵单元效率明显下降，整层可能更慢。反过来，保留多个高效计算 body，用设备任务图缩短暴露的阶段间隔，也可能更接近端到端最优。两者都必须用同一层的 trace 和流量证明。

将模型落到 H/I：H 的 CV/FIX/MTE overlap 可影响未遮蔽等待和局部搬运，I 的 TMA/MMA/warp overlap、CLC 工作分配可影响 pipeline 气泡与尾部；这些机制都不能免掉权重/KV 的必要读取或依赖链。比较时应分别测 H 的 UB/L1/L0 及 I 的寄存器/SMEM/TMEM 占用、有效并发和等待，不能把两种硬件的峰值代入同一个参数后直接排序。[H-mixed] [H-fa] [I-gqa] [I-dynamic]

### 6.6 用 softmax 和 decode attention 定量说明：究竟省了什么

#### softmax：单 launch、少 GM、长行并行是三个独立优化

只按 FP32、无额外复制、行统计量充分复用来算，第 9 章五 executable softmax 的逻辑 global 访问约为：

```text
max: 读 X
sub: 读 X，写 shifted
exp: 读 shifted，写 exponent
sum: 读 exponent
div: 读 exponent，写 Y
合计约 8*M*N 个 FP32 元素 + O(M) 行统计量访问
```

片上能保留整行的融合版本可接近读 X、写 Y，即 `2*M*N` 个元素。历史例子的 `4096×128` 对应主体逻辑访问约 **16 MiB → 4 MiB**。这只是根据代码推导的流量，不是 profiler 已测得的 HBM 流量；L2 命中、广播、事务粒度及编译生成方式会改变真实流量。[E8（PyPTO on GPU）]

第 2.2 节超长行两遍算法则至少读 X 两遍、写 Y 一遍，即主体约 `3*M*N`；同时增加指数/重标定计算，但解除整行驻留要求。对 `M=8` 的长行：

- 每行单核/单 CTA 循环能解决容量问题，但可能只有少量独立工作，不能充分利用全部核/SM。
- split-reduce 可增加并行度，却引入 partial stats、归并、同步和再次访问。
- 用 Simpler/TileFwk 推进 partial→merge→normalize，减少的可能是 Host 提交，不自动消除 GM partial。
- 单合作 kernel 内归并也要证明跨核进展和存储可见性；不能用普通 block barrier 替代。
- 是否更快取决于长行长度、独立行数、内存/指数瓶颈与同步成本，不存在“一 kernel 必胜”的一般规则。

这说明 B（PyPTO2-block版）的完整 `MAX_N=512` softmax、C（PyPTO3（Simpler））固定行 tile 教学例、F（Triton-Ascend）的单行 reduce 都只能证明各自覆盖的局部问题；不能据它们的短代码直接推导超长 reduce 的整机最优解。

#### decode attention：GQA 复用、page 流式算法和阶段流水往往比 launch 计数更重要

对每请求一条 query、无 KV 量化，忽略 softmax 的标量运算，设元素字节数为 `s`，各请求长度为 `L_b`：

```text
QK + PV 运算量约 = 4 * Hq * D * Σ_b L_b
理想 KV 读取量约 = 2 * s * Hkv * D * Σ_b L_b
理想 KV 算术强度约 = 2 * Hq / (s * Hkv)
```

这里假设每个请求的 KV 被其 GQA query heads 复用，且没有重复 page 读取；跨请求共享前缀、L2 驻留和 page cache 会改变实际 HBM 量。BF16/FP16、GQA=5 时，上式约为 **5 FLOP/byte**，不是硬件实测。这使“如何读页和复用 KV”成为低 batch decode 的重要优化方向。

因此，逐 Q head 的多个 kernel 即使都很快，也需检查 KV 是否被重复请求；但**不能把重复逻辑读取直接乘成重复 HBM**，因为 L2 可能复用。一个大 kernel 即使减少 launch，若为每个 query head 保留重复 K/V tile，也可能没有获得预期的数据复用。

C（PyPTO3（Simpler））的 native PA 以一个 KV head 的 5 个 Q heads 组成有效 `[5,128]`、物理 `[16,128]` tile，并把 4 个 page 拼成 softmax stack，这是具体的 GQA/流水选择；E（PyPTO on GPU）则有静态 bucket 与合并/分头编译分支；F（Triton-Ascend）把 GQA 排进 `BLOCK_M` 并在设备按实际 page_count 循环。三者优化自由度和浪费来源不同。[C16（PyPTO3（Simpler））] [E10（PyPTO on GPU）] [E12（PyPTO on GPU）] [F6（Triton-Ascend）]

还要特别区分：

- **取消完整 score/probability 矩阵的物化**：online attention 的核心算法收益。
- **取消每个小 stack 的跨执行单元 GM transfer**：更低一层的通信优化，取决于硬件与流水路径。
- **复用一块 GM ring**：降低 workspace 峰值，不等于这块 ring 不产生访问。

C（PyPTO3（Simpler））当前 PA 的 score/probability/PV 三个 transfer buffer 容量合计约 **3.9375 MiB**，尚未包含同步空间；该数字来自 `1152×512×4 + 1152×512×2 + 1152×128×4`，是**分配容量而非每请求流量**。其存在清楚说明：融合 task 已做成，进一步消除/缩减跨阶段搬运仍是独立问题。[C17（PyPTO3（Simpler））]

H/I 的例子也支持“先看字节与算法，再数 launch”：H 连续 FA 在局部保留在线统计与累加，不代表已承担 page-table 间接访问；I softmax 教程的部分变体先将 exp 物化到输出 GM，kernel 8 改为在线统计后重读输入，同样是单个计算 kernel 却有不同流量。I GQA 的两次 launch 则换取 split 工作分解并付出 partial GM/归并成本；是否值得取决于具体负载，本文没有 I 的实测性能。[H-fa] [I-softmax] [I-gqa]

### 6.7 整层 Transformer 的真正难点：换并行分解，而不只是把函数 inline

整层目标在本文限定为单设备的一层 decoder：norm→QKV→Q/K norm/RoPE/cache append→PA→Wo→residual/norm→SwiGLU MLP→residual；不把多卡通信、采样或 embedding 偷算进“这一层”，也不把它们的缺失算成该层算法错误。

| 阶段/边界 | 常见高效分解 | 强行统一到一个固定分解的代价 | 向 megaKernel 推进需要的能力 |
| --- | --- | --- | --- |
| norm→QKV | norm 按行归约；投影按输出 tile 或 split-K | norm 输出可见性、跨 hidden 分块归约，GEMM 等待；局部 tile 可能不能直接被下一阶段消费 | 归约完成 token、细粒度消费者依赖、数值允许的代数改写 |
| QKV→cache/PA | 投影输出列分块；PA 按请求/KV head/序列块 | output 分片变换、RoPE、cache 写发布、GQA 打包 | layout/索引协议、细粒度发布与跨核可见性 |
| PA→Wo | PA 的 head 独立输出；Wo 消费整个 hidden | 收齐 head 或流式累加；partial output 与归并 | producer-consumer pipeline、split-K/partial-reduce 支持 |
| Wo→residual/norm | 大投影后小 Vector 运算与行归约 | Cube/Vector 比例骤变，固定角色可能空闲 | epilogue 融合、角色重分配、与后续阶段重叠 |
| gate/up→SwiGLU→down | 两个投影可并行；SwiGLU 逐元素；down 要消费 intermediate | 扩维中间值很大，整块常驻不现实；down 的 K 分块要求不同 | 分块产生/消费、片上与 GM 抉择、环形缓冲/原子或归并 |
| 本层→下一层 | 残差/归一化可与下一层部分工作关联 | 跨层 workspace 复用和 WAR 依赖；错误早释造成数据污染 | 跨层 liveness、版本/epoch、明确完成事件和回收协议 |

最值得关注的不是“有没有 for/if”，而是**不同阶段能否以各自合适的分工推进，并交换足够细粒度的数据就绪信息**。

以 dense、非量化、GQA、SwiGLU 层作另一个分析界限：hidden 为 `h`，KV hidden 为 `h_kv=Hkv*D`，intermediate 为 `f`，投影权重元素量约为：

```text
P_layer ≈ 2*h*h + 2*h*h_kv + 3*h*f
投影 FLOPs ≈ 2*B*P_layer
读取一次权重字节量 ≈ s_weight*P_layer
投影的权重算术强度 ≈ 2*B/s_weight
```

不计 norm/RoPE 等小权重；假设权重不能全部留在可用 cache，并在该批次中读取一次。BF16 且 `B=1` 时该理想强度约 1 FLOP/byte；**把层变成一个 kernel 不会自动消除必须读取的大权重**。batch 增大、量化、权重布局、跨请求复用、L2 行为会改变约束，这些都可能比再少一次 dispatch 更关键。若权重已驻留 cache，必须换用该层级的字节量，而不能继续套 HBM 下界。

C（PyPTO3（Simpler））的 `decode_fwd` 已提供比概念图更强的证据：它延后 RMS 的标量因子、组织 split-K 投影、显式安排 TaskId，并将上层输出与下一层输入处理关联起来。但这些是**作者主动设计的模型优化**，不等于给任意模型加 `@pl.jit` 都会自动得到同样优化；有限精度下的乘法重排还需要数值验证。[C20（PyPTO3（Simpler））]

H/I 的资源配置使“换阶段需重做分解”有更具体的输入：第 8.12 节按 H 默认 FA 声明计算的 UB 对齐占用约 205 KiB，且 L1/L0 各自另计；这是源码分配预算，不能当成最终峰值或硬件容量。I GQA 作者先扣除 Q/P/统计量/barrier，再据剩余 SMEM 求 KV stage 数。若继续融合 Wo、residual 或 norm，新增生命周期可能迫使两者减槽、改 tile 或经 GM 交接，不能只把函数内联后假设资源仍 fit。[H-fa] [H-scratch] [I-gqa]

### 6.8 三种实现策略：任务图、合作 kernel、自动融合，各有不同的最优区间

| 机制 | 极致性能机会 | 主要代价/退化条件 | 主要对应路线 |
| --- | --- | --- | --- |
| 设备任务图 + 各阶段专门 kernel | 各阶段保留合适 tile、核心数和代码资源；可重叠独立任务、隐藏 Host 间隔 | task 太细则构图/依赖/派发占比高；task 边界多 GM；资源组/affinity 不佳影响流水 | A（PyPTO2-tensor版）、C（PyPTO3（Simpler）） |
| 大合作 kernel / persistent SPMD，含角色分工 | 阶段间直接通信、细粒度流水和局部复用；减少部分调度边界 | 全局同步、资源最重分支、代码/寄存器压力、动态不均衡、进展/退出协议 | B（PyPTO2-block版）、D（CANNBot DSL）；C（PyPTO3（Simpler））的 SPMD task；F（Triton-Ascend）须在后端/协议支持范围内；H（CATLASS DSL）的 CV 合作、I（CuTe DSL）的 warp/persistent/CLC/TS 也在此层比较，均不据此认定整层已实现。[H-mixed] [I-task] [I-dynamic] |
| 图编译器逐步扩大自动融合域 | 客户少改代码；可以统一搜索融合/tiling，避免人为过早切 kernel | 模式/模板/成本模型覆盖不足就拆分或 fallback；编译搜索预算；跨阶段 effect/循环难 | G（AutoFuse + Inductor）、F（Triton-Ascend）的 Inductor 路径；E（PyPTO on GPU）当前受集成 emitter 限制 |

这三种策略可组合，而不是只能三选一。**任务图可以包含强融合合作 kernel；自动图编译器也可以生成任务图或合作 kernel，只是本地实现不能用理论可扩展性代替现状。**

两个特别容易被忽略的上限：

1. **调度带宽上限。** 若实测任务派发能力是 `Q_dispatch` 个 task/s，则一个请求 `N_task` 的依赖推进存在相应吞吐约束；串行链还受每条边可暴露延迟限制。粗化 task 可缓解，但也可能降低并行和增加资源需求。A（PyPTO2-tensor版）/C（PyPTO3（Simpler））不能只报告“所有核忙”，应报告 task 粒度、派发速率、关键链空洞和 scheduler 占用。
2. **驻留/进展上限。** 一个巨大的 GPU kernel 的寄存器/shared memory、线程数和代码路径会限制可驻留 CTA。仅在普通 launch 中给跨 CTA 自旋加原子/fence，并不能保证未被调度的生产者有机会运行；合作 launch 或其他经过证明的协议才构成正确的前提。CUDA 的 cooperative grid 同步有明确 launch/资源约束，并非普通 grid 的默认能力。[CUDA Cooperative Groups](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html)

NPU 不能照搬 GPU 的寄存器占用公式或 grid 限制，但同样要审核 AIC/AIV 参与集合、资源组、等待事件来源及 worker 协议。`sync_start`、`allow_early_resolve`、AIV sub-block 都是具体契约，不是“加了同步即可安全”的装饰参数。[C6（PyPTO3（Simpler））] [C14（PyPTO3（Simpler））]

### 6.9 九条路线逐项评估：当前现状、极致性能可能性、用户代价与差距

当前实现概览见第 6.2 节，调用链和源码见第 4 章；以下集中比较性能上探、用户代价与差距。“可能性”是补齐缺口后可争取的方向，不是已达到的硬件峰值比例。

#### A（PyPTO2-tensor版）：程序级组织已有基础，极限取决于 task 粒度与核内质量是否协同

- **性能上探**：能够保留多个专门阶段，用设备侧推进/重叠来接近整层关键路径下界；对形状不齐、阶段分工差异大的程序，具有避免手写整层大 body 的架构空间。核内生成质量、跨 task GM 和调度粒度仍决定实际极限。
- **用户代价**：客户迁移到 Tensor DSL 并处理布局/动态契约；普通算子作者可少写显式流水，但追求极限时仍需 tile/pass 选项、task 切分和图级诊断。调度框架降低的是跨阶段组织的重复实现，不免除性能工程。
- **差距**：本次没有整层统一 trace、task 数/代价、GM 读写和最优库基线，不能声称整层性能已领先。下一步应先输出一个真实层的 task DAG/GM 图，定位是 dispatch-limited、memory-limited 还是 kernel-limited，再决定粗化或融合哪一段。

#### B（PyPTO2-block版）：低层控制直接，整层协作协议与可复用策略仍需作者/框架承担

- **性能上探**：针对固定/分桶负载，可主动安排 double buffer、核心分工、矩阵/向量流水和局部数据传递，争取接近手工优化核内实现的效果；没有“SPMD 必然比 MPMD 低一个性能上限”的理论约束。
- **用户代价**：更容易把性能意图写到核内，也更直接承担有效窗口、地址容量、局部别名、同步和策略输入一致性。prefill 的 Host 分配策略依赖实际长度时，服务系统要提供/更新这些信息，不能无代价读取设备长度后再回 Host。
- **差距**：已找到的 PA 例子不是 decode 专项性能证明；整层还缺分阶段重分片、全局归并、正确进展、统一 workspace 生命周期及可维护封装的直接证据。适配为 Simpler extern task 可以补外层组织，但需要第 10.3 节的 ABI 工作，不能直接复用原 Host `<<<...>>>` launcher。

#### C（PyPTO3（Simpler））：已有整层/多层程序和强融合 PA，现阶段最需要拆清“完成到哪一层”

- **性能上探**：外层任务图 + 内层强融合构成很宽的优化空间：阶段保留不同资源分工，在合适边界融合、提前发布和流水重叠，而不必一次重写整层物理 body。可以追求很低的整层延迟，但不能从“空间宽”推出调度、PTOAS 和生成代码已经最优。
- **用户代价**：普通 Tensor/InCore 写法与极限模型写法差别很大。现有模型使用 `manual_scope`、显式 TaskId 数组、early-resolve、split-K、GM transfer ring、cache/fence/sync 协议；这实际上把相当一部分调度/内存责任交回专家作者。库客户可以不感知，维护库的团队仍要承担。
- **具体差距**：当前 PA 仍有三块 GM transfer；`manual_scope` 下跨 task 依赖必须手工正确；批次大于 `BATCH_PAD` 时按窗口连续推进，scratch-ready 依赖约束复用，不能据“支持更大 batch”推出已经最佳吞吐；这里的 Qwen 融合 PA 驱动仍限定 A2/A3，不能用另一份通用 PA 的 A5 通过结果替代它。应测整层关键链、窗口串行化、跨阶段流量、scheduler 成本及并发安全，再评估能否缩减 GM/同步或改变分工。

#### D（CANNBot DSL）：适合表达深度流水，已有 paged decode 对象，整层仍需验证

- **性能上探**：作者可控制 buffer 槽、数据格式、搬运路径和流水时序；对资源/数据流清楚的子图，具备争取低搬运、高重叠的表达手段。若某硬件/后端支持直接 CV 传递，可减少部分 GM，但这不是所有 A2/A3/A5 场景的默认保证。
- **用户代价**：相对 Tensor/框架表达，需要更多 kernel 结构设计和资源推理；Channel 帮助表达所有权，不替作者决定最佳切块、分工、Host 分发和全局同步。完整整层会把工程复杂度放大到跨阶段，而不只是多写几个算术 API。
- **差距**：本次 native paged kernel 及补充 decode 已通过；接下来需扩大页尾/异长/GQA/长尾负载的覆盖，再验证两种不同分工阶段间的合作及性能，仍不能直接评为整层已就绪。`@aicpu` 能编 AICPU 函数不等于已有任务 runtime。

#### E（PyPTO on GPU）：要区分 GPU/TensorIR 的潜力与当前 PyPTO emitter 的覆盖差距

- **性能上探**：TensorIR/CUDA Tile 的 tile、矩阵计算和 GPU 执行资源允许研究更大融合、GQA 复用和静态专门化；**不能从这个底座的潜力倒推当前 PyPTO 接口已支持通用整层**。TensorIR 自身 README 也明确处于 early release，不作为生产性能承诺。[E14（PyPTO on GPU）]
- **用户代价**：客户需理解当前支持模式、shape/stride 规格、bucket/分头行为；NPU Tensor 写法未必可直接编译。极限优化很可能要同时修改高层 emitter、typed module 构造和下游调度，而不仅调一个 grid。
- **差距顺序**：先让所需局部融合图可靠到达后端，再补跨算子图/多函数支持、动态契约、跨 CTA 协作与安全 launch，最后才有通用整层比较资格。GPU 不需要 AICPU，但如果选择动态设备任务图，仍需实现相应 GPU 调度协议；现有硬件 CTA 调度不会自动识别任意 task DAG。

#### F（Triton-Ascend）：客户表达紧凑，kernel 内复杂性由后端承担；整层需要超出普通 program 映射

- **性能上探**：作者保留 program/GQA/tile/online 算法控制，后端可跨 tile 运算优化资源与流水，减少显式物理工程；高性能复杂单算子是有代码基础的方向。实际能否接近手写极限，要看生成的 CV/布局/缓冲和静态/动态循环优化。
- **用户代价**：原生 Triton 作者要设计工作分解、mask、constexpr、并行与算法；比显式地址代码紧凑，不代表不用调优。通过 Inductor 的客户迁移成本较低，但后端选择与 fallback 仍需审计。
- **差距**：普通 `program_id` 和 AutoBlockify 不提供任意跨 program 动态依赖图。完整层需要安全的跨阶段重分片/协作和融合选择；单 kernel 也可能被后端插入 CV GM workspace，所以不能仅凭两个 `tl.dot` 在一个函数里就声称零 GM。还需统一 A2/A3 与 A5 的实际生成/验证。

#### G（AutoFuse + Inductor）：自动化用户成本最有吸引力，性能极限取决于融合域与策略覆盖

- **性能上探**：若图覆盖、成本模型和资源规划足够，可以自动生成接近手工安排的融合区间，并在不同 shape 自动选择策略；不存在“客户不显式写 buffer，所以理论上永远不能极致”的结论。
- **用户代价**：客户保留 PyTorch 的成本低；达到某模型极限时，工作转移给 lowering/template/tiler/fusion planner 开发者。若必须依靠不透明 pattern 或频繁 fallback，用户调试性能的成本会上升。
- **具体差距**：所选 Inductor 接入当前明确拒绝 matmul prologue、多处 indirect indexing 的合并，以及若干 indirect indexing/归约组合。[G12（AutoFuse + Inductor）] 这些是 PA 与整层融合需要跨越的实际入口约束，不只是抽象的“还需更多优化”；不能把该入口的限制推广到所有 AutoFuse 入口。仍需证明 native paged decode、动态图/状态写入/复杂归约的覆盖及选择质量。Inductor 捕获一整层不代表送给 AutoFuse 一个整层融合图，送进去也不保证一个物理 kernel；应给出 graph-break/extern/generated-kernel 清单和全层 trace，不用“模型可 torch.compile”代替。

#### H（CATLASS DSL）：物理融合和 A5 流水有具体实现，整层推进还需要额外设计

连续 FA 展示了 QK→softmax→PV→rescale 的 mixed kernel；`basic_mixed` 则把 MMAD 结果经 FIXPIPE 送入 UB，与另一个输入相加后写回。对第 6.6 节的性能模型而言，H 可以具体控制 KV tile、L1/L0/UB 多缓冲、矩阵/向量交接和寄存器计算，以减少选定阶段的 GM 往返。StreamK 还展示了改变工作分解与经 GM workspace 归并 partial 的另一种选择，不能把 H 概括为只会静态单 tile GEMM。[H-fa] [H-mixed] [H-streamk]

从这些算子扩到整层仍须安排投影、norm、attention、MLP 各阶段的分解转换、全局完成条件、GM 生命周期和资源再分配；一个 `tla.jit` helper 内联不会完成这些工作。H 的当前优势方向是让算法作者明确表达硬件流水，代价是布局、事件协议、尾块、工具链版本与特化组合的维护。**单算子通过只为 K4/K5 的局部实现提供验证对象，不能替代整层 K1—K5 的验收。**

#### I（CuTe DSL）：persistent 与 warp 专门化补充合作 kernel 路线，仍需辨明全局边界

I 有完整 softmax、GEMM、连续 GQA、分页 MLA，以及静态/动态 persistent 和实验性 TS。用户可以控制 copy/MMA atom、线程与值分区、TMA/SMEM/TMEM、生产/消费阶段与 warp 角色，因此更容易把第 6.7 节的合作 kernel 方案落实成可读代码。它比“一个 CTA 只做一个 tile”的单一例子有更丰富的工程工具，但这只是相对于该简单写法的能力增量，不是已测出的性能优势。[I-softmax] [I-gqa] [I-mla] [I-static] [I-dynamic] [I-task]

persistent 的收益取决于局部效率和工作不均衡；更多专门化 warp、barrier 和 TMEM/SMEM 占用也可能减少可驻留 CTA。整层不同阶段的输出分区仍需重排；跨 CTA 依赖与全局归约不能自动从 warp 级 TS 推出。I 的 GQA simple 选择两个 launch 加 GM partial，就是为了用明确的归并阶段衔接 split-KV；减少第二个 launch 是否值得，要与资源和进展协议的代价一起测量。[I-gqa]

### 6.10 用户使用代价：必须把“客户只调用一行”与“谁维护这一行后面”拆开

下表是基于接口职责的**定性成本分析**，不是人天估计。已有成熟算子库可显著降低客户侧成本，但不会消除维护者成本。

| 路线 | 客户接入/迁移 | 追求整层极限的专家工作 | 动态能力带来的持续维护 | 主要排错对象 |
| --- | --- | --- | --- | --- |
| A（PyPTO2-tensor版） | Tensor DSL/布局/运行集成 | tile/pass、任务粒度、融合、核内生成质量 | shape/valid-window、任务展开、缓存与资源配置 | 图/子图/task、设备控制与算子两个层次 |
| B（PyPTO2-block版） | 调用库可很轻；原生开发需 block版 DSL | TileGroup/地址/mutex、Host partition、跨阶段协议 | 容量、metadata 策略一致性、kernel 变体 | 核内内存/同步、Host 参数及编译 ABI |
| C（PyPTO3（Simpler）） | 可封装模型 API；开发需 Tensor/scope 概念 | 普通 scope 到 manual TaskId/extern/SPMD 的能力跨度很大 | task/workspace 生命周期、batch window、page/shape 变体 | Orchestration、Simpler、InCore/PTOAS、库层手工协议 |
| D（CANNBot DSL） | 库调用轻；原生开发需 Host/Device staging | Channel/Buffer/VF、layout、流水、全局协作 | 动态 TensorSpec、有效访问、策略及架构特化 | Python staging→CANNIR→生成 AscendC→硬件 |
| E（PyPTO on GPU） | 支持模式内轻，NPU 程序迁移受限 | emitter/图覆盖、Tile IR 调度、CTA 协作、分桶 | shape/stride cache、bucket、合并/分头分支 | 模式拒绝、编译 artifact、CUDA trace |
| F（Triton-Ascend） | PyTorch 路径较轻；原生用 Triton | 算法/program/tile、后端 layout/CV/sync 协同 | constexpr、mask、数据循环、编译分支 | TTIR/适配IR/HIVM、workspace/launch 与原图 |
| G（AutoFuse + Inductor） | 保持 PyTorch 的迁移最少这一类 | 模板/lowering/自动 tiling/融合成本模型 | guards、图覆盖、tiling 与外部算子变化 | Inductor 分组、ASCIR/生成代码、fallback |
| H（CATLASS DSL） | 调用已有封装可轻；原生开发需 TLA 物理路径 | layout/搬运、L1/L0/UB/Vector、CV 协议与整层资源复用 | 静态容量/动态逻辑范围、FA 常量变体、分页及尾块 | TLA→mixed/HIVM/AVE→产物、manifest/ABI、数值与异步协议。[H-passes] [H-fa] |
| I（CuTe DSL） | 封装可简化调用；kernel 作者需掌握 CuTe layout/atom | 线程/warp/cluster、pipeline、persistent/CLC、SMEM/TMEM 与整层分解 | SM 目标、constexpr/tile/stride、变量序列/split，experimental 接口 | Host/kernel staging、layout/资源、CUDA 产物、pipeline/TS 协议；检查器不替数值测试。[I-dsl] [I-ts-memory] [I-checker] |

若某条路线只展示“客户一行调用”的优化库，另一条展示“从零手写 kernel”，不能据代码行数比较开发成本。应固定三个任务：**使用现成算子、为新 shape 扩展算子、实现新的整层融合**，分别记录谁改哪层代码。

服务场景还有独立成本：warmup/首次编译、编译缓存规模、workspace 峰值、并发资源独占、异常取消/超时与观测能力。长驻计算可能降低单请求间隔，也可能影响多租户公平和资源交还；这部分在本次源码审计中没有完成同口径验证，不能默认任一路线免费解决。

### 6.11 从现状到目标：用具体工程关卡衡量差距，而不是给“支持 megaKernel”打勾

| 验收关卡 | 必须交付的证据 | 当前本地证据与仍缺的部分 |
| --- | --- | --- |
| 完整数学/状态语义 | 整层数值、KV cache 写入、残差、尾块与 shape 契约 | C（PyPTO3（Simpler））有模型程序；A（PyPTO2-tensor版）有组合程序；其他路线的局部例子不能替代完整层验收；H 的 Q=1 连续 decode 已失败，应先处理数值；I 的源码/协议检查不等于分页 MLA 或整层数值通过。[H-run-diagnostics] [I-checker] |
| 用户入口与图覆盖 | 哪些节点进编译器、哪些外部调用、哪些 graph break | G（AutoFuse + Inductor）/F（Triton-Ascend）需展开 Inductor 分组；E（PyPTO on GPU）先过 emitter 支持边界 |
| 提交与调度收敛 | Host API、设备 entry、控制/worker、内部 task 分别计数 | A（PyPTO2-tensor版）/C（PyPTO3（Simpler））已有 device runtime；整层 K2 仍需 trace；H mixed 产物的函数数不能代替事件数，I GQA 两次源码 launch 仍需目标 trace。[H-run-artifacts] [I-gqa] |
| 跨阶段数据融合 | buffer 生命周期图、分配峰值、实际 GM/L2/HBM 流量 | C（PyPTO3（Simpler））已有显式 transfer，可定位差距；其他后端必须检查生成物，不能只数源码 tensor；H 的静态分配和 I 的 phase alias 可检查局部复用责任，仍需最终峰值、GM 及硬件计数。[H-scratch] [I-ts-memory] |
| 全设备合作正确性 | 每个等待的生产者/参与集合、资源驻留/进展、压力测试 | 不能从单 softmax/FA 的正确性升级为整层无死锁保证；H 的局部 AutoSync、I 的 TS 有界 checker 各有作用域，不能据其存在免除整层进展测试。[H-auto-sync] [I-checker] |
| 端到端性能优势 | 同硬件/精度/输入/缓存/并发的最优分 kernel 基线；p50/p99/吞吐 | 本次没有跨路线结果；不提供性能名次 |
| 用户和维护可承受性 | 新 shape/新模型的改动层、编译成本、排错路径、回归矩阵 | 由接口职责可判断成本在哪里，实际人天和故障率仍需项目数据 |
| 可替代/可共用性 | 后端/内存规划/scheduler 适配原型及回归 | 第 10 章是边界评估，不是兼容性验收 |

更有区分度的原型不是“把三个 pointwise inline”，而是：

1. **长 reduce softmax**：少量行、跨核 partial/merge/normalize，测进展与小任务代价。
2. **动态 paged decode**：固定 metadata shape，只改 `actual_seq_len`，再独立改 `B/P/T`，测编译/tiling与负载分配。
3. **PA→Wo→residual/norm**：并行分解发生两次变化，能看出任务图和合作 kernel 的真实组织差异。
4. **完整 Transformer 层**：保留最优分阶段基线，逐边界扩大融合；每一步同时记录“省下什么、新增什么”。

这些是下一步取证方案，不意味着本次已实现这些原型或九条路线已跑通同一测试。

### 6.12 本章的判断：最可能获得收益的边界，各路线并不相同

不要预设任务图、手工 DSL 或自动编译的性能胜负；应按目标负载联合优化算法、分解、流水、数据驻留和调度。最优形态可能是一个合作 kernel，也可能是设备程序中的几个强融合 kernel；各路线当前基础、优化接缝和责任位置分别见第 6.2、6.9、6.10 节。

推广成本取决于专家责任能否通过稳定库/API/诊断工具封装。**已有局部 kernel、已有设备程序、全层正确性、同口径性能、动态/并发稳健性**仍须分开验收，不能互相替代。

H 的直接机会是将已有 CV 数据路径扩展到自然相邻的计算链，同时先解决连续 decode 的精度和分页/动态契约；I 的直接机会是用 layout/warp/pipeline 与 persistent 工具优化已明确的合作计算。两者与 A/C 的设备任务组织可在理论上互补，但当前不把 H 的构建成功或 I 的 TS 工具当作整层性能结论。第 6.14 节进一步比较其调度粒度。[H-run-results] [I-task] [I-dynamic]

<a id="scheduler-performance"></a>
### 6.13 专题：设备动态 scheduler 到底让 megaKernel 的“极致性能可达成”容易了多少？

**有条件地支持“PyPTO3 更容易走向程序级 megaKernel”这个判断：现成的设备任务体系减少了组织整层的基础设施工作，显式 InCore/SPMD/extern 接口又提供了核内与跨任务协同优化的接缝。但相对于同样有设备任务体系的 PyPTO2-tensor版，不能仅凭 scheduler 存在判定 PyPTO3 更容易达到性能极限。**

#### 先区分四种“更容易”，它们不是同一结论

| “更容易”的具体含义 | 设备 task runtime 能提供的帮助 | 仍然需要解决的事情 |
| --- | --- | --- |
| 更容易表达整层 | 允许不同阶段保留不同 kernel/资源分工，以依赖连接 | DSL/算子覆盖、状态副作用、动态循环与参数契约 |
| 更容易做到设备侧推进 | 已有 task 构造、就绪/派发/完成机制，不必由客户从零实现 | 任务生成速度、设备控制开销、worker 协议、并发/失败处理 |
| 更容易逐步逼近性能最优 | 可以局部融合、改变粒度、重叠独立任务，而不必每次重写整个合作 body | 核内代码质量、数据驻留、布局转换、重分片、调度与内存协同 |
| 更容易维护/推广给用户 | 若库封装稳定，客户可以只调用模型接口 | 专家手工依赖和局部协议的维护成本仍在；接口开放得多不等于自动化更多 |

前两项是已有体系很直接的工程优势；后两项是**条件性机会**。特别是“有较多可调旋钮”不等于“找到最优点的搜索成本低”。

#### 对当前 PyPTO3 的一个具体反例：外层动态，不等于核内工作自动被再切分

当前 native PA 先把 attention 作为 `ATTN_SPMD_BLOCKS=24` 的 SPMD task 提交，随后在 kernel 内执行以下循环。摘录来自原实现，`core` 是 logical block ID，相关形状/常量在入口已定义：[C16（PyPTO3（Simpler））] [C23（PyPTO3（Simpler））]

```python
for task in pl.range(core, num_tasks, ATTN_SPMD_BLOCKS):
    batch = task // NUM_KV_HEADS
    kv_head = task % NUM_KV_HEADS
    seq_len = pl.read(seq_lens, [batch])
    page_count = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    stack_count = (seq_len + STACK_TOKENS - 1) // STACK_TOKENS
```

这里 `num_tasks = active_batch * NUM_KV_HEADS`。变量名 `task` 指 **kernel 内的请求/head 工作项**，不是每项都向 Simpler 提交一个独立 TaskId。真实长度决定本 block 的 page/stack 循环；它没有在这里把长序列再次切成可被外层 scheduler 独立窃取/派发的任务。

以该实现的 `NUM_KV_HEADS=8` 为例，若 active batch 为 1，主 PA 循环只有 8 个请求/head 工作项；24 个 logical blocks 中只有 8 个会进入这段循环。这里仅分析该循环，不把此前 norm/RoPE/cache 阶段或 AIC/AIV 子角色也算成空闲，也不把 logical block 数当物理核心数。**再长的序列也不会仅因外面有动态 scheduler，就自动把这 8 份工作拆成 24 份。** 要增加该段并行度，需要 split-sequence/partial attention + merge，或其他算法分工改造。

同理，不同请求长度很不均衡时，固定 stride 分配可能留下长尾；Simpler 能管理这个 SPMD task 的外层依赖，不代表能任意拆开正在运行的 page 循环重新均衡。`sync_start=True`/资源组协议有助于合作任务进展，也会约束任务什么时候能一起启动，不能当作零代价的“随时填空闲核”。

这也解释一个看似相反的现象：block版 的所选 paged prefill 示例虽然没有通用设备 DAG scheduler，却在 Host `build_work_ranges` 中主动做了分工策略；某种静态/分桶长度分布下，这类分工可能很有效。反过来，若长度只有设备端知道、请求变化快，Host 策略元数据维护又可能增加成本。两者必须比较**实际采用的分工算法**，不能只凭“动态 scheduler 对静态 launch”预测负载均衡。[B15（PyPTO2-block版）]

#### 性能极限：scheduler 能改善哪项，不能替代哪项

作为瓶颈分析而非性能预测，可写出一个简化下界：

```text
T_layer ≥ max(
    T_dependency_path,
    cube_work / effective_cube_throughput,
    vector_work / effective_vector_throughput,
    slow_memory_bytes / effective_memory_bandwidth,
    dispatch_units / effective_dispatch_throughput
)
```

这里的 throughput 必须对应实际资源/工作类型；内存字节数指所分析层级的真实流量，不是仅按源代码数组大小推算。`dispatch_units` 要按 runtime 真正派发的粒度统计，不能直接用 Python 调用数或唯一 TaskId 数替代。该式忽略了一部分争用/串行化，不能当实际耗时预测；各项也不能不考虑重叠就全部相加。

- **外层调度**可减少 Host 逐阶段推进间隙、在已有合法并行性中选择 ready work、配合任务粗化/提前解析/异构资源组织缩短空洞。
- **核内优化**决定真正的 Cube/Vector 吞吐、搬运重叠、寄存器/UB/L1/L0 使用与数值实现。scheduler 不会把低效 GEMM 自动变成高效 GEMM。
- **算法/分解**决定有没有足够并行性，是否重复读权重/KV，是否需要 partial/merge，阶段之间怎样重分片。scheduler 不会创造依赖图中不存在的独立工作。
- **内存与融合**决定中间值是否落 GM、是否能复用/直接传递。runtime 能正确管理 GM 生命周期，不等于消除了这些访问；反之，过度融合也可能增加资源压力或破坏阶段并行。
- **调度自身**也可能成为瓶颈。task 太细、描述生成/依赖解析/队列写入太密集时，省掉的 Host launch 可能被设备调度与中间 GM 成本抵消；把五个很短的 softmax 数学步骤硬拆成五个 task，就是需要先检验的形态。

对 PyPTO3 而言，较有利的组合不是“尽量多切 task”，而是 **外层保留值得动态组织的阶段，内层融合那些需要局部数据复用且工作粒度足够大的片段**。当前 `pypto-lib` 的强融合 PA + 多阶段模型程序已经体现这种组合思路；但 transfer ring、手工依赖和 batch window 等现状说明 K3 程序级组织并不自动完成 K5 数据融合与最优吞吐。

#### 当前对比结论与仍需补齐的证据

| 对比问题 | 当前可以下的结论 | 不能省略的验证 |
| --- | --- | --- |
| PyPTO3 对 block版/CANNBot 的程序级组织优势 | 已有外层设备任务体系与整层/多层程序入口，可少从零实现一套任务 runtime；block版/CANNBot 的当前主路径没有同等外层体系 | 最优强融合 kernel/Host 提交基线、调度开销、跨阶段 GM、并发与进展；不能直接推为更快 |
| PyPTO3 对 PyPTO2-tensor版的优势 | 显式 TaskId/SPMD/extern 组合在当前库中有直接示范；tensor版也有 tiling、合图和设备任务体系 | 同目标任务分解、相近核内质量下的 ready/dispatch/finish trace；分别找双方最佳分解，不只比较默认教学代码 |
| 单物理合作 kernel 是否比任务图更优 | 某些固定负载可少付 dispatch/中间传递代价；任务图则可保留阶段特化与更灵活组织 | 全局进展、资源预算、重分片、代码体积、尾部不均衡；二者理论上没有固定胜负关系 |
| 哪一种用户代价更低 | 使用已封装库都可以很轻；新增 kernel/形状/整层策略的责任分布不同 | 区分客户接入、算子作者调优、编译/runtime 团队维护，不能拿“一行调用”比较底层工程量 |

若要进一步区分 PyPTO2-tensor版与 PyPTO3，建议做两组对照：第一组尽可能统一核内算法/数据布局，观察任务组织本身；第二组允许各自选择最佳融合/tiling/任务粒度，观察真实整层可达性能与实现代价。前一组帮助定位 scheduler，后一组才代表路线整体；只做其中一组都可能误判。当前没有这些 NPU 对照结果，本文的判断停留在**已实现机制、具体差距和工程可达路径**，不把推断写成性能名次。

---

相对 H/I，本节“设备任务体系减少整层组织工作”的判断仍成立于跨计算入口这一层；但若目标是一个已划定合作 kernel 的局部角色流水，则 I 的 TS/pipeline 也能减少协议维护，H 也有受限 AutoSync。应按具体被省去的工程工作比较，而不是把存在不同粒度的 scheduler 归为同一种优势；相应源码与性能模型映射见第 6.14 节。[H-auto-sync] [I-task-manager]

### 6.14 CATLASS / CuTe 的 scheduler：调度粒度与责任层次

#### 6.14.1 五种具体机制，不用一个 scheduler 标签合并

| 机制 | 被调度的工作 | 决策位置 / 进展来源 | 对整层 megaKernel 的实际帮助与边界 |
| --- | --- | --- | --- |
| A/C 设备任务体系 | 具有核内入口、参数和依赖的计算任务 | 设备控制/调度与 worker 按就绪状态推进 | 减少跨算子组织与 Host 参与；task 内部分工仍由该 kernel 负责 |
| H FA 的固定步长循环 | `(batch, Q tile, head)` 工作项 | `block_idx()` 起点、`block_num()` 步长；无逐项就绪队列 | 一组物理核重复处理更多工作；不自动拆分长请求或调度任意后继算子 |
| H StreamK / I 静态 persistent | GEMM tile 或 K 区间 | 算法制定工作映射；静态 persistent 的后续 tile 可由固定规则求出 | 缓解末轮/分解不均衡或复用调度结构；不是运行时检查依赖后的任意任务派发 |
| I CLC 动态 persistent | 待执行 CTA/cluster 对应的 tile | 利用 Cluster Launch Control 获取尚未启动的工作，并处理取消/接管结果 | 动态补充 tile，改善某些长尾；工作域与 kernel 已确定，不是多算子 ready DAG |
| I 实验性 TS | 同一 kernel 内由若干连续 warp 执行的任务及资源阶段 | 编译期声明 schedule/dependency，运行时执行 acquire/commit/wait/release 协议 | 降低复杂 warp 专门化流水的协议编写成本；不能直接当成设备端模型任务 runtime |

来源：[H-fa] [H-streamk] [I-static] [I-dynamic] [I-task] [I-schedule]；A/C 的证据与性能分析见第 4.8/4.10/6.13 节。

这也使 SPMD/MPMD 的讨论更具体：I 的 CTA grid 可以总体执行同一 kernel，CTA 内又让不同 warp 执行 TMA、MMA、softmax、修正等不同程序段；H mixed kernel 则由 AIC/AIV 执行不同角色。二者都可体现**局部角色专门化**。A/C 外层派发不同核内入口时的 MPMD 属于另一层；不能依据局部角色分工就宣布已有相同的整层任务系统。

#### 6.14.2 CuTe TS 的价值是可声明、可检查的异步资源协议

`Task` 把一段连续 warp 范围与输入/输出资源绑定；`@schedule` 构造 schedule，作者仍编写实际搬运和计算。资源的生产者 acquire/commit 与消费者 wait/release 定义 buffer 何时可覆写、何时可读取；`TaskManager` 据这些声明组织执行和依赖。与 D 的 Channel、H 的 mutex/flag 一样，关键问题都是**异步所有权、槽位轮转与资源可用性**，但 TS 把多角色关系提升到了专门的编程抽象。[I-task] [I-schedule] [I-task-manager]

它还提供检查器枚举抽象执行状态，报告阻塞、资源竞争等问题；这对复杂合作 kernel 的可维护性很有价值。不过检查对象是已声明的模型，并不能证明用户任意地址运算/底层指令都正确。实现还记录状态探索上限及 `hit_state_limit`，应连同覆盖范围检查；“在探索状态内未发现问题”不能省略为无条件的死锁/竞争自由证明。[I-checker]

`work_tile_loop` 与 `domain_loop` 又是两条不同的轴：前者推进 tile，后者遍历一个 tile 内的 K/序列等计算域。静态 WorkQueue 可以不需要单独的工作获取流水；CLC 队列需要动态获取工作及配套协议；dynamic domain 则可能只是在当前 tile 内读取运行时 offset 后循环。**运行时循环次数变化，不必意味着工作分配也是动态的。**[I-schedule] [I-ts-tutorial]

#### 6.14.3 用关键路径与资源模型评估调度机制

假设相同算法与工作集合下，固定分配产生的末尾空闲为 `T_tail`，动态机制能够减少其中的 `ΔT_tail`，但增加工作获取、同步及资源占用导致的时间，则可用下式作一阶分析：

```text
动态分配的潜在净收益
  ≈ ΔT_tail
    − T_work_fetch
    − ΔT_sync
    − ΔT_resource_pressure
    − ΔT_local_compute_or_memory
```

这是分析框架，不是本次测量结果。I 的 CLC 主要直接作用于 tile 工作分配，TS 主要降低局部异步合作的实现难度并影响流水效率；H 的 StreamK 改变工作分解，并同时引入 partial 归并。它们可能与 A/C 的外层任务调度互补，但无法代替 QK/PV 的高效实现、GQA 复用、页访问局部性及整层重分片。

验收时应保持算法、dtype、shape 和实际工作相同，分别记录静态/动态策略的 tile 分配、空闲尾部、额外同步、workspace、寄存器/片上峰值与总耗时。I 的示例本次仅作源码参照；H 的 correctness 也不直接提供这些性能计量。

<a id="dynamic-tiling"></a>
## 7. 动态 shape 与 tiling：是否另写函数、由谁计算、具体怎么写

先直接回答客户最关心的问题：**动态 shape 并不天然要求用户另写一个 tiling 函数；SPMD 也不天然要求独立 Host tiler。真正决定用户要写什么的，是 kernel 的覆盖范围、分工策略、参数传递方式和交付接口。**

一个固定容量的 tile，配合运行时循环次数、地址和尾块有效范围，就能处理一族 shape；若想随 shape 选择不同物理模板、跨核切分长归约、按各请求长度均衡工作，则需要策略逻辑，但它可以在普通 Python、分阶段 Host 程序、设备 kernel、Orchestration 或独立 tiling task 中。只有某些接入契约明确要求用户提供单独的回调函数。

本章先给责任矩阵，再逐路线展示 tiling 接缝；完整 softmax/attention 计算体见第 2 章，代码分类与验证口径沿用第 2.1、12.1 节。

### 7.1 六种“动态”，分别看支持

| 动态类型 | 例子 | 主要责任层 |
| --- | --- | --- |
| 维度元数据动态 | `M/B/P/T` 调用时变化 | 类型/guard/ABI/allocator |
| tensor内容动态 | `actual_seq_len`、block_table内容变化 | 设备load/控制流/索引；可能需要数据相关tiling |
| 物理tile有效窗口动态 | `valid_rows/valid_cols` | kernel/后端mask、padding、合法布局 |
| 算法/策略动态选择 | 短行单核、长行split-reduce、不同page堆叠 | Host/device策略、tiling key、编译变体 |
| 程序/任务数动态 | 按实际seq_len生成page任务 | Orchestration/runtime或kernel循环 |
| 资源派发动态 | ready task分配给空闲核心 | scheduler；不等于上述所有动态 |

### 7.2 客户是否必须单独提供 tiling 函数

| 路线 | 必须另写一个注册式 tiling 函数吗 | 当前策略位置 / 接口 | 物理tile与缓存边界 |
| --- | --- | --- | --- |
| A（PyPTO2-tensor版） | 不作为此编程模型的普遍要求 | tile设置、Tensor程序/loop、编译器分解 | 动态标记与CheckArgs/产物选择仍重要 |
| B（PyPTO2-block版） | **Python JIT 不强制；文档所示离线自定义算子包接入需要 Host C++ TilingFunc** | 普通Python计算blockDim/TilingData/work_ranges；或kernel内计算；包接入按Host ABI填参 | 动态axis可复用；静态参数/tiling key/物理tile变化可产生变体 |
| C（PyPTO3（Simpler）） | 不强制；也绝不是“不需要tiling” | 编译tile、Orchestration、Host Python tiler、AIV tiling task均有例子 | 动态GM descriptor ≠ 动态物理TileType；库例子也可能专门化 |
| D（CANNBot DSL） | 不强制C++ TilingFunc | Python表达的分阶段Host程序、动态TensorSpec、结构数据、kernel循环；有bounded tiler描述符 | 直接JIT取具体shape契约；Dim与静态capacity内的运行时tile是不同能力 |
| E（PyPTO on GPU） | 用户不提供NPU式tiler，但提供静态bucket/geometry/schedule | operator wrapper / graph编译的tile策略 | high-level shape/stride专门化；内容mask可以动态 |
| F（Triton-Ascend） | 不强制单独TilingFunc | Python launch grid/autotune/constexpr + 后端资源规划 + runtime循环 | constexpr与specialization控制缓存；不能因runtime参数就保证仅一个binary |
| G（AutoFuse + Inductor） | 客户通常不手写；编译器会生成 | ATT/模板生成Host tiling；wrapper调用并管理workspace | 静态Top-N与动态策略不同；Inductor guards仍会影响compile次数 |
| H（CATLASS DSL） | 不强制独立注册 TilingFunc | 普通 Python/dataclass/编译常量、block_num，以及 kernel 动态逻辑范围/循环 | MMAD 两组 M/N/K 复用同一产物；FA 当前读取 Host 全局常量，实际长度参数未用于设备体，二者不能合称全动态。[H-mmad-example] [H-fa] [H-run-artifacts] |
| I（CuTe DSL） | 不强制 NPU 式 TilingFunc | Python/Host JIT 选 tile、atom、stage、grid/split；kernel 读动态 metadata，调度器领取工作 | runtime shape/stride、constexpr 资源与 CLC 工作调度是三种不同契约；动态长度不自动搜索新 tile。[I-tensor-runtime] [I-gqa] [I-dynamic] |

### 7.3 block版 的具体 tiling 提供方式，不止 dataclass

block版 的 Python JIT 可使用普通 launch 策略、运行时标量、TilingData、work_ranges tensor 或 kernel 内计算；具体写法与限制见第 7.8.1—7.8.5 节。TilingData 不是任意 Python 对象，也不存在通用 `@jit(tiling_func=...)` 注册接口。[B7（PyPTO2-block版）] [B8（PyPTO2-block版）]

文档所示离线自定义算子包则要求 Host C++ TilingFunc，见第 7.8.6 节；自动生成数据结构不等于自动翻译 Python 策略。策略读取设备内容的成本、计划更新与缓存契约统一见第 7.13 节。[B9（PyPTO2-block版）]

### 7.4 PyPTO3 的动态能力要看具体入口，不把示例变量名当复用证明

`DynVar / bind_dynamic`、动态 shape 参数和 Orchestration runtime read 均有实现；物理 `TileType` 仍受静态容量约束，动态 `valid_shape` 不会扩展已分配的 UB/L1。[C2（PyPTO3（Simpler））] [C7（PyPTO3（Simpler））] [C8（PyPTO3（Simpler））] [C9（PyPTO3（Simpler））] [C10（PyPTO3（Simpler））]

native PA 的 `B/P/T` 测试与几何编译签名见第 2.6 节；Host、Orchestration、SPMD 内部及 AIV task 四种 tiling 位置见第 7.9 节。将 Host 策略迁入设备仍需 ABI/依赖适配，不能由“有 AICPU”直接推出兼容。[C17（PyPTO3（Simpler））]

### 7.5 “重新tiling”至少有三种成本

```text
同一个已有策略函数，以新shape再运行一次
  ≠ 新写一个策略函数
  ≠ 重新编译一个kernel变体
```

对动态系统的验收要分别记录这三项，以及Host/设备同步、CPU策略耗时、JIT cache hit、设备scratch峰值。只记录“支持dynamic=True”无法估算服务运行成本。

### 7.6 先定义“tiling 函数”在谈哪一层

| 实际工作 | 输入 → 输出 | 一定需要单独的函数吗 | 动态 shape 时的典型处理 |
| --- | --- | --- | --- |
| 选择物理计算模板 | dtype、布局、硬件、算法 → TileType、UB/L1容量、流水级数 | 不一定，常写成编译常量或模板参数 | 固定模板覆盖一个范围；超出范围选择其他变体 |
| 计算本次有效工作量 | runtime shape / actual length → 循环次数、最后一块有效范围 | 不需要，kernel内部几行整数运算即可 | 每次执行重新读取并计算 |
| 划分核心或任务的工作 | 工作集合及代价 → blockDim、task集合、work_ranges、split-KV | 不一定，但复杂策略适合独立封装 | Host计算、设备生成计划，或运行时任务派发 |
| 满足框架 launch ABI | shape、资源限制 → TilingData、TilingKey、workspace、blockDim | **由交付接口决定** | 例如自定义算子包的Host TilingFunc，或AutoFuse自动生成的tiling函数 |

这里的“函数”不能按名字判断。普通 Python 的 `build_work_ranges(...)` 是算法意义上的 tiler；block版 的 `TilingData` 是数据结构，不是算法；CANNBot 的 `make_bounded_tiler(...)` 是有界分块描述符构造接口，不等于一个会自动搜索最佳分块的 Host 回调。

**不另写 tiler 的常见充分条件**是：输入始终符合已编译的 dtype/layout/shape 契约；物理 tile 容量固定且合法；所有需要变化的循环、偏移和尾块窗口都由运行时值导出；分工覆盖所有工作且互不重复；没有依赖本次长度却未更新的缓存计划。在这些条件下，改变 shape 可以只改变参数和执行次数。

反过来，仅把类型中的数字改成 `DYNAMIC`，不会自动补齐跨块归约、split-KV 合并、workspace 分配或负载均衡算法。设备 scheduler 同样不能替代这些语义。

### 7.7 PyPTO2-tensor版：把分块写在 Tensor 程序中，不要求另注册 Host tiler

tensor版不是“用户完全不管 tile”。用户可以设置 `set_vec_tile_shapes`、Cube tiling，也可以显式写 `loop → view(valid_shape) → 计算 → assemble`。编译器和 TileFwk 执行体系继续承担 Tensor 算子分解、依赖及任务执行；用户没有因此获得 block版 那样完整的物理 buffer / TileGroup 编程责任。[A8（PyPTO2-tensor版）] [A8b（PyPTO2-tensor版）]

下面按 `dynamic_mul_kernel` 改写，将 `tile_b` 固定为 16，突出“输入动态、分块策略不变”。[A20（PyPTO2-tensor版）]

```python
import pypto

@pypto.frontend.jit
def dynamic_mul_fixed_tile(
    x: pypto.Tensor([pypto.DYNAMIC, 128], pypto.DT_FP16),
    out: pypto.Tensor([pypto.DYNAMIC, 128], pypto.DT_FP16),
):
    tile_b = 16
    rows = x.shape[0]
    pypto.set_vec_tile_shapes(1, 128)
    for idx in pypto.loop((rows + tile_b - 1) // tile_b):
        offset = idx * tile_b
        valid_rows = min(tile_b, rows - offset)
        part = pypto.view(
            x, [tile_b, 128], [offset, 0],
            valid_shape=[valid_rows, 128],
        )
        result = pypto.mul(part, 2.0)
        pypto.assemble(result, [offset, 0], out)
```

例如 rows 从 33 变成 65，上述程序本身表达了 3 块变成 5 块以及尾部 1 行；用户不需要再实现一次“给每个核分配哪几块”的 C++ TilingFunc。这里说明的是表达与职责，单次编译复用还应核对实际入口的参数检查和缓存，不以示例名字或注释代替 cache 实验。

对应两个重点算子：

- **尾轴 softmax：**第 2 章原生例子的 `Tensor([DYNAMIC, ...])` 只把首轴声明为动态，其余轴按静态策略处理。batch 变化与最后一维从 256 变成 513，不是同一个动态承诺。客户通过 Tensor reduction 和 tile 配置表达计算，不需要手写 block版 风格的 load/store/TileGroup。[A12（PyPTO2-tensor版）]
- **paged attention：**已有程序从 `act_seq` 读取每个请求长度，再写 KV 循环、view 和有效范围；动态行为就在 Tensor 程序中。用户仍要表达怎样遍历 page、怎样做在线 softmax，不能把“存在 AICPU”理解为运行时会自动发明 PA 算法。[A13（PyPTO2-tensor版）]
- **非常长的 reduction：**Tensor reduction 可以保留较高层表达，但最终采用怎样的分解和融合，要看编译产物。不能把高层 `softmax` 可构图，等同于已经得到理想的跨核长行实现。

因此，tensor版的主要用户体验是“给 Tensor 程序动态边界与 tiling 提示/切片”，而不是“编写独立的分核描述数据，再直接 launch 自己管理的物理 kernel”。

### 7.8 PyPTO2-block版：Python JIT 下的五种写法，以及何时才需要 Host C++

#### 7.8.1 写法一：直接读动态 shape，Host 只算 blockDim

这是回答“block版 是否必需 Python tiling API”的最小反例：第 2 章完整 `softmax_tile_group_kernel` 根本没有 TilingData 参数。它声明两个动态轴，用固定 `TileType[16,512]`，从 `x.shape` 计算 row-tile 数，从 `get_block_idx/get_block_num` 进行步进分工，并设置 `[valid_rows, cols]`。[B14（PyPTO2-block版）]

可与该完整 kernel 放在同一模块的 Host 包装如下。分核函数是本文改写的普通 Python，不是框架注册回调；范围取自原 softmax 的 `TILE_ROWS=16, MAX_N=512`。

```python
from pypto_pro.runtime.platform import get_platform_info

def softmax_launch_policy(rows, cols, vector_cores):
    if rows < 0 or not 1 <= cols <= 512:
        raise ValueError("requires rows >= 0 and 1 <= cols <= 512")
    if vector_cores < 1:
        raise ValueError("valid vector core count is required")
    return min(vector_cores, (rows + 15) // 16)

def launch_pro_softmax(x, y):
    if tuple(x.shape) != tuple(y.shape):
        raise ValueError("input and output shapes must match")
    rows, cols = map(int, x.shape)
    blocks = softmax_launch_policy(
        rows, cols, get_platform_info().vector_core_num,
    )
    if blocks == 0:
        return y
    softmax_tile_group_kernel[None, blocks](x, y)
    return y
```

`get_platform_info` 是实际接口；查询失败可能得到 0，上例明确拒绝，不把未知硬件容量当作可 launch 数。此包装只针对原测试中的纯 Vector softmax；混合 Cube/Vector kernel 要按混合组的 blockDim 及 subblock 语义选择资源，不能直接套用 `vector_core_num`。[B22（PyPTO2-block版）] [B13（PyPTO2-block版）]

在这个明确边界内：

| 输入变化 | 用户需要做什么 | 不需要做什么 |
| --- | --- | --- |
| rows：777 → 2049 | 同一个包装重算 row-tile 数与blockDim；kernel重算尾行 | 不用新写tiler，也不因rows值本身新建物理tile |
| cols：300 → 512 | 传新shape；kernel把有效列改成512 | 不用把TilingData塞进签名 |
| cols：512 → 513 | 选择/实现另一个能覆盖513列的方案 | **不能**只把valid_cols设成513，越过512容量 |
| rows：正数 → 0 | 包装直接返回空结果，不发0-block launch | 不需要用NPU kernel处理空工作 |

“另一个方案”可以是更宽且资源合法的模板，也可以是分段算法。单纯加宽还必须核对所有临时 tile、双缓冲、reduce 布局及 UB 容量；把常量改成 1024 不是资源可行性的证明。原源码测试标为 A5/950，本章没有据此宣称同一内核可在 A2/A3 原样运行。

#### 7.8.2 写法二：普通 Python 策略 + TilingData，数据结构由框架打包

如果希望 Host 预计算额外参数，可以自己写一个普通函数，返回 blockDim 和 dataclass。下面是对上一例的**接口改造示例**，并非要求这样写；它刻意用简单的 row-tile 数演示数据流。

```python
from dataclasses import dataclass

@dataclass
class SoftmaxRuntimeTiling:
    num_row_tiles: int

def make_softmax_tiling(rows, cols, vector_cores):
    blocks = softmax_launch_policy(rows, cols, vector_cores)
    tiling = SoftmaxRuntimeTiling(num_row_tiles=(rows + 15) // 16)
    return blocks, tiling

def launch_pro_softmax_with_tiling(softmax_with_tiling, x, y, vector_cores):
    rows, cols = map(int, x.shape)
    blocks, tiling = make_softmax_tiling(rows, cols, vector_cores)
    if blocks:
        softmax_with_tiling[None, blocks](x, y, tiling)
    return y
```

传入的 `softmax_with_tiling` 是用户改写后的 kernel，不是内置 API：将原 kernel 签名的最后一项增加为 `tiling: SoftmaxRuntimeTiling`，把内部 `num_tiles = (rows + TILE_ROWS - 1) // TILE_ROWS` 改为 `num_tiles = tiling.num_row_tiles`，其余完整计算体不变。这个简单例子中 Host 预计算几乎没有表达上的必要；复杂的分段参数、算法开关或每核分工才更值得封装。

实际机制和限制必须一起看：[B7（PyPTO2-block版）] [B4（PyPTO2-block版）]

- 一个 kernel 最多一个 TilingData，且必须放在参数列表末尾。
- 字段支持 `int/float/bool` 及其定长数组，不是任意 Python 对象、list-of-dict 或可变长度树；当前数组长度要求字面常量 1—2048。`int[N]` 等写法需按文档使用延迟注解，不能当作普通 Python 内置类型的运行时下标语义。
- runtime 按布局序列化 dataclass；`_pack_tiling_arg` 中可见 CPU 字节缓冲、`torch.uint8` tensor 以及向输入设备的转换。**Python化编写不等于参数打包、分配及H2D传递没有成本。**
- 参数传入后，字段是运行时值。它可以决定循环和分支，但不会自动重建已经编译好的物理 tile 类型、UB 地址或流水布局。
- 仅需要一两个参数时，也可用签名中的 `pl.DT_INT32` 等运行时标量；`x.shape` 已能提供的维度通常不必再复制一遍。多个来源表达同一个长度时，用户必须保证一致。

#### 7.8.3 写法三：每核工作描述用 tensor，不受 TilingData 定长数组形式束缚

原生 `build_work_ranges` 就是已存在的 Python tiling 实践，而不是需要另设计的接口。它接收 Q/KV 长度列表、mask spans、head数和核数，输出 `int32[num_cores,4]`，四列为 `work_start, n_items, split_mode, two_c`。[B23（PyPTO2-block版）]

源码存在两种策略：

- 当 `B * n_head_q > num_cores` 时，默认使用正反配对的步进分工；该分支主要依赖工作总数与核数，不需要逐项 KV 代价模型。
- 另一分支按每个 Q tile 将遍历的 KV block 数估算代价，搜索连续区间划分；不是简单“每核同样多 tile”。

对应调用点的摘录如下；输入、`total_work` 和缓存张量均由原 `_run_case` 构造，第 2 章已有完整 attention 计算体。[B24（PyPTO2-block版）]

```python
work_ranges = build_work_ranges(
    seq_q_list, seq_kv_list, spans_per_batch, n, num_cores,
).to(device)

flex_attention_bf16[None, min(num_cores, total_work)](
    q.to(device), kc, vc, o, cu_seqlens_q, block_ids,
    seqlens_kvcache, spans_dev, zero_mask, work_ranges,
)
```

这份例子是 **paged prefill**，不能冒充已验证的专用 decode 实现；它足以证明 block版 可以要求算子作者写 Python 分工策略并以 tensor 交给 kernel。

这里有一个非常重要的更新边界：

- 只改变 KV 长度，kernel读取新的 `seqlens_kvcache` 决定真实计算；若 Q 工作集合不变且原分工仍完整覆盖它，旧计划可能仍正确，只是代价均衡不再合适。
- 改变 Q 长度，使 `ceil(seq_q/TS)` 和工作编号变化，就不能默认旧 `work_ranges` 仍覆盖完整工作集合。跨过分块边界时尤其需要重建或证明可复用。
- 核数变化时，要同时更新分工描述与 launch；不能只改方括号里的 blockDim。
- 若新长度在原策略定义域内，通常是**重跑同一个 `build_work_ranges`**，不是为每个长度新写函数或新编译一个kernel。

#### 7.8.4 写法四：TilingKey 选择有限个专门化变体，不把实际长度都塞进 key

block版 已有 `@pl.jit(tiling_key=...)`，与不存在的通用 `tiling_func=` 注册参数是两回事。当前文档的真实使用形式为：[B25（PyPTO2-block版）]

```python
from pypto_pro.runtime.tilingkey import TilingKeyField

class FaTilingKey:
    NeedAttnMask = TilingKeyField(bits=1, values=[0, 1])

# 对文档中用 @pl.jit(tiling_key=FaTilingKey) 定义的 fa_kernel：
# fa_kernel[None, num_cores, {"NeedAttnMask": 1}](q, k, v, o, tiling)
```

字段在编译时被折叠为常量，具体 key 对应独立专门化产物。相同思路可用于有限的 tile 模板选择，但可编译性和资源合法性仍由具体实现保证。频繁变化的 `actual_seq_len` 通常应该留在设备 tensor/运行时参数里，不应为每个长度创建一个 key。

三种最容易混淆的东西如下：

| 对象 | 作用时机 | 改变内容的通常影响 |
| --- | --- | --- |
| `TilingData` / 普通运行时标量 / work_ranges tensor | kernel运行时读取 | 改变循环、分工或运行时分支；值本身不是TilingKey |
| `TilingKey` | 编译时折叠、launch时选变体 | 首次遇到该key可能编译；AOT要事先交付所需变体 |
| Tensor的`STATIC`轴或`...`静态尾轴 | 参数绑定及编译专门化 | 该轴变化可能选择/创建另一shape变体 |

源码中的该 JIT 对象进程内 variant key 是 `(static_signature, dtype_hash, tilingkey_packed)`；blockDim 和 TilingData 字段值不作为该 key 的组成。这个结论不替代完整的磁盘缓存、编译选项、架构和依赖身份检查。[B26（PyPTO2-block版）]

特别注意：`Tensor[[DYNAMIC, DYNAMIC], ...]` 与 `Tensor[[DYNAMIC, ...], ...]` 不同；后者的省略号会展开成静态尾轴。对应单元测试明确检查了“首轴 Var，尾轴 Const，尾轴参与 static_signature”。这正是“看上去都写了动态，换一个维度却重编译”的具体来源之一。[B27（PyPTO2-block版）]

#### 7.8.5 写法五：直接在 kernel 中计算长度、分块和分工

SPMD kernel完全可以自己读取 `actual_seq_len`，计算 `ceil(L/S)`，循环访问实际 page；也可以按固定步进分配 `(batch,head)`。第 2 章的 block版 attention 展示了设备长度参数与循环。此时没有独立 Host tiler，只有用户写在核内的 tiling 逻辑。

其取舍是：避免读取长度回 Host，不代表工作会自动均衡；每个核心重复计算复杂策略也可能浪费资源。如果改成设备生成一张全局分工表，还需要前置 kernel、同步或合法的共享协议。block版 当前的直接 launch 和 TileGroup 本身，不等于已经提供 Simpler 那样的通用跨任务 DAG scheduler。

#### 7.8.6 什么时候确实需要写 C++ TilingFunc：文档所示离线自定义算子包接入

**Python JIT 使用方式**与**交付一个接入 CANN 算子工程的离线包**不能混为一谈。后者的当前文档明确要求实现 Host tiling，并把生成的 kernel 二进制、Host 实现及 aclnn 接口打包。[B9（PyPTO2-block版）]

文档回调的关键接口摘录如下，workspace赋值整理为策略变量。`total_length / tile_num / block_dim / user_workspace_bytes` 都要由 Host 按输入及策略求出；这段展示 ABI 填充，不是已经包含策略计算的完整 C++ 工程：

```cpp
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto *tiling = context->GetTilingData<AddExampleTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->total_length = total_length;
    tiling->tile_num = tile_num;
    context->SetTilingKey(GET_TPL_TILING_KEY(0));
    context->SetBlockDim(block_dim);
    size_t *workspace_size = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace_size);
    workspace_size[0] = user_workspace_bytes;
    return ge::GRAPH_SUCCESS;
}
```

用户具体需要完成的是：读取本次输入形状及平台资源，计算长度/分块/分核策略，填写与 Python dataclass 一致的字段，选择合法且已交付的 key，设置 blockDim，报告足够的 workspace。若所调用接口另需系统 workspace，必须一并查询和计入，不能照搬上例中的零用户workspace假设。

block版 自动生成的是 **TilingData / TilingKey 的 C++ 头文件和布局**，不是把任意 `make_softmax_tiling` 或 `build_work_ranges` Python 算法自动翻译成 Host C++。`SetTilingKey` 接受打包值；候选值 `[16,64,128]` 中的 64 编码为候选下标 1，不能直接把 64 当作最终 key。

所以，对“block版 如何要求用户提供 tiling”最准确的回答是：**JIT 下按普通函数、标量、dataclass或tensor提供即可；该算子包接入路径按Host C++回调提供。不能把后一种要求推广成所有 block版 调用甚至所有 AOT 形式的必选项。**

### 7.9 PyPTO3（Simpler）：同一任务体系中至少有四种 tiling 位置

| 位置 | 本地具体实现 | 谁写策略 | actual_seq_len 变化后发生什么 |
| --- | --- | --- | --- |
| 融合SPMD kernel内部 | pypto-lib native PA | 算子作者写pl.read、page/stack循环与分工 | 重新读取设备长度，改变循环；不要求Host tiler |
| Orchestration程序 | 动态PA的InCore多阶段示例 | 作者写按batch/page产生计算调用的程序 | 运行时循环改变任务实例集合 |
| Host Python | Simpler高性能SPMD PA的`make_pa_nd_decode_tiling` | 算子/集成作者 | 重算元数据、分核选择，传入同一任务体系 |
| 独立AIV tiling task | pypto-lib CCE PA | CCE tiling实现作者；Python声明extern并编排依赖 | 设备读长度、写metadata；后续attention task等待它 |

**第一种，不另写函数。**native PA真实摘录是：[C23（PyPTO3（Simpler））]

```python
seq_len = pl.read(seq_lens, [batch])
page_count = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
stack_count = (seq_len + STACK_TOKENS - 1) // STACK_TOKENS
```

这些值控制 page/stack 循环；外层固定步进的局部工作项不是可独立派发的 Simpler task。完整分工及长尾分析见第 6.13 节。

**第二种，放在 Orchestration。**动态多阶段 PA 从 `context_lens` 读取 `cur_seq`，计算 `bn_this_batch`，然后在运行时循环里调用 QK、softmax准备、PV和在线更新等 InCore 函数。[C9（PyPTO3（Simpler））] 用户没有另注册 Host tiler，但编排程序本身就在表达动态工作分解。builder捕获的 `q_tile/head_dim/block_size` 与动态 Tensor descriptor 是不同层面的约束，不能仅因类型中有动态变量就把物理几何也当成任意运行时值。

**第三种，已有真正的 Host Python tiler。**调用点摘录如下；变量来自原场景的 `generate_args`，其中 `context_lens` 本来就是 CPU tensor，不需要从设备取回。[C24（PyPTO3（Simpler））]

```python
tiling, effective_block_dim = make_pa_nd_decode_tiling(
    batch=batch,
    kv_seq_lens=context_lens.tolist(),
    num_heads=num_heads,
    kv_heads=num_kv_heads,
    head_dim=head_dim,
    head_dim_v=head_dim,
    num_blocks=k_page.shape[0],
    block_size=block_size,
    max_blocks_per_query=block_table.shape[1],
    scale=scale,
    block_dim=block_dim,
    device="cpu",
    dtype=dtype,
)
ws = workspace_sizes(batch, num_heads, head_dim, head_dim, block_dim)
```

该函数会依据长度、head数和核数在不同分核/split-KV策略之间选择，生成 int32 tiling tensor，并返回有效 blockDim；workspace 另有计算函数。场景把这些数据放进 TaskArgs；AICPU Orchestration读取 `effective_block_dim`，设置 `args.launch_spec.set_block_num(...)` 后提交混合核任务。[C11（PyPTO3（Simpler））] [C12（PyPTO3（Simpler））]

这直接说明：**有 Simpler 与有 Host tiler 是可以同时成立的。**元数据中已经包含长度相关计划时，必须按其语义更新；scheduler不会自动替用户重跑这个Python函数。

**第四种，设备 tiling task。**pypto-lib 的 `build_paged_attention_metadata` 包含以下完整 scope 摘录：[C19（PyPTO3（Simpler））]

```python
with pl.spmd(
    1, name_hint="pa_tiling", allow_early_resolve=True,
    deps=[prev_reader_tid[0]],
) as tiling_tid:
    metadata = paged_attention_tiling_cce(
        seq_lens, metadata, max_blocks_per_seq, num_blocks,
        batch_offset, batch_count,
    )
```

`paged_attention_tiling_cce` 由 `@pl.jit.extern(core_type="aiv", source=...)` 声明；真正的 tiling 体是对应 CCE 源码，不能把这段 Python 包装称为“任意 Python tiling 算法自动在 AIV 执行”。后面的 attention SPMD scope 通过 `deps=[tiling_tid]` 依赖它。metadata跨窗口复用时还要等前一批 reader，源码里的 `prev_reader_tid` 就在处理这条关系。

这一方案能让长度保留在设备侧，同时把 tiling→attention 的依赖纳入任务系统；代价是额外的设备计算、任务/依赖、metadata存储及其生命周期。它比“有没有独立tiler”这个二分法更能解释 PyPTO3 的架构特点，也更能说明动态 scheduler 对整层执行组织的帮助及其边界。

### 7.10 CANNBot DSL：动态输入契约与动态 tile 参数分别怎么写

#### 7.10.1 固定分块，输入 M 动态：Dim + TensorSpec + kernel循环

当前明确的受约束动态入口是 `.compile(TensorSpec(...))`。以下整理自原生动态尾块测试，省略测试装饰器；`channel_utils` 是该仓库测试目录的辅助模块，不是承诺随公共wheel导出的API。这个最小copy例子用于隔离动态机制，softmax/attention计算体见第2章。[D20（CANNBot DSL）]

```python
import torch
import torch_npu
import cannbotdsl
from cannbotdsl import dtypes, jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.tensor import ceil_div, tile_view
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.types import Tensor
from channel_utils import ub_channel_at

@kernel
def dyn_tail_copy_kernel(x: Tensor, y: Tensor):
    ub = ub_channel_at(0, dtypes.float32, shape=(32, 32))
    for coord in range(ceil_div(x.shape[0], 32)):
        mem_copy(ub, tile_view(x, (32, 32), (coord, 0)))
        mem_copy(tile_view(y, (32, 32), (coord, 0)), ub)

@jit
def dyn_tail_copy_host(x: Tensor, y: Tensor):
    dyn_tail_copy_kernel[1](x, y)

M = cannbotdsl.Dim("M")
spec = cannbotdsl.TensorSpec((M, 32), dtypes.float32, stride=(32, 1))
compiled = dyn_tail_copy_host.compile(spec, spec)
try:
    for rows in (32, 33, 65, 100):
        x = torch.randn(rows, 32, dtype=torch.float32).npu()
        y = torch.zeros_like(x)
        compiled(x, y)
        torch.npu.synchronize()
        torch.testing.assert_close(y, x)
finally:
    compiled.close()
```

这里不需要单独 tiler：M 从输入元数据进入 ABI，循环次数是运行时值，尾块实际范围由 `tile_view` 的边界语义及 lowering 处理。这个例子只 launch 一个逻辑 block，用于说明正确性机制，不是高性能分核方案。

`Dim("M")` 默认正数范围；若改成 `Dim("M", multiple_of=32)`，则是主动缩窄允许输入的契约，并不代表编译器自动把 33 行补成 64 行。该仓库另有测试确认不满足倍数的输入会在 Host 契约检查处拒绝。[D7（CANNBot DSL）]

直接用真实 tensor 调用普通 JIT 时，源码先构造具体 shape/stride 的静态契约；不能只在Python函数中读取 `x.shape` 就推断已有“一个二进制覆盖所有M”。显式 Dim 契约、直接 JIT 专门化以及 kernel 的尾块能力，必须分别核对。[D6（CANNBot DSL）]

#### 7.10.2 连 tile 大小也想运行时改变：make_bounded_tiler 与静态 capacity

当前还有比“固定32行，只有最后一块变短”更直接的接口。原生 matmul测试以静态 `capacity=(128,64,64)` 约束M/K/N分块容量，接受运行时 `tile_m/tile_k/tile_n`。其中一部分声明如下，摘自完整 kernel：[D21（CANNBot DSL）]

```python
a_tiler = make_bounded_tiler(
    (tile_m, tile_k), capacity=(128, 64), alignment=(16, 16),
)
l1_a = Channel(
    MemLoc.L1,
    shape=(tile_m, tile_k),
    capacity=(128, 64),
    dtype=dtypes.float16,
    depth=2,
)
```

`shape` 是本次逻辑分块，`capacity` 决定静态可分配容量，两者不是一回事。完整实现也为 B/C、L0A/L0B/L0C 建立相应声明；循环用实际 tile 步长，尾部再按输入范围裁剪。

其 Host 编译与调用形式是：

```python
# bounded_tiler_matmul 为原测试中的完整 Host @jit 函数。
spec_a = cannbotdsl.TensorSpec((250, 90), dtypes.float16)
spec_b = cannbotdsl.TensorSpec((130, 90), dtypes.float16)
spec_c = cannbotdsl.TensorSpec((250, 130), dtypes.float32)
compiled_mm = bounded_tiler_matmul.compile(
    spec_a, spec_b, spec_c,
    dtypes.int64, dtypes.int64, dtypes.int64,
)
try:
    for tile_m, tile_k, tile_n in ((128, 64, 64), (80, 48, 32), (64, 32, 48)):
        compiled_mm(a_npu, b_npu, c_npu, tile_m, tile_k, tile_n)
finally:
    compiled_mm.close()
```

这里 `a_npu/b_npu/c_npu` 按上述shape/dtype准备，计算为 `A @ B.T`；代码是原场景的调用摘录，不独立提供张量初始化。三个 `dtypes.int64` 指定的是运行时标量 ABI，并不是把某个 tile 值预先固化。原测试针对**固定输入shape、多种运行时tile**；前一例针对**动态输入M、固定tile**。两项证据不能偷换为已实测所有两者组合。

接口约束是 `0 < tile <= capacity` 且 `tile % alignment == 0`，而最后一块的实际有效范围可以不对齐。[D22（CANNBot DSL）] 因此：

- 不必为 tile 从128变80重写函数或扩大已分配L1；同一个模板可以保留128容量而使用80逻辑行。
- `make_bounded_tiler` 不负责决定80是否最优；Host或device策略作者仍要选择数值。
- 原测试对正数但不对齐的 `tile_m=24` 检查了数据操作不执行、输出保持零的 fail-closed 行为，**不是抛异常且得到正确结果**。调用者应提前校验；不能把非法值当作自动调参请求。
- 超出capacity、改变不受支持的布局或改变物理流水结构，仍可能需要新的模板/产物。

与 block版 的相近点是“运行时参数 + 有界片上容量 + 用户分工”。具体差异是，CANNBot 在此提供命名的 BoundedTiler/Channel capacity 描述，block版 的既有softmax用固定TileType和set_validshape。不能把“某API形态不同”概括成一方完全不能写动态分块。

另一个使用差异是策略的执行方式：block版例子的普通Host函数由Python执行；CANNBot的Host `@jit` 属于分阶段编译链，不能把其Python源码直接理解为每次launch都由CPython解释执行。两者外层都还可以有普通Python包装；选择把shape策略写在外层还是编译的Host程序内，会影响可用Python特性、交付方式与调用成本。[D9（CANNBot DSL）]

### 7.11 Triton-Ascend：grid / constexpr / runtime标量，已经构成一种Python tiling写法

独立编写Triton kernel时，作者通常直接写Host策略：选 `BLOCK_SIZE`、构造grid、调用kernel。原softmax教程就是这样；将与动态无关的缓存包装去掉后，调用可以写成下面形式，配套 `softmax_kernel` 为第2章已有完整实现。[F5（Triton-Ascend）]

```python
def launch_triton_softmax(x):
    rows, cols = map(int, x.shape)
    if rows < 1 or cols < 1:
        raise ValueError("this example requires nonempty rows and columns")
    y = torch.empty_like(x)
    block_size = triton.next_power_of_2(cols)
    num_programs = min(32, rows)
    softmax_kernel[(num_programs, 1, 1)](
        y, x, x.stride(0), y.stride(0), rows, cols, block_size,
    )
    return y
```

这不是 NPU 自定义算子包的 TilingFunc，但当然包含 tiling 策略。原教程的32是该示例的program配置，不是所有NPU的物理核数，也不能把该grid直接解释为GPU式32个SM。

kernel 中 `n_rows/n_cols` 是运行时标量，`BLOCK_SIZE: tl.constexpr` 是编译期值。列数300与500都选512宽模板；513会选1024，因此Host会选择另一种编译配置。即使两个列数落在同一bucket，也不能直接保证只产生一个binary：本地JIT还有普通参数特化与对齐特化，缓存综合 specialization/options。[F17（Triton-Ascend）] 上游公开接口也区分 `do_not_specialize` 和 `do_not_specialize_on_alignment`；这里仅作为概念佐证，Ascend实际行为仍以本地实现为准。[Triton JIT 官方接口](https://triton-lang.org/main/python-api/generated/triton.jit.html)

如果N大到当前单行模板无法有效容纳，`next_power_of_2` 只给出一个数，不等于已设计可运行且高性能的分段归约。需要另一个长行算法/模板或确认后端生成的分解满足目标，不能只拿一个编译期常量的变化作证明。

**Decode attention有更直接的“不读回长度”例子。**unified attention的Host不把Q长度数组搬回CPU精确求grid，而用：

```python
total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs
# 原调用grid：
# kernel_unified_attention_2d[(total_num_q_blocks, num_kv_heads)](...)
```

设备kernel再定位请求、读取 `seq_lens_ptr`，计算实际 `num_blocks=ceil(seq_len/BLOCK_SIZE)`。[F18（Triton-Ascend）] 这是“Host用shape给保守grid，device用内容给真实工作”的现成做法；以可能多出的逻辑program换取不必在Host精确实现数据相关grid。它不承诺自动实现KV成本均衡或split-KV。

如果客户走 `torch.compile → Inductor → Triton-Ascend`，上述策略可由Inductor/模板生成，客户不再亲自写Triton Host包装。这与独立Triton算子作者的责任必须分开。

### 7.12 PyPTO on GPU与AutoFuse：两种都不要求客户手写NPU tiler，但原因不同

#### 7.12.1 PyPTO on GPU：当前高层路径主要是静态产物与动态内容

此处限定本文的 `pypto_kernels` 集成入口。当前 `launch_graph` 会比较真实tensor的shape/stride与artifact描述；不一致时抛出 `differs from static Artifact ABI`。因此不能拿固定artifact直接换任意shape，并期待某个隐藏tiler修复它。[E16（PyPTO on GPU）]

已有paged decode的调用形式为以下摘录，实际参数准备见第2章和benchmark。[E10（PyPTO on GPU）] [E11（PyPTO on GPU）]

```python
out = paged_attention_decode(
    query, key_cache, value_cache,
    req_to_token, request_index, valid_tokens, virtual_to_physical,
    kv_heads=kv_heads,
    bucket_tokens=bucket_tokens,
    stream=stream,
)
```

`valid_tokens` 是设备 INT64 tensor，每个batch一个长度；其内容可以在同一静态bucket内变化，接口约定 `1 <= valid_tokens[b] <= bucket_tokens`。这与改变其shape、query的B、缓存geometry或bucket容量不同。

客户/算子集成作者在这里写的是**bucket与几何选择、静态graph/tiling配置**，不是NPU式Host TilingFunc。同一个operator函数可负责选取或构建匹配的新产物；但缓存中旧artifact的ABI并不会变成动态。若选择bucket要先读回device长度，仍可能引入同步；也可从服务侧已知上界选bucket，让实际长度留在设备mask中。

静态 bucket 可能保留填充区域的计算/访存，长度变短不必然同比例减少工作。launch 数还受第 2.6 节的合并/分头分支影响，不能只数 Python 调用。

#### 7.12.2 AutoFuse + Inductor：tiling函数确实存在，但通常由编译器生成

客户仍写PyTorch。以下函数定义可放入第2章已初始化的NPU/PyTorch环境，表示请求动态shape编译；不单独提供客户tiler：

```python
import torch

def pytorch_softmax(x):
    return torch.softmax(x, dim=-1)

compiled_softmax = torch.compile(
    pytorch_softmax,
    backend="inductor",
    dynamic=True,
    options={"npu_backend": "ascendc"},
)
```

这条路径由Inductor/AutoFuse生成图、kernel及Host tiling；实际wrapper中的动态分支会调用 `tiling_fn`，写回 `tiling_data/workspace_size/block_dim`，然后分配workspace并launch。[G5（AutoFuse + Inductor）] [G7（AutoFuse + Inductor）]

```text
本次符号shape实参 + device资源限制
    → 生成的Host tiling_fn
    → tiling_data、workspace_size、block_dim
    → workspace分配/绑定 → 生成的kernel launch
```

此处是 **“无需客户编写，但存在独立 tiling 函数”**，与block版/CANNBot中“作者自己写kernel循环，所以不必有独立函数”不是同一种自动化。`task_queue` 分支可把这段动态Host工作放入Host侧队列回调，不能据此把它误认成AICPU tiler。

`dynamic=True` 是前端编译请求，不保证每种动态控制流、数据相关长度或gather都被AutoFuse支持，也不保证guard永不失败。若代码调用 `actual_seq_len.item()` 再走Python分支，不能因为后端会自动tiling，就认定该数据相关行为被自动转成了设备PA循环。当前paged decode融合范围仍按第2章及API边界章节的证据，不扩张为完整native PA支持。

**AscendC在这里的位置：**它可以作为生成的kernel代码与底层API承载；Host tiling是否由客户编写取决于上层接入。block版离线算子包要求客户填写Host回调，AutoFuse生成Host tiling，CANNBot提供自己的Host/Device编译链；不能把“都使用AscendC/CCE工具链”推成相同的客户tiling要求。

### 7.13 同一个softmax / paged decode，哪些变化要重算、哪些变化要重写

先把“新写函数”“重跑原函数”“选择新二进制”分开，再讨论算子。

| 变化 | 保持算法/物理模板时的常见处理 | 是否要重跑策略 | 何时需要新策略/实现 |
| --- | --- | --- | --- |
| softmax的M变化 | 改row循环、task数或grid，尾行裁剪 | Host算grid或显式row计划时重算；device公式会自然重算 | 小M大N导致并行不足，决定改跨核N切分时 |
| softmax的N在模板容量内变化 | 改有效列、mask或分段循环 | 取决于策略是否依赖N；不等于必需独立tiler | 原算法只支持整齐列/固定N，尚未实现正确尾部时 |
| softmax的N超出当前单行容量 | 选择更宽合法模板，或多段归约/在线算法 | 通常要选择已有变体/分段参数 | 没有覆盖该范围的算法，或要从单核升级为split-reduce时 |
| decode仅`actual_seq_len`内容变化 | device读取新L，改变page循环/mask | 若计划缓存了长度、页数或split-KV边界就需更新；仅用于代价均衡时可正确但不再最优 | 原策略不覆盖新增长度范围、需要新split-KV/合并方案时 |
| decode的B变化 | 动态ABI、循环/工作集合、输出及workspace随之变化 | 显式工作计划通常需更新；固定步进可device重算 | 固定B产物无对应动态契约，或计划格式/metadata槽位容量不够时 |
| KV pool的P变化 | 更新缓存shape/stride/基址；有效page id必须合法 | 不一定影响实际计算分工；检查是否参与workspace/计划公式 | ABI是静态P、物理格式变化或缓存索引能力超界时 |
| block_table的T变化 | 更新行跨度和可访问page容量 | 依赖T的描述/分配需要更新，不依赖者无需为T发明新算法 | 静态T产物需换变体；旧代码无法表达新stride/范围时 |
| Hq/Hkv、D、page大小S变化 | 检查GQA关系、Cube几何、layout、局部容量及签名 | 只要参与策略就需要更新 | 通常更容易触及专门化或物理模板边界；不能保证仅改TilingData即可 |
| 可用核数变化 | 同步调整blockDim、步进或work_ranges | 每核描述依赖核数时必须更新 | 混合核配比/同步协议变了，已超出原模板契约时 |

表中“常见处理”是跨路线的契约分析，不宣称每个当前实现都拥有该项动态接口。GPU静态artifact、tensor版静态尾轴、block版的MAX_N和CANNBot的Dim/capacity分别约束着可用范围。

#### 7.13.1 超长softmax：困难在归约语义，不在有没有Host回调

如果一行分成多个N方向的块，不能分别做局部softmax再把结果拼起来。至少需要一致的全行最大值和分母。例如每块先产生 `m_i=max(x_i)` 与 `l_i=sum(exp(x_i-m_i))`，再合并：

```text
m = max_i(m_i)
l = sum_i(exp(m_i - m) * l_i)
y[j] = exp(x[j] - m) / l
```

可以重读输入输出归一化，也可以按算法保存所需中间状态；不同方案的GM流量与workspace不同。若跨核完成，还要处理partial结果的生命周期、合并依赖及输出分工。

AICPU任务系统有利于组织“partial → merge → normalize”，但用户或编译器仍需提供正确分解；SPMD也可用多kernel、已有模板或合法的核间机制表达，但独立tiler只计算分块参数，并不能取代上述算法。第6章的megaKernel性能分析必须把这部分数据搬运和同步计入，不能只统计Host函数数量。

#### 7.13.2 actual_seq_len：是否在Host可获得，决定用户代价

| 信息来源 | Host是否通常已有该信息 | 写Host策略的额外代价 | 可选表达 |
| --- | --- | --- | --- |
| `tensor.shape / stride` | 通常是Host可访问的tensor元数据 | 不需要为了这些元数据读取tensor元素 | Python/C++算grid、workspace、模板选择 |
| 服务调度器维护的请求长度 | 若本来就有且与设备状态一致，则已有 | 维护一致性；传递必要参数 | Host按长度均衡、构造work_ranges |
| 上游device kernel刚产出的长度tensor | 通常只有设备内容可直接使用 | 取回元素可能产生D2H及等待/同步 | device循环；保守grid/bucket；设备tiling task |
| 缓存的历史分工计划 | 有计划，但不代表有最新输入语义 | 校验依赖并重算必要字段 | 按内容版本/几何/核数管理计划有效性 |

读 `x.shape` 与执行设备tensor的 `.item()/.cpu()/.tolist()` 不是一类成本；后者是否实际同步、能否异步搬运、是否已有CPU镜像，要按具体数据生产链和stream依赖测量。不能在一个GPU/NPU来源的长度tensor上随手写 `tolist()`，再把结果当作“纯Python函数、没有设备交互”。

PA计划的正确性至少要包含这些契约：batch索引一致；GQA映射合法；每个L在该kernel支持的范围；`ceil(L/S) <= T`；实际访问的page id在P范围内；尾页mask正确；work_ranges完整且无重复；split-KV合并状态与本次分区一致。**更换B/P/T并不天然需要新写tiler，但必须重新检查这些契约和编译产物的允许范围。**

#### 7.13.3 四种缓存不能共用一个“shape没变”判断

| 缓存对象 | 保持有效需要关注什么 | 典型反例 |
| --- | --- | --- |
| 编译产物 | dtype/layout、静态维度、模板/key、目标与选项 | block版动态轴变化可复用，但STATIC尾轴或TilingKey变化选新变体 |
| launch参数包 | 当前指针、shape、stride及产物ABI | GPU内容可变而packet仍有效；分配新buffer需要匹配新指针 |
| tiling/work_ranges计划 | 所有被策略消费的shape、长度/内容版本、核数及模式 | L的tensor地址和shape不变，旧计划却缓存了旧page数 |
| workspace/metadata及其复用 | 容量、别名、读写依赖、并发调用生命周期 | PyPTO3前次reader未结束就覆盖同一metadata |

所以，**同一binary可复用，不等于同一tiling计划可复用；同一计划可复用，也不等于同一buffer可被并发覆盖。**这比一句“支持动态shape”更接近服务系统的实际合同。

H/I 应放进同一个缓存与成本记录：H MMAD 改 M/N/K 已有同产物通过的样本，FA 改 Q/KV 则按当前 Host 常量形成特化；I 的动态 tensor 元数据与编译期 tile/atom/角色也要分开。尤其 I 的 CLC 是运行中分配工作，并不替代 Host 的编译参数/资源策略选择；动态工作均衡不等于无重编译、无 tiling 成本。[H-run-artifacts] [H-fa] [I-tensor-runtime] [I-dynamic]

### 7.14 从用户成本和架构相近性看，动态tiling的结论

- **PyPTO2-tensor版与PyPTO3（Simpler）**都允许把动态工作分解留在程序/设备任务层，客户不普遍承担Host注册式tiler。tensor版用户也能显式tile切分；PyPTO3当前更直接展示了InCore/Orchestration、外部SPMD和前置AIV tiling task的接缝。设备scheduler解决已表达工作如何推进，不等于自动决定所有切分。
- **PyPTO2-block版与CANNBot DSL**在“用户决定核内模板和分工、runtime值控制循环、有界局部容量”上更接近。两者都不因SPMD强制独立Host TilingFunc；具体提供方式分别包括block版标量/dataclass/tensor/TilingKey，以及CANNBot Host staging/Dim契约/BoundedTiler。block版文档中的离线算子包则额外暴露Host C++ tiling责任。
- **独立Triton-Ascend**把相当一部分用户策略放在Python grid/constexpr/autotune与kernel循环；**Inductor→Triton-Ascend和AutoFuse**可把这些职责转给框架/编译器。用户代码更少不代表tiling不存在，也不保证生成策略覆盖所有算法。
- **当前PyPTO on GPU高层路径**主要通过静态产物/bucket承载动态内容；其客户代价首先是明确geometry和bucket范围，而不是寻找一个尚未提供的通用动态Host tiler接口。
- **CATLASS DSL 与 CuTe DSL**也不普遍要求注册式 tiler，代价集中在作者选择物理 tile、layout、槽数与角色，并声明哪些元数据可动态变化。H 的 MMAD 动态样本和 FA 常量路径要分开，I 的 runtime shape、constexpr 资源与 CLC 工作分配也要分开；各自例子不能代替完整动态域的验收。[H-mmad-example] [H-fa] [I-tensor-runtime] [I-dynamic]
- **极致性能下通常更需要讨论“策略能写到哪里”而不是“有没有tiling函数”。**固定步进最省元数据但可能不均衡；Host成本模型可能改善分工但依赖长度可获得性；设备tiling task减少Host往返但有自身开销；多变体提升专门化空间但增加编译、缓存和测试成本。没有哪一种位置仅凭架构就自动最优。

对开发/客户/测试的最低交付要求应是：每个kernel列出可复用shape域、编译期几何、runtime参数、策略输入依赖、workspace公式、计划失效条件、空输入/尾块/非法值行为，以及是否需要Host读取设备内容。分别记录“重新计算策略”“选择已有变体”“首次编译变体”的发生次数和耗时，再比较动态能力与用户代价。



### 7.15 H/I：动态 layout、固定资源和动态工作队列分别验收

| 变化 | H（CATLASS DSL） | I（CuTe DSL） | 对客户的影响 |
| --- | --- | --- | --- |
| 地址或内容变化、类型/布局契约不变 | 编译样本与运行实参分开 | DLPack/runtime tensor 与编译类型分开 | 可以设计重复调用；仍须满足设备、stream、生命周期和 ABI |
| 逻辑 shape/stride 变化 | `mark_layout_dynamic` / `mark_compact_shape_dynamic`；kernel 读取 `origin_shape` 等动态值 | 对应动态 tensor 元数据接口，保留指定连续维/对齐约束 | 标注哪些值动态，再检查 kernel 是否真正用动态值；不是所有 shape 自动共用一份产物 |
| 物理 tile/槽数/执行角色变化 | `allocate` 容量静态；例子 tiling 字段或 Python 全局量决定特化 | Copy/MMA tile、warp/cluster、SMEM/TMEM 布局通常进入编译配置 | 经常需要另一个特化与资源计划，不能因 GM 动态就运行时任意扩大片上 tile |
| 每请求 KV 长度不同 | 当前 FA 的长度参数未被设备读取；语言级动态能力不能补齐例子 | MLA 的 cache sequence / var-seq 路径提供具体对象；另查可变 split 的约束 | 应在同一产物下改变设备长度内容，再检查输出、迭代数及 cache key |
| 工作项数或单项迭代数变化 | basic MMAD 的动态 GM 与固定步长遍历；FA 特化按源码实际处理 | persistent work-tile 域与 dynamic domain 分开 | 动态元数据、Host tiling、设备取工作是三种不同开销 |

来源：[H-tensor-runtime] [H-api-allocate] [H-mmad-example] [H-fa] [I-tensor-runtime] [I-mla] [I-schedule]。

#### 7.15.1 H 的已有两种写法：basic MMAD 与 FA 不能一起标成“全动态”

basic MMAD 从 GM tensor 的 `origin_shape` 取得 M/N/K，用 `_tiling.l1_tm/l1_tn/l1_tk` 等固定块确定片上容量，再让核按 `block_idx/block_num` 遍历输出块与 K 段；Host 负责构造 tensor、tiling 参数及 launch。这样可以做到**逻辑范围动态、物理策略固定**，不要求客户另写 C++ TilingFunc。实际可复用范围仍受 dtype、layout、拷贝对齐和 tile 配置限制；Host 的 `create_tla_tensor()` 明确调用 `mark_layout_dynamic()`。[H-mmad-example] [H-common-utils] [H-tensor-runtime]

FA 则在 `apply_shape_args()` 里改写全局 `Q_SEQ/KV_SEQ/HEAD_NUM` 等，并在编译时读取。这里已有 `compute_tiling()` 和长度 tensor，但设备端当前没有消费它们。要支持同一二进制下 `[127,513,...]` 的异长请求，不能只换 Host 数组：应修改设备取长度、batch 偏移、task 映射和尾块逻辑，再做相应验证。此处明确的是**现有例子的实现边界**，不是 H 语言在原理上做不到。[H-fa] [H-fa-tiling]

本轮 `M/N/K=128/128/256` 与 `384/160/272` 的 MMAD 均通过，且记录到同一 cache key 和 `kernel.o` 路径（第 12.1.3 节）。这给固定物理策略覆盖不同逻辑尺寸提供了具体对象；没有据两个样本外推完整 shape 域。

#### 7.15.2 I 的参数分类：动态形状并不替作者选择 tile

CuTe runtime tensor 可标记动态 shape/layout；`Constexpr` 则把算法与元编程选择放到编译期。MLA 还使用 page-table、cache sequence 等设备内容，持久化 scheduler 与 var-seq/split 配置共同决定工作。在连续 GQA simple 中，Host `compute_gqa_decode_grid()` 又根据 B/head/序列规模与 SM 数选择 split 数、g tile 和网格。这说明 **CuTe 同样存在 tiling 策略函数，只是不必是 NPU 风格注册式 TilingFunc**。[I-tensor-runtime] [I-mla] [I-gqa]

若动态 domain 只改变一个 CTA 内的 K 循环，它不会自动解决不同 CTA 间的长尾；若把固定队列换成 CLC，也不会自动把 static layout 变成任意动态物理布局。客户必须分别验证“缓存是否复用”“Host 是否重算策略”“设备是否读取新长度”“是否重新分配工作”四件事，沿用第 7.13.3 节的四种缓存区分。

<a id="intra-kernel-fusion"></a>
## 8. 用户怎样写出核内融合：多个 Vector 计算如何复用片上数据

**九条路线在各自支持域内，都有将多个 Vector 计算放进同一核内区域、消除部分中间 GM tensor 的源码路径；区别在谁划区域、选 tile、表达存储与同步。** 第 8.3 节逐项列出用户与编译器的分工，不承诺任意算子链或 shape 都能融合。

这一层需要与第 6 章的设备侧任务编排分开：**核内融合解决 producer 与 consumer 之间的数据留存；scheduler 解决任务何时、在哪个执行资源上推进。** 持久 worker 内执行了五个 task，不代表它们的中间 tensor 已不经 GM。反过来，没有 AICPU scheduler 的 kernel DSL，也可以很好地实现局部 Vector 融合。

### 8.1 先把“融合成功”的四种含义分开

| 观察层次 | 能证明什么 | 不能证明什么 |
| --- | --- | --- |
| 同一个 Python 函数、JIT 入口或完整捕获图 | 用户提交了一段程序，编译器可能看到多个操作 | 一个 InCore、一个物理 kernel、无中间 GM |
| 同一个核内 task / kernel 的计算体包含多个操作 | 计算不必按每个数学操作分别派发；可以做核内数据复用 | 中间值一定片上驻留；物理 kernel 中仍可有 GM scratch / spill |
| producer 的值直接由区域内 consumer 使用，不再物化成中间 GM tensor | 消除了对应的显式 GM 写入/读取；这是本问题最关心的融合收益 | 零片上读写、零同步、零寄存器溢出，或整个输入只读一遍 |
| 连续计算进一步在寄存器 / VF 中复用 | 可以继续减少 UB/shared 等片上存储访问 | 任意归约、跨核通信、Cube/Vector 交换都能这样处理 |

这里的“task / kernel”按各路线实际层级解释：Simpler 的 task 实例、核内函数、设备 worker 启动和 CUDA kernel launch 不是同一计数单位。对于 mixed Cube/Vector kernel，还必须看目标工具链是否形成多个核种入口。纯 Vector 算子的核内融合，不应直接外推成 mixed kernel 的全部执行事件只有一个。

用户提出的“一个 task 包含多个 Vector 逻辑，并通过片上内存协作”是重要的典型情况，但不应把 `UB / L1 / shared memory` 当作同义词：

- **NPU Tile 向量计算：**常以 UB 为操作数和中间结果的载体；L1/L0 更多涉及 Cube 数据路径和受目标约束的搬运。使用了 L1 只能说明某段数据路径，不能单独证明 Vector 链已融合。
- **A5 的寄存器 / VF 路径：**还可以在 UB 之上，把连续的向量操作组成寄存器计算链。PyPTO2-block版 文档明确区分这两级：Tile 链可能反复读写 UB，Reg 链在入口/出口与 UB 交互。这个接口及目标支持不能直接套给 A2/A3。[B28（PyPTO2-block版）]
- **GPU：**逐元素 producer/consumer 可能由同一线程通过寄存器直接连接，完全不需要 shared memory；跨线程归约、重排等才可能需要 warp 通信或 shared memory。普通 shared memory 属于 CTA，而不是“一个 kernel 内所有 SM 的公共中间数组”。[CUDA 编程模型](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)

因此，应问“这条中间数据边落在哪个地址空间、谁生产/消费、生命周期到哪里”，而不是仅问“代码里有没有申请 UB/shared”。

### 8.2 以最小 Vector 链说明收益，以及 ATen 的准确边界

先用不含归约的数学链排除跨核归约干扰：

```python
def vector_chain(x):
    a = x + 1.0
    b = a * 0.5
    return torch.exp(b)
```

对于稠密、同 dtype、无额外输出的 `N` 个元素，假设 eager 路径把这三项分别实现为 kernel，则逻辑数据路径是：

```text
逐项执行：
GM x → add → GM a → mul → GM b → exp → GM y

核内融合，每个 tile：
GM x_tile → [add → mul → exp，区域内保留中间值] → GM y_tile
```

在忽略标量常量、cache、spill、分配和其他辅助操作的模型下，前者为 `6 × N × sizeof(dtype)` 的逻辑 GM 读写，后者可降到 `2 × N × sizeof(dtype)`。这是访问量模型，不是“显存 DRAM 实测流量必减为三分之一”，更不是三倍加速承诺；跨 kernel 的 GM 访问也可能命中 cache，但依然没有获得相同的显式片上生命周期保证。

**“用户只写基础 ATen，就必然逐节点跨 kernel”仅对某些未融合的 eager 调用链成立，不能泛化。** 单次 `torch.softmax` 本来就可能使用融合实现；一个 ATen 操作也可能内部调用多个 kernel。`torch.compile` 则可以捕获同一段基础算子表达，把合法的 pointwise/reduction 组合交给后端生成融合 kernel。[PyTorch 性能指南：算子融合](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#fuse-operations)

注意，`out=` 或 in-place 复用 GM allocation，解决的是分配/存储复用，不自动消除相邻 kernel 的 GM 读写；graph replay、同 stream 顺序执行或一次 Host 调用，也不自动取得核内融合的效果。

### 8.3 九条路线：用户究竟需要提供什么

| 路线 | 能否达成本章的核内融合 | 用户实际写法与控制点 | 主要交给工具的工作 | 用户代价及不能依赖的假设 |
| --- | --- | --- | --- | --- |
| A（PyPTO2-tensor版） | 能，已有 Tensor 图纵向合图及手动 scope 控制 | 在同一可编译 Tensor 程序表达连续计算；配置兼容的 tile；必要时用 `sg_set_scope` | Tensor/Tile 图分解、合图、核内 codegen、任务执行计划 | 不必逐项分配 UB，但要理解 tile 依赖、合图失败及代价；一个 Tensor op 不必对应一个 task |
| B（PyPTO2-block版） | 能，原生 softmax 已把五项计算写在一个核内函数 | 一个 `@pl.jit` kernel 内用 Tile/TileGroup 装载、连续运算、存储；需要时进一步写 VF | IR/backend、自动 mutex 等受支持同步处理、目标代码生成 | 作者负责 tile、UB 地址/slot、尾块、核间分工；多个单独 launch 的 kernel 不因同处 Python 函数而融合 |
| C（PyPTO3（Simpler）） | 能，pypto-lib softmax 已提供单个 InCore 区域内的五项计算 | 在同一个 `pl.at(CORE_GROUP)` / InCore 计算体组织 producer/consumer，外层再做并行/任务编排 | InCore outlining、Tensor/Tile lowering、PTOAS 内存/同步、Simpler 派发 | 作者决定哪些计算跨 task，哪些必须核内组合；`pl.scope()` 不是核内融合开关 |
| D（CANNBot DSL） | 能，原生 Buffer softmax 显式保留 UB 中间量 | 一个 `@kernel` 体内用 Buffer 表达计算临时量，Channel 表达相应搬运/所有权阶段，连续调用向量 API | MLIR 片上规划、Channel/同步 lowering、适用的 VF 分组和代码生成 | 更直接承担内存空间、layout、生命周期/流水；一个 Host `@jit` 调多个 kernel 不是自动核间合图 |
| E（PyPTO on GPU） | 能，已有 sigmoid-mul 等组合计算图 | 把支持的 Tile 计算合写进一个 kernel 图；由包装层编译并 launch 该图 | PyPTO → TensorIR/Tile IR、线程/layout/资源分配、CUDA launch | 要符合当前 lowering 模式、shape/stride/tile 契约；不能把 NPU 的完整多 InCore 编排直接视为 GPU 单 kernel |
| F（Triton-Ascend） | 能，既有手写 fused softmax，也有 Inductor 入口 | kernel 作者在同一个 `@triton.jit` 内组合 `tl.load → 计算链 → tl.store`；PyTorch 客户也可由 Inductor 代写 | Triton/AscendNPU-IR lowering、bufferization、布局、内存与流水规划 | 手写时负责逻辑 program、tile、mask、归约方案；自动路径受 Inductor 和后端共同约束 |
| G（AutoFuse + Inductor） | 能，入口明确为区域内中间量跳过 Store/Output | 用户写可捕获的 PyTorch 基础操作，选择对应后端；不必手写 UB 或 kernel DSL | Inductor 合法合图、ASC 图生成、AutoFuse schedule/tiling/buffer/codegen | 最终融合区域由工具决定；支持的 lowering、索引和归约组合不等于全部 ATen；仍可能拆分/fallback/报错 |
| H（CATLASS DSL） | 能，已有 mixed MMAD+Vector add 与 FA 子过程 | 在同一个 `@tla.kernel` 组织计算，设备 helper 内联，显式安排局部 buffer/搬运与 CV 交接 | TLA/mixed/Vector lowering、受支持局部 AutoSync、目标代码生成 | 作者负责空间/容量/槽及跨核协议；内联 helper 不扩大同步作用域，也不保证寄存器全程复用。[H-mixed] [H-fa] [H-auto-sync] |
| I（CuTe DSL） | 有源码路径，softmax/MLA 显式保留核内状态；本机未运行 | 在同一 `@cute.kernel` 组合 fragment 计算、Copy/MMA、归约与 pipeline；Host 调用边界另算 | CuTe 编译与寄存器分配，pipeline/TS helper 可生成声明的资源协议 | 作者负责线程/值映射、资源/同步、尾块；一个 Host JIT 可多 launch，GQA simple 的 partial 仍经 GM。[I-softmax] [I-mla] [I-gqa] [I-task] |

表中“能”指已有源码路径，验证状态见第 12 章。下面摘录计算体；完整 kernel、Host 调用及 shape 契约见第 2、7 章。

#### 8.3.1 PyPTO2-tensor版：写 Tensor 链，tile 与合图策略共同决定核内区域

原生 softmax 的 `amax → sub → exp → sum → div` 在一个 Tensor helper 中连续表达，调用者设置 `set_vec_tile_shapes`；并非先要求客户写五个独立的设备 task。[A12（PyPTO2-tensor版）]

下面是按公开 `sg_set_scope` API 改写的**合图控制示意**，需置于tensor版 JIT Tensor 程序中：

```python
def softmax_with_scope(x):
    # The caller selects compatible vector tile shapes before this region.
    pypto.set_pass_options(sg_set_scope=1)
    row_max = pypto.amax(x, dim=-1, keepdim=True)
    shifted = x - row_max
    numerator = pypto.exp(shifted)
    denominator = pypto.sum(numerator, dim=-1, keepdim=True)
    y = numerator / denominator
    pypto.set_pass_options(sg_set_scope=-1)
    return y
```

默认先使用自动纵向合图即可，不是每段计算都必须手工标 scope。需要干预时，`sg_set_scope=1` 对后续图操作附加分组信息，`-1` 恢复未显式分组的状态；它不是 Python 词法 `with` 块，也不是设备侧 mutex。公开接口还提供是否允许并行合图、是否允许与未标记区域合图的 tuple 控制。[A21（PyPTO2-tensor版）] [A22（PyPTO2-tensor版）]

落地条件不只有 scope ID：producer/consumer 的 tile 划分、shape/layout、循环和归约依赖必须允许合法地合并。将长归约切成多个 tile 后，局部数据依赖可能已不是一一对应；强行给所有操作相同 ID，不等于物理资源足够，也不等于能消掉所有 partial GM。

源码在 `AddRawOperation` 捕获 scope，分图阶段合并同 scope 节点；混合 Cube/Vector 是否允许还有平台判断。因此，不能把旧文档对某类 CV 分离平台的限制扩大成“tensor版永远不能融合 Matmul 与 Vector”。[A15（PyPTO2-tensor版）] [A16（PyPTO2-tensor版）]

**用户感知：**主要操作 Tensor、tile 策略和图分组；为了这个局部融合，不要求用户直接写 UB 分配器或 AICPU scheduler。相比 PyPTO3，核内边界更多是图编译策略的结果，而非一个显式 InCore 函数体。

#### 8.3.2 PyPTO2-block版：在一个核内函数中把 UB Tile 计算串起来

原生 softmax 中，同一个 `@pl.jit(auto_mutex=True)` kernel 的 Vector 区域内，已经为输入、输出、归约和 scratch 建好 TileGroup 并设置 valid shape。以下是其中连续计算的摘录：[B14（PyPTO2-block版）]

```python
pl.load(in_slot, x, [row_off, 0])

pl.row_max(red_slot, in_slot, tmp_slot)
pl.row_expand_sub(out_slot, in_slot, red_slot)
pl.exp(out_slot, out_slot)
pl.row_sum(red_slot, out_slot, tmp_slot)
pl.row_expand_div(out_slot, out_slot, red_slot)

pl.store(y, out_slot, [row_off, 0])
```

这里 `red_slot/out_slot/tmp_slot` 是核内 Vec/UB Tile，不是交给另一个 task 的 GM tensor。用户没有为 max、sub、exp、sum、div 各 launch 一次；例子由每个 core 循环处理分配给它的行 tile。实际完整代码会在 load 后选择其余 slot、设置尾块，摘录省略了这部分设置，不能脱离第 2 章完整实现执行。

还可以进一步降低 UB 往返。原生 LayerNorm 的 `@pl.vector_function` 中，归一化部分直接用 RegTensor 串联；以下摘录发生在已完成 mean/std、已设 mask 的寄存器循环中：[B29（PyPTO2-block版）]

```python
reg = vf.load_align(in_tile, base + r * LANES)
gamma = vf.load_align(gamma_tile, r * LANES)
beta = vf.load_align(beta_tile, r * LANES)
xc = vf.sub(reg, mean_b, mreg)
norm = vf.div(xc, std_b, mreg)
out = vf.mul(norm, gamma, mreg)
out = vf.add(out, beta, mreg)
vf.store_align(out_tile + (base + r * LANES), out, mreg)
```

这个子链的 `xc/norm/out` 直接传递寄存器值；但完整 LayerNorm 仍多遍读取输入 UB，不能描述成“整个 LayerNorm 只读一次”。相比 Tile 版，用户要承担更多 mask、寄存器块宽度、地址偏移、归约顺序及目标 VF 支持责任。**block版 的显式核内组合与其可选的寄存器优化，应分两级评价；不是“只有手写 VF 才能省 GM”。** [B28（PyPTO2-block版）]

#### 8.3.3 PyPTO3（Simpler）：把五项计算放进一个 InCore，而非只放进一个 JIT

`pypto-lib` softmax 的核心写法就是直接答案；下面摘录原生 kernel 的循环体，输入/输出声明及常量见第 2 章：[C15（PyPTO3（Simpler））]

```python
for r in pl.parallel(0, ROWS, ROW_TILE):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="softmax_rows"):
        tile_x = x[r : r + ROW_TILE, :]
        row_max = pl.row_max(tile_x)
        shifted = pl.row_expand_sub(tile_x, row_max)
        exp_shifted = pl.exp(shifted)
        denom = pl.row_sum(exp_shifted)
        y[r : r + ROW_TILE, :] = pl.row_expand_div(exp_shifted, denom)
```

五个数学步骤出现在同一个 InCore 区域，编译器可以在该核内任务中安排相应 Tile 及中间值；外层行 tile 的并行仍可形成多个 task 实例，**不是整个输入只有一个 task，也不是每个数学步骤都各有一个 task**。源码中的静态 `COLS/ROW_TILE` 是示例契约，不是任意长行都会自动 fit。

反面例子有明确单元测试：同一函数中两个相邻 `pl.at(CORE_GROUP)` 区域，outlining 结果是两个 InCore 函数，加一个依次调用它们的 Orchestration。这个 pass 不会因为它们相邻就把 `add` 与 `mul` 自动并回一个 InCore。[C25（PyPTO3（Simpler））] [C26（PyPTO3（Simpler））]

所以，若用户原来把 softmax 拆成五个独立 InCore，则要重新把需复用的计算合写在一个核内区域，或用能在该区域内展开的 helper；仅将五个调用包装到一个 `pl.jit`、一个 `pl.scope()` 或同一个依赖链中不够。helper 即使内联，若展开后仍含五个独立 InCore scope，边界仍然存在。

`pl.scope()` 管 Orchestration 的依赖追踪和中间内存生命周期，而且不能放在 InCore 区域里；这与本节的“核内融合计算区域”不是一个概念。[C27（PyPTO3（Simpler））]

**用户感知：**需要有意识地区分“核内生产消费”和“跨 task 生产消费”。Simpler 能更高效地推进后者，但不自动把它变成前者。若任务通过 GM 参数传递结果，同一 AICore/worker 先后执行它们，也不构成 UB 值可跨 task 隐式留存的契约。

#### 8.3.4 CANNBot DSL：一个 Device kernel，局部临时量用 Buffer，搬运阶段用 Channel

原生 softmax 明确把输入/输出边界建成 UB Channel，把 Vector 内部临时量建成 UB Buffer。下面为计算阶段摘录，声明、GM 搬运和调用入口见第 2 章及原文件：[D10（CANNBot DSL）] [D11（CANNBot DSL）]

```python
_ub_x_r = ch_x.wait()
reduce_max(_ub_m, _ub_x_r, axis=1)
expand(_ub_m_full, _ub_m, axis=1)
sub(_ub_x_shift, _ub_x_r, _ub_m_full)
ch_x.release(_ub_x_r)

exp(_ub_e, _ub_x_shift)
reduce_sum(_ub_s, _ub_e, axis=1)
expand(_ub_m_full, _ub_s, axis=1)

_ub_y = ch_y.acquire()
div(_ub_y, _ub_e, _ub_m_full)
ch_y.commit(_ub_y)
```

`_ub_x_shift/_ub_e/_ub_s` 不需要先成为跨 kernel 的 GM tensor；最后再把输出 Channel 的结果搬到 GM。这个收益来自 **同一 Device kernel 内的计算组合、显式局部存储与合法生命周期**，不是 `Channel` 这个名称自动带来跨 kernel 融合。

它还不止于手动 UB 串联：当前默认 MLIR pipeline 包含 `cannir-vf-grouping → cannir-form-vf-regions → cannir-vf-transform`，可继续形成 VF 并处理寄存器计算。分组源码会检查支持的向量操作、数据/写依赖、Channel 阶段、DMA/同步/控制流边界及归约组合；区域外仍需要的值要保留输出。[D23（CANNBot DSL）] [D24（CANNBot DSL）] [D25（CANNBot DSL）] [D26（CANNBot DSL）]

因此，**一个 kernel、一个 UB 计算链、一个 VF group 仍可能是三个不同范围**。原例中的 Channel release/acquire 就可能切开更细的 VF 分组，但并未因此变成多个 Host kernel launch，也未因此要求把全部中间量搬到 GM。默认 pipeline 可被配置改变；有分组 pass 不意味着任何链都已整段保留寄存器。

**用户感知：**与 block版 相近，必须理解核内存储、同步和切片；但 CANNBot 主要通过 Buffer/Channel/layout 与 MLIR passes 组织这些语义，不能把它等同为 block版 的 TileGroup/mutex 或同一编译实现。

#### 8.3.5 PyPTO on GPU：将连续表达放进同一受支持计算图，底层再分配线程和存储

当前仓库不只有“五阶段各 launch 一次”的实验。原生 `sigmoid_mul_kernel` 把装载后的 Tile 转成 FP32，连续执行下面的计算，最后只显式存结果；摘录位于同一个 `pl.at(CORE_GROUP)` 的 tile 循环中：[E17（PyPTO on GPU）]

```python
value_tile = pl.load(value, [row, block * 128], [1, 128])
gate_tile = pl.load(gate, [row, block * 128], [1, 128])
value_wide = pl.cast(value_tile, target_type=pl.FP32)
gate_wide = pl.cast(gate_tile, target_type=pl.FP32)
gate_neg = pl.neg(gate_wide)
gate_exp = pl.exp(gate_neg)
denominator = pl.add(gate_exp, 1.0)
sigmoid = pl.recip(denominator)
result_wide = pl.mul(value_wide, sigmoid)
result = pl.cast(result_wide, target_type=pl.BF16)
pl.store(result, [row, block * 128], out)
```

对本文讨论的 BF16 输入场景，它是 `value * sigmoid(gate)` 的 FP32 中间计算版本；不要与逐项 BF16 舍入的表达无条件视为逐位等价。包装层校验 shape/stride，末轴须为 128 的倍数，编译缓存包含相关信息，调用一次 `launch_graph` 提交该产物。中间 SSA 值没有被作者声明为多个 GM 输出；最终是否有 shared 临时量、寄存器 spill，以及实际 grid/block，需要看生成物，不由这些 Python 变量名保证。[E17（PyPTO on GPU）] [E1（PyPTO on GPU）] [E4（PyPTO on GPU）]

这里的用户改法是合写一个**当前 NVIDIA lowering 能接受的图**，不是把五个独立编译/launch 的函数在 Python 外面包起来。第 9 章的五阶段旧实验只能证明那一种写法的五次 launch，不能推导“PyPTO on GPU 天生不能融合多个 Vector”。反过来，这个已存在的点算子也不能证明任意 NPU 风格五 InCore 编排或 softmax 归约图都已被同一 GPU 路径接受。

#### 8.3.6 Triton-Ascend：手写一个 logical program 的计算体，或让 Inductor 生成它

原生 fused softmax 教程中，一个 program 的行处理包含连续的五项计算；行循环、grid 及完整调用见第 2 章。下列摘录已有输入行指针及输出行指针的计算上下文：[F5（Triton-Ascend）]

```python
mask = col_offsets < n_cols
row = tl.load(input_ptrs, mask=mask, other=-float("inf"))
row_minus_max = row - tl.max(row, axis=0)
numerator = tl.exp(row_minus_max)
denominator = tl.sum(numerator, axis=0)
softmax_output = numerator / denominator
tl.store(output_ptrs, softmax_output, mask=mask)
```

这些 `row_minus_max/numerator/denominator` 是 kernel 内的数据流值，不是 Python eager Tensor 操作或五个 Host kernel 调用。用户不用为它们逐项写 UB 地址；Triton/AscendNPU-IR 再做 lowering、bufferization、资源和流水规划，但具体存储与是否使用 GM workspace 必须检验产物。[F8（Triton-Ascend）]

手写用户负责 `BLOCK_SIZE`、program 分工、尾轴 mask、归约算法和资源可行性；逻辑 program 不应等同物理 NPU core。若继续只写 PyTorch，则使用下一节的 Inductor 入口，以上 kernel DSL 工作转移给编译器，但随之受自动融合策略约束。

#### 8.3.7 AutoFuse + Inductor：客户写数学链，后端明确区分区域内部值和外部输出

客户不需先写 UB Buffer。原生 smoke test 已有 `h = x + residual; return h.abs().sum(dim=-1)` 的编译表达，也有多步 LayerNorm 和 softmax；这是实际入口，不是给终端客户要求手工构造 ASC 图。[G13（AutoFuse + Inductor）]

与本问题最直接相关的源码链是：

```text
Inductor 的预融合节点列表
  → NPUScheduling.codegen_nodes(...)
  → NPUKernel(...).tracing_asc()
  → _store_buffer：记入 _local_stores；
                    不是区域输出，就跳过 Store/Output
  → _load_contiguous：命中局部值时直接重用/reshape
  → ASC 图的 schedule/tiling/buffer/codegen
  → wrapper 调用生成产物
```

这说明不仅“画了一个包含多节点的图”，入口已经有消除对应显式中间 GM 物化的实现。最终的局部地址空间、scratch、额外 layout kernel 则仍由后续处理决定。[G14（AutoFuse + Inductor）] [G15（AutoFuse + Inductor）] [G16（AutoFuse + Inductor）] [G6（AutoFuse + Inductor）]

当前 `can_fuse_vertical` 对多重间接索引、gather 前后操作类型、归约组合和 matmul prologue 等有明确限制；codegen 还存在独立 transpose prologue 的路径。因此，一份 PyTorch 图可以落成多个融合 kernel，而不是所有基础操作必合一。[G12（AutoFuse + Inductor）] [G16（AutoFuse + Inductor）]

#### 8.3.8 AscendC 基础层补充：多个 Vector API 调用不是多个 kernel launch

AscendC 是多条路线使用的底层 API/编译接口，此处不另算一条完整路线。AutoFuse 的 AscendC 扩展库提供了一个直观例子：`AxpyExtend` 是 `inline __aicore__` 函数，输入输出为 `AscendC::LocalTensor`，其非 half 分支连续执行：[G17（AutoFuse + Inductor）]

```cpp
AscendC::Muls(dst, src_1, alpha, count);
AscendC::Add(dst, src_0, dst, count);
```

这里两次 API 调用组成同一核内计算子链，`dst` 的中间结果由局部 Tensor 传给下一项；不是每调一次 `AscendC::Muls/Add` 就从 Host launch 一个 kernel。这个片段也没有证明它们是一条机器指令或全程寄存器运算。若用户直接用 AscendC 实现，需要在外层安排 GM/片上搬运、buffer、核间分工、尾块和必要的同步，并选择 launch/tiling 策略；库 helper 不替代这些工作。

这进一步定位了上层融合工具的职责：**把足够多的生产消费逻辑及局部生命周期交给同一个 kernel 编译单元**。如果上层早已把它们分成两个独立入口、只用 GM tensor 连接，下游单独编译这两个入口，通常不会自动把它们变成上述局部数据链。MLIR 或 AscendC 的名字本身都不消除这个边界。

#### 8.3.9 CATLASS DSL：helper 在同一 kernel 展开，显式连接局部数据路径

`basic_mixed.py` 将 MMAD 结果从 L0C 经 FIXPIPE 交给 UB，Vector 等跨核通知后与 addend 相加，再写最终输出；FA 的 Vector 子过程则组合 max、exp、sum 与在线状态更新。用户需要把这些计算及 buffer/flag 协议写在同一个 `@tla.kernel` 的合作体内，`@tla.jit` helper 可在编译时内联。现成示例已经给出局部中间值不经 GM 的具体路径，不能把同一个 Host 函数先后调用两份产物当成相同效果。[H-mixed] [H-fa] [H-dsl]

H 后端还含 AVE 组合等优化，但能否进一步消掉 UB 往返、哪个寄存器值可以复用，要看相应 IR/指令和依赖。`auto_sync="v0"` 也仅覆盖规定局部域；作者仍负责 CV 与线程协议。第 8.12 节保留完整数据路径和资源预算，第 12.1.3 节给出已测及失败边界。[H-passes] [H-auto-sync]

#### 8.3.10 CuTe DSL：在线统计、fragment 运算与角色合作必须在设备体内成立

CuTe softmax 教程以线程/值分区、寄存器计算、warp shuffle/shared 归约来组合局部逻辑；其 kernel 8 在线更新 max/sum 后重读输入生成输出，避免完整 exp 中间量。分页 MLA 则将搬运、矩阵计算、softmax 与修正置于相应 warp/pipeline 协作中。kernel 作者需要同时选 fragment/layout 和资源/归约协议，Host 的 `@cute.jit` 本身不规定这一融合区域。[I-softmax] [I-mla]

连续 GQA simple 恰好展示边界：decode 内已有强融合，但 split partial 仍写 GM，Host 再 launch reduction。TS 可以帮助生成已声明的 warp/资源协议，不会自动把两个 Host launch 合并，也不能替任意输入保证资源可行。I 本机未运行，以上只证明源码表达与责任位置。[I-gqa] [I-task] [I-ts-memory]

### 8.4 如果客户坚持只写 PyTorch，应怎样获得这个效果

两条 NPU Inductor 路线可共用下列**完整客户表达与选择入口**；需正确安装对应后端及依赖，并在独立进程中分别运行：[F7（Triton-Ascend）] [G1（AutoFuse + Inductor）]

```python
import torch
import torch_npu

def vector_chain(x):
    a = x + 1.0
    b = a * 0.5
    return torch.exp(b)

def compile_vector_chain(route, dynamic=False):
    backend_name = {
        "Triton-Ascend": "default",
        "AutoFuse + Inductor": "ascendc",
    }[route]
    return torch.compile(
        vector_chain,
        backend="inductor",
        fullgraph=True,
        dynamic=dynamic,
        options={"npu_backend": backend_name},
    )

# Select one route per process for an isolated comparison.
compiled = compile_vector_chain("AutoFuse + Inductor", dynamic=True)
x = torch.randn(257, 300, device="npu", dtype=torch.float32)
y = compiled(x)
torch.npu.synchronize()
torch.testing.assert_close(y, vector_chain(x), rtol=1e-3, atol=1e-3)
```

当前 torch_npu 注册的 `default` 指向 Triton loader，`ascendc` 指向相应扩展入口；这里故意使用真实选择值，而非假设存在一个同名的 `"triton-ascend"` PyTorch backend。[F19（Triton-Ascend）]

客户要做的是让 producer/consumer 同时进入可捕获程序、保持编译器能处理的索引/形状/副作用语义，并检查输出产物：

- 普通局部变量 `a/b` 不要求成为 GM tensor；若客户把它们也作为可观察的最终输出返回或写到外部状态，则对应结果必须按接口要求可见，但主计算链仍可能部分融合。
- Python helper 不一定是融合边界；独立的 opaque custom op、外部库 kernel、graph break 或 Host 取设备标量，则可能让后端看不到整段可融合计算。
- `fullgraph=True` 要求整段图捕获，遇到不支持的捕获可直接报错；它不要求一个 kernel。`dynamic=True` 也不是“关闭资源约束和 specialization”。
- Inductor 中 `Scheduling/can_fuse_vertical` 是**编译期的分组和代码生成决策**，不是前文的 AICPU 设备动态 scheduler，也不是 CUDA SM 的硬件 block 调度。[F10（Triton-Ascend）] [G12（AutoFuse + Inductor）]

其他七条路线提供 Python 入口，不等于在本文已验证一条“任意 ATen 程序原样输入，就自动转成该路线融合 kernel”的前端。客户使用已有算子库时可以不接触 kernel DSL；若要自定义融合链，通常需 kernel 作者用该路线的原生语言实现，或另外开发框架 lowering/集成。这项成本应由提供算子或编译服务的一方计算，而非归为“Python 用户自然就会”。

对 H/I，tensor bridge 解决的是指针、dtype/layout、动态元数据与参数传递；它不等于 PyTorch 自动发现并融合这些 DSL kernel，也不自动提供 autograd 或通用 Inductor lowering。尤其 torch_npu 的 `catlass_template.py` 生成 CATLASS C++，不能据同名将其算作 H 的 Python TLA 集成。需要原生 DSL 自定义融合时，作者/框架接入成本仍按本节计入。[H-tensor-runtime] [H-inductor-template] [I-tensor-runtime]

### 8.5 softmax、paged decode 与动态 shape：融合区域为什么会被迫改变

| 场景 | 用户需要怎样组织代码 | 可以争取消除的中间量 | 不能承诺消失的访问/成本 |
| --- | --- | --- | --- |
| 尾轴 softmax，单行或行 tile 可放入目标局部资源 | 把 max/sub/exp/sum/div 放入同一可融合计算区域；归约排除无效 lane，末尾 masked/valid store | shifted、exp、行统计等不必逐阶段全量物化成 GM tensor | 原始输入/最终输出、局部 scratch、必要的归约通信、可能的 spill |
| 归约轴很大，一个 tile 放不下 | 固定容量 chunk 循环，或显式 split-reduce；不能只扩大 tile 或强设 scope | 可以只保存统计量、重新读取/计算输入，避免完整 exp tensor 的 GM 中间量 | 多遍输入读、跨核 partial GM、额外归约 kernel，或更复杂的合作同步 |
| paged decode 的一个 KV chunk 内 | 将 QK 后的缩放/mask/softmax 更新等放进可协作的局部区域；PV 与在线统计的联系也要明确 | 适用实现可避免完整 score/probability 矩阵的 GM 物化 | KV/page table 读取、必要的 CV 交换、跨分区 partial、split-K 合并；同一 kernel 仍可能有 GM ring |
| 同一几何范围内只改尾部有效长度 | 复用物理 tile，动态 loop/valid_shape/mask 随实际长度改变 | 已经合法的局部融合链通常可继续使用 | 必须保证所有消费者、归约 identity 和 store 都遵守尾部语义，不能只 mask 第一个 load |
| `actual_seq_len` 内容改变，或其他 tensor shape 变大 | 前者区分运行时边界/工作量；后者复核 tile、layout、缓存键、分工与 workspace | 有界变化不必把每项计算拆回独立 kernel | 超过容量或改变布局/归约方式时需重算策略、选变体或重新编译，详见第 7 章 |

例如长行 softmax 可以在**同一个核内任务**中先逐 chunk 求 max，再逐 chunk 累加 exp 的和，最后重新读取输入归一化输出：没有必要保存完整的 exp GM tensor，但输入读了多遍。若改为多个核同时归约一个长行，则各核需要交换 partial；把函数名字合并不会凭空产生合法的跨核同步或共享 UB。

PyPTO2-tensor版与 Inductor 路线需要编译器支持这种分解/融合，并不由“语法能写 reduce”就保证生成理想算法；block版/CANNBot/手写 Triton/InCore 作者可更直接设计它，但也相应承担数值、分工和同步责任。第 2 章的公共长行算法是语义参考，不能冒充所有后端都会自动采用的实现。

对于 paged decode，`actual_seq_len` 影响有效 KV 工作量，不天然要求把 exp、sum 各自放到不同 task；更关键的是 QK、softmax、PV 的生产消费分工和片上资源是否兼容。当前 PyPTO3 native PA 就保留 GM transfer ring，是“同一合作 task 仍有 GM 中间通信”的实际反例，见第 8.10 节。[C16（PyPTO3（Simpler））] [C23（PyPTO3（Simpler））]

**新增一个 Vector 操作是否需要新的 tiling 函数？** 单纯在现有 tile 上新增受支持的 pointwise 计算，不会在接口层必然增加一个 Host tiler；但它可能增加 live buffer、寄存器占用、同步或改变最佳 tile。block版/CANNBot 等显式策略作者需要重新核查现有函数/参数是否仍有效；自动路径由 compiler/planner 重新评估，可能改变计划或变体。不能把“用户没多写 tiling 函数”误解为“底层计划无需改变”。

H/I 也面临上述选择：H 的固定 UB/L1/L0 与 I 的 SMEM/TMEM/寄存器预算，使新增 Vector/fragment 操作可能压缩缓冲槽或拉长生命周期；没有新增 Host tiler 接口不等于计划无需复核。H 的 KV=513 尾块通过与 Q=1 精度失败是不同配置的事实，不能合成“任意动态 attention 已正确”；I 的不同 split/变量序列需按各自 kernel 约束验收。[H-run-results] [H-fa] [I-gqa] [I-mla]

### 8.6 从极致性能与使用成本看：应融合哪一段，而非一味扩大 scope

局部融合的收益来源是减少中间物化、launch/派发，以及必要时减少片上读写。代价则可能是更长的值生命周期、更多寄存器/UB 占用、较小的并行驻留数、layout 转换、重复计算、同步等待和资源竞争。Cube 与 Vector 的最优 tile 或核间分工不同，还可能使“统一成一个更大的 kernel”比多个精调 kernel 更慢。

对用户可采用分层做法：

1. **先在一个 tile 的自然生产消费链内融合。** 纯 pointwise 链、softmax 行内链、matmul 的可支持 epilogue，优先查中间值是否真正不物化到 GM。
2. **再检查是否值得进一步做 VF/寄存器融合。** UB 已消除 GM 往返后，瓶颈可能转到 UB 带宽、指令吞吐或归约；此时 block版 的 VF、CANNBot 的 VF passes、其他后端的寄存器优化才是下一层问题。
3. **最后讨论跨 tile、跨核、跨任务的大区域。** 明确需要哪些 partial、哪些 GM transfer、什么同步以及如何保持负载均衡，再使用第 6 章的 megaKernel 与 scheduler 分析，不能用“设备侧有动态调度器”替代这一步。

这也解释了路线相似性：**block版 与 CANNBot 在作者主动建设核内融合区域这一层更近；tensor版与两条 Inductor 路线在根据较高层数据流做编译合图这一层有相似性；PyPTO3 的 InCore 与各 kernel DSL 在局部计算层相近，但其整个任务程序不能只按 kernel DSL 比较。** 局部 Vector 融合本身不足以区分谁更有整层 megaKernel 能力，也不足以决定第 14 章按整体架构选择的最相似者。

在同一尺度上，H 与 block版/CANNBot 都由作者主动连接核内物理数据路径；I 则以线程/值 fragment、atom 和 pipeline 承担相近责任，TS 可辅助资源协议。它们增加了设计融合区域的控制面，并未消除扩大区域造成的占用、同步和重分片代价；第 8.12 节的预算与第 6 章的关键路径模型应一起使用。[H-mixed] [I-gqa] [I-task]

### 8.7 验证方法：不能仅凭 profiling 中“一个 kernel”下结论

| 检查层次 | 最低检查项 | 能排除的误判 |
| --- | --- | --- |
| 用户代码 / 捕获图 | 操作是否都进入同一编译区域；中间值是否外部可观察；是否调 opaque/extern | 同一个 Python 函数被误认作同一个融合 kernel |
| 分区 / 核内 IR | InCore/fused region 数量；各函数输入输出；producer/consumer tile 与归约依赖 | 把一个 Orchestration、五个 InCore 当成一个核内任务 |
| 生成代码及存储 | GM load/store、workspace 参数、UB/L1/shared/reg 分配、跨 pipeline 搬运及 spill | 一个 kernel 仍对中间 tensor 做 GM 往返；局部 buffer 误认作寄存器 |
| 运行 trace | launch、task 实例、核种入口、grid/block、stream、辅助 layout/归约 kernel | 一次 Host 提交或 persistent worker 启动被误认成一次融合执行 |
| 性能计数与正确性 | 实际 GM/cache 流量、UB/shared 访问、占用和等待；误差、mask、动态边界 | 用逻辑访问模型代替 DRAM 实测，或融合后舍入/归约语义变化未校验 |

上述结论是源码路径与用户契约；具体 shape/目标上的 spill、kernel 数和性能仍须按表验收，已记录验证见第 12 章。

H 本次已留存源级 kernel、lowered MLIR、ABI manifest 与产物 hash，可据此检查 mixed 拆分和资源表达，但没有设备 profiler 的事件/性能证据；I 当前只有源码，尚不能填写真实 spill、占用或流量。两者都不应借用 E 的旧 SM89 trace 补齐本表的运行栏。[H-run-artifacts] [I-executor]

### 8.8 仍须区分四个不同的内存分配问题

| 问题 | 示例 | 典型owner | 为什么不可直接互换 |
| --- | --- | --- | --- |
| 物理片上地址/寄存器分配 | UB / L1 / L0 / shared / registers | kernel作者 + codegen + backend | 地址空间、alignment、bank、异步生命周期及指令约束不同 |
| 一个kernel的GM scratch | CV通信、split-reduce partial、sync-lock | kernel编译器 + launch wrapper | 大小表达式/ABI/同步协议与这个kernel绑定 |
| 跨kernel/task中间tensor | QKV、attention结果、MLP中间值 | framework图编译器 / task runtime | 生命周期由依赖图、stream、alias和in-place语义决定 |
| 长寿命模型状态 | KV cache、权重、page table | 应用/框架/allocator | 不能被局部“最后一次使用”错误回收；有别名和跨调用存活 |

此外，GPU local memory 通常指线程私有的地址空间，可能落设备内存，不等于 NPU 片上 LocalTensor/UB；shared memory 的作用域见第 8.1、9.1 节。

### 8.9 各路线的资源规划具体在哪里

| 路线 | 片上规划及流水 | GM/workspace | 需要重点检验的边界 |
| --- | --- | --- | --- |
| A（PyPTO2-tensor版） | TileFwk图/核内codegen及tile策略 | 图与runtime计划、参数tensor | task依赖改变后内存复用是否仍合法 |
| B（PyPTO2-block版） | TileType、地址、TileGroup、mutex/pipeline；作者控制强 | 调用者分配；workspace/TilingData经参数传 | 显式地址重叠、valid_shape、双缓冲slot生命周期 |
| C（PyPTO3（Simpler）） | PyPTO passes + PTOAS memory/sync；可手写mixed pipeline | create_tensor、scope/runtime、显式transfer ring | auto依赖与manual_scope混用；WAR/WAW；跨task可见性 |
| D（CANNBot DSL） | Buffer/Channel、layout、MLIR allocation/同步推断 | Host runtime/调用者及传入tensor | Channel槽位数量≠任意全局task并发；release时机 |
| E（PyPTO on GPU） | TensorIR/Tile compiler计划tile/layout/workers | framework参数和中间tensor；operator包装 | bucket/stride符合产物；跨launch中间值实际GM流量 |
| F（Triton-Ascend） | bufferization、PlanMemory、CVPipelining、MarkMultiBuffer、同步passes | compiler-generated workspace及sync-lock描述，driver分配 | 逻辑grid、物理block和workspace索引必须同一契约 |
| G（AutoFuse + Inductor） | BufQueAllocator/MemReuseManager、schedule/模板 | 生成GetWorkspaceSize/tiling；Inductor分配外部buffer | 融合边界内/外生命周期不可混用，动态表达式溢出及guard |
| H（CATLASS DSL） | 作者静态 allocate/slot + TLA 按地址空间对齐递增偏移；显式 CV 协议，局部 AutoSync 有限制 | 调用者与算法显式安排，如 StreamK partial | 该 scratch pass 不做自动 last-use 复用；声明量/最终峰值、mask/alias/异步消费须分别核对。[H-scratch] [H-streamk] [H-auto-sync] |
| I（CuTe DSL） | 作者给 layout/atom/pipeline；TS 可做 SMEM/TMEM 分配和显式 phase alias | Host/框架与 kernel ABI，如 GQA split partial/reduction | phase 生命周期靠实际同步成立；SMEM/TMEM/register 不可混算，TS 不规划任意模型图的 GM。[I-ts-memory] [I-gqa] |

F（Triton-Ascend） 的 [F8（Triton-Ascend）] 明确区分 `GLOBAL_WORKSPACE_PLAN` 等规划位置；G（AutoFuse + Inductor） 的 [G6（AutoFuse + Inductor）] 用生命周期及可复用条件安排buffer/queue。这些都不是简单的“按tensor字节数顺序放置”。

### 8.10 一个具体的跨层例子：PyPTO3 native PA

```text
上游投影TaskId
      ↓ deps
SPMD attention task（24 logical blocks）
      ├─ 每个core的UB/L1/L0：装载、矩阵运算、softmax，局部buffer
      ├─ score/probability/PV transfer：GM三槽ring，按core/slot索引
      ├─ event/FFTS workspace：同步状态
      └─ out / KV cache：应用级数据及可变状态
      ↓ TaskId完成
下游projection / residual / MLP
```

把PTOAS的UB planner换掉，不会自动改变GM transfer ring；把Simpler scheduler换掉，不会自动生成新的核内barrier；减少一个task也未必减少一份GM tensor。这是“共用内存规划”必须先约定层级的直接原因。

### 8.11 三种同步分别验收

PyPTO2-tensor版 本次“前端更新后 cache 参考通过、板端却接近旧 cache”的现象，为下面的同步与别名分析增加了具体检查对象；重建 view 的局部修正有效，但仍不足以把根因直接归到某一种同步机制。[RUN-A-fail] [RUN-A-front] [RUN-A-pass]

| 同步范围 | 典型对象 | 不能替代 |
| --- | --- | --- |
| 核内 / pipeline | Vector/Cube/MTE事件、warp/block barrier、局部buffer释放 | 其他核或其他task完成 |
| 同一合作task/kernel跨核 | mixed-core event、跨block锁、受支持全局barrier | 任意并发kernel之间的通用依赖 |
| 跨task / kernel | TaskId、ready queue、stream/event | task内部指令顺序和cache可见性协议 |

C（PyPTO3（Simpler））的native PA在Phase 0后明确执行cache/fence/sync操作：[C16（PyPTO3（Simpler））]。这说明“所有核到达”与“GM数据对消费者可见”也不是可以随便混用的一条语义。

H 的局部 flag/mutex、CV cross-core flag、`vec.func` 内线程/内存屏障和 AscendCL stream 属于不同作用域；AutoSync 明确不接管全部层次。I 的 pipeline/mbarrier、CTA/cluster 合作、CUDA stream 也需逐层区分，TS 只能在已声明资源/参与者范围生成协议。到达、异步写完成、消费者可见和全局进展仍是分别要证明的条件。[H-auto-sync] [H-fa] [H-runtime] [I-pipeline] [I-task]

### 8.12 H/I 怎样实现核内融合、片上复用与同步

#### 8.12.1 H：已有 mixed GEMM 与 FA，把四种融合证据落到源代码

`basic_mixed.py` 的数据路径可以用下图表示；它描述源码中的局部数据流，不是本次 profiler trace：

```text
A/B 的 GM → L1 → L0A/L0B → MMAD / L0C
                                  ↓ FIXPIPE，按 AIV 分工送 UB
addend 的 GM ───────────────────→ UB → Vector load/add/store → UB → 输出 GM
                                  ↑ cross_core flag 标记结果可读
```

作者先为 L1/L0/UB 分配容量，再用 layout 建立视图；Cube 完成 MMAD 后搬到 UB，Vector 等待跨核通知，将结果与 addend 相加。它让第 8.1 节的“同一核内协作区域”“中间结果不经 GM”有了具体实现对象。但 UB 到寄存器仍有 load/store，片上流量并未凭空消失；FA 的 softmax 也明确写入部分 UB 状态，再配合同核内存屏障和跨阶段事件。[H-mixed] [H-fa]

H 的 `@tla.kernel(auto_sync="v0")` 可在约束域内自动插入局部 mutex；它不负责跨核或 `vec.func` 内线程同步，且不允许与显式 local flag/mutex 混用，extern 也不在支持域。这意味着可以比较 **同一算法的显式同步与局部自动同步**，不能写成“开启后自动完成任意整层流水”。当前 FA 已自行写显式协议，不能简单叠加 AutoSync。[H-dsl] [H-auto-sync]

#### 8.12.2 H 的片上分配算法：静态容量与编译器选偏移不等于自动生命周期复用

`planTlaScratchAllocations()` 遍历 `tla.alloc_ptr`，按地址空间各自维护 next offset，为每次分配做对齐并递增。源码中这一层没有根据 last-use 将不同分配复用到相同区间；`TlaLowerPtrPass` 消费这些偏移，UB 还有对应 scratch symbol。因此至少要分清：

- 用户选择空间、容量、对齐和槽数；TLA 分配器给出静态偏移。
- 用户可在算法内重复使用同一已分配 buffer，但仍要保证异步消费者已结束。
- 两个不同 `allocate` 不会仅因源码作用域先后，就在这个分配器中自动获得相同物理空间。
- 下游额外优化或最终峰值以生成物为准，不能拿“采用 MLIR”作为全局最优内存规划的证据。

来源：[H-scratch] [H-ptr-pass] [H-api-allocate]。这为第 8.9/10.2 节的公共 planner 讨论提供了一个具体的起点：可以设计更强的生命周期复用，但必须把 flag、异步 copy、跨 CV 读写和 alias 效应作为输入。

用默认 FA 的 `Q_BLOCK=KV_BLOCK=HEAD_DIM=128`、`Q_BLOCK_SUB=64`、b16 输入做一次**源码预算计算**，逐个 `allocate` 乘 dtype 字节数，并按当前分配器的 512 B 对齐累加，可得到：

| 局部空间 | 源码中主要声明 | 按该分配规则计算的字节数 |
| --- | --- | --- |
| L1 | Q 单槽、K/V 各双槽、P 三槽；每槽 `128×128×2 B` | 262,144 B = 256 KiB |
| L0A / L0B | 各两个 32 KiB 槽 | 各 65,536 B = 64 KiB |
| L0C | QK 双槽与 PV 双槽；每槽 `128×128×4 B` | 262,144 B = 256 KiB |
| UB | S/P/PV 缓冲、acc、max/sum 等状态、tmp 与 mask 声明 | 原始元素量 207,616 B；逐分配对齐后 209,920 B = 205 KiB |

这里统计源级声明，**没有把不同地址空间或各个物理核的容量相加，也没有把它当成硬件总容量/最终峰值**。例如 mask buffer 虽被声明，当前数学路径没有使用输入 mask；最终是否保留以 lowering/二进制为准。`ub_out_f16_ptr` 由 acc pointer 做 `recast_ptr`，没有增加独立 `allocate`，但“可以换类型复用同一 buffer”仍须由算法保证旧值已无需保留。[H-fa] [H-scratch]

这个对象让第 6.7 节的整层问题更具体：不能仅把下一阶段函数内联进 FA，就假设它自动获得足够 UB/L1；需要决定哪些分配结束、哪些槽可复用、状态是否必须经 GM 交接，再计算加入新流水后的峰值。

#### 8.12.3 I：layout/atom、pipeline 与 TS 分别承担不同工作

在 CuTe 中，同一 kernel 里选择 local tile、线程/值分区、Copy/MMA atom 与寄存器运算，可以让生产者结果继续交给消费者；TMA/MMA 等异步阶段依赖 pipeline/mbarrier 协议。分页 MLA 的 page-table pipeline、Q/KV 载入、矩阵计算、softmax 与输出修正是具体组合。普通 GQA simple 则把 split 输出放在 GM，再 launch reduction，所以它既有 kernel 内融合，也有 kernel 间中间存储。[I-layout] [I-copy] [I-mma] [I-pipeline] [I-mla] [I-gqa]

TS 的内存模块进一步提供资源声明、SMEM 字节/TMEM 列分配和显式 phase alias group：同一 phase 的分配同时存活，不同 phase 可复用一个物理区域。这是一种**用户声明生命周期关系、分配器计算布局**的工具；不是自动发现任意模型图上所有可复用内存。声明的 phase 顺序还需实际同步保证，检查器的边界见第 6.14.2 节。[I-ts-memory] [I-checker]

连续 GQA simple 还有一个可对照的 tiling 计算：先为 Q、P、max/sum 和各 pipeline barrier 预留 SMEM，再用剩余预算除以每个 KV stage 的数据与同步开销，求 `KV_stages`。这相当于 `floor((容量−固定占用)/(每槽数据+每槽同步))`，因此改变 head dimension、g tile 或 stage 数会影响可用缓冲深度。它是具体作者策略，不能泛化成 CuTe 自动找到所有算法的最优 stage；预算值也应与最终分配和占用率复核。[I-gqa]

| 规划/同步层 | H | I | 本文的验收方式 |
| --- | --- | --- | --- |
| 跨 kernel GM/workspace | Host/算子安排，例如 StreamK partial | Host/算子安排，例如 GQA split partial | 记录分配、生命周期与真实读写字节 |
| 核内片上空间 | 静态 allocate + TLA 偏移规划 | SMEM/TMEM 布局，TS 可显式 phase alias | 核对空间峰值、对齐、alias 和消费者完成时点 |
| 寄存器与局部计算 | `vec.func`、寄存器 mask/load/store 等 | thread/value fragment、warp shuffle/寄存器计算 | 检查最终指令、spill 与片上访问，不只看源代码变量 |
| 异步/跨角色同步 | 本地 flag/mutex、CV cross flag、线程/内存屏障 | pipeline/mbarrier/CTA/cluster，TS 资源协议 | 分别验收内存可见性、buffer 安全与执行进展 |

H/I 都为深度融合提供了更细的控制；控制越细，越需要第 8.7 节的产物与硬件证据来判断是否兑现收益。

<a id="gpu-launch"></a>
## 9. GPU 多 SM、实际 launch 与 NPU block 的区别

### 9.1 grid、CTA、warp 和 SM 的准确关系

CTA（Cooperative Thread Array）通常就是 thread block。`grid=(32,1,1)` 表示本次launch有32个block，不是32个SM。普通SIMT kernel的每block线程数是 `blockDim.x*blockDim.y*blockDim.z`；warp为32线程。一个CTA的线程不跨SM执行；多个CTA可以驻留同一SM，映射和次序不能靠block编号推断。[CUDA编程模型](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)

两个各32线程的CTA即使恰好同驻一个SM，各自的普通shared memory也不能随意互访。显式cluster/distributed shared memory是另一套受硬件与launch约束的机制，不是“碰巧同SM”的结果。[CUDA编程模型](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)

```text
grid.x=32
  → 32个逻辑CTA工作实例
  → GPU按寄存器/shared/warp等资源及调度约束分配
  → 可能分布在多个SM；每SM可能接收多个CTA或多个wave
  → CTA数不等于同时活跃SM数，也不证明饱和
```

E（PyPTO on GPU） 通过 [E4（PyPTO on GPU）]/[E5（PyPTO on GPU）] 提交这类grid，并非由Python逐SM launch，也非由Simpler ready queue分派到SM。`SPREAD` 调度偏好不构成每个SM都执行、每个SM负载均匀的保证。

### 9.2 CUDA Tile 的逻辑 block 参数与 profiler 物理线程

E（PyPTO on GPU） 的历史产物记录 `artifact_launch_abi_block=(1,1,1)`，profiler却显示 `block=(128,1,1)`。这不是少用了127个线程：CUDA Tile由编译器安排tile内线程，Host遵循Tile launch ABI；不能把该逻辑参数按普通SIMT线程数解释。[CUDA Tile kernel启动](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-tile-kernels.html) [E6（PyPTO on GPU）]

因此也不能反过来把profiler的128随意写回Tile ABI。分析要同时记录“调用侧ABI”和“最终物理worker配置”。

### 9.3 五阶段 softmax：历史 trace 核对结果

本节列出历史 GPU 实验的完整表格、数值及失败诊断，用于分析 launch ABI、物理 worker 与程序边界。原始 trace/summary 未迁入当前环境；下列解析结果来自历史实验，当前环境未重跑 profiling 或重新验证旧二进制。记录来源为[历史实验归档][GPU-history]。

实验输入 `[4096,128]` FP32；设备为 RTX 1000 Ada Generation Laptop GPU，20 SM、sm89。[E7（PyPTO on GPU）]

| 阶段 | 计算kernel名 | 实际grid | profiler物理block | 物理warp/block | CUPTI stream ID |
| --- | --- | --- | --- | --- | --- |
| row max | `pypto_row_reduction` | `(32,1,1)` | `(128,1,1)` | 4 | 13 |
| subtract | `pypto_fused_pointwise` | `(4096,1,1)` | `(128,1,1)` | 4 | 13 |
| exp | `pypto_fused_pointwise` | `(4096,1,1)` | `(128,1,1)` | 4 | 13 |
| row sum | `pypto_row_reduction` | `(32,1,1)` | `(128,1,1)` | 4 | 13 |
| divide | `pypto_fused_pointwise` | `(4096,1,1)` | `(128,1,1)` | 4 | 13 |

旧环境记录中，原始trace重新解析得到 **5次 `cuLaunchKernelEx`、5个kernel事件**，与summary一致；CUPTI stream ID不是Host raw stream指针，不能拿两者数值直接比较。该实验选择了非默认stream。

golden记录最大绝对误差 `1.1920928955078125e-7`，`atol=2e-7,rtol=2e-5`。这只属于该次历史实验与其输入，不是本次重跑或所有shape的正确性证明。[E7（PyPTO on GPU）] [E9（PyPTO on GPU）]

当时的两条失败记录：

```text
5 InCore + 1 Orchestration:
TensorIR emission rejected: program must contain exactly one function

该实验的 one-scope FP32 softmax:
TensorIR emission rejected: native RMSNorm input/output must be identical rank-2 BF16 tensors
```

第一条属于所选GPU emitter的多函数程序限制；第二条是该尝试进入当前模式检查后的诊断，不说明softmax在GPU上必须满足RMSNorm的条件。源码和历史构建未做二进制hash一一绑定，后续HEAD改变需要重新复现后才能称为当前运行结果。

### 9.4 NPU 不能直接套 CUDA grid/block 定义

B（PyPTO2-block版）/D（CANNBot DSL）/G（AutoFuse + Inductor） 的 `blockDim` 是NPU launch并行配置，必须结合AIC/AIV/mixed和sub-block理解，不是“一个block内有多少CUDA线程”。

F（Triton-Ascend） 更明确地分开：

```text
Triton用户logical grid
 → coalescing等可能调整
 → 若启用auto-map，Host将physical blockNum裁到对应资源数
 → AutoBlockify在kernel里增加logical-block循环
 → 每个physical block处理多个logical programs
```

[F3（Triton-Ascend）]/[F9（Triton-Ascend）] 是这一编译器—runtime联合契约的证据。故 `tl.program_id` 数量、CANN物理blockDim、AIV sub-block数量和GPU CTA数量不是同一个计数器。

对所有路线，都应该并列输出 `logical work / physical resources / launch config / stream / actual trace`，而不是只报告“用了32核”。

H 的 artifact 调用同样传 `block_num/stream`，kernel 用 block ID 与混合核角色分工；它不是 CUDA 的每 CTA 线程数。本机 H manifest 中的 `arch_scope=aic.c310` 与实际 lowered IR 中的 `_mix_aic/_mix_aiv` 需要合看，不能只读这个字段便判为纯 Cube，也不能将编译函数数当物理启动次数。[H-runtime] [H-execution] [H-run-artifacts]

### 9.5 I（CuTe DSL）与 E（PyPTO on GPU）：同是 CUDA launch，作者控制面不同

第 9.1 节关于 grid/CTA/warp/SM 的区分同样适用于 I。CuTe 的 `.launch()` 显式给 grid、block，并可带 cluster、动态 shared-memory 和 stream；作者还可在 kernel 里分配 warp 角色。E 的历史实验则由 PyPTO/TensorIR 编译与 runtime 决定相应映射。不能把 E 历史 trace 中的物理线程数套到 I 的 MLA/GQA，更不能把 Blackwell 的 TMA/TMEM/CLC 当作旧 Ada `sm89` 实验已验证的能力。[I-dsl] [I-gqa] [I-mla] [I-dynamic]

CuTe GQA simple 的源码给 decode 配置 12 个 warp（384 线程），然后以另一份 launch 配置执行 reduction。这个数能说明该源码的角色分配，不能说明 CTA 同时驻留多少 SM；同理，persistent grid 控制工作驻留与重复领取的策略，仍须结合每 CTA 的寄存器/SMEM/TMEM、cluster 约束及目标硬件分析占用率。[I-gqa]

本次没有 NVIDIA GPU，I 的 launch 数判断限于所引 Host 程序的静态调用结构，尚无运行 trace；E 的第 9.3 节历史 profiling 原样保留，二者不能相互充当实测来源。

与 H 横向比较时，应先写清单位：I GQA 的 384 是该 decode 配置的 CUDA 线程数，H FA 的 `block_num=8` 是 NPU launch 配置；两者都不是“同时占用多少 SM/Cube/Vector”的直接测量。H/I 的作者都能选择工作分工，但物理映射、mixed/cluster 协议和驻留约束各自验收。[I-gqa] [H-run-results] [H-runtime]

<a id="replacement"></a>
## 10. 技术替代与共用：明确改哪一层、不改哪一层

以下是源码基础上的工程评估，不是已经实现的适配器清单。成本用相对范围描述，未经原型不估人月。评价的是接口跨度，不评价团队能力。

### 10.1 替换后端：不是换一个可执行文件路径

| 改造 | 可保留 | 必须重做 / 对齐 | 相对跨度 |
| --- | --- | --- | --- |
| C（PyPTO3（Simpler））的InCore后端从PTOAS换成AscendNPU-IR | PyPTO用户表达、外层Orchestration、Simpler可望保留 | PTO/custom IR→目标dialect；tile/layout/effects；ABI、资源、sync/workspace；与现有wrapper衔接 | 中高；需要限定算子子集 |
| F（Triton-Ascend）改用PTOAS | Triton语法、部分TTIR/Inductor入口 | TTIR或适配IR→PTO；显式memory/layout责任；dynamic loop、dot/reduce、metadata/launch协议 | 高；不能只把HIVM文件喂给PTOAS |
| B（PyPTO2-block版）接PTOAS或AscendNPU-IR | block版前端、自有IR部分、Host API可能保留 | private ops、地址/TileGroup/mutex/VF语义；CCECodegen位置新增lowering；验证架构覆盖 | 中高 |
| D（CANNBot DSL）接另一MLIR后端 | Python tracing、部分标准arith/scf/memref表示 | CANNIR/Channel/缓冲所有权及AscendC专用语义转换；Host Device边界 | 中高；同为MLIR降低基础设施摩擦，不消除语义差异 |
| G（AutoFuse + Inductor）把Device codegen改为PTO/HIVM | Inductor入口、ASCIR、已有schedule/Host tiling可部分保留 | 明确转换发生在schedule前还是后；重写kernel renderer或较高层lowering；保持tiling/workspace ABI | 高；不能让两个后端重复做互相矛盾的规划 |
| E（PyPTO on GPU）复用NPU后端 | 只能共享目标无关语义与上层分析部分 | SM/CTA/warp、地址空间、tensor指令、同步与二进制全部目标相关 | 跨硬件重定向，不是局部插件替换 |
| A（PyPTO2-tensor版）换核内codegen后端 | Tensor/TileFwk图任务体系理论上可保留 | 从已分解计算导出稳定kernel IR/ABI、动态参数、与设备runtime的调用协议 | 中高；若同时替runtime则显著扩大 |
| H（CATLASS DSL）改接另一 NPU 核内后端 | Python staging、算法、可被目标表达的 TLA 语义 | 选定转换层，重建 layout tag、地址空间/MMAD/Vector、mixed split、异步效应、ABI/metadata；接 PTOAS 也不是只换命令 | 中高；与 F 共用部分下游不等于 TLA 和 TTIR 输入可互换。[H-passes] [H-mixed-pass] |
| E（PyPTO on GPU）改用 I（CuTe DSL）生成核内实现 | 上层数学接口、部分 wrapper 与正确性契约 | 从 PyPTO 图/模式选 CuTe 模板或生成 layout/atom/线程分区，适配编译缓存、参数、CUDA artifact/launch | 中高至高，取决于限定模板还是通用新后端；当前仅工程设想。[I-dsl] [I-executor] |
| H（CATLASS DSL）与 I（CuTe DSL）跨硬件重定向 | 数学语义、tile/资源/异步依赖的部分描述 | NPU CV/MTE/FIX/L0/UB 与 GPU warp/TMA/TMEM 的计算、存储、同步及进展映射 | 跨硬件编译设计；不能把 layout API 改名即视为移植完成。[H-api-copy] [I-copy] [I-mma] |

任何一项都应先回答：**IR进来时哪些决策已经确定，哪些允许后端推翻？** 如果前端已确定UB地址和pipeline槽位，后端不能再无条件bufferize成另一套生命周期；如果把较高层图交给后端，前端就必须让出对应规划职责。

### 10.2 共用内存规划：先共享契约，再判断是否共享算法实现

一个可共用的planner输入至少需要：

- 逻辑buffer、alias/view关系和读写effects；
- 大小表达式、dtype、layout、alignment、bank及memory space；
- 使用点和异步完成点，不仅是文本指令顺序；
- 单核、sub-block、合作组、task/stream等作用域；
- pipeline slot与并发实例数量；
- 动态上界、workspace计算方式和溢出检查；
- 外部owner、in-place、跨调用存活限制。

输出至少要区分地址 / offset计划、pool与容量、生命周期证明所依赖的同步条件、workspace/launch元数据。

| 层 | 更现实的共用切入点 | 不应直接搬用 |
| --- | --- | --- |
| kernel内UB/L1等 | buffer-effect-liveness通用模型、算法库、验证器、target resource descriptor | 绑定PTO/CANNIR/HIVM/ASCIR节点类型的pass与结果 |
| kernel内GM scratch | size-expression/ABI及allocator接口、sync-state描述 | 单纯把“总字节数”传给另一kernel而丢失布局 |
| task图GM | 统一依赖/alias/完成事件模型，再做跨task复用 | kernel内linear liveness结果 |
| 框架图GM | Inductor buffer ownership、stream/event完成与cache契约 | 把Simpler scope释放规则当通用框架allocator语义 |

A（PyPTO2-tensor版）/C（PyPTO3（Simpler）） 在跨task tensor计划方面接近；B（PyPTO2-block版）/D（CANNBot DSL）/F（Triton-Ascend）/G（AutoFuse + Inductor）在局部buffer领域有重叠；E（PyPTO on GPU）共享的是分析思想与部分算法，目标内存约束仍不同。不是“同一个planner一定更好”，而是统一输入/输出契约后才有可比较的正确性和维护边界。

H/I 的具体 planner 也应接入上述契约：H 的对齐递增分配可作为最基础的 offset 计划，若加生命周期复用，必须补 alias/异步完成分析；I 的 TS phase alias 已让作者表达部分复用关系，但 SMEM 字节、TMEM 列及 warp 资源仍有独立目标约束。两者都可参与局部资源描述和验证器的共用讨论，不能直接把结果当 task 图 GM 的释放计划。[H-scratch] [I-ts-memory]

### 10.3 把 B（PyPTO2-block版）/D（CANNBot DSL）/F（Triton-Ascend）/G（AutoFuse + Inductor）/H（CATLASS DSL）kernel 接入 Simpler：可行方向，但当前不是直接兼容

C（PyPTO3（Simpler））已有extern kernel、SPMD task、mixed kernel和tiling task这些接缝。[C6（PyPTO3（Simpler））] [C13（PyPTO3（Simpler））] [C19（PyPTO3（Simpler））] 一个分阶段原型可以这样界定：

```text
Host编译 / tiling / allocation（先保留现有工具）
  → kernel artifact + task adapter
  → Simpler可调用的device entry
  → 多block task提交
  → logical ID / core role / scratch / completion协定
  → 上下游TaskId依赖
```

必须处理八项，而不是只改一个函数签名：

| 事项 | 具体问题 |
| --- | --- |
| device-callable产物 | 原本全局launch入口能否变成worker可调用函数或兼容entry？不能在AICore里调用Host launch wrapper |
| 逻辑ID | 原kernel使用硬件blockIdx还是任务逻辑ID？persistent worker里的硬件索引不一定等于task索引 |
| 资源组 | block数量、AIC/AIV角色、sub-block、sync_start是否按原kernel预期一起可用 |
| 参数ABI | tensor descriptor、shape/stride、scalar、tiling data、workspace布局和对齐 |
| 内部同步 | 事件编号、全局barrier、sync-lock是否与runtime或其他并发task冲突 |
| GM可见性 | task完成通知前的数据发布、消费者cache/fence、异步DMA完成语义 |
| tiling位置 | 保留Host tiler、生成device tiler、或设备读取长度；不能默认为任意Host策略可搬到AICPU |
| 生命周期和异常 | scratch归谁、何时回收、异步完成、超时/失败如何传到scheduler |

B（PyPTO2-block版）/D（CANNBot DSL）通常需要把直接launch模型封装为task ABI；F（Triton-Ascend）还需维护compiler metadata、AutoBlockify与workspace索引的契约；G（AutoFuse + Inductor）需保留或替换生成tiling及wrapper中的职责。

这是一条有明确接口工作的集成路线，不是架构不可能；但也没有足够证据说当前block版/CANNBot/Triton/AutoFuse产物已经可不改地放入Simpler。A（PyPTO2-tensor版）/C（PyPTO3（Simpler）） scheduler互换更大：其任务图格式、依赖、内存计划、设备代码及控制协议一起耦合。

H 应按同样八项验收：现成 PyACL/AscendCL artifact 是 Host 启动对象，需另外解决 worker 可调用入口、参数打包、逻辑 block ID、AIC/AIV 资源组、flag/workspace 重入及完成通知。当前并未完成该适配。I 的 CUDA cubin 则不能直接送入 NPU Simpler worker；可借鉴其资源协议或在 GPU 任务体系中设计适配，跨硬件实现转换是另一项工程。[H-execution] [H-runtime] [H-mixed-pass] [I-executor] [I-task]

### 10.4 接入另一个 scheduler，不一定把所有源码“统一”

| 组合方式 | 用户感知 | 能取得什么 | 不能自动取得什么 |
| --- | --- | --- | --- |
| Host按顺序调用各路线kernel | 大部分原API保留 | 快速组合、数值/ABI验证 | 消除Host launch、统一task图或片上融合 |
| 外层任务runtime调用extern kernel | 需要task wrapper/metadata | 统一依赖、设备派发、部分GM管理 | 核内IR统一、跨kernel自动片上复用 |
| 共享kernel IR / 后端 | 可能保留多个前端 | 共用局部优化/目标覆盖 | 相同用户抽象或相同scheduler |
| 统一高层程序IR再lower到多kernel后端 | 用户入口可选择 | 更一致的图级融合/内存策略 | 低开发成本；这是范围最大的体系工程 |

优先做哪个不是本文替管理层决策；可先用一个带动态长度和workspace的attention子图验证接口跨度，再判断收益是否抵得过迁移与维护成本。

H 可以作为 NPU kernel 提供者参与 Host 组合或 extern 适配原型；是否进入 Simpler 取决于上一节契约，不取决于同用 Python/MLIR。I 可在 GPU 上保留自己的 CUDA kernel/Host 编译，TS 则服务核内资源协作；将它们纳入统一程序 IR 或跨设备调度，需要额外的目标与完成事件语义。第 10.5 节保留这些方向的逐项可保留/需重做清单。[H-execution] [I-executor] [I-task]

### 10.5 H/I 的替代关系与公共契约

| 设想 | 可以保留什么 | 还需具体实现什么 | 当前判断 |
| --- | --- | --- | --- |
| H kernel 接入 Simpler | TLA 核内算法、物理 layout、部分资源策略 | `kernel_entry(args)`/参数打包、逻辑 block ID、mixed entry、workspace/flag、任务完成与重入协议 | 设计方向可行；当前 PyACL launch artifact 不等于 Simpler extern ABI 已兼容 |
| H 与 F 共用下游编译 | 部分 HIVM/AVE 语义和目标转换 | 固定版本/构建配置、layout/地址空间/异步效应契约、双方进入下游时已经做出的 tiling 选择 | 有共同基础设施对象；不是直接交换 Python frontend 就能共用全部 passes |
| H/D/B 共用局部 planner | 空间、容量、对齐、读写/异步生命周期、槽数等描述 | 各 IR 的效应提取、alias、硬件通路、合法 barrier；把计划重新编码回各 DSL/IR | 可先共享测试与算法输入契约，不能把 H 的简单偏移分配器当现成全能 planner |
| E 改用 I 作为核内实现 | 客户算子/模型接口与部分数学规格 | PyPTO TensorIR 模式到 CuTe layout/atom/线程分区的转换，CUDA Host ABI/cache/export 适配 | 能讨论新后端或模板调用方案；当前无现成兼容证据 |
| H/I 共用流水描述 | 数据依赖、buffer 所有权、stage、生命周期/资源约束 | NPU CV/FIX/MTE 与 GPU warp/TMA/TMEM 的目标映射、可见性和进展模型 | 最有价值的共用方向之一；物理空间及同步不能机械同名映射 |
| I 的 TS 思路用于整层 runtime | 资源声明、依赖可视化、协议检查的设计经验 | 任意任务入口、全局就绪/完成、跨阶段 GM、失败处理和跨核资源调度 | 可借鉴方法；不能把 CTA 内 TaskManager 直接等同 A/C 的 runtime |

H 的编译产物可以由 Host 装载/执行，只能证明其现有 launch ABI 完整，不能证明它已经满足任意设备 runtime 的调用规约。I 的 JIT executor 提供 cubin 与导出相关逻辑，也不能消除设备二进制架构、参数布局、stream/上下文和全局同步上的差异。[H-runtime] [H-execution] [I-executor] [I-task] [H-scratch]

H/I 的资源规划、layout、异步协议和 kernel artifact 为第 10.1—10.4 节的公共契约提供了具体代码参照。适配应先在一个小 kernel 上检验动态参数、重入和完成通知，再扩大到 PA/整层；这些原型结果才是判断现成替代关系的依据。

## 11. 多维相似性、优化空间和开发成本

### 11.1 相似性必须按层回答

| 比较维度 | 最明显的相似组 | 容易被遗漏的差别 |
| --- | --- | --- |
| 前端代码/基础IR血缘 | A（PyPTO2-tensor版）—B（PyPTO2-block版）；C（PyPTO3（Simpler））—E（PyPTO on GPU，独立 checkout） | 血缘并不决定runtime；C（PyPTO3（Simpler））/E（PyPTO on GPU）能力演进未自动同步 |
| 程序级设备任务系统 | A（PyPTO2-tensor版）—C（PyPTO3（Simpler）） | TileFwk与Simpler不是同一套实现 |
| 显式核内控制 | B（PyPTO2-block版）—D（CANNBot DSL）；C（PyPTO3（Simpler））的SPMD/InCore亦接近 | C（PyPTO3（Simpler））还带外层task语义；B（PyPTO2-block版）不以MLIR为主IR |
| MLIR贯穿较长kernel编译链 | D（CANNBot DSL）—F（Triton-Ascend） | D（CANNBot DSL）偏显式硬件/Host staging，F（Triton-Ascend）偏逻辑tile program；dialect不兼容 |
| 自有前端IR→专用MLIR后端 | C（PyPTO3（Simpler））—E（PyPTO on GPU） | C（PyPTO3（Simpler））的PTOAS不负责整个task系统；E（PyPTO on GPU）当前受pattern限制 |
| Inductor用户入口/融合kernel生成 | F（Triton-Ascend）—G（AutoFuse + Inductor） | F（Triton-Ascend）有独立kernel DSL，G（AutoFuse + Inductor）有自动生成Host tiler；两者有extern/template边界 |
| 自动核内buffer/同步与直接launch | E（PyPTO on GPU）—F（Triton-Ascend）；G（AutoFuse + Inductor）也有同类分工 | GPU/NPU硬件映射不同；F（Triton-Ascend）/G（AutoFuse + Inductor）自动化责任不完全相同 |
| 固定物理tile + runtime有效尺寸 | B（PyPTO2-block版）—C（PyPTO3（Simpler））—D（CANNBot DSL）—F（Triton-Ascend） | 语法、合法上界、cache特化和同步协议各异 |
| 单独生成Host tiling | G（AutoFuse + Inductor）最明确；B（PyPTO2-block版）的AOT客户场景接近 | G（AutoFuse + Inductor）是编译器生成；B（PyPTO2-block版）通常是客户普通Python或Host C++策略 |
| tiling也可作为设备task | C（PyPTO3（Simpler））有直接实例 | 不是Simpler自动替客户写所有tiling |
| Python layout/atom/显式局部资源 | H（CATLASS DSL）—I（CuTe DSL）；同硬件工程再对照 B/D | H 更侧重物理 layout tag 与 NPU 通路；I 有线程/值 layout 代数与 GPU atom，名字相近不证明代码血缘。[H-api-layout] [I-layout] |
| 显式 NPU 核内控制且较早进入 MLIR | D（CANNBot DSL）—H（CATLASS DSL）；下游基础设施另比较 F | D 的 Channel/Host staging/CANNIR 与 H 的 TLA/helper/mixed/HIVM 不同；共享 MLIR 不等于共享语义。[H-dsl] [H-passes] |
| 固定资源上的动态工作与资源调度 | H 的 block-stride/StreamK、I 的 static persistent/CLC/TS | 工作 tile 分配、warp 资源协议与 A/C 的跨程序 ready-task 调度属于不同层；不是自动组成同一任务系统。[H-streamk] [I-dynamic] [I-task] |

这使某两条路线能够同时“在MLIR方面很像，在调度方面很不像”，并不矛盾。最后一节的单一最相似者只是明确权重后的压缩表达。

### 11.2 各路线的主要优化空间，不等于已兑现性能

| 路线 | 可直接发力的维度 | 典型代价 / 风险 |
| --- | --- | --- |
| A（PyPTO2-tensor版） | 图/task粒度、依赖、tile、GM复用、跨阶段并发 | 编译与runtime耦合较深；过细task导致调度/metadata开销 |
| B（PyPTO2-block版） | 物理tile、VF/Cube布局、地址、double-buffer、角色分工 | 用户与算子维护者承担资源证明；架构特化和尾块组合增多 |
| C（PyPTO3（Simpler）） | task融合/拆分、Simpler并发、SPMD子图、核内PTO优化、manual依赖 | 显式依赖和GM一致性难度；大SPMD task可降低外层调度弹性 |
| D（CANNBot DSL） | Channel/SWP、搬运布局、memory reuse、低层vector/cube控制 | 生命周期与事件协议复杂；高性能例子与动态/AOT契约需同时维护 |
| E（PyPTO on GPU） | pattern覆盖、tile/layout、CTA规模、bucket、head合并、减少多launch/GM | 静态专门化与编译器覆盖；更大融合可能增加资源/编译压力 |
| F（Triton-Ascend） | kernel算法、grid/tile/autotune、CV流水、buffer与同步passes | 高层tile可能产生不理想底层分解；编译器成本模型需随硬件更新 |
| G（AutoFuse + Inductor） | Inductor融合/模板、ASCIR schedule、自动tiling、buffer复用、避免extern边界 | 用户省去低层工作，复杂性转移给compiler；不支持的图仍会拆分 |
| H（CATLASS DSL） | CV 分工、物理 layout/搬运、L1/L0/UB tile/槽、寄存器链、StreamK 归并与局部内存复用 | 固定资源/尾块/跨 CV 证明及工具链适配；当前 Q=1 精度失败须先处理，不能用局部融合源码给性能背书。[H-fa] [H-streamk] [H-run-diagnostics] |
| I（CuTe DSL） | atom/线程分区、warp 角色、TMA/MMA overlap、persistent/CLC、SMEM/TMEM 及 split 策略 | 架构特化、资源/同步、编译展开与实验接口维护；TS 检查范围和本机无 GPU 实测需明确。[I-gqa] [I-dynamic] [I-ts-memory] [I-checker] |

小batch decode、超长reduce和大GEMM不一定需要同一最佳策略。更大fusion可能减少launch却损失局部资源效率；更多task可能提高负载均衡却增加scheduler开销。只能按实际输入分布测量。

### 11.3 成本分客户、算子作者和基础设施三方

| 成本承担者 | A（PyPTO2-tensor版）/C（PyPTO3（Simpler）） | B（PyPTO2-block版）/D（CANNBot DSL） | E（PyPTO on GPU）/F（Triton-Ascend） | G（AutoFuse + Inductor） | H（CATLASS DSL） | I（CuTe DSL） |
| --- | --- | --- | --- | --- | --- | --- |
| 客户写数学程序 | 需学习Tensor/scope/task模型；库封装可降低负担 | 若直接写kernel则门槛高；调用封装库则低 | E（PyPTO on GPU）取决于pattern/API；F（Triton-Ascend）可直接DSL或PyTorch | PyTorch入口最少改写，但可见fallback/动态guard | 封装调用可轻；自写需 TLA/物理存储与编译契约 | 封装调用可轻；自写需 CuTe layout/atom/Host-device staging |
| 高性能算子作者 | 同时理解任务粒度和核内分工 | 低层资源/同步/尾块责任更多 | 理解tile program、compiler lowering和特化策略 | 若只用现成图很少写kernel；扩模板/新lowering责任转到后端开发者 | NPU 数据路径、静态容量、mask/尾块、CV/线程同步 | 线程/值映射、架构 atom、warp/pipeline、split/合并与资源预算 |
| 编译器开发者 | 图语义、核内编译、runtime契约联动 | 核内IR、硬件语义与JIT/Host边界 | 跨IR及target lowering、autotune与ABI | Inductor接入+图schedule+tiling+codegen+模板覆盖 | TLA passes、mixed/Vector lowering、固定 NPU IR/CANN 兼容 | CuTe/相关 dialect 到目标编译、Host launcher、配套组件版本 |
| runtime开发者 | A（PyPTO2-tensor版）/C（PyPTO3（Simpler））的任务依赖/派发/内存/异常体系较重 | 相对集中在JIT/cache/ABI/launch；复杂合作kernel仍有协议 | E（PyPTO on GPU）为CUDA artifact/launch；F（Triton-Ascend）为CANN及compiler metadata | Host wrapper、generated tiling、workspace、框架/stream集成 | 参数打包、artifact/cache、AscendCL、mixed metadata/stream | JIT/export/cubin、CUDA context/stream、workspace 与 kernel 资源协议 |
| 测试与维护 | task和kernel两级验证 | 物理资源、同步、shape边界 | 图覆盖/特化/target/toolchain组合 | 前端分组/模板/extern、动态tiling、框架版本组合 | 数值/动态复用/同步/产物一致性；现有 Q=1 失败回归 | SM/布局/尾块/变量序列/split，TS 有界检查加数值与硬件验证 |

这是责任位置比较，不是“代码越少开发成本越低”的排名；性能调优、诊断、版本适配和客户支持可能大于首个kernel实现成本。

H/I 两列可结合实际分工核对：[H-dsl] [H-passes] [H-runtime]、[I-dsl] [I-executor] [I-task]。H 的六通过/一失败不是维护成本的数值估计，I 的 TS checker 也不是故障率或开发效率测量；它们只是说明数值回归、资源协议和工具链分别由谁维护。

### 11.4 H/I 的相似性、优化空间与工程成本

| 维度 | H（CATLASS DSL） | I（CuTe DSL） |
| --- | --- | --- |
| 最接近的既有局部工作 | B/D 的显式核内工程；F 的部分 AscendNPU-IR 下游 | E 的 CUDA artifact/launch；更细的核内角色/资源控制可与 B/D/H 对照 |
| 两者之间最明显的相似性 | Python 元编程、显式 tensor/layout、局部 buffer/流水、编译产物 | 相同责任族；CuTe layout 代数、thread/value 分区更明确，物理目标不同 |
| 可直接优化的对象 | CV 分工、L1/L0/UB tile/槽、FA 状态、布局搬运、StreamK 归并 | atom/线程分区、TMA/MMA overlap、warp 角色、persistent/CLC、SMEM/TMEM 占用 |
| 客户调用成本 | 调用已有封装可以很低；直接实现 kernel 需要学习物理路径 | 现成算子/封装可降低调用成本；手写高性能 kernel 需掌握 layout/warp/pipeline |
| 算子作者成本 | 尾块、mask、buffer/flag、跨 CV 交接；本例 PA 还需补分页与设备长度逻辑 | 架构特定 tile/atom、线程合作、barrier、split/归并；标准 GQA PA 不能直接以 MLA 替代 |
| 编译器/运行库成本 | TLA passes、AscendNPU-IR 固定版本、CANN、ABI/cache/mixed metadata | Python DSL 与配套编译组件版本、CUDA/SM 目标、JIT/export、实验性接口演进 |
| 可维护性工具 | 受限 AutoSync、具体 IR/manifest、现有 end-to-end golden | pipeline helper、TS 资源/schedule/检查器、可调度示例；检查器不替代数值/硬件测试 |

H 的公开说明将当前实现定位为 TLA API 封装，更完整的 CATLASS C++ 分层抽象仍有后续建设空间。因此不能把 C++ 模板库里所有 schedule/算子自动计入 H。I 也应分开核心 CuTe API、`cute_ext`/experimental 和具体架构示例；依赖文件与目录名称本身不是接口长期稳定性的保证。[H-readme] [I-requirements] [I-task] [I-gqa]

从 megaKernel 的工程成本看，**显式控制与自动化之间存在多种责任分配方式**：H 已有受限自动同步，I TS 可自动组织已声明的资源协议；它们都不是简单的“所有同步全手写”，也都没有因此自动获得任意模型整层融合。相应成本应分别计入客户、算子作者和基础设施三方。

## 12. 证据口径、验证结果与后续测试边界

### 12.1 验证范围与已记录结果

本节汇总源码核对、历史 CPU/GPU 实验和 A5 环境及算子验证，分别标注来源与适用范围；A5 日志提供随文快照。理论模型、源码机制、编译执行正确性和硬件性能是四类不同证据，不能相互替代。总表覆盖 A—I：A—G 的测试逐项列出，H 的构建与用例结果详见第 12.1.3 节，I 仅有源码分析；没有九路线同口径的整层或跨硬件性能实验。

| 事项 | 已记录状态 | 能支持什么结论 |
| --- | --- | --- |
| 九条主链、关键分支、IR/launch/tiling源码 | 已检查本地HEAD及相关函数/调用点 | 实现结构、接口约束、责任归属；H/I 的具体对象另列下方两行，固定来源见第 13 章 |
| tensor版/block版/PyPTO3/CANNBot 重点对比 | 已核对tile/scope、TileGroup/Channel、PA分工、动态策略、核内融合及片上复用 | 用户控制与实际编译/运行责任；不据此给性能排序 |
| block版 SPMD / PyPTO3 MPMD 任务组织 | 已核对 block版 分核/launcher、PyPTO3 三阶段 SPMD 示例、Simpler kernel-ID 派发及完成路径；未执行新增 NPU 片段 | SPMD kernel 与多程序任务体系可组合；调度粒度不等于核内循环粒度 |
| 九路线前端/API 边界 | 已核对公开表面、参数/目标/融合限制、原生计算链及PyTorch入口 | 区分有 Python DSL、构图接口、API/后端支持；不是完整 ISA 覆盖测试；包含 H 的 TLA/AutoSync 与 I 的 layout/atom/Host-device/TS 边界 |
| 动态tiling写法及契约 | 已核对block版 dataclass/key/cache/离线Host回调、CANNBot Dim/bounded tiler、Simpler两种显式tiler及Triton/GPU/AutoFuse入口 | 谁需写函数、怎样传参、策略与产物何时失效 |
| 用户代码到核内融合 | 已核对tensor版纵向合图/scope、block版 Tile/VF、InCore outlining、CANNBot VF passes、GPU组合算子、Triton softmax和AutoFuse局部值/输出处理 | 存在何种融合机制、怎样编写及限制在哪里；不证明特定shape的kernel数、无spill或性能 |
| softmax Host分核公式 | 源码分析阶段CPU执行162组合法输入、4组拒绝输入，检查tile覆盖与尾行总数 | 本文Host策略及整数分工公式；不证明原生kernel尾块正确 |
| block版原生Python work_ranges | 源码分析阶段抽取原函数与hybrid_bounds在CPU执行360组计划；按kernel编号公式检查无漏项/重复 | 两种分工模式的元数据覆盖性；不证明NPU执行、最优均衡或性能 |
| pypto-lib softmax/native PA/CCE tiling task | 已核对完整入口和关键实现 | C（PyPTO3（Simpler））同时容纳task与SPMD、不同tiling位置 |
| Markdown中的Python片段 | 全部Python代码块逐段AST解析；不导入执行NPU DSL | 排除语法损坏；不证明上下文依赖完整、后端支持或 NPU 数值正确 |
| MLIR 专章 | 核对本地 PTO/CANNIR/HIVM/TensorIR 源码与官方机制文档；未编译本文 IR 摘录 | 表示/编译/执行的边界及具体使用位置 |
| PTOAS / AscendNPU-IR 编译与 launch 边界 | 核对 Group/多函数输入、VPTO Host stub/fatobj、HFusion 多 kernel/tiling、HACC Host launch、HIVMC Host 编译及 Triton launcher；未运行新增目标测试 | 区分后端本体与集成入口、函数数与 launch 数；不证明任意模型图可运行或存在 AICPU 动态任务 runtime |
| megaKernel 定量推导 | 复核 softmax 逻辑访问、PA transfer 容量、GQA/权重强度公式 | 分析模型；不是 PMU 实测、性能预测或路线名次 |
| 公共softmax语义参考 | 历史 GPU 环境中的 CPU 实验，torch `2.11.0+cu128`；算法与结果见下文 | 分段算法与torch.softmax一致 |
| 公共paged decode参考与padded表达 | 历史 GPU 环境中的 CPU 实验，同shape不同L内容 | 数值在给定容差内一致，不是逐位等价 |
| 既有GPU原始trace/summary | 保留旧环境文档中的解析记录；本次未迁入原始材料，未重跑profiling | 说明当时记录的五阶段launch/grid/block/stream；不能称为本次原始trace复核 |
| NPU原生kernel编译及执行 | 已完成选定A5 PA的编译执行与数值校验，逐项见下表；A2/A3未在本轮执行 | 支持指定用例和环境的正确性结论；不能推广到所有硬件、shape或整层性能 |
| 九路线整层Transformer、跨硬件性能 | 未执行 | 不提供速度、利用率或成熟度数值排名 |
| 后端替换、planner复用、scheduler适配 | 仅做接口跨度评估 | 不称为已验证兼容或可直接替换 |
| H（CATLASS DSL）源码与 A5 验证 | TLA/编译/runtime、layout/同步/资源已核对；构建安装完成，七组主用例六通过、一组 Q=1 连续 decode 精度失败，见第 12.1.3 节 | 支持这些指定 MMAD/mixed/连续 FA 的结果及动态产物复用；未建立原生分页 PA 或整层性能证据。[H-run] |
| I（CuTe DSL）固定源码分析 | 核对 Host/device、layout/atom、softmax、分页 MLA、连续 GQA、persistent/CLC/TS；本机未安装/运行 CUDA kernel | 可判断接口责任、算法与静态 launch 结构；不能据此宣称数值通过、kernel 性能或模型 runtime 已完成。[I-dsl] [I-mla] [I-gqa] [I-task] |

公共softmax测试：

| shape | 分段结果对torch.softmax最大绝对差 |
| --- | --- |
| `[777,300]` | `2.9802322387695312e-8` |
| `[8,131075]` | `2.3283064365386963e-10` |
| `[3,1]` | `0` |
| `[1,1025]` | `9.313225746154785e-10` |

以上为历史 GPU 环境中的 FP32 CPU 测试：`torch.manual_seed(0)` 后按表中顺序生成 `torch.randn` 输入，`torch.set_num_threads(4)`、`chunk=1024`，校验 `atol=1e-6,rtol=1e-5`。PA 使用本文 `make_decode_case` 的 BF16 输入/输出，`L=[1,129,513]` 和 `[127,128,512]` 的最大绝对差分别为 `1.9073486328125e-6` 和 `0.000244140625`，均通过 `atol=2e-2,rtol=2e-2`。CPU 线程/归约实现会影响有限精度结果，不要求逐位一致。

这些测试只验证本文公共golden，不是B（PyPTO2-block版）/D（CANNBot DSL）/F（Triton-Ascend）等NPU DSL实现的验证。这些描述对应旧 GPU 环境的 CPU 检查；当前 A5 的依赖安装、工具链构建及本地兼容性改动另见本节环境记录。

已记录的CPU计划检查使用现有 `gpu/PyPTO-LOVE-TensorIR/envs/ada-sm89` 环境，torch `2.11.0+cu128`，单CPU线程；不导入block版 DSL或运行NPU代码。softmax检查覆盖M为`0/1/15/16/17/33/65/777/2049`、N为`1/300/512`、核数为`1/2/8/24/32/64`的笛卡尔积，并拒绝负M、零N、N超容量和零核数。work_ranges检查覆盖Q长度列表`[1]`、`[1,1,1]`、`[127,128,129]`、`[512,256,200]`、`[128]*8`，head数`1/4/16`、核数`1/2/8/24`、默认/连续/步进配对三种选择以及两组KV长度/可见范围。检查的是每个工作编号恰好覆盖一次、单核编号递增及未launch核心无工作，不是完整attention数值测试或成本模型最优性证明。

历史文档校验记录包括58个 Python 代码块 AST、196个本地引用、61张表格、99个代码块及13个导航锚点；这些是对应快照的统计。当前文档校验同样覆盖代码块、AST、引用、表格和导航。4个 MLIR 代码块及 AscendC 摘录仍未逐段调用目标工具链编译；SPMD/派发及 PTOAS/HACC/HFusion 的额外测试也不能由通用 PA 通过代为验收。文档检查不等于浏览器渲染或硬件测试。


#### 12.1.1 A5 环境、基础算子与逐路线 PA 验证

设备为 `Ascend950PR_957b`，物理 NPU 4 映射为逻辑 `npu:0`；CANN 为 `9.2.0-weekly.20260902.01`，驱动为 `25.7.rc1.2`。多数路线使用 Torch 2.10 / torch_npu 2.10.post4；AutoFuse 使用独立的 Torch / torch_npu 2.12 环境。PyPTO3 使用 Simpler 的固定子模块和 PTOAS v0.57；Triton-Ascend 为本地构建的 3.6.0。源码、工具链及本地构建改动的版本摘要见[环境记录][RUN-env]。这些是共享机器上的直接正确性运行，pytest 总耗时不代表算子延迟。

##### 环境准备阶段的基础算子

以下为本 session 已完成的基础验证，结果与9份原始日志追加归档于[基础算子证据][RUN-smoke]。它们使第2章的 softmax 表达和分工分析有实际运行对象；CANNBot 此次基础测试取自 arena 的 RMSNorm/MatMul，不把它记作本章 Channel softmax 已运行。

| 路线 | 本次实际输入 | 已记录结果与对应对象 |
| --- | --- | --- |
| A：tensor版 | FP32 softmax `[32,32,1,256]` | 原 softmax 例子通过；对应 Tensor 运算与 tile 配置。[A12（PyPTO2-tensor版）] |
| B：block版 | FP32 softmax `[2048,64]`、`[4096,128]`、`[1000,200]`、`[777,300]`、`[100,512]`、`[2049,100]` | 原测试6组通过；对应固定物理 tile、动态 valid shape 及多核尾行。[B14（PyPTO2-block版）] |
| C：PyPTO3 | FP32 softmax `[512,256]`，另有 hello world `[1024,512]`、matmul `[256,256]` | pypto-lib 的3个例子以 A5 入口通过；softmax 对应本章 `pl.parallel/CORE_GROUP` 写法。[C15（PyPTO3（Simpler））] |
| D：CANNBot | arena RMSNorm：FP16 `[4096,2304]`、FP32 `[768,12288]`；MatMul：M/K/N=`4096/3840/384`，FP16/BF16 | 原 arena 数值测试通过；为 DSL 编译、混合精度矩阵/向量入口提供对象。[D-arena-rms] [D-arena-mm] |
| E：PyPTO on GPU | 本轮 NPU 环境未执行 CUDA 基础算子 | 旧 SM89 softmax 记录仍见第 9.3 节，不计入本轮 A5 通过数。[E7（PyPTO on GPU）] [E9（PyPTO on GPU）] |
| F：Triton-Ascend | FP32 masked row softmax `[777,300]`、`[31,512]`；kernel仓 BF16 MatMul M/K/N=`2048/7168/16384` | softmax误差分别为 `2.9802322387695312e-8`、`7.450580596923828e-9`；MatMul在记录的API兼容性补丁后通过 |
| G：AutoFuse + Inductor | FP32 `torch.compile(fullgraph=True)` softmax `[777,300]` | 误差 `2.9802322387695312e-8`；确认注册的NPU调度来自源码扩展，未由此推断任意图都全融合 |
| H：CATLASS DSL | f16/f16→f32 MMAD 两组 M/N/K=128/128/256、384/160/272；f32 mixed MMAD+add 32³；FA 另见第 12.1.3 节 | 三组矩阵用例通过；两组 MMAD 相同 cache key/kernel.o。未把 FA 内 softmax 计成独立 softmax 测试。[H-run-results] [H-run-artifacts] |
| I：CuTe DSL | 本机 NPU 环境，没有对应 CUDA 执行输入 | 源码包含 softmax 等示例；本轮未运行，不计入 A5 基础算子通过数。[I-softmax] |

这些基础测试与下面PA测试使用各自的输入和容差。日志含编译及runtime诊断，总耗时不是纯kernel延迟；原CPU计划检查也继续保留，二者分别验证Host分工公式与设备计算。

##### PagedAttention

| 标识与路线 | 完整实现 / 原测试来源 | 本轮 PA 结果 | 已验证的范围 |
| --- | --- | --- | --- |
| A — PyPTO2-tensor版 | [ctrl_perf_kernel][A-pa] | **原版精度失败；本地修正版通过** | 前处理→KV写入→PA；FP16，B=4，Hq/Hkv=8/1，D=128，页大小128，L=127/128/129/513 |
| B — PyPTO2-block版 | [flex_attention_bf16][B-pa]、[页大小测试][B-pa-test] | **通过** | BF16；页大小128/256/512的prefill，以及补充的单token decode；后者L=127/128/129/513 |
| C — PyPTO3（Simpler） | [通用 PA 程序][C-pa]、[PTOAS 测试][C-pa-test] | **2例通过** | B=64，Hq/Hkv=16/1，D=128，页大小128，L=8192及8100；BF16输入、FP32输出 |
| D — CANNBot DSL | [paged kernel][D-pa]、[原测试][D-pa-test] | **2个原用例及1个decode用例通过** | FP16，D=128，页大小128；无mask多query attention；B=4、Hq/Hkv=9/1、Q长度1、L=512的decode |
| E — PyPTO on GPU | [paged_attention_decode][E-pa]、[benchmark入口][E-pa-test] | **本轮未执行** | 当前是NPU环境；保留代码事实，历史GPU测量另行说明 |
| F — Triton-Ascend | [paged_attention_fwd][F-pa]、[原测试][F-pa-test] | **8例通过** | FP16，页大小16；MHA/GQA、causal prefill/decode；QK维32/48、V维32，含KV尾页 |
| G — AutoFuse + Inductor | [Inductor接入][G-register]、[融合调度][G-scheduler]；使用第2.4节的分页图表达 | **图级2组长度通过** | BF16，B=3，Hq/Hkv=40/8，D=128；L=[1,129,513]及[127,128,512]；含外部gather/bmm |
| H — CATLASS DSL | [连续 FA][H-fa] | **没有原生分页 PA 的运行结果** | 连续 FA Q=117 指定用例通过、Q=1 失败；无 page table，不能用这批结果给 PA 打通过。[H-run-results] |
| I — CuTe DSL | [分页 FP16 MLA][I-mla]、[FP8 MLA][I-mla-fp8]；[连续 GQA][I-gqa] | **本轮未执行，只有源码证据** | MLA latent/rope 与标准 GQA ABI 不同；连续 GQA 两阶段不算分页 GQA 验证 |

这张表不能简化成“所有原版 PA 已通过”。A 有已复现的原版失败；C 的实测对象是通用多任务 PA；G 验证的是图表达及其混合执行结果。各路线的布局、dtype、scale、reference 和规模不一致。[完整结果与容差][RUN-results]

##### 三项实测结论与能力边界

**PyPTO2-tensor版：运行成功不等于 KV 更新语义正确。** 原文件是控制CPU性能看护程序，未提供 attention golden。新增完整参考计算后，原版最大绝对误差为 `0.7950679659843445`；板端输出却与更新前cache的PA吻合，误差为 `0.0001582503318786621`。使用更新后cache的golden时，前端解释器日志为 `index 13 result PASS`，板端失败。[原代码][A-pa]、[原版板端日志][RUN-A-fail]、[前端日志][RUN-A-front]

在 `LOOP_PRE` 完成后、PA 开始前重建两个 view，完整参考计算通过，最大误差为 `0.0002454519271850586`，仍使用 `atol=1e-3, rtol=1e-2`。下面两行来自本轮本地修正，**未合入上游**；只验证了B=4，不替代原B=16性能门禁，也未定位到具体编译Pass或runtime根因。[补丁][RUN-A-patch]、[修正版日志][RUN-A-pass]

```python
k_cache_2d = pypto.reshape(k_cache, kv_2d_shape, inplace=True)
v_cache_2d = pypto.reshape(v_cache, kv_2d_shape, inplace=True)
```

**CANNBot：当前已找到原生分页实现。** `FlashAttnNoquant(paged=True)` 的QK/PV阶段通过 `block_table[batch_idx,n_idx]` 读取物理页，测试将逻辑序列写入打乱的物理页，并传入设备端 `seqused_kv`。此次decode的KV长度为512，未验证KV尾页或同批异长请求。两个多query用例是无mask full attention，不能写成causal prefill验证。实现位于DSL仓的examples；arena的连续FlashAttention样例不能代替这条证据。[寻址实现][D-pa]、[输入构造与校验][D-pa-test]

**AutoFuse：图完整捕获后仍可包含外部算子。** 本轮导入源码版 `inductor_npu_ext` 后，确认NPU调度类属于该扩展；`torch.compile(fullgraph=True, dynamic=False)` 的两组长度均通过。生成wrapper包含7个不同AutoFuse函数、9个生成函数调用点，同时有2次 `aten.index_select`、2次 `aten.bmm`、2次 `aten.repeat_interleave`，以及arange、bitwise_not、reshape。这是wrapper静态调用点计数，**不是设备kernel launch计数**；生成函数名含有matmul也不能证明矩阵乘已被融合。[注册代码][G-register]、[生成代码检查结果][RUN-G-lowering]

##### 验证边界

- block版 的 prefill 和单 token decode 均有通过记录；页大小测试还检查三个布局的输出逐位一致。[block版原测试][B-pa-test]、[本地结果][RUN-results]
- PyPTO3通用PA原测试使用 `scale=1.0`，同一用例内请求长度相同。它验证了8192/8100长度，不能推出整个动态shape矩阵已通过。[测试与reference][C-pa-test]
- PyPTO3的Qwen3-14B融合SPMD PA也有完整源码，含前置处理与缓存写入；其测试CLI只接受 `a2a3/a2a3sim`，本轮未移植或运行其A5路径。通用PA通过不能替它背书。[融合PA][C-qwen]、[平台限制][C-qwen-platform]
- 旧环境的Ada SM89五阶段softmax trace/summary本轮未迁入当前GPU checkout，不再列为可复核的当前测量；当前GPU代码按本次checkout及其公开source lock核对。[GPU版本锁][E-lock]

#### 12.1.2 实测证据归档与复跑

本仓库的 `tests/npu_gpu_programming_stacks_comparison.a5.json` 归档了脱敏后的[环境记录][RUN-env]、[PA清单][RUN-results]、驱动源码、[复跑脚本][RUN-script]、[Triton JUnit][RUN-F-xml]、[PyPTO3 JUnit][RUN-C-xml]、[CANNBot JUnit][RUN-D-xml]及原版/修正版日志与补丁。实验引用固定到该快照的GitHub commit，用于核验本机测量；各工具仓库的代码引用用于核验实现。二进制、张量及完整生成目录仍保留在本机。

以下命令用于本session已经完成安装的workspace，在根目录运行，每次创建新的日志目录。快照中的驱动和脚本可供核对与恢复；仅克隆本文所在仓库不会自动得到工具链及运行环境。tensor版明确选择本地修正版；不替换仓库原始文件。

```bash
bash .npu-stack/paged-attention/run.sh pypto2_rebind 4
bash .npu-stack/paged-attention/run.sh pypto2_pro 4
bash .npu-stack/paged-attention/run.sh pypto3 4
bash .npu-stack/paged-attention/run.sh cannbot 4
bash .npu-stack/paged-attention/run.sh triton 4
bash .npu-stack/paged-attention/run.sh autofuse 4
```

将第一条的`pypto2_rebind`改为`pypto2`可复现未修改原版，当前预期在KV-append精度校验处非零退出。修正版已通过该复跑入口再次验证。[RUN-script] [RUN-results]

#### 12.1.3 H/I 本次核对与验证

**H 已完成构建安装和板端验证；七组主用例中六组通过，Q=1 连续 decode 精度失败。**本轮使用独立 CATLASS venv，公共 `catlass.tla` 可导入，已从当前 CATLASS checkout 构建 TLA 编译工具、Python bridge 与 MLIR 绑定。AscendNPU-IR 使用第 13 章的固定子模块及其 LLVM/Triton 版本，并应用该版本要求的 Triton 补丁。[H-run-build] [H-ir-build] [H-ir-patches]

本机为物理 NPU 4（进程内逻辑 0），`Ascend950PR_957b`，28 Cube / 56 Vector；CANN 为 `9.2.0-weekly.20260902.01`，末端 `hivmc-a5` 报 `0.3.0 / Release build`。Host 编译器使用本地安装的 Clang 19.1.1，按用户指定 **16 并发**完成构建；AscendNPU-IR 为 Release + PIC + RTTI，assertions 关闭，TLA extension 为 Debug editable 安装。Python 3.12.3、torch 2.10.0+cpu、torch_npu 2.10.0.post4；具体版本、编译选项和文件 hash 见快照。[H-run-env] [H-run-build]

GCC 13.3.0 的先前尝试在两处模板特化代码上失败。切换 Clang 后撤回了临时兼容性补丁，最终构建没有额外修改 `bishengir` 编译器源码；这些失败日志与最终成功状态一起归档。CATLASS 示例的 kernel、Host 输入构造和 golden 均未修改；外围驱动只传入 CLI 参数并采集结果。[H-run]

以下 FA 均为 `B=1,Hq/Hkv=8/1,D=128`、连续 KV、全零 mask。表中的最大绝对误差是对原驱动参考结果的诊断量，判定仍采用原驱动的完整标准。[H-run-results]

| 用例 | 输入/输出类型 | 尺寸 | block_num | 原始精度判定 | 最大绝对误差 |
| --- | --- | --- | --- | --- | --- |
| MMAD | f16/f16 → f32 | M/N/K=128/128/256 | 2 | **通过** | `1.220703e-04` |
| MMAD 动态尺寸 | f16/f16 → f32 | M/N/K=384/160/272 | 2 | **通过** | `1.831055e-04` |
| MMAD + Vector add | f32 | M/N/K=32/32/32 | 1 | **通过** | `1.525879e-05` |
| 连续 FA，多 query / 无 mask | f16 | Q=117，KV=512 | 8 | **通过** | `2.638102e-04` |
| 连续 FA，多 query / 无 mask | bf16 | Q=117，KV=512 | 8 | **通过** | `2.073646e-03` |
| 连续 decode | f16 | Q=1，KV=512 | 8 | **失败** | `8.130878e-03` |
| 连续 FA，KV 尾块 | f16 | Q=117，KV=513 | 8 | **通过** | `2.626181e-04` |

MMAD/mixed 的这三组均为 f32 输出、K<2048，共享比较函数使用 `abs(error) <= (1/256) * max(1, abs(reference))`。FA 则分别计算 NPU 输出和分块低精度参考相对完整 FP32 attention 的 MARE（最大相对误差）、MERE（平均相对误差）、RMSE，再比较 `error_NPU / max(error_blocked_reference, eps)`；阈值依次为 `2.0/1.2/1.2`，f16 的 eps 为 `2^-7`，bf16 为 `2^-6`。这些不是 A—G 的统一容差，也不构成跨路线精度排名。[H-common-golden] [H-fa]

外围检查还确认全部输出有限；四组主 FA 的输出均已覆盖哨兵值。Q=1 那组尽管 1,024 个输出全部写入，仍因精度失败返回非零，不能把完成 launch 当作通过。[H-run-results]

**decode 失败的复核：**固定原驱动随机种子 42、相同 Q/K/V 与 cache key `4a86ee93a8ed579a`，8 核复跑仍失败，最大绝对误差从首次的 `8.130878e-3` 变为 `6.538039e-2`；仅改成 1 核也失败，误差为 `6.415969e-2`。首次 MARE ratio 为 `2.175862 > 2.0`，8 核复跑为 `16.324497`。随后默认 Q=117 的 FP16 FA 对照复跑通过，误差与首轮一致。当前只确认这个固定配置存在可复现的精度问题、且误差随运行变化，尚未定位到具体 kernel 语句、lowering 或运行时机制。[H-run-diagnostics] [H-run]

**实测怎样支持前面的理论：**两组 MMAD 的 cache key 均为 `02db0bc37b247666`，对应同一 `kernel.o` 路径，为第 7.15 节的动态逻辑范围提供了对象。mixed 与 FA 的实际 `lowered.mlir` 含 `_mix_aic`、`_mix_aiv` 和 Vector helper，为第 4.11/8.12 节的异构核内合作提供了产物证据；这仍不是设备 profiler 的 launch 计数。编译产物、ABI manifest、函数声明及 hash 均已归档。[H-run-artifacts]

脱敏后的完整记录保存在本仓库 `tests/npu_gpu_programming_stacks_comparison.catlass.json`，文中引用固定到该文件的 GitHub commit。[H-run] 本机已有环境的复跑方式如下；脚本内容也包含在快照中，恢复到另一 workspace 时需替换 `<MEGA_ROOT>`、`<USER_HOME>` 并准备对应依赖。[H-run-replay]

```bash
# 从本 session 的 mega workspace 根目录执行
source .npu-stack/catlass/activate.sh 4
cd pto_qcy
export PYPTO_BUILD_JOBS=16
source .claude/skills/testing/load-env.sh
python ../.npu-stack/catlass/run_examples.py
python ../.npu-stack/catlass/run_examples.py --decode-diagnostics
python ../.npu-stack/catlass/run_examples.py --prefill-control
```

主用例脚本和 decode 复核脚本当前预期返回非零，并分别留下 `results.json`、`decode-diagnostics.json`；默认 FA 对照通过，记录在 `prefill-control.json`。如需重建，本机入口为 `PYPTO_BUILD_JOBS=16 bash .npu-stack/catlass/build-ir.sh` 和 `PYPTO_BUILD_JOBS=16 bash .npu-stack/catlass/build-dsl.sh`，从 workspace 根目录运行。快照中的过程总耗时包含编译、Host golden 和检查，**不是 kernel latency**。[H-run-replay]

I 的分析依据固定 CUTLASS checkout 的代码。本机未安装/运行 CuTe 的 CUDA kernel，也没有其性能数据；E 的历史 GPU 实验不代替 I 的实测。H 的连续 attention、I 的分页 MLA 与 A—G 的 PA 用例采用不同契约，分别记录，不构成同口径性能排名。

### 12.2 后续若要测性能，建议统一验收表

| 类别 | 必须记录 |
| --- | --- |
| 算子边界 | PA是否含QK norm/RoPE/KV append；softmax是否含输入输出转换 |
| 数据条件 | dtype、layout、stride、`B/Hq/Hkv/D/S/P/T`、长度分布、page乱序/共享 |
| 动态分类 | 仅 `L` 内容变、`B` 变、`P/T` 变、`D/S` 变，分别测 |
| 冷热成本 | 编译耗时、tiling耗时、cache hit、首次launch、warm执行 |
| 执行数量 | Host API、物理kernel事件、CTRL/SCHE/worker启动、task实例数 |
| 并行 | logical grid/task、physical block/SM/core、wave、尾部不均衡 |
| 内存 | UB/L1/L0/shared/register预算，spill、GM scratch峰值、跨阶段字节数 |
| 核内融合 | 图分组与InCore边界、局部值与外部输出、显式中间GM是否消除、是否进一步消除UB往返；不能只数launch |
| 时序 | kernel内pipeline等待、task ready/dispatch延迟、Host同步、stream依赖 |
| 数值 | 长行极值、page尾块、长度边界、GQA映射、输出与cache canary |
| megaKernel | 分别验收K1—K5，禁止用graph replay API数替代kernel数 |
| 版本 | 源码HEAD、submodule gitlink、实际编译器/库/驱动版本和产物hash |
| H 当前验收入口 | 先复现/定位 Q=1 连续 decode 失败；分别验收独立 softmax、连续 FA 与新增分页/设备异长，不以 MMAD 动态复用代替 FA 动态验证。[H-run-diagnostics] [H-fa] |
| I 当前验收入口 | 准备符合具体 kernel 架构条件的 NVIDIA GPU；区分分页 MLA 与标准 GQA，核对真实 launch/partial/占用，再测试 TS 的数值、安全和进展。[I-mla] [I-gqa] [I-checker] |

当前已具备A5环境及可复跑的PA输入，可以把上述分析落实到生成代码、launch、同步和内存行为的检查。若要比较A2/A3与A5，还需对应硬件的实验；一台NPU不能替另一个硬件桶完成证明。不同路线先统一算子边界及输入分布，再讨论性能差异。

### 12.3 本文的限制与待确认问题

- HEAD只是源码快照；实际加载的工具链、wheel、CANN库和缓存二进制可能不同，不能以邻接目录推断版本绑定。
- example/test存在是接口和算法证据，不是本次执行结果；某些源码docstring滞后，本文优先函数体。
- CANNBot已找到并执行paged实现；AutoFuse已执行分页图表达，但仍含外部算子。二者现在有不同层次的真实对象，仍须分别验证尾页/异长覆盖和原生融合边界。
- 未验证各任务系统在极端形状、并发请求、取消/失败恢复下的行为。
- 仍需实测的关键开放项是：整层K2/K4/K5、动态分布下的性能、C（PyPTO3（Simpler））/PTOAS与F（Triton-Ascend）/AscendNPU-IR替换原型、extern-task ABI适配成本。
- 本文不比较分布式多卡通信，也不将模型库内的业务常量误认成框架限制。
- 不对团队组织、人员能力、代码数量作价值判断；技术替代关系应由同一验收契约的原型结果补充。

- H 的连续 FA 尚未形成本文标准分页 GQA PA 对象；设备实际长度/mask 参数在当前例子未生效，Q=1 精度失败的根因尚未定位。两组 MMAD 复用只证明给定样本，不外推整个动态域。[H-fa] [H-run-diagnostics] [H-run-artifacts]
- I 本机未运行，所选 MLA 有 latent/rope、Q/head/page/split 约束，TS checker 有界；旧 E 的 GPU 数值/trace 不能补成 I 的正确性、无死锁或性能证据。[I-mla] [I-checker]

## 13. 版本快照与关键源码索引

### 13.1 版本快照

#### 13.1.1 历史 GPU 环境版本与实验来源

下表是2026-09-06文档记录的版本与旧目录布局，保留用于追溯；不是当前A5安装版本。

| 对象 | 目录 | Git HEAD |
| --- | --- | --- |
| A（PyPTO2-tensor版）/B（PyPTO2-block版） | `npu/pypto2` | `d8586de9743e34fcab949378f46f6b6a58ce3114` |
| C（PyPTO3（Simpler）） | `npu/pypto` | `b8165168ec16198fa2b26a0f88b1c08415d5668c` |
| C（PyPTO3（Simpler）） runtime | `npu/pypto/runtime` | `77fa0171c24a4e1c323fb29a6a86239df93edb58` |
| C（PyPTO3（Simpler）） 用户算子库 | `npu/pypto-lib` | `57e9d6a9294d9c38edd042f1edb5bdc50b4622bb` |
| D（CANNBot DSL） | `npu/cannbot-dsl_ma` | `9b89be7b7cc54ad3bd1193d7929cdf4a29d7dc8f` |
| F（Triton-Ascend） | `npu/triton-ascend` | `132ebe0d7a7e8476260bc3717f7255e81a31b33c` |
| 独立 NPU IR 后端 | `npu/AscendNPU-IR` | `90037fe3371cb88d50c42cd4e165075cbafe83b1` |
| 独立 PTOAS 后端 | `npu/PTOAS` | `dc15ee5b9e459c025eb4f714f2f892b535d93eb0` |
| G（AutoFuse + Inductor） | `npu/graph-autofusion` | `3b5e6a387e7dc2a0eef7d052a0f7a50c06947db2` |
| G（AutoFuse + Inductor） 接入源码 | `npu/torchair` | `cf2acaee5fe6139617fa7014ea0d09933cf8b35c` |
| F（Triton-Ascend）/G（AutoFuse + Inductor） 后端选择及 F（Triton-Ascend） 接入 | `npu/torch_npu` | `399ff8efe984f66ddedf36ae0381af72d425e7c9` |
| E（PyPTO on GPU） 集成 | `gpu/PyPTO-LOVE-TensorIR` | `0bd6951b00433820d8be7ad8e314222b15969fc8` |
| E（PyPTO on GPU） 编译器 | `gpu/PyPTO-LOVE-TensorIR/.sources/pypto` | `322daf842042ad9f6a70086412c38b2f42d17c58` |
| E（PyPTO on GPU） TensorIR | `gpu/PyPTO-LOVE-TensorIR/.sources/pypto/3rdparty/nvidia/tensor-ir` | `d0b21f3786921f37719a1fe3bcff229a3df7426c` |
| E（PyPTO on GPU） CUDA Tile IR | `gpu/PyPTO-LOVE-TensorIR/.sources/pypto/3rdparty/nvidia/cuda-tile` | `af2417041cc939b87ef56d92cfdcf61737c5457e` |

Triton-Ascend HEAD中 `third_party/ascend/AscendNPU-IR` 的gitlink是 `aea934a66646e837c54fea11e87db54d42eb3221`，与上表独立 `npu/AscendNPU-IR` HEAD不同；CMake又允许外部源码路径。因此本文分别报告集成调用链和独立后端源码能力，不称它们已经在本地编译为同一套二进制。

旧环境记录中，`benchmarks/operators/profile_softmax_task_pipeline_ada_sm89.py` 为未跟踪脚本，trace/summary也属于本地实验材料。当前A5 workspace未迁入这些原始材料；第9章保留其旧文档记录，不把它们改写为上游已发布测试。当前已完成工具链安装和必要兼容性改动，与旧环境“仅源码核对”的工作状态不同。


#### 13.1.2 当前A5工作区及本次源码核对版本

以下版本是本次实际核对的checkout；短SHA仅用于阅读，文件链接使用完整SHA。除CANNBot外，代码来源均指向GitHub或GitCode。

| 对象 | 本轮源码版本 | 固定版本代码入口 |
| --- | --- | --- |
| A/B：cann/pypto | `85c9484e236e` | [tensor版][A-entry] / [block版][B-jit] |
| C：hw-native-sys/pypto | `9f657f37ed20` | [PTO后端][C-backend] / [通用PA][C-pa] |
| C：hw-native-sys/simpler | `4e4d3a4ad1e5` | [A5 scheduler][R-dispatch] / [A5 executor][R-executor] |
| C：hw-native-sys/pypto-lib | `c6bc0bf50d6b` | [Qwen模型程序][C-decode-layer] |
| D：CANNBot DSL本地工作区 | `e87c6ed5ec1a`；以工作区文件内容为准 | [本地PA][D-pa] / [本地测试][D-pa-test] |
| E：PyPTO-LOVE-TensorIR | `b8830d9551b4` | [集成代码][E-compile] / [source lock][E-lock] |
| E：公开bundle内PyPTO fork | `c27629e993a5` | [GitHub中的源码bundle][E-bundle]；[恢复脚本][E-bootstrap] |
| F：triton-lang/triton-ascend | `c747daae7f67` | [编译stages][F-compiler] |
| F：Ascend/triton-ascend-kernels | `f7d13ba7c7b6` | [PA kernel][F-pa] / [测试][F-pa-test] |
| G：cann/graph-autofusion | `3b5e6a387e7d` | [AutoFuse代码生成][G-pyautofuse] |
| G：Ascend/torchair | `cf2acaee5fe6` | [Inductor扩展][G-register] |
| torch_npu源码分析 | `9f15aa301f6c` | [NPU组合调度][F-inductor]；安装wheel版本另见环境记录 |
| 独立AscendNPU-IR源码分析 | `90037fe3371c` | [HIVM pipeline][N-pipeline]；非本轮CANN二进制的源码版本声明 |
| 独立PTOAS源码分析 | `30a83c586cc9` | [driver][P-driver]；本轮实测工具为[v0.57发布包][P-release] |
| H：cann/catlass Python TLA DSL | `337cc89254d4` | [Python DSL][H-dsl] / [FA][H-fa]；不包含全库 C++ 算子覆盖声明 |
| H：固定 AscendNPU-IR 子模块 | `a07821269ede` | [固定版本构建说明][H-ir-build] / [子模块配置][H-ir-submodules]；使用 Clang 构建，与独立 NPU-IR checkout 分开；GCC 曾在 [模板特化][H-ir-matmul] 处失败，不计入可用工具链 |
| H：配套 LLVM / Triton | `9c0841dc3fa2` / `c3c476f357f1` | [LLVM][H-llvm] / [Triton][H-triton]，再应用固定 NPU-IR 的 [Triton 补丁][H-ir-patches] |
| I：NVIDIA/cutlass CuTe DSL | `59e3a3338d51` | [DSL 入口][I-dsl] / [CuTe 示例][I-examples]；源码核对，未 GPU 执行 |
| I：配套 wheel 要求 | `nvidia-cutlass-dsl==4.8.0.dev0` | [仓库 requirements][I-requirements]；是源码的安装要求，不是本机安装成功记录 |

GPU bundle的`origin_url`标识上游来源，不保证fork commit可在上游GitHub文件页访问。因此本文引用已公开的bundle与source lock，底层文件路径在第9节给出；已按锁文件恢复并核对bundle SHA256及head tree。旧版的`.sources/pypto`本地路径不再当作可点击来源。[E-lock] [E-bootstrap] [E-bundle]

当前Triton的构建过程会应用仓库内Ascend补丁：JIT源码分析需同时看基线[JIT][F-jit]、[构建入口][F-build]和[Ascend补丁][F-patch]；不能把构建后的文件当作基线Git blob逐字一致。PyPTO3的实测PTOAS为v0.57发布包，独立PTOAS源码分析版本单列；torch_npu源码分析版本也不替代实际安装wheel版本。[RUN-env]


#### 13.1.3 GPU bundle、历史记录及源码定位

当前公开的[PyPTO bundle][E-bundle]与[TensorIR bundle][E-tensor-bundle]分别包含 fork head `c27629e993a52b47d41fb898c749279dce44221b` 和 `db41d0733eb73971ee03a74faca81d1af6e6aef7`。`origin_url`是上游来源，不保证这些fork commit在上游仓库有独立文件页。以下旧索引现在链接到实际发布的bundle，文件路径用于恢复后定位，原摘录的行号仅保留作旧版本导航；当前以符号和函数体为准。

| 索引 | 公开源码包 | 包内文件 / 关键内容 |
| --- | --- | --- |
| [E2（PyPTO on GPU）]、[E15（PyPTO on GPU）] | PyPTO | `src/codegen/nvidia/tensor_ir_codegen.cpp`：模式与静态shape/单函数检查 |
| [E3（PyPTO on GPU）] | PyPTO | `src/compiler/nvidia_producer_bridge.cpp`：typed module构造及tile编译 |
| [E4（PyPTO on GPU）] | PyPTO | `src/runtime/nvidia/executable.cpp`：grid及launch ABI |
| [E5（PyPTO on GPU）] | PyPTO | `src/runtime/nvidia/driver_api.cpp`：`cuLaunchKernelEx` |
| [E6（PyPTO on GPU）] | PyPTO | `docs/en/dev/backend/02-nvidia-executable.md`：runtime约定 |
| [E13（PyPTO on GPU）]、[E14（PyPTO on GPU）] | TensorIR | `include/tensor_ir/Dialect/TensorDialect.td`、`README.md` |

[恢复脚本][E-bootstrap]及[source lock][E-lock]提供bundle路径、SHA256、shallow boundary和head tree。原GPU profiler及trace未随当前checkout提供，因此[E7（PyPTO on GPU）]—[E9（PyPTO on GPU）]链接到已发布的历史文档记录；第2、9章保留相应内容与限制，不用当前bundle冒充当时测量的二进制来源。PyPTO2-tensor版的新增完整golden驱动则保存在[A5证据快照][RUN-A-driver]。

### 13.2 编译与runtime主链定位

| 对象 | 关键位置与符号 |
| --- | --- |
| A（PyPTO2-tensor版）前端 | [A1（PyPTO2-tensor版）] `jit(new_ir=True)`；[A2（PyPTO2-tensor版）] `compile_new/compile`；[A3（PyPTO2-tensor版）] PIL pipeline |
| A（PyPTO2-tensor版）编译/执行 | [A4（PyPTO2-tensor版）]/[A4b（PyPTO2-tensor版）] finalize与compile queue；[A5（PyPTO2-tensor版）] CCE codegen；[A6（PyPTO2-tensor版）] DeviceLauncher |
| A（PyPTO2-tensor版）任务 | [A9（PyPTO2-tensor版）] `Dispatch/SendTask/SetReadyQueue`；[A10（PyPTO2-tensor版）] KernelModule/CheckArgs |
| A（PyPTO2-tensor版）动态/tile/共享IR | [A7（PyPTO2-tensor版）]/[A8（PyPTO2-tensor版）]/[A8b（PyPTO2-tensor版）] dynamic/tile设置；[A11（PyPTO2-tensor版）] `pypto_impl.ir` |
| B（PyPTO2-block版）编译/launch | [B1（PyPTO2-block版）] `generate_single`；[B2（PyPTO2-block版）] .so调用/Host launch；[B3（PyPTO2-block版）] 方括号launch与默认block |
| B（PyPTO2-block版）共享/动态 | [B5（PyPTO2-block版）]/[B5b（PyPTO2-block版）]/[B5c（PyPTO2-block版）] bootstrap/IR/loader；[B6（PyPTO2-block版）] shape policy；[B12（PyPTO2-block版）] 缓存测试 |
| B（PyPTO2-block版）tiling/核内索引 | [B4（PyPTO2-block版）]/[B7（PyPTO2-block版）]/[B8（PyPTO2-block版）]/[B9（PyPTO2-block版）] 打包/JIT/AOT；[B10（PyPTO2-block版）]/[B11（PyPTO2-block版）]/[B13（PyPTO2-block版）] 索引/尾块/多核 |
| C（PyPTO3（Simpler））后端 | [C1（PyPTO3（Simpler））] `_run_ptoas`、kernel wrapper；[C2（PyPTO3（Simpler））] PTO动态参数；[C3（PyPTO3（Simpler））] Orchestration submit |
| C（PyPTO3（Simpler））runtime | [C4（PyPTO3（Simpler））]/[C4b（PyPTO3（Simpler））] SimplerWorker；[C5（PyPTO3（Simpler））] runtime目录；[C6（PyPTO3（Simpler））] spmd_submit；[C14（PyPTO3（Simpler））] logical ID |
| C（PyPTO3（Simpler））动态/extern | [C7（PyPTO3（Simpler））]/[C8（PyPTO3（Simpler））]/[C9（PyPTO3（Simpler））]/[C10（PyPTO3（Simpler））] DynVar/cache/PA/TileType约束；[C13（PyPTO3（Simpler））] extern ABI |
| D（CANNBot DSL）主链 | [D1（CANNBot DSL）] 调用矩阵；[D2（CANNBot DSL）] kernel materialize；[D3（CANNBot DSL）]/[D3b（CANNBot DSL）] MLIR；[D4（CANNBot DSL）] AscendC translation |
| D（CANNBot DSL）动态/Host | [D5（CANNBot DSL）] AICPU kernel；[D6（CANNBot DSL）]/[D7（CANNBot DSL）] 动态契约；[D8（CANNBot DSL）] datastruct；[D9（CANNBot DSL）] AOT/cache |
| E（PyPTO on GPU）主链 | [E1（PyPTO on GPU）] compile_graph/launch_graph；[E2（PyPTO on GPU）] emitter；[E3（PyPTO on GPU）] typed TensorIR bridge |
| E（PyPTO on GPU）launch/实验 | [E4（PyPTO on GPU）] NvidiaExecutable；[E5（PyPTO on GPU）] Driver API；[E6（PyPTO on GPU）] Tile ABI；[E7（PyPTO on GPU）]/[E8（PyPTO on GPU）]/[E9（PyPTO on GPU）] 实验三件套 |
| F（Triton-Ascend）编译 | [F1（Triton-Ascend）] stages；[F2（Triton-Ascend）] TTIR适配；[F4（Triton-Ascend）] CMake依赖；[F12（Triton-Ascend）] 工具查找 |
| F（Triton-Ascend）后端/launch | [F3（Triton-Ascend）] CANN launch；[F8（Triton-Ascend）] HIVM pipeline；[F9（Triton-Ascend）] AutoBlockify；[F11（Triton-Ascend）] RegBase LLVM lowering |
| F（Triton-Ascend）框架入口 | [F7（Triton-Ascend）] torch_npu backend loader；[F10（Triton-Ascend）] Triton/CATLASS选择 |
| G（AutoFuse + Inductor）入口/IR | [G1（AutoFuse + Inductor）] 安装/启用；[G2（AutoFuse + Inductor）] Inductor→ASCGraph；[G3（AutoFuse + Inductor）] Autofuser/三份代码输出 |
| G（AutoFuse + Inductor）tiling/内存/launch | [G4（AutoFuse + Inductor）] compile_ascendc；[G5（AutoFuse + Inductor）] tiling生成；[G6（AutoFuse + Inductor）] buffer lifecycle；[G7（AutoFuse + Inductor）] wrapper；[G8（AutoFuse + Inductor）] compile_adapter |
| PTOAS | [P1（PTOAS）] emitc/vpto/arch选项；[P2（PTOAS）] memory规划；[P3（PTOAS）] modern规划实现 |
| H（CATLASS DSL）前端 / 编译 | [H-dsl] `CatlassBaseDSL/TlaDSL`；[H-compile] `CompileCallable`；[H-passes] `buildTlaPipeline` |
| H（CATLASS DSL）内存 / runtime | [H-scratch] 静态地址空间偏移；[H-execution] `_compile_kernel`；[H-runtime] `launch_kernel` |
| I（CuTe DSL）编译 / runtime | [I-dsl] `CutlassBaseDSL` / pipeline；[I-compiler] compiler；[I-executor] cubin/JIT；[I-runtime] CUDA load/launch |
| I（CuTe DSL）工作分配 / 资源 | [I-static] 静态 persistent；[I-dynamic] CLC；[I-task] / [I-schedule] / [I-task-manager] TS；[I-ts-memory] 内存；[I-checker] 协议检查 |

### 13.3 用户完整示例入口

| 对象 | Softmax | Attention / 动态 / 模型 |
| --- | --- | --- |
| A（PyPTO2-tensor版） | [A12（PyPTO2-tensor版）] tensor版完整例子 | [A13（PyPTO2-tensor版）] 动态paged attention所在的多阶段程序 |
| B（PyPTO2-block版） | [B14（PyPTO2-block版）] 双动态、multicore、TileGroup完整测试 | [B15（PyPTO2-block版）] paged prefill；[B16（PyPTO2-block版）] 动态TND actual_seq |
| C（PyPTO3（Simpler）） | [C15（PyPTO3（Simpler））] pypto-lib示例 | [C16（PyPTO3（Simpler））] native decode；[C17（PyPTO3（Simpler））] 动态驱动；[C18（PyPTO3（Simpler））] 讲解；[C19（PyPTO3（Simpler））] AIV tiling task；[C20（PyPTO3（Simpler））] decode_fwd |
| C（PyPTO3（Simpler））旧分task例子 | 第2章及[C9（PyPTO3（Simpler））] | [C11（PyPTO3（Simpler））]/[C12（PyPTO3（Simpler））] Simpler Host Python tiler + SPMD task |
| D（CANNBot DSL） | [D10（CANNBot DSL）] Buffer版；[D11（CANNBot DSL）] NPU golden入口 | [D12（CANNBot DSL）] PA设计边界；[D13（CANNBot DSL）] 完整FlashAttention蓝本；[D7（CANNBot DSL）] 动态AOT |
| E（PyPTO on GPU） | [E8（PyPTO on GPU）] 五阶段完整实验 | [E10（PyPTO on GPU）] paged decode API；[E11（PyPTO on GPU）] 完整benchmark用法 |
| F（Triton-Ascend） | [F5（Triton-Ascend）] kernel与Host教程 | [F6（Triton-Ascend）] 原生paged unified attention、decode参数和golden |
| G（AutoFuse + Inductor） | 第2.3节客户程序；[G1（AutoFuse + Inductor）]入口文档 | 第2.4节客户表达；[G9（AutoFuse + Inductor）] 仅作外部FA/ACLGraph集成证据 |
| H（CATLASS DSL） | [H-fa] FA 内 online-softmax；没有据此宣称独立长行 softmax 已验证 | [H-fa] 连续 FA；[H-fa-tiling] Host tiler；[H-mmad-example] 动态 GM；[H-mixed] CV 交接；[H-streamk] 工作切分/归并 |
| I（CuTe DSL） | [I-softmax] 八种实现及 launcher，源码证据 | [I-mla] / [I-mla-fp8] 分页 MLA；[I-gqa] 连续 GQA 两阶段；[I-ts-tutorial] persistent/domain/warp 调度教程 |

链接行号对应上述HEAD；源码以后移动时优先按符号定位。相对路径便于本文随pto工作区一起阅读。

[A1（PyPTO2-tensor版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/python/pypto/frontend/parser/entry.py#L1188
[A2（PyPTO2-tensor版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/python/pypto/frontend/parser/entry.py#L451
[A3（PyPTO2-tensor版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/python/pypto/pil/compile_pipeline.py#L29
[A4（PyPTO2-tensor版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/framework/src/interface/tensor/ir.cpp#L207
[A4b（PyPTO2-tensor版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/framework/src/interface/tensor/ir_finalize.cpp#L39
[A5（PyPTO2-tensor版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/framework/src/codegen/codegen.cpp#L21
[A6（PyPTO2-tensor版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/framework/src/machine/runtime/launcher/device_launcher.cpp#L482
[A7（PyPTO2-tensor版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/python/pypto/frontend/__init__.py#L72
[A8（PyPTO2-tensor版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/python/pypto/_controller.py#L46
[A8b（PyPTO2-tensor版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/docs/zh/guide/programming_guide/tensor/development/tiling.md#L9
[A9（PyPTO2-tensor版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/framework/src/machine/device/dynamic/aicore_manager.h#L994
[A10（PyPTO2-tensor版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/python/src/bindings/runtime.cpp#L430
[A11（PyPTO2-tensor版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/python/pypto/ir.py#L135
[A12（PyPTO2-tensor版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/examples/02_intermediate/operators/softmax/softmax.py#L76
[A13（PyPTO2-tensor版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/python/tests/st/test_ctrl_cpu_perf.py#L73
[B1（PyPTO2-block版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/python/pypto_pro/runtime/jit.py#L951
[B2（PyPTO2-block版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/python/pypto_pro/runtime/jit.py#L601
[B3（PyPTO2-block版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/python/pypto_pro/runtime/jit.py#L1595
[B4（PyPTO2-block版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/python/pypto_pro/runtime/jit.py#L467
[B5（PyPTO2-block版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/python/pypto_pro/_bootstrap.py#L14
[B5b（PyPTO2-block版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/python/pypto_pro/ir/__init__.py#L36
[B5c（PyPTO2-block版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/python/pypto/_loader.py#L60
[B6（PyPTO2-block版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/python/pypto_pro/runtime/shape_policy.py#L239
[B7（PyPTO2-block版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/docs/zh/guide/programming_guide/pro/development/tile_based_python_programming/TilingData.md#L22
[B8（PyPTO2-block版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/python/pypto_pro/runtime/jit.py#L1395
[B9（PyPTO2-block版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/docs/zh/guide/programming_guide/pro/development/compilation_and_execution/offline_binary_compilation.md#L139
[B10（PyPTO2-block版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/framework/src/interface/pypto_pro/backend/backend_cce_ops.cpp#L834
[B11（PyPTO2-block版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/docs/zh/guide/programming_guide/pro/development/tile_based_python_programming/tail_block_handling.md#L25
[B12（PyPTO2-block版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/python/tests/ut/pypto_pro/runtime/test_shape_policy_codegen.py#L109
[B13（PyPTO2-block版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/docs/zh/guide/programming_guide/pro/development/tile_based_python_programming/multi_core_partitioning_and_Tiling.md#L74
[B14（PyPTO2-block版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/python/tests/st/pypto_pro/frontend/tile_vector/test_softmax.py#L77
[B15（PyPTO2-block版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/python/tests/st/pypto_pro/frontend/fa/test_flex_attention_prefill.py#L14
[B16（PyPTO2-block版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/python/tests/st/pypto_pro/frontend/fa/test_fa_tnd_dn.py#L364
[C1（PyPTO3（Simpler））]: https://github.com/hw-native-sys/pypto/blob/9f657f37ed20ce148b46fb7229c267a152a0644e/python/pypto/backend/pto_backend.py#L14
[C2（PyPTO3（Simpler））]: https://github.com/hw-native-sys/pypto/blob/9f657f37ed20ce148b46fb7229c267a152a0644e/src/codegen/pto/pto_codegen.cpp#L1078
[C3（PyPTO3（Simpler））]: https://github.com/hw-native-sys/pypto/blob/9f657f37ed20ce148b46fb7229c267a152a0644e/src/codegen/orchestration/orchestration_codegen.cpp#L198
[C4（PyPTO3（Simpler））]: https://github.com/hw-native-sys/pypto/blob/9f657f37ed20ce148b46fb7229c267a152a0644e/python/pypto/runtime/task_interface.py#L17
[C4b（PyPTO3（Simpler））]: https://github.com/hw-native-sys/pypto/blob/9f657f37ed20ce148b46fb7229c267a152a0644e/python/pypto/runtime/worker.py#L220
[C5（PyPTO3（Simpler））]: https://github.com/hw-native-sys/simpler/blob/4e4d3a4ad1e54c1db3d50e72decc025a9075bfa0/src/a2a3/runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler.cpp#L1
[C6（PyPTO3（Simpler））]: https://github.com/hw-native-sys/pypto/blob/9f657f37ed20ce148b46fb7229c267a152a0644e/python/pypto/language/scope.py#L210
[C7（PyPTO3（Simpler））]: https://github.com/hw-native-sys/pypto/blob/9f657f37ed20ce148b46fb7229c267a152a0644e/python/pypto/language/typing/dynamic.py#L20
[C8（PyPTO3（Simpler））]: https://github.com/hw-native-sys/pypto/blob/9f657f37ed20ce148b46fb7229c267a152a0644e/python/pypto/jit/cache.py#L225
[C9（PyPTO3（Simpler））]: https://github.com/hw-native-sys/pypto/blob/9f657f37ed20ce148b46fb7229c267a152a0644e/examples/models/06_paged_attention_dynamic.py#L262
[C10（PyPTO3（Simpler））]: https://github.com/hw-native-sys/pypto/blob/9f657f37ed20ce148b46fb7229c267a152a0644e/src/ir/transforms/init_memref.cpp#L576
[C11（PyPTO3（Simpler））]: https://github.com/hw-native-sys/simpler/blob/4e4d3a4ad1e54c1db3d50e72decc025a9075bfa0/tests/st/a2a3/tensormap_and_ringbuffer/spmd_paged_attention_highperf/kernels/pa_tiling.py#L255
[C12（PyPTO3（Simpler））]: https://github.com/hw-native-sys/simpler/blob/4e4d3a4ad1e54c1db3d50e72decc025a9075bfa0/tests/st/a2a3/tensormap_and_ringbuffer/spmd_paged_attention_highperf/kernels/orchestration/paged_attention_highperf_orch.cpp#L63
[C13（PyPTO3（Simpler））]: https://github.com/hw-native-sys/pypto/blob/9f657f37ed20ce148b46fb7229c267a152a0644e/docs/en/dev/language/04-external-kernels.md#L15
[C14（PyPTO3（Simpler））]: https://github.com/hw-native-sys/simpler/blob/4e4d3a4ad1e54c1db3d50e72decc025a9075bfa0/src/a2a3/runtime/tensormap_and_ringbuffer/common/intrinsic.h#L53
[C15（PyPTO3（Simpler））]: https://github.com/hw-native-sys/pypto-lib/blob/c6bc0bf50d6b1b58bde4698cdd4d8ff0f5d33b80/examples/intermediate/softmax.py#L24
[C16（PyPTO3（Simpler））]: https://github.com/hw-native-sys/pypto-lib/blob/c6bc0bf50d6b1b58bde4698cdd4d8ff0f5d33b80/models/qwen3_14b/paged_attention_pypto.py#L67
[C17（PyPTO3（Simpler））]: https://github.com/hw-native-sys/pypto-lib/blob/c6bc0bf50d6b1b58bde4698cdd4d8ff0f5d33b80/models/qwen3_14b/test_paged_attention_pypto.py#L161
[C18（PyPTO3（Simpler））]: https://github.com/hw-native-sys/pypto-lib/blob/c6bc0bf50d6b1b58bde4698cdd4d8ff0f5d33b80/docs/models/qwen3_14b/paged_attention_pypto.md#L1
[C19（PyPTO3（Simpler））]: https://github.com/hw-native-sys/pypto-lib/blob/c6bc0bf50d6b1b58bde4698cdd4d8ff0f5d33b80/models/qwen3_14b/paged_attention_cce.py#L172
[C20（PyPTO3（Simpler））]: https://github.com/hw-native-sys/pypto-lib/blob/c6bc0bf50d6b1b58bde4698cdd4d8ff0f5d33b80/models/qwen3_14b/decode_fwd.py#L287
[D1（CANNBot DSL）]: ../../cannbot_dsl/cannbot-dsl/python/cannbotdsl/lang/jit.py#L11
[D2（CANNBot DSL）]: ../../cannbot_dsl/cannbot-dsl/python/cannbotdsl/core/compiler/kernel_materialize.py#L155
[D3（CANNBot DSL）]: ../../cannbot_dsl/cannbot-dsl/python/cannbotdsl/core/compiler/builder.py#L419
[D3b（CANNBot DSL）]: ../../cannbot_dsl/cannbot-dsl/python/cannbotdsl/core/compiler/passes.py#L23
[D4（CANNBot DSL）]: ../../cannbot_dsl/cannbot-dsl/lib/CANNIR/Translate/TranslateToAscendC.cpp#L2854
[D5（CANNBot DSL）]: ../../cannbot_dsl/cannbot-dsl/python/cannbotdsl/aicpu/kernel.py#L9
[D6（CANNBot DSL）]: ../../cannbot_dsl/cannbot-dsl/docs/design/dynamic_shape_design.md#L89
[D7（CANNBot DSL）]: ../../cannbot_dsl/cannbot-dsl/test/cannbotdsl/test_aot_dynamic_shape.py#L35
[D8（CANNBot DSL）]: ../../cannbot_dsl/cannbot-dsl/docs/design/datastruct.md#L3
[D9（CANNBot DSL）]: ../../cannbot_dsl/cannbot-dsl/docs/design/jit_aot_end_to_end.md#L102
[D10（CANNBot DSL）]: ../../cannbot_dsl/cannbot-dsl/test/cannbotdsl/test_softmax_buffer.py#L25
[D11（CANNBot DSL）]: ../../cannbot_dsl/cannbot-dsl/test/cannbotdsl/test_softmax_npu.py#L39
[D12（CANNBot DSL）]: ../../cannbot_dsl/cannbot-dsl/docs/paged_attention_design.md#L1
[D13（CANNBot DSL）]: ../../cannbot_dsl/cannbot-dsl/examples/flash_atten/test_flash_attention_channel_first_swp_npu.py#L1
[E1（PyPTO on GPU）]: https://github.com/zhaosiying12138/PyPTO-LOVE-TensorIR/blob/b8830d9551b4f342f5511b5e1272cac5982e1728/packages/pypto-kernels/src/pypto_kernels/_boot.py#L365
[E2（PyPTO on GPU）]: https://github.com/zhaosiying12138/PyPTO-LOVE-TensorIR/blob/b8830d9551b4f342f5511b5e1272cac5982e1728/vendor/git/pypto.bundle
[E3（PyPTO on GPU）]: https://github.com/zhaosiying12138/PyPTO-LOVE-TensorIR/blob/b8830d9551b4f342f5511b5e1272cac5982e1728/vendor/git/pypto.bundle
[E4（PyPTO on GPU）]: https://github.com/zhaosiying12138/PyPTO-LOVE-TensorIR/blob/b8830d9551b4f342f5511b5e1272cac5982e1728/vendor/git/pypto.bundle
[E5（PyPTO on GPU）]: https://github.com/zhaosiying12138/PyPTO-LOVE-TensorIR/blob/b8830d9551b4f342f5511b5e1272cac5982e1728/vendor/git/pypto.bundle
[E6（PyPTO on GPU）]: https://github.com/zhaosiying12138/PyPTO-LOVE-TensorIR/blob/b8830d9551b4f342f5511b5e1272cac5982e1728/vendor/git/pypto.bundle
[E7（PyPTO on GPU）]: https://github.com/nalinaly/pypto/blob/098b896ffb84ffebd0dac87af62835b7e33195a9/tests/npu_gpu_programming_stacks_comparison.md#L3960
[E8（PyPTO on GPU）]: https://github.com/nalinaly/pypto/blob/098b896ffb84ffebd0dac87af62835b7e33195a9/tests/npu_gpu_programming_stacks_comparison.md#L433
[E9（PyPTO on GPU）]: https://github.com/nalinaly/pypto/blob/098b896ffb84ffebd0dac87af62835b7e33195a9/tests/npu_gpu_programming_stacks_comparison.md#L3960
[E10（PyPTO on GPU）]: https://github.com/zhaosiying12138/PyPTO-LOVE-TensorIR/blob/b8830d9551b4f342f5511b5e1272cac5982e1728/packages/pypto-kernels/src/pypto_kernels/attention.py#L1714
[E11（PyPTO on GPU）]: https://github.com/zhaosiying12138/PyPTO-LOVE-TensorIR/blob/b8830d9551b4f342f5511b5e1272cac5982e1728/packages/pypto-kernels/benchmarks/paged_attention_sm120.py#L38
[F1（Triton-Ascend）]: https://github.com/triton-lang/triton-ascend/blob/c747daae7f67fb7acd1013bf9793505aec1dc9e2/third_party/ascend/backend/compiler.py#L1385
[F2（Triton-Ascend）]: https://github.com/triton-lang/triton-ascend/blob/c747daae7f67fb7acd1013bf9793505aec1dc9e2/third_party/ascend/backend/compiler.py#L194
[F3（Triton-Ascend）]: https://github.com/triton-lang/triton-ascend/blob/c747daae7f67fb7acd1013bf9793505aec1dc9e2/third_party/ascend/backend/driver.py#L1156
[F4（Triton-Ascend）]: https://github.com/triton-lang/triton-ascend/blob/c747daae7f67fb7acd1013bf9793505aec1dc9e2/third_party/ascend/CMakeLists.txt#L7
[F5（Triton-Ascend）]: https://github.com/triton-lang/triton-ascend/blob/c747daae7f67fb7acd1013bf9793505aec1dc9e2/third_party/ascend/tutorials/02-fused-softmax.py#L59
[F6（Triton-Ascend）]: https://github.com/triton-lang/triton-ascend/blob/c747daae7f67fb7acd1013bf9793505aec1dc9e2/third_party/ascend/unittest/pytest_ut/test_triton_unified_attention.py#L37
[F7（Triton-Ascend）]: https://gitcode.com/Ascend/pytorch/blob/9f15aa301f6c69f7632b57e1ff198a84c27c3f79/torch_npu/_inductor/__init__.py#L125
[F8（Triton-Ascend）]: https://gitcode.com/Ascend/AscendNPU-IR/blob/90037fe3371cb88d50c42cd4e165075cbafe83b1/bishengir/lib/Dialect/HIVM/Pipelines/HIVMPipelines.cpp#L250
[F9（Triton-Ascend）]: https://gitcode.com/Ascend/AscendNPU-IR/blob/90037fe3371cb88d50c42cd4e165075cbafe83b1/bishengir/lib/Dialect/HIVM/Transforms/AutoBlockifyParallelLoop.cpp#L54
[F10（Triton-Ascend）]: https://gitcode.com/Ascend/pytorch/blob/9f15aa301f6c69f7632b57e1ff198a84c27c3f79/torch_npu/_inductor/codegen/npu_combined_scheduling.py#L30
[F11（Triton-Ascend）]: https://gitcode.com/Ascend/AscendNPU-IR/blob/90037fe3371cb88d50c42cd4e165075cbafe83b1/bishengir/lib/Tools/bishengir-compile/regbase/PassPipeline.cpp#L239
[F12（Triton-Ascend）]: https://github.com/triton-lang/triton-ascend/blob/c747daae7f67fb7acd1013bf9793505aec1dc9e2/third_party/ascend/backend/utils.py#L470
[G1（AutoFuse + Inductor）]: https://gitcode.com/Ascend/torchair/blob/cf2acaee5fe6139617fa7014ea0d09933cf8b35c/experimental/_inductor_npu_ext/README.md#L1
[G2（AutoFuse + Inductor）]: https://gitcode.com/Ascend/torchair/blob/cf2acaee5fe6139617fa7014ea0d09933cf8b35c/experimental/_inductor_npu_ext/python/inductor_npu_ext/npu.py#L2243
[G3（AutoFuse + Inductor）]: https://gitcode.com/cann/graph-autofusion/blob/3b5e6a387e7dc2a0eef7d052a0f7a50c06947db2/autofuse/compiler/py_module/pyautofuse.cpp#L188
[G4（AutoFuse + Inductor）]: https://gitcode.com/Ascend/torchair/blob/cf2acaee5fe6139617fa7014ea0d09933cf8b35c/experimental/_inductor_npu_ext/python/inductor_npu_ext/compiler/_compiler.py#L739
[G5（AutoFuse + Inductor）]: https://gitcode.com/cann/graph-autofusion/blob/3b5e6a387e7dc2a0eef7d052a0f7a50c06947db2/autofuse/codegen/codegen_tiling.cpp#L508
[G6（AutoFuse + Inductor）]: https://gitcode.com/cann/graph-autofusion/blob/3b5e6a387e7dc2a0eef7d052a0f7a50c06947db2/autofuse/optimize/buffer_allocate/buf_que_allocator.cpp#L87
[G7（AutoFuse + Inductor）]: https://gitcode.com/Ascend/torchair/blob/cf2acaee5fe6139617fa7014ea0d09933cf8b35c/experimental/_inductor_npu_ext/python/inductor_npu_ext/codegen/_asc_codegen.py#L610
[G8（AutoFuse + Inductor）]: https://gitcode.com/cann/graph-autofusion/blob/3b5e6a387e7dc2a0eef7d052a0f7a50c06947db2/autofuse/compiler/python/compile_adapter.py#L577
[G9（AutoFuse + Inductor）]: https://gitcode.com/Ascend/torchair/blob/cf2acaee5fe6139617fa7014ea0d09933cf8b35c/experimental/_inductor_npu_ext/tests/smoke/test_fa3_fwd_reduce_overhead.py#L24
[P1（PTOAS）]: https://github.com/hw-native-sys/PTOAS/blob/30a83c586cc9af801a9db0eb9137820b4cd8554c/tools/ptoas/ptoas.cpp#L334
[P2（PTOAS）]: https://github.com/hw-native-sys/PTOAS/blob/30a83c586cc9af801a9db0eb9137820b4cd8554c/lib/PTO/Transforms/PTOPlanMemory.cpp#L1
[P3（PTOAS）]: https://github.com/hw-native-sys/PTOAS/blob/30a83c586cc9af801a9db0eb9137820b4cd8554c/lib/PTO/Transforms/PTOPlanMemoryModern.cpp#L1

[D14（CANNBot DSL）]: ../../cannbot_dsl/cannbot-dsl/examples/flash_attn_noquant/flash_attn_noquant.py#L56
[D15（CANNBot DSL）]: ../../cannbot_dsl/cannbot-dsl/test/cannir/cannir-opt/convert_to_ascvec_softmax.mlir#L22
[E12（PyPTO on GPU）]: https://github.com/zhaosiying12138/PyPTO-LOVE-TensorIR/blob/b8830d9551b4f342f5511b5e1272cac5982e1728/packages/pypto-kernels/src/pypto_kernels/attention.py#L183
[E13（PyPTO on GPU）]: https://github.com/zhaosiying12138/PyPTO-LOVE-TensorIR/blob/b8830d9551b4f342f5511b5e1272cac5982e1728/vendor/git/tensor-ir.bundle
[E14（PyPTO on GPU）]: https://github.com/zhaosiying12138/PyPTO-LOVE-TensorIR/blob/b8830d9551b4f342f5511b5e1272cac5982e1728/vendor/git/tensor-ir.bundle
[F13（Triton-Ascend）]: https://gitcode.com/Ascend/AscendNPU-IR/blob/90037fe3371cb88d50c42cd4e165075cbafe83b1/bishengir/test/Dialect/HIVM/plan-memory.mlir#L1
[G10（AutoFuse + Inductor）]: https://gitcode.com/Ascend/torchair/blob/cf2acaee5fe6139617fa7014ea0d09933cf8b35c/experimental/_inductor_npu_ext/tests/smoke/inductor_npu_ext_test.py#L515
[P4（PTOAS）]: https://github.com/hw-native-sys/PTOAS/blob/30a83c586cc9af801a9db0eb9137820b4cd8554c/test/samples/PyPTOIRParser/paged_attention_example_kernel_softmax_prepare.pto#L1

[A14（PyPTO2-tensor版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/framework/src/interface/configs/tile_fwk_config_schema.json#L199
[A15（PyPTO2-tensor版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/framework/src/interface/function/function.cpp#L2120
[A16（PyPTO2-tensor版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/framework/src/passes/tile_graph_pass/graph_partition/supernode_graph_builder.cpp#L1092
[A17（PyPTO2-tensor版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/python/pypto/op/__init__.py#L13
[A18（PyPTO2-tensor版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/python/pypto/op/conv.py#L191
[A19（PyPTO2-tensor版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/python/pypto/op/matmul.py#L147
[B17（PyPTO2-block版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/python/pypto_pro/language/__init__.py#L12
[B18（PyPTO2-block版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/python/pypto_pro/language/_api.py#L1396
[B19（PyPTO2-block版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/python/pypto_pro/language/parser/_buffer_parser.py#L10
[B20（PyPTO2-block版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/python/pypto_pro/language/parser/_ast_parser.py#L89
[B21（PyPTO2-block版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/framework/src/interface/pypto_pro/backend/backend_cce_vf_ops.cpp#L1569
[C21（PyPTO3（Simpler））]: https://github.com/hw-native-sys/pypto/blob/9f657f37ed20ce148b46fb7229c267a152a0644e/python/pypto/language/op/__init__.py#L12
[C22（PyPTO3（Simpler））]: https://github.com/hw-native-sys/pypto/blob/9f657f37ed20ce148b46fb7229c267a152a0644e/src/backend/910B/backend_910b_ops.cpp#L29
[C23（PyPTO3（Simpler））]: https://github.com/hw-native-sys/pypto-lib/blob/c6bc0bf50d6b1b58bde4698cdd4d8ff0f5d33b80/models/qwen3_14b/paged_attention_pypto.py#L260
[D16（CANNBot DSL）]: ../../cannbot_dsl/cannbot-dsl/python/cannbotdsl/channel.py#L10
[D17（CANNBot DSL）]: ../../cannbot_dsl/cannbot-dsl/python/cannbotdsl/core/frontend/compiler.py#L44
[D18（CANNBot DSL）]: ../../cannbot_dsl/cannbot-dsl/python/cannbotdsl/ops/__init__.py#L9
[D19（CANNBot DSL）]: ../../cannbot_dsl/cannbot-dsl/python/cannbotdsl/ops/reg/store.py#L155
[E15（PyPTO on GPU）]: https://github.com/zhaosiying12138/PyPTO-LOVE-TensorIR/blob/b8830d9551b4f342f5511b5e1272cac5982e1728/vendor/git/pypto.bundle
[F14（Triton-Ascend）]: https://github.com/triton-lang/triton-ascend/blob/c747daae7f67fb7acd1013bf9793505aec1dc9e2/third_party/ascend/backend/__init__.py#L80
[F15（Triton-Ascend）]: https://github.com/triton-lang/triton-ascend/blob/c747daae7f67fb7acd1013bf9793505aec1dc9e2/third_party/ascend/language/cann/extension/semantic.py#L101
[F16（Triton-Ascend）]: https://github.com/triton-lang/triton-ascend/blob/c747daae7f67fb7acd1013bf9793505aec1dc9e2/third_party/ascend/language/cann/extension/__init__.py#L16
[G11（AutoFuse + Inductor）]: https://gitcode.com/Ascend/torchair/blob/cf2acaee5fe6139617fa7014ea0d09933cf8b35c/experimental/_inductor_npu_ext/python/inductor_npu_ext/asc_ops.py#L49
[G12（AutoFuse + Inductor）]: https://gitcode.com/Ascend/torchair/blob/cf2acaee5fe6139617fa7014ea0d09933cf8b35c/experimental/_inductor_npu_ext/python/inductor_npu_ext/npu.py#L3343

[A20（PyPTO2-tensor版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/examples/02_intermediate/controlflow/others/dynamic.py#L107
[B22（PyPTO2-block版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/python/pypto_pro/runtime/platform.py#L157
[B23（PyPTO2-block版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/python/tests/st/pypto_pro/frontend/fa/test_flex_attention_prefill.py#L1362
[B24（PyPTO2-block版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/python/tests/st/pypto_pro/frontend/fa/test_flex_attention_prefill.py#L1593
[B25（PyPTO2-block版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/docs/zh/guide/programming_guide/pro/development/tile_based_python_programming/multi_core_partitioning_and_Tiling.md#L277
[B26（PyPTO2-block版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/python/pypto_pro/runtime/jit.py#L1746
[B27（PyPTO2-block版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/python/tests/ut/pypto_pro/runtime/test_shape_policy_jit.py#L31
[C24（PyPTO3（Simpler））]: https://github.com/hw-native-sys/simpler/blob/4e4d3a4ad1e54c1db3d50e72decc025a9075bfa0/tests/st/a2a3/tensormap_and_ringbuffer/spmd_paged_attention_highperf/test_spmd_paged_attention_highperf.py#L295
[D20（CANNBot DSL）]: ../../cannbot_dsl/cannbot-dsl/test/cannbotdsl/test_aot_p1b_dyn_tail.py#L45
[D21（CANNBot DSL）]: ../../cannbot_dsl/cannbot-dsl/test/cannbotdsl/test_aot_p1b_dyn_tail.py#L110
[D22（CANNBot DSL）]: ../../cannbot_dsl/cannbot-dsl/python/cannbotdsl/tensor.py#L1568
[E16（PyPTO on GPU）]: https://github.com/zhaosiying12138/PyPTO-LOVE-TensorIR/blob/b8830d9551b4f342f5511b5e1272cac5982e1728/packages/pypto-kernels/src/pypto_kernels/_boot.py#L541
[F17（Triton-Ascend）]: https://github.com/triton-lang/triton-ascend/blob/c747daae7f67fb7acd1013bf9793505aec1dc9e2/python/triton/runtime/jit.py#L399
[F18（Triton-Ascend）]: https://github.com/triton-lang/triton-ascend/blob/c747daae7f67fb7acd1013bf9793505aec1dc9e2/third_party/ascend/unittest/pytest_ut/test_triton_unified_attention.py#L258

[A21（PyPTO2-tensor版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/docs/zh/guide/programming_guide/tensor/debug/performance.md#L408
[A22（PyPTO2-tensor版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/docs/zh/api/tensor_api/config/pypto-set_pass_options.md#L43
[B28（PyPTO2-block版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/docs/zh/guide/programming_guide/pro/development/tile_based_python_programming/Reg_vector_computation.md#L1
[B29（PyPTO2-block版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/python/tests/st/pypto_pro/frontend/vf_api/test_layernorm_tile_group_vf.py#L71
[C25（PyPTO3（Simpler））]: https://github.com/hw-native-sys/pypto/blob/9f657f37ed20ce148b46fb7229c267a152a0644e/src/ir/transforms/outline_incore_scopes_pass.cpp#L303
[C26（PyPTO3（Simpler））]: https://github.com/hw-native-sys/pypto/blob/9f657f37ed20ce148b46fb7229c267a152a0644e/tests/ut/ir/transforms/test_outline_incore_scopes.py#L86
[C27（PyPTO3（Simpler））]: https://github.com/hw-native-sys/pypto-lib/blob/c6bc0bf50d6b1b58bde4698cdd4d8ff0f5d33b80/docs/pypto-coding/pypto-coding-style.md#L684
[C28（PyPTO3（Simpler））]: https://github.com/hw-native-sys/pypto/blob/9f657f37ed20ce148b46fb7229c267a152a0644e/python/pypto/language/op/unified_ops.py#L1135
[C29（PyPTO3（Simpler））]: https://github.com/hw-native-sys/pypto/blob/9f657f37ed20ce148b46fb7229c267a152a0644e/python/pypto/language/typing/memref.py#L46
[C30（PyPTO3（Simpler））]: https://github.com/hw-native-sys/pypto/blob/9f657f37ed20ce148b46fb7229c267a152a0644e/docs/en/dev/passes/11-convert_tensor_to_tile_ops.md#L3
[D23（CANNBot DSL）]: ../../cannbot_dsl/cannbot-dsl/python/cannbotdsl/core/utils/env.py#L93
[D24（CANNBot DSL）]: ../../cannbot_dsl/cannbot-dsl/lib/AscVec/Transforms/VfGrouping.cpp#L58
[D25（CANNBot DSL）]: ../../cannbot_dsl/cannbot-dsl/lib/CANNIR/Transforms/FormVfRegions.cpp#L145
[D26（CANNBot DSL）]: ../../cannbot_dsl/cannbot-dsl/lib/AscVec/Transforms/VfTransformPass.cpp#L123
[E17（PyPTO on GPU）]: https://github.com/zhaosiying12138/PyPTO-LOVE-TensorIR/blob/b8830d9551b4f342f5511b5e1272cac5982e1728/packages/pypto-kernels/src/pypto_kernels/sigmoid_mul.py#L22
[F19（Triton-Ascend）]: https://gitcode.com/Ascend/pytorch/blob/9f15aa301f6c69f7632b57e1ff198a84c27c3f79/torch_npu/_inductor/__init__.py#L379
[C31（PyPTO3（Simpler））]: https://github.com/hw-native-sys/pypto/blob/9f657f37ed20ce148b46fb7229c267a152a0644e/python/pypto/backend/pto_backend.py#L1380
[P5（PTOAS）]: https://github.com/hw-native-sys/PTOAS/blob/30a83c586cc9af801a9db0eb9137820b4cd8554c/test/lit/vpto/section_sugar_multi_func.pto#L1
[P6（PTOAS）]: https://github.com/hw-native-sys/PTOAS/blob/30a83c586cc9af801a9db0eb9137820b4cd8554c/tools/ptoas/VPTOHostStubEmission.cpp#L126
[P7（PTOAS）]: https://github.com/hw-native-sys/PTOAS/blob/30a83c586cc9af801a9db0eb9137820b4cd8554c/tools/ptoas/ObjectEmission.cpp#L680
[P8（PTOAS）]: https://github.com/hw-native-sys/PTOAS/blob/30a83c586cc9af801a9db0eb9137820b4cd8554c/test/vpto/cases/vmi_new/private-call-argument-boundary-store/launch.cpp#L28
[F20（AscendNPU-IR）]: https://gitcode.com/Ascend/AscendNPU-IR/blob/90037fe3371cb88d50c42cd4e165075cbafe83b1/bishengir/include/bishengir/Dialect/HACC/IR/HACCAttrs.td#L55
[F21（AscendNPU-IR）]: https://gitcode.com/Ascend/AscendNPU-IR/blob/90037fe3371cb88d50c42cd4e165075cbafe83b1/bishengir/lib/Dialect/HFusion/Transforms/OpFusion.cpp#L249
[F22（AscendNPU-IR）]: https://gitcode.com/Ascend/AscendNPU-IR/blob/90037fe3371cb88d50c42cd4e165075cbafe83b1/bishengir/include/bishengir/Tools/bishengir-compile/Options.td#L132
[F23（AscendNPU-IR）]: https://gitcode.com/Ascend/AscendNPU-IR/blob/90037fe3371cb88d50c42cd4e165075cbafe83b1/bishengir/test/Dialect/HFusion/AutoSchedule/test-host-multiple-tiling.mlir#L1
[F24（AscendNPU-IR）]: https://gitcode.com/Ascend/AscendNPU-IR/blob/90037fe3371cb88d50c42cd4e165075cbafe83b1/bishengir/lib/Conversion/HACCToLLVM/HACCToLLVM.cpp#L214
[F25（AscendNPU-IR）]: https://gitcode.com/Ascend/AscendNPU-IR/blob/90037fe3371cb88d50c42cd4e165075cbafe83b1/bishengir/lib/Conversion/HACCToLLVM/GetOrCreateUtils.cpp#L27
[F26（AscendNPU-IR）]: https://gitcode.com/Ascend/AscendNPU-IR/blob/90037fe3371cb88d50c42cd4e165075cbafe83b1/bishengir/hivmc/bishengir/lib/Tools/hivmc/A3/HIVMCMainA3.cpp#L936
[F27（AscendNPU-IR）]: https://gitcode.com/Ascend/AscendNPU-IR/blob/90037fe3371cb88d50c42cd4e165075cbafe83b1/bishengir/hivmc/bishengir/lib/Tools/hivmc/A5/HIVMCMainA5.cpp#L861
[F28（AscendNPU-IR）]: https://gitcode.com/Ascend/AscendNPU-IR/blob/90037fe3371cb88d50c42cd4e165075cbafe83b1/bishengir/lib/Dialect/HFusion/Pipelines/HFusionPipelines.cpp#L289
[F29（AscendNPU-IR）]: https://gitcode.com/Ascend/AscendNPU-IR/blob/90037fe3371cb88d50c42cd4e165075cbafe83b1/bishengir/include/bishengir/Dialect/HFusion/Transforms/Passes.td#L62
[F30（AscendNPU-IR）]: https://gitcode.com/Ascend/AscendNPU-IR/blob/90037fe3371cb88d50c42cd4e165075cbafe83b1/bishengir/test/Conversion/HACCToLLVM/hacc-to-llvm.mlir#L1
[F31（AscendNPU-IR）]: https://gitcode.com/Ascend/AscendNPU-IR/blob/90037fe3371cb88d50c42cd4e165075cbafe83b1/bishengir/lib/Dialect/HFusion/Pipelines/regbase/HFusionRegbasePipelines.cpp#L502
[F32（Triton-Ascend）]: https://github.com/triton-lang/triton-ascend/blob/c747daae7f67fb7acd1013bf9793505aec1dc9e2/third_party/ascend/backend/compiler.py#L717
[F33（Triton-Ascend）]: https://github.com/triton-lang/triton-ascend/blob/c747daae7f67fb7acd1013bf9793505aec1dc9e2/third_party/ascend/backend/driver.py#L450
[F34（AscendNPU-IR）]: https://gitcode.com/Ascend/AscendNPU-IR/blob/90037fe3371cb88d50c42cd4e165075cbafe83b1/bishengir/lib/Dialect/HIVM/Transforms/SplitMixKernel.cpp#L746
[F35（AscendNPU-IR）]: https://gitcode.com/Ascend/AscendNPU-IR/blob/90037fe3371cb88d50c42cd4e165075cbafe83b1/bishengir/include/bishengir/ExecutionEngine/Passes.td#L23
[G13（AutoFuse + Inductor）]: https://gitcode.com/Ascend/torchair/blob/cf2acaee5fe6139617fa7014ea0d09933cf8b35c/experimental/_inductor_npu_ext/tests/smoke/inductor_npu_ext_test.py#L523
[G14（AutoFuse + Inductor）]: https://gitcode.com/Ascend/torchair/blob/cf2acaee5fe6139617fa7014ea0d09933cf8b35c/experimental/_inductor_npu_ext/python/inductor_npu_ext/npu.py#L1858
[G15（AutoFuse + Inductor）]: https://gitcode.com/Ascend/torchair/blob/cf2acaee5fe6139617fa7014ea0d09933cf8b35c/experimental/_inductor_npu_ext/python/inductor_npu_ext/npu.py#L1081
[G16（AutoFuse + Inductor）]: https://gitcode.com/Ascend/torchair/blob/cf2acaee5fe6139617fa7014ea0d09933cf8b35c/experimental/_inductor_npu_ext/python/inductor_npu_ext/npu.py#L3762
[G17（AutoFuse + Inductor）]: https://gitcode.com/cann/graph-autofusion/blob/3b5e6a387e7dc2a0eef7d052a0f7a50c06947db2/autofuse/ascendc/api/axpy.h#L15
[B30（PyPTO2-block版）]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/python/pypto_pro/runtime/jit.py#L755
[C32（PyPTO3（Simpler））]: https://github.com/hw-native-sys/simpler/blob/4e4d3a4ad1e54c1db3d50e72decc025a9075bfa0/src/a2a3/runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler_dispatch.cpp#L124
[C33（PyPTO3（Simpler））]: https://github.com/hw-native-sys/simpler/blob/4e4d3a4ad1e54c1db3d50e72decc025a9075bfa0/src/a2a3/runtime/tensormap_and_ringbuffer/aicore/aicore_executor.cpp#L21
[C34（PyPTO3（Simpler））]: https://github.com/hw-native-sys/pypto/blob/9f657f37ed20ce148b46fb7229c267a152a0644e/tests/st/runtime/cross_core/test_spmd.py#L109
[C35（PyPTO3（Simpler））]: https://github.com/hw-native-sys/simpler/blob/4e4d3a4ad1e54c1db3d50e72decc025a9075bfa0/src/a2a3/runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler_completion.cpp#L178




<!-- A5 execution evidence and additional fixed-revision sources. -->
[A-entry]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/python/pypto/frontend/parser/entry.py#L1188
[B-jit]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/python/pypto_pro/runtime/jit.py#L951
[C-backend]: https://github.com/hw-native-sys/pypto/blob/9f657f37ed20ce148b46fb7229c267a152a0644e/python/pypto/backend/pto_backend.py#L14
[C-qwen]: https://github.com/hw-native-sys/pypto-lib/blob/c6bc0bf50d6b1b58bde4698cdd4d8ff0f5d33b80/models/qwen3_14b/paged_attention_pypto.py#L67
[C-decode-layer]: https://github.com/hw-native-sys/pypto-lib/blob/c6bc0bf50d6b1b58bde4698cdd4d8ff0f5d33b80/models/qwen3_14b/decode_fwd.py#L288
[E-compile]: https://github.com/zhaosiying12138/PyPTO-LOVE-TensorIR/blob/b8830d9551b4f342f5511b5e1272cac5982e1728/packages/pypto-kernels/src/pypto_kernels/_boot.py#L365
[E-pa]: https://github.com/zhaosiying12138/PyPTO-LOVE-TensorIR/blob/b8830d9551b4f342f5511b5e1272cac5982e1728/packages/pypto-kernels/src/pypto_kernels/attention.py#L1714
[E-pa-test]: https://github.com/zhaosiying12138/PyPTO-LOVE-TensorIR/blob/b8830d9551b4f342f5511b5e1272cac5982e1728/packages/pypto-kernels/benchmarks/paged_attention_sm120.py#L38
[F-compiler]: https://github.com/triton-lang/triton-ascend/blob/c747daae7f67fb7acd1013bf9793505aec1dc9e2/third_party/ascend/backend/compiler.py#L1385
[F-jit]: https://github.com/triton-lang/triton-ascend/blob/c747daae7f67fb7acd1013bf9793505aec1dc9e2/python/triton/runtime/jit.py#L399
[F-inductor]: https://gitcode.com/Ascend/pytorch/blob/9f15aa301f6c69f7632b57e1ff198a84c27c3f79/torch_npu/_inductor/codegen/npu_combined_scheduling.py#L30
[G-scheduler]: https://gitcode.com/Ascend/torchair/blob/cf2acaee5fe6139617fa7014ea0d09933cf8b35c/experimental/_inductor_npu_ext/python/inductor_npu_ext/npu.py#L2243
[G-pyautofuse]: https://gitcode.com/cann/graph-autofusion/blob/3b5e6a387e7dc2a0eef7d052a0f7a50c06947db2/autofuse/compiler/py_module/pyautofuse.cpp#L237
[P-driver]: https://github.com/hw-native-sys/PTOAS/blob/30a83c586cc9af801a9db0eb9137820b4cd8554c/tools/ptoas/ptoas.cpp#L334
[N-pipeline]: https://gitcode.com/Ascend/AscendNPU-IR/blob/90037fe3371cb88d50c42cd4e165075cbafe83b1/bishengir/lib/Dialect/HIVM/Pipelines/HIVMPipelines.cpp#L250
[A-pa]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/python/tests/st/test_ctrl_cpu_perf.py#L71
[B-pa]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/python/tests/st/pypto_pro/frontend/fa/test_flex_attention_prefill.py#L840
[B-pa-test]: https://gitcode.com/cann/pypto/blob/85c9484e236e35aaf4f6f04e3099c0a9442fd452/python/tests/st/pypto_pro/frontend/fa/test_flex_attention_prefill.py#L1741
[C-pa]: https://github.com/hw-native-sys/pypto/blob/9f657f37ed20ce148b46fb7229c267a152a0644e/examples/models/04_paged_attention.py#L230
[C-pa-test]: https://github.com/hw-native-sys/pypto/blob/9f657f37ed20ce148b46fb7229c267a152a0644e/tests/st/runtime/framework_and_models/test_paged_attention.py#L835
[C-qwen-platform]: https://github.com/hw-native-sys/pypto-lib/blob/c6bc0bf50d6b1b58bde4698cdd4d8ff0f5d33b80/models/qwen3_14b/test_paged_attention_pypto.py#L955
[R-dispatch]: https://github.com/hw-native-sys/simpler/blob/4e4d3a4ad1e54c1db3d50e72decc025a9075bfa0/src/a5/runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler_dispatch.cpp#L108
[R-executor]: https://github.com/hw-native-sys/simpler/blob/4e4d3a4ad1e54c1db3d50e72decc025a9075bfa0/src/a5/runtime/tensormap_and_ringbuffer/aicore/aicore_executor.cpp#L21
[D-pa]: ../../cannbot_dsl/cannbot-dsl/examples/flash_attn_noquant/flash_attn_noquant.py#L337
[D-pa-test]: ../../cannbot_dsl/cannbot-dsl/examples/flash_attn_noquant/test_flash_attn_noquant.py#L187
[E-lock]: https://github.com/zhaosiying12138/PyPTO-LOVE-TensorIR/blob/b8830d9551b4f342f5511b5e1272cac5982e1728/vendor/source-lock.json#L1
[E-bundle]: https://github.com/zhaosiying12138/PyPTO-LOVE-TensorIR/blob/b8830d9551b4f342f5511b5e1272cac5982e1728/vendor/git/pypto.bundle
[E-bootstrap]: https://github.com/zhaosiying12138/PyPTO-LOVE-TensorIR/blob/b8830d9551b4f342f5511b5e1272cac5982e1728/tools/bootstrap_release.py#L31
[F-pa]: https://gitcode.com/Ascend/triton-ascend-kernels/blob/f7d13ba7c7b6b4590b1d3910f93a2fd66fa6e9a3/src/triton_ascend_kernels/attention/paged_attention.py#L461
[F-pa-test]: https://gitcode.com/Ascend/triton-ascend-kernels/blob/f7d13ba7c7b6b4590b1d3910f93a2fd66fa6e9a3/tests/attention/test_paged_attention.py#L268
[G-register]: https://gitcode.com/Ascend/torchair/blob/cf2acaee5fe6139617fa7014ea0d09933cf8b35c/experimental/_inductor_npu_ext/python/inductor_npu_ext/__init__.py#L20
[F-build]: https://github.com/triton-lang/triton-ascend/blob/c747daae7f67fb7acd1013bf9793505aec1dc9e2/setup_ascend.py#L139
[F-patch]: https://github.com/triton-lang/triton-ascend/blob/c747daae7f67fb7acd1013bf9793505aec1dc9e2/third_party/ascend/patch/triton-ascend-3.6.0.patch#L1120
[P-release]: https://github.com/hw-native-sys/PTOAS/releases/tag/v0.57
[RUN-env]: https://github.com/nalinaly/pypto/blob/6d4a092ead2d6d43645c14cbdba84cabed37d75d/tests/npu_gpu_programming_stacks_comparison.a5.json#L5
[RUN-results]: https://github.com/nalinaly/pypto/blob/6d4a092ead2d6d43645c14cbdba84cabed37d75d/tests/npu_gpu_programming_stacks_comparison.a5.json#L150
[RUN-script]: https://github.com/nalinaly/pypto/blob/6d4a092ead2d6d43645c14cbdba84cabed37d75d/tests/npu_gpu_programming_stacks_comparison.a5.json#L1291
[RUN-A-fail]: https://github.com/nalinaly/pypto/blob/6d4a092ead2d6d43645c14cbdba84cabed37d75d/tests/npu_gpu_programming_stacks_comparison.a5.json#L869
[RUN-A-front]: https://github.com/nalinaly/pypto/blob/6d4a092ead2d6d43645c14cbdba84cabed37d75d/tests/npu_gpu_programming_stacks_comparison.a5.json#L887
[RUN-A-patch]: https://github.com/nalinaly/pypto/blob/6d4a092ead2d6d43645c14cbdba84cabed37d75d/tests/npu_gpu_programming_stacks_comparison.a5.json#L1029
[RUN-A-pass]: https://github.com/nalinaly/pypto/blob/6d4a092ead2d6d43645c14cbdba84cabed37d75d/tests/npu_gpu_programming_stacks_comparison.a5.json#L906
[RUN-G-lowering]: https://github.com/nalinaly/pypto/blob/6d4a092ead2d6d43645c14cbdba84cabed37d75d/tests/npu_gpu_programming_stacks_comparison.a5.json#L753
[RUN-F-xml]: https://github.com/nalinaly/pypto/blob/6d4a092ead2d6d43645c14cbdba84cabed37d75d/tests/npu_gpu_programming_stacks_comparison.a5.json#L1360
[RUN-C-xml]: https://github.com/nalinaly/pypto/blob/6d4a092ead2d6d43645c14cbdba84cabed37d75d/tests/npu_gpu_programming_stacks_comparison.a5.json#L1361
[RUN-D-xml]: https://github.com/nalinaly/pypto/blob/6d4a092ead2d6d43645c14cbdba84cabed37d75d/tests/npu_gpu_programming_stacks_comparison.a5.json#L1362
[GPU-history]: https://github.com/nalinaly/pypto/blob/098b896ffb84ffebd0dac87af62835b7e33195a9/tests/npu_gpu_programming_stacks_comparison.md#L3960
[E-tensor-bundle]: https://github.com/zhaosiying12138/PyPTO-LOVE-TensorIR/blob/b8830d9551b4f342f5511b5e1272cac5982e1728/vendor/git/tensor-ir.bundle
[RUN-A-driver]: https://github.com/nalinaly/pypto/blob/6d4a092ead2d6d43645c14cbdba84cabed37d75d/tests/npu_gpu_programming_stacks_comparison.a5.json#L1044

[RUN-smoke]: https://github.com/nalinaly/pypto/blob/41f37a8d65df526d20d7cff8cff933666997befe/tests/npu_gpu_programming_stacks_comparison.a5.json#L1364
[D-arena-rms]: ../../cannbot_dsl/cannbot-arena/test/rms_norm/test_rms_norm.py
[D-arena-mm]: ../../cannbot_dsl/cannbot-arena/test/matmul/matmul/test_matmul_precision.py

[H-readme]: https://gitcode.com/cann/catlass/blob/337cc89254d46f5d6041d117aca64edf7197027a/python/tla_dsl/README.md#L1
[H-dsl]: https://gitcode.com/cann/catlass/blob/337cc89254d46f5d6041d117aca64edf7197027a/python/tla_dsl/catlass/catlass_dsl/catlass.py#L15
[H-compile]: https://gitcode.com/cann/catlass/blob/337cc89254d46f5d6041d117aca64edf7197027a/python/tla_dsl/catlass/base_dsl/compiler.py#L8
[H-execution]: https://gitcode.com/cann/catlass/blob/337cc89254d46f5d6041d117aca64edf7197027a/python/tla_dsl/catlass/execution.py#L312
[H-runtime]: https://gitcode.com/cann/catlass/blob/337cc89254d46f5d6041d117aca64edf7197027a/python/tla_dsl/catlass/base_dsl/runtime/ascend.py#L137
[H-api-layout]: https://gitcode.com/cann/catlass/blob/337cc89254d46f5d6041d117aca64edf7197027a/python/tla_dsl/catlass/core_api.py#L3810
[H-api-copy]: https://gitcode.com/cann/catlass/blob/337cc89254d46f5d6041d117aca64edf7197027a/python/tla_dsl/catlass/core_api.py#L4421
[H-api-mmad]: https://gitcode.com/cann/catlass/blob/337cc89254d46f5d6041d117aca64edf7197027a/python/tla_dsl/catlass/core_api.py#L5522
[H-api-allocate]: https://gitcode.com/cann/catlass/blob/337cc89254d46f5d6041d117aca64edf7197027a/python/tla_dsl/catlass/core_api.py#L7575
[H-tensor-runtime]: https://gitcode.com/cann/catlass/blob/337cc89254d46f5d6041d117aca64edf7197027a/python/tla_dsl/catlass/tla/runtime.py#L274
[H-passes]: https://gitcode.com/cann/catlass/blob/337cc89254d46f5d6041d117aca64edf7197027a/python/tla_dsl/csrc/mlir/lib/Passes/PassRegistry.cpp#L44
[H-mixed-pass]: https://gitcode.com/cann/catlass/blob/337cc89254d46f5d6041d117aca64edf7197027a/python/tla_dsl/csrc/mlir/lib/Passes/TlaSplitMixedFuncPass.cpp#L1
[H-extern-pass]: https://gitcode.com/cann/catlass/blob/337cc89254d46f5d6041d117aca64edf7197027a/python/tla_dsl/csrc/mlir/lib/Passes/TlaLowerExternCallPass.cpp#L1
[H-auto-sync]: https://gitcode.com/cann/catlass/blob/337cc89254d46f5d6041d117aca64edf7197027a/python/tla_dsl/csrc/mlir/lib/Passes/TlaInsertAutoMutexPass.cpp#L1
[H-scratch]: https://gitcode.com/cann/catlass/blob/337cc89254d46f5d6041d117aca64edf7197027a/python/tla_dsl/csrc/mlir/lib/Passes/TlaScratchAllocation.cpp#L31
[H-ptr-pass]: https://gitcode.com/cann/catlass/blob/337cc89254d46f5d6041d117aca64edf7197027a/python/tla_dsl/csrc/mlir/lib/Passes/TlaLowerPtrPass.cpp#L129
[H-fa]: https://gitcode.com/cann/catlass/blob/337cc89254d46f5d6041d117aca64edf7197027a/python/tla_dsl/examples/end_to_end/flash_attention_infer/flash_attention_infer.py#L90
[H-fa-readme]: https://gitcode.com/cann/catlass/blob/337cc89254d46f5d6041d117aca64edf7197027a/python/tla_dsl/examples/end_to_end/flash_attention_infer/README.md#L1
[H-fa-tiling]: https://gitcode.com/cann/catlass/blob/337cc89254d46f5d6041d117aca64edf7197027a/python/tla_dsl/examples/end_to_end/flash_attention_infer/fa_tiling.py#L63
[H-mmad-example]: https://gitcode.com/cann/catlass/blob/337cc89254d46f5d6041d117aca64edf7197027a/python/tla_dsl/examples/end_to_end/basic_mmad/basic_matmul.py#L36
[H-common-golden]: https://gitcode.com/cann/catlass/blob/337cc89254d46f5d6041d117aca64edf7197027a/python/tla_dsl/examples/end_to_end/common/golden.py#L23
[H-common-utils]: https://gitcode.com/cann/catlass/blob/337cc89254d46f5d6041d117aca64edf7197027a/python/tla_dsl/examples/end_to_end/common/utils.py#L95
[H-mixed]: https://gitcode.com/cann/catlass/blob/337cc89254d46f5d6041d117aca64edf7197027a/python/tla_dsl/examples/end_to_end/basic_mixed/basic_mixed.py#L52
[H-streamk]: https://gitcode.com/cann/catlass/blob/337cc89254d46f5d6041d117aca64edf7197027a/python/tla_dsl/examples/end_to_end/basic_mmad_streamk/basic_mmad_streamk.py#L1
[H-ir-build]: https://gitcode.com/cann/catlass/blob/337cc89254d46f5d6041d117aca64edf7197027a/python/tla_dsl/docs/zh/dsl_development/build_guide/ascend_npu_ir.md#L1
[H-cmake]: https://gitcode.com/cann/catlass/blob/337cc89254d46f5d6041d117aca64edf7197027a/python/tla_dsl/csrc/mlir/CMakeLists.txt#L1
[H-ir-submodules]: https://gitcode.com/Ascend/AscendNPU-IR/blob/a07821269ede7a5e683ac02c8a2d291608083741/.gitmodules#L1
[H-ir-matmul]: https://gitcode.com/Ascend/AscendNPU-IR/blob/a07821269ede7a5e683ac02c8a2d291608083741/bishengir/lib/Conversion/HFusionToHIVM/Matmul.cpp#L510
[H-ir-patches]: https://gitcode.com/Ascend/AscendNPU-IR/tree/a07821269ede7a5e683ac02c8a2d291608083741/build-tools/patches/triton
[H-llvm]: https://gitcode.com/Ascend/llvm-project/blob/9c0841dc3fa2a224595b41e9e0641b796d2afeb9/llvm/CMakeLists.txt#L1
[H-triton]: https://github.com/triton-lang/triton/blob/c3c476f357f1e9768ea4e45aa5c17528449ab9ef/CMakeLists.txt#L1
[H-inductor-template]: https://gitcode.com/Ascend/pytorch/blob/9f15aa301f6c69f7632b57e1ff198a84c27c3f79/torch_npu/_inductor/codegen/catlass/catlass_template.py#L156
[I-dsl]: https://github.com/NVIDIA/cutlass/blob/59e3a3338d516ca6ce0e073af8da65289678a35c/python/CuTeDSL/cutlass/cutlass_dsl/cutlass.py#L567
[I-layout]: https://github.com/NVIDIA/cutlass/blob/59e3a3338d516ca6ce0e073af8da65289678a35c/python/CuTeDSL/cutlass/cute/core.py#L3684
[I-mma]: https://github.com/NVIDIA/cutlass/blob/59e3a3338d516ca6ce0e073af8da65289678a35c/python/CuTeDSL/cutlass/cute/nvgpu/tcgen05/mma.py#L1
[I-copy]: https://github.com/NVIDIA/cutlass/blob/59e3a3338d516ca6ce0e073af8da65289678a35c/python/CuTeDSL/cutlass/cute/nvgpu/cpasync/copy.py#L1
[I-pipeline]: https://github.com/NVIDIA/cutlass/blob/59e3a3338d516ca6ce0e073af8da65289678a35c/python/CuTeDSL/cutlass/pipeline/sm100.py#L1
[I-compiler]: https://github.com/NVIDIA/cutlass/blob/59e3a3338d516ca6ce0e073af8da65289678a35c/python/CuTeDSL/cutlass/base_dsl/compiler.py#L1
[I-executor]: https://github.com/NVIDIA/cutlass/blob/59e3a3338d516ca6ce0e073af8da65289678a35c/python/CuTeDSL/cutlass/base_dsl/jit_executor.py#L139
[I-runtime]: https://github.com/NVIDIA/cutlass/blob/59e3a3338d516ca6ce0e073af8da65289678a35c/python/CuTeDSL/cutlass/base_dsl/runtime/cuda.py#L600
[I-tensor-runtime]: https://github.com/NVIDIA/cutlass/blob/59e3a3338d516ca6ce0e073af8da65289678a35c/python/CuTeDSL/cutlass/cute/runtime.py#L213
[I-requirements]: https://github.com/NVIDIA/cutlass/blob/59e3a3338d516ca6ce0e073af8da65289678a35c/python/CuTeDSL/requirements.txt#L1
[I-softmax]: https://github.com/NVIDIA/cutlass/blob/59e3a3338d516ca6ce0e073af8da65289678a35c/examples/python/CuTeDSL/experimental/primitives/tutorial/06_softmax.py#L86
[I-mla]: https://github.com/NVIDIA/cutlass/blob/59e3a3338d516ca6ce0e073af8da65289678a35c/examples/python/CuTeDSL/cute/blackwell/kernel/attention/mla/mla_decode_fp16.py#L3378
[I-mla-fp8]: https://github.com/NVIDIA/cutlass/blob/59e3a3338d516ca6ce0e073af8da65289678a35c/examples/python/CuTeDSL/cute/blackwell/kernel/attention/mla/mla_decode_fp8.py#L1
[I-gqa]: https://github.com/NVIDIA/cutlass/blob/59e3a3338d516ca6ce0e073af8da65289678a35c/examples/python/CuTeDSL/cute_ext/blackwell/attention/gqa_decode_simple.py#L88
[I-static]: https://github.com/NVIDIA/cutlass/blob/59e3a3338d516ca6ce0e073af8da65289678a35c/examples/python/CuTeDSL/helpers/static_persistent_tile_scheduler.py#L1
[I-dynamic]: https://github.com/NVIDIA/cutlass/blob/59e3a3338d516ca6ce0e073af8da65289678a35c/python/CuTeDSL/cutlass/utils/dynamic_persistent_tile_scheduler.py#L1
[I-task]: https://github.com/NVIDIA/cutlass/blob/59e3a3338d516ca6ce0e073af8da65289678a35c/python/CuTeDSL/cutlass/experimental/task_scheduling/task.py#L546
[I-schedule]: https://github.com/NVIDIA/cutlass/blob/59e3a3338d516ca6ce0e073af8da65289678a35c/python/CuTeDSL/cutlass/experimental/task_scheduling/schedule_builder.py#L15
[I-task-manager]: https://github.com/NVIDIA/cutlass/blob/59e3a3338d516ca6ce0e073af8da65289678a35c/python/CuTeDSL/cutlass/experimental/task_scheduling/task_manager.py#L1
[I-ts-memory]: https://github.com/NVIDIA/cutlass/blob/59e3a3338d516ca6ce0e073af8da65289678a35c/python/CuTeDSL/cutlass/experimental/task_scheduling/memory.py#L15
[I-checker]: https://github.com/NVIDIA/cutlass/blob/59e3a3338d516ca6ce0e073af8da65289678a35c/python/CuTeDSL/cutlass/experimental/task_scheduling/exhaustive_checker.py#L1
[I-ts-tutorial]: https://github.com/NVIDIA/cutlass/blob/59e3a3338d516ca6ce0e073af8da65289678a35c/examples/python/CuTeDSL/experimental/task_scheduling/blackwell/tutorial/03_persistent_scheduling_dynamic_domain_ts/README.md#L1
[I-examples]: https://github.com/NVIDIA/cutlass/tree/59e3a3338d516ca6ce0e073af8da65289678a35c/examples/python/CuTeDSL

[H-run]: https://github.com/nalinaly/pypto/blob/fcb8d1b0070c7921a56c103a43eafb3cf309a8c8/tests/npu_gpu_programming_stacks_comparison.catlass.json#L1
[H-run-env]: https://github.com/nalinaly/pypto/blob/fcb8d1b0070c7921a56c103a43eafb3cf309a8c8/tests/npu_gpu_programming_stacks_comparison.catlass.json#L5
[H-run-build]: https://github.com/nalinaly/pypto/blob/fcb8d1b0070c7921a56c103a43eafb3cf309a8c8/tests/npu_gpu_programming_stacks_comparison.catlass.json#L88
[H-run-results]: https://github.com/nalinaly/pypto/blob/fcb8d1b0070c7921a56c103a43eafb3cf309a8c8/tests/npu_gpu_programming_stacks_comparison.catlass.json#L217
[H-run-diagnostics]: https://github.com/nalinaly/pypto/blob/fcb8d1b0070c7921a56c103a43eafb3cf309a8c8/tests/npu_gpu_programming_stacks_comparison.catlass.json#L479
[H-run-artifacts]: https://github.com/nalinaly/pypto/blob/fcb8d1b0070c7921a56c103a43eafb3cf309a8c8/tests/npu_gpu_programming_stacks_comparison.catlass.json#L620
[H-run-replay]: https://github.com/nalinaly/pypto/blob/fcb8d1b0070c7921a56c103a43eafb3cf309a8c8/tests/npu_gpu_programming_stacks_comparison.catlass.json#L3441

<a id="nearest"></a>
## 14. 最终选择：每种方式最相似的另一种是谁

这里按 **整体执行组织/责任分层 → 用户编程与tiling → 核内编译分工 → 前端代码血缘** 的顺序判断。megaKernel也按程序级与物理kernel级分开考虑。下表比较 A—I 九种方式的“最相似者”；这是有方向的选择，不强制成对，不是性能或优劣排名。各维度的依据与适用范围见第 14.1 节。

| 方式 | 最相似的另一种 | 首要理由 | 最重要的不相同 | 若换一个维度，谁会更近 |
| --- | --- | --- | --- | --- |
| A（PyPTO2-tensor版） | **C（PyPTO3（Simpler））** | 都以Tensor/程序分解连接设备任务执行体系，可组织多阶段模型程序 | 自有IR、编译器和runtime实现不同；TileFwk不是Simpler | 按代码/基础IR共享，B（PyPTO2-block版）更近 |
| B（PyPTO2-block版） | **D（CANNBot DSL）** | 显式核内资源、tile/循环、混合计算、Host直接launch，作者承担性能工程 | B（PyPTO2-block版）自有IR/CCE；D（CANNBot DSL）更早进入MLIR且有Host staging/Channel | 按代码血缘选A（PyPTO2-tensor版）；按task内SPMD核内风格，C（PyPTO3（Simpler））的对应子路径也很近 |
| C（PyPTO3（Simpler）） | **A（PyPTO2-tensor版）** | 外层程序/依赖/设备派发是整体架构识别度最高的部分 | C（PyPTO3（Simpler））的PTOAS、Simpler及显式TaskId/SPMD/extern接缝不同 | 只看pypto-lib的融合SPMD attention，B（PyPTO2-block版）/D（CANNBot DSL）更近；看前端血缘则E（PyPTO on GPU）更近 |
| D（CANNBot DSL） | **H（CATLASS DSL）** | 都由用户显式安排 NPU 核内计算、缓冲/流水与 launch，较早进入 MLIR；动态 tile 窗口不强制独立 TilingFunc | D 的 MLIR Host/Device staging、Channel、CANNIR/AscendC，与 H 的 TLA、内联设备 helper、HIVM 不同 | 按较长的 MLIR 编译链可对照 F（Triton-Ascend）；按 TileGroup/Channel 的核内作者责任可对照 B（PyPTO2-block版），区别见第 14.1 节 |
| E（PyPTO on GPU） | **F（Triton-Ascend）** | 当前都偏向kernel/program级tile编程与专用后端lowering，runtime直接launch，非AICPU任务图主链 | GPU CTA/SM与NPU program/block映射不同；E（PyPTO on GPU）模式覆盖及动态入口更受限 | 按PyPTO前端血缘选C（PyPTO3（Simpler））；按客户Inductor入口则E（PyPTO on GPU）不如G（AutoFuse + Inductor）接近F（Triton-Ascend） |
| F（Triton-Ascend） | **G（AutoFuse + Inductor）** | 当前都在PyTorch/Inductor的NPU kernel生成位置承担较完整的lowering、资源与launch分工 | F（Triton-Ascend）有独立Triton DSL/MLIR链，G（AutoFuse + Inductor）有ASCIR/自动Host tiling/AscendC生成 | 只看独立tile-kernel编程及专用MLIR后端，E（PyPTO on GPU）更近；看显式MLIR硬件链也可对比D（CANNBot DSL） |
| G（AutoFuse + Inductor） | **F（Triton-Ascend）** | 客户可保留PyTorch表达，Inductor选择融合/模板/extern并生成NPU kernel | G（AutoFuse + Inductor）不是Triton frontend；ASCIR、ATT tiling及生成代码方式不同 | 只看生成AscendC及局部buffer语义，D（CANNBot DSL）的下层更近 |
| H（CATLASS DSL） | **D（CANNBot DSL）** | NPU 显式 layout/资源/搬运/流水，较早进入 MLIR，Host 直接调用产物 | H 的 TLA/HIVM、局部 AutoSync 与 D 的 Channel/Host staging/CANNIR/AscendC 不同 | 按 Python layout/核内元编程看 I 更近；按部分下游编译基础设施看 F 更近 |
| I（CuTe DSL） | **H（CATLASS DSL）** | Python 元编程、显式 layout/搬运/矩阵与 pipeline，直接编译/launch，作者承担局部资源责任 | GPU thread/value layout/atom/warp/TS 与 NPU 物理 tag/AIC/AIV/局部 AutoSync 不同 | 按 CUDA artifact/launch 看 E 更近；TS 与 A/C 仅按资源/任务粒度对照 |

按责任层归类，**A（PyPTO2-tensor版）与 C（PyPTO3（Simpler））属于程序/任务执行家族；B（PyPTO2-block版）、D（CANNBot DSL）与 H（CATLASS DSL）接近显式 NPU 核内工程家族；I（CuTe DSL）承担相近的 GPU layout/warp/流水责任；F（Triton-Ascend）与 G（AutoFuse + Inductor）接近 Inductor kernel 生成家族；E（PyPTO on GPU）的整体形态最靠近 F，前端血缘靠近 C。** 这些分组并不否定各家内部的其他模式，特别是 C 已有与 B/D 接近的 SPMD 核内实现。它们用于定位哪些层可以共用、哪些层存在替代关系，不是互斥的架构标签。


### 14.1 相似性判断的依据与适用范围

按本章优先级，H（CATLASS DSL） 首先选 D（CANNBot DSL），I（CuTe DSL） 首先选 H（CATLASS DSL），D（CANNBot DSL） 首先选 H（CATLASS DSL）。B（PyPTO2-block版）/D（CANNBot DSL） 同样在显式核内资源与流水责任上相近，但 D（CANNBot DSL） 使用 MLIR Host/Device 链和 Channel，B（PyPTO2-block版） 使用共享 PyPTO2 基础 IR 上的 block版操作与 TileGroup；D（CANNBot DSL）/H（CATLASS DSL） 则同时接近 NPU 显式资源和较早 MLIR 入口这两个维度。局部相似性与整体选择的权重不同，因此不要求双向匹配。

| 对象 / 比较角度 | 相似对象 | 理由与边界 |
| --- | --- | --- |
| H（CATLASS DSL） 的整体执行与核内责任 | **D（CANNBot DSL）** | 同为 NPU 显式硬件/内存/流水 + 较早进入 MLIR + Host launch；H（CATLASS DSL） 的 TLA/HIVM 链、D（CANNBot DSL） 的 CANNIR/AscendC 与 Host staging 不同 |
| I（CuTe DSL） 的整体核内编程责任 | **H（CATLASS DSL）** | Python 元编程、显式 layout/搬运/矩阵与流水、直接 kernel artifact/launch 相近；I（CuTe DSL） 的 GPU 线程/warp 与 TS、H（CATLASS DSL） 的 AIC/AIV/物理 tag 仍有显著差异 |
| D（CANNBot DSL） 的整体架构与核内责任 | **H（CATLASS DSL）**；B（PyPTO2-block版） 也是同硬件参照 | D（CANNBot DSL）/H（CATLASS DSL） 同时接近显式资源与 MLIR 主链；若优先比较 TileGroup/Channel 的作者责任，B（PyPTO2-block版）/D（CANNBot DSL） 也很相近 |
| H（CATLASS DSL）/I（CuTe DSL） 仅看 layout 和 Python 接口 | **互为重点比较对象** | 不能从名字相近推出 layout 代数同等覆盖、代码血缘或后端兼容 |
| H（CATLASS DSL） 仅看下游编译基础设施 | **F（Triton-Ascend） 的 AscendNPU-IR 路径** | 输入抽象、已完成的物理分解、固定版本与 launch metadata 不同 |
| I（CuTe DSL） 的 TS 与 A（PyPTO2-tensor版）/C（PyPTO3（Simpler）） 的任务系统 | **按粒度对照，不能合为同一整体家族** | 单 kernel 的 warp/资源协议，与跨计算入口的设备任务依赖/派发分别减少不同工程工作 |

九路线覆盖 **A（PyPTO2-tensor版）/C（PyPTO3（Simpler）） 的程序与任务组织、B（PyPTO2-block版）/D（CANNBot DSL）/H（CATLASS DSL） 的显式 NPU 核内工程、F（Triton-Ascend）/G（AutoFuse + Inductor） 的 Inductor kernel 生成、E（PyPTO on GPU） 的 PyPTO/GPU 接口，以及 I（CuTe DSL） 的 CuTe layout/warp/persistent 工具**。这些实现分别说明 megaKernel 的不同层次：“程序级推进、单 kernel 合作、片上数据复用、最终性能”需要各自的证据，不能相互替代。

九路线的有向选择为 **A（PyPTO2-tensor版）→C（PyPTO3（Simpler））、B（PyPTO2-block版）→D（CANNBot DSL）、C（PyPTO3（Simpler））→A（PyPTO2-tensor版）、D（CANNBot DSL）→H（CATLASS DSL）、E（PyPTO on GPU）→F（Triton-Ascend）、F（Triton-Ascend）→G（AutoFuse + Inductor）、G（AutoFuse + Inductor）→F（Triton-Ascend）、H（CATLASS DSL）→D（CANNBot DSL）、I（CuTe DSL）→H（CATLASS DSL）**。D（CANNBot DSL）/H（CATLASS DSL）/I（CuTe DSL） 的判断依据是显式资源、Host/kernel 边界与实际编译链；B（PyPTO2-block版）/D（CANNBot DSL） 的同硬件核内责任比较见第 4.9 节。这些选择不依据名字、未测性能或相同 ISA。[H-dsl] [H-passes] [H-api-layout] [I-dsl] [I-layout] [I-task]
