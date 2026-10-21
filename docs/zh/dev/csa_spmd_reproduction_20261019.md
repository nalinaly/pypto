# CSA 单 SPMD 配套分支（2026-10-19）

本分支 `dsv4-csa-spmd-repro-20261019` 固定 nalinaly/vllm-ascend 的 CSA
单 SPMD 实测工具链。它基于调试分支 `d01de8523`，保留已采用的本地修复，
不代表这些修改已合入 PyPTO 主线。

完整克隆、构建、环境绑定和既有 CSA 测试命令见
[CSA 复现说明](https://github.com/nalinaly/vllm-ascend/tree/dsv4-flash-pto-v0.25.1rc1/tests/pypto_test/csa_toolchain_repro_20261019)。
该目录的 `versions.json` 固定三仓提交，`verification.json` 区分本次源码
重建检查与既有真机验证。

## 配套依赖

- Simpler fork：`https://github.com/nalinaly/simpler.git`；
  提交 `57e499238fd4ee3548947ad860be456aaeeb16b5`。
- `.gitmodules` 的 `runtime` URL、gitlink 和
  `python/pypto/_kernel_abi.py` 的 revision 同步到该版本。
- Simpler 的 256 Tensor + 16 scalar ABI 必须与当前 PyPTO 同时使用；
  同名包的版本号 `0.1.0` 不能替代 Git 提交号。
- PTOAS 使用 nalinaly/PTOAS 的同名复现分支，具体提交见上述版本清单。
- 当前实测 CANN 为 9.2.0-beta.2，Torch 2.10.0、torch_npu 2.10.0.post2；
  不是会话初期的 CANN 9.0 环境。

## 本地补丁范围

1. 为显式 MIX Group 的共享 InOut 返回建立正确 SSA 参数绑定。
2. 保留动态维度算术形成的 reshape 元数据与 inline Tile 参数注解。
3. 为普通本地 GM 信号提供 `system.signal_notify/signal_wait`，
   明确信号发布/等待不代替任意业务数据的完整同步。
4. 修复分支 view 句柄、条件原地 gather 及行/列 carry 的 PTOAS 代码生成。
5. 保持原地写入及 view 的别名链，补齐 While 内 For/Acc 规则递归；
   修正分支局部生存期和内存窗口范围，保留容量、冲突与 pinned-buffer 约束。
6. 将 signed/INDEX FloorMod 的生成结果修正为与 Python/IR 一致的 floor 余数。
7. 继承 `d46a813ca` 的常量 slice 起点折叠，以及 `d01de8523` 的无用
   Tile 传递闭环删除。

部分补丁用于备用 lowering 路径；保留它们是为了复现已验证的工具链源码，
不声称每条补丁都是每个 CSA 档位的运行必需条件。相关回归用例和中英文
开发文档一并保存。临时导入重定向脚本不属于编译器实现。

问题记录包括 PyPTO #3014、#3015、#3016、#3017、#3020、#3026、#3027、
#3028、#3037。复现分支的发布不会自动关闭这些上游 issue，也不创建上游 PR。

## 官方修复接入（2026-10-21）

公共 CSA 工具链继续保留 kernel 调试分支的运行时 ABI。
接入上游通用共享目的视图修复后，删除旧的 move 专用 reshape 生成逻辑，
由 `PrepareTileDestination` 统一选择一次目标。保留全零 replay sentinel
的动态有效范围表达；此规则不推广到部分维度为零的 valid shape。

如果只将某个 GM 参数的部分 Scalar 写循环改成 DMA 会造成混写，
则保留该函数原有的 Scalar 循环。判断复用最终一致性检查所使用的别名分析，
不关闭对源程序已有 DMA/Scalar 混写的拒绝。

kernel 调试分支尚无完整 Buffer IR lowering。导入的重编译/Buffer 阶段测试
在 `lower_tile_to_buffer` 能力缺失时明确跳过；Group 输出与共享存储测试
独立执行，不把该能力缺口计为回归通过。

