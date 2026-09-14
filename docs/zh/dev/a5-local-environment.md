# nalinaly/pypto 的 A5 本地运行环境

本记录对应 2026-09-05 的 `/home/q00473782/inductor` 工作区，目标是恢复
`nalinaly/pypto` 的基本 A5 真机能力，供 `inductor_pto` 开发使用。
验证范围是少量代表算子的正确性，不是全量回归或性能验收。

## 运行依据和依赖版本

实际运行目录是 `pto_qcy/pypto`。`pto/` 下的上游 main 仅供代码参考，
不参与 Python 导入、动态库加载、构建或编译缓存复用。

| 组件 | 本地位置（相对工作区） | 版本或源码依据 |
| --- | --- | --- |
| PyPTO | `pto_qcy/pypto` | `https://github.com/nalinaly/pypto`，基础提交 `9cece0b730a96fe1a52c2637537132f524ffe1ea` |
| Simpler | `pto_qcy/pypto/runtime` | fork 基线 `06c8b29698c547e339ed78b0acaefa6ca52e0d49`；A5 L1 HBG 与头文件依赖修复 `9d53975cf65ab67975230a6c70d5c48d6220e25c` |
| PTOAS 源码 | `pto_qcy/PTOAS` | v0.57，`307d0484a9e7d5e36f01b253d2bebe4d2f45fe81` |
| PTOAS 可执行文件 | `pto_qcy/.venv/bin/ptoas` | v0.57 官方 CPython 3.12 x86_64 wheel |
| PTO-ISA 源码 | `pto_qcy/pto-isa` | `f51c92f610827daad0ddfb383072e03d514b4ae9` |
| Simpler 实际构建的 PTO-ISA | `pto_qcy/pypto/runtime/build/pto-isa` | 同一 SHA，由 `runtime/pto_isa.pin` 锁定 |
| Python 环境 | `pto_qcy/.venv` | Python 3.12.3，独立 venv，不继承 system-site-packages |
| PyTorch / torch-npu | 同一 venv | `2.12.0+cpu` / `2.12.0` |
| CANN | `/home/q00473782/Ascend/cann-9.2.0-weekly.20260902.01/cann-9.2.0` | 现有 9.2.0 weekly 工具链 |

PTOAS wheel 保存在 `pto_qcy/downloads/`，文件名为
`ptoas-0.57-cp312-cp312-manylinux_2_34_x86_64.whl`，SHA256：

```text
50973353d15ec78e81745ddef64029b918a9963e1c5735bfa7e65c00a94a7f92
```

`toolchain/versions.env` 中的 x86_64 SHA 对应 CPython 3.10 wheel，
不能用它校验此处的 CPython 3.12 wheel。没有安装独立 LLVM，
也没有改动 cannbot_dsl session 的工具链或 `/home/q00473782/llvm`。

Simpler 修复目前是本地提交，父仓库的 gitlink 已锁定它，但尚未推送到远端。
迁移到另一台机器时应同时携带这个 submodule 提交，不能只克隆远端 PyPTO 基础提交。

## 激活和重建

```bash
cd /home/q00473782/inductor
source pto_qcy/env.sh
```

该入口加载现有 CANN，再激活独立 venv；清除搜索路径中的 `pto/` 条目和
CANN 自带的同名 `pypto` 包路径。`PYPTO_ROOT`、`SIMPLER_ROOT`、`PTOAS_ROOT`、
`PTO_ISA_ROOT` 均指向 `pto_qcy`，TorchInductor/PTO 缓存也使用该目录下的 `.cache/`。
重复 source 不会累积重复路径。`PTO_PLATFORM` 未设置时默认 `a5`；显式指定其他平台
会保留，因此真机调用应明确检查平台。

venv 已安装构建工具，包括 CMake、Ninja、nanobind、scikit-build-core、pybind11，
以及 numpy、pytest 和 ruff。只重建当前 checkout，不自动升级或切换依赖：

```bash
bash /home/q00473782/inductor/pto_qcy/rebuild.sh
```

等价的核心命令如下；默认构建并行度为 2，可由机器本地的 testing.env 调整：

```bash
cd /home/q00473782/inductor/pto_qcy/pypto
source ../env.sh
source .claude/skills/testing/load-env.sh
python -m pip install --no-build-isolation -e runtime \
    --config-settings=build.targets=build_package_a5
python -m pip install --no-build-isolation -e .
python -m pip install --no-build-isolation --no-deps -e ../../inductor_pto
python -m pip check
```

PyPTO 原生扩展和 Simpler 均从本 checkout 构建，没有复制上游目录的 `.so`。
`runtime/build/lib/pto_isa_build.json` 记录真实 PTO-ISA pin 和构建目录。

## 真机验证及复现

机器为 Ascend950PR A5，`npu-smi` 版本 25.7.rc1.2。当前驱动不接受芯片查询的
`-c 0` 参数，预检会在明确收到该错误时重试 `npu-smi info -t board -i 0`，
并通过 CANN 的 `Short_SoC_version=Ascend950` 确认 A5。

运行前用 `npu-smi info` 选择无其他任务的卡。本次使用物理卡 1，并映射为逻辑卡 0。
本机没有 `task-submit`，因此按用户要求直接串行执行，不重置其他卡、不终止其他 session。

PyPTO 的 `RunConfig()` 默认是 `a2a3sim`，**不会读取 `PTO_PLATFORM`**。
直接调用 PyPTO 时必须明确写 `RunConfig(platform="a5", device_id=0)`。
以下复用已有示例中的 JIT 定义，覆盖向量加法、行归约 softmax 和 Cube 矩阵乘法：

```bash
cd /home/q00473782/inductor/pto_qcy/pypto
source ../env.sh
ASCEND_RT_VISIBLE_DEVICES=1 python - <<'PY'
import importlib
import torch
from pypto.runtime import RunConfig

cfg = RunConfig(platform="a5", device_id=0, runtime="tensormap_and_ringbuffer")
torch.manual_seed(0)
cases = [
    ("examples.beginner.01_hello_world", "tile_add", 128, lambda a, b: a + b, 1e-5),
    ("examples.intermediate.02_softmax", "tile_softmax", 64, None, 1e-5),
    ("examples.beginner.05_matmul", "matmul_64", 64, lambda a, b: a @ b, 1e-3),
]
for module, name, size, golden, tolerance in cases:
    fn = getattr(importlib.import_module(module), name)
    a, b = torch.randn(size, size), torch.randn(size, size)
    out = torch.zeros_like(a)
    if golden is None:
        fn(a, out, config=cfg)
        expected = torch.softmax(a, dim=-1)
    else:
        fn(a, b, out, config=cfg)
        expected = golden(a, b)
    torch.testing.assert_close(out, expected, rtol=tolerance, atol=tolerance)
    print("PASS A5", name)
PY
```

结果：三个算子均通过数值校验，进程正常退出。日志在 `pto_qcy/logs/pypto-a5-smoke.log`。
A5 的 TRB 和 HBG 均已构建并通过代表真机验证。CANN CPU_TOPO 查询暂不可用时，
运行时结合设备侧 OCCUPY 与自带的 A5 拓扑配置推导可用线程。

## 同进程设备指针接口

这个 fork 的公共 `Worker.run` 使用不携带地址的 Buffer/Tensor 标识。
手工构造的 `DeviceTensor(data_ptr, shape, dtype)` 不能经过该 wire ABI。
为接入 PyTorch 已分配的 NPU 内存，`CompiledProgram` 和 `compiled[name]` 提供
`build_chip_args(*args)`，复用已有参数数量、shape、dtype 和标量类型校验，返回：

```python
chip_args, owners, return_style = compiled.build_chip_args(*args)
# chip_args: simpler.task_interface.ChipStorageTaskArgs
# owners: 持续保留至同步 run 返回的参数/输出对象
```

该结果仅供同进程 `simpler.task_interface.ChipWorker.run` 使用。
`DeviceTensor` 以 `child_memory=True` 携带地址，Tensor 与 Scalar 分池保持各自顺序。
调用方负责持有原始存储，并处理 PyTorch 与 Simpler 间的设备同步；接口不会替调用方
推断地址属于哪个设备。公共 Worker 的所有权验证保持生效。

此接口补了两个针对性单测：设备地址/标量分池正确，错误 shape 在打包前被拒绝。
Inductor 的接入和六个代表用例结果见 `inductor_pto/docs/a5-fork-runtime.md`。
以上为 L2 路径的接入记录；当前 L1 与双前端能力见下节。

## A5 HBG L1 与双前端

runtime gitlink 已包含 HBG 借用流生命周期、不可变图恢复与 A5 平台启动 ABI。
公共 `@pl.jit(execution="l1", runtime="host_build_graph")` 现在接受
`RunConfig(platform="a5", device_id=0, runtime="host_build_graph")`。
直接写 PyPTO 时每次调用显式传 config；不传 config 的 JIT 仍默认 A2/A3。
用法与 capture 生命周期见 [L1 指南](../user/04-l1-aclgraph.md)。

Inductor 和官方 `pytorch/helion` 的适配器在 A5 上共用 HBG L1/TaskQueue 默认路径；
`PTO_RUNTIME=tensormap_and_ringbuffer` 显式选择 L2。Helion 本体在
`/home/q00473782/inductor/helion`；没有使用 `pto/` 作为运行依据，也未新建 LLVM 工具链。

少量代表检查均通过：底层 L1 eager 与三次图回放；公共 JIT eager 自动输出分配、
warmup 后追加第二个 callable、PyTorch/add/mul/PyTorch 串联 capture/replay；
双前端禁止 L2 worker/张量包装/前端同步的 L1 transport 检查；Helion 64×64 matmul；
显式 Inductor L2/TRB。另有两个 JIT CPU 检查覆盖保留 A2/A3 默认和 A5 显式 lowering。
日志在 `pto_qcy/logs/a5-l1-*` 和 `a5-l2-trb-regression.json`。

当前 runtime 同时保留 fork 的 A2/A3 close 后 worker 退出修复（`06c8b29`），
公共 L1 report 布局为 128 字节。A5 AIC/AIV 自定义编译命令已补齐 runtime 与
平台头文件依赖，增量构建不会继续复用按旧 64 字节布局编译的对象文件。
合并并重建后，公共 JIT eager allocator 与 ACLGraph replay 再次通过，耗时
8.94 秒，日志为 `pto_qcy/logs/pypto-push-header-a5.log`。
runtime 提交位于 `nalinaly/simpler` 的 `a5-l1-hbg` 分支。

选择空闲物理卡 1 后，可复现公共 JIT 验证：

```bash
cd /home/q00473782/inductor/pto_qcy/pypto
source ../env.sh
source .claude/skills/testing/load-env.sh
export ASCEND_PROCESS_LOG_PATH="$PWD/../logs/a5-l1-public-jit-device"
mkdir -p "$ASCEND_PROCESS_LOG_PATH"
ASCEND_RT_VISIBLE_DEVICES=1 PYPTO_L1_JIT_TEST_RUNTIME=host_build_graph \
  python -m pytest tests/st/runtime/l1/test_l1_jit_aclgraph.py \
  --platform=a5 --device=0 -q
```

A5 L1 当前使用 HBG scheduler。direct-AIV 优化、A5 TRB L1、并发 replay、SDMA、
分布式和 DFX 不属于本次支持范围，也没有宣称全量算子、仿真或性能回归通过。
