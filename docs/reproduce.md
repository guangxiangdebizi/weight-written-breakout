# 复现方法

先按根目录 README 创建 Python 3.13 虚拟环境并安装 `requirements.txt`。以下命令都在仓库根目录执行。`work/` 已被 Git 忽略，用于新的运行结果，避免改写归档数据。

## V3：推荐入口

```bash
python experiments/v3/verify.py --model experiments/v3/recurrent-0.01.npz --reference experiments/v3/reference-hand.npz
python experiments/v3/benchmark.py --model experiments/v3/recurrent-0.01.npz --kind compiled_fast --plan experiments/v3/protocol.json --split heldout --output work/v3-ours
python experiments/v3/benchmark.py --model experiments/v3/reference-hand.npz --kind reference --plan experiments/v3/protocol.json --split heldout --output work/v3-reference
```

`--split stress` 使用预设压力测试；`--split reproduction` 使用公开基线复跑的种子。测试脚本保存每一局得分、统计量和第一局完整轨迹。

权重构造可独立运行：

```bash
python experiments/v3/build_weights.py --output work/rebuilt-v3
```

不传 `--output` 时会写入脚本所在目录。为保留归档权重，建议始终显式指定新的输出目录。

## V2

V2 保留六个候选。最终选择 `extended_old.npz`。

```bash
python experiments/v2/verify_revision.py
python experiments/v2/run_batch.py --minatar . --model experiments/v2/extended_old.npz --plan experiments/v2/protocol.json --split heldout --output work/v2-ours
python experiments/v2/run_batch.py --minatar . --model experiments/v2/original.npz --plan experiments/v2/protocol.json --split heldout --output work/v2-original
```

这里 `--minatar .` 将仓库根目录加入模块搜索路径，环境实际从当前虚拟环境中安装的 MinAtar 包导入。也可将其替换为 MinAtar 源码根目录。

## V1

```bash
python experiments/v1/verify.py
python experiments/v1/evaluate.py --minatar . --plan experiments/v1/protocol.json --result work/v1-rerun
```

V1 的脚本会运行协议中所有策略的主测试和压力测试。

## 统计与权重检查

```bash
python -m unittest discover -s tests -v
```

测试会检查记录的均分、样本量、模型哈希、V3 配对增益及重建权重。它们不重新运行数千局完整评测，也不能替代独立复现。

## 重建示例动图

```bash
python -m pip install Pillow
python scripts/render_replay.py --output work/replay.gif
```

动图只使用仓库内已记录的真实轨迹，并明确标为重绘与抽帧。这个步骤不影响模型成绩。

## 报告复现差异

请提供 Python/NumPy/SciPy/MinAtar 版本、模型文件 SHA256、完整命令、种子范围、`summary.json` 和是否使用快速后端。不要上传凭据或无关个人信息。
