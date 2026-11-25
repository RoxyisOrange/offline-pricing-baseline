# 离线定价仿真 · 小组实验指南

E1–E4 从同一份数据发布读取训练集和测试集。选好场景后，可用 `train.py` 跑现有的 Naive 和 Python DR，或用公共读取器训练自己的模型，再调用 `evaluate_round`。E0 的历史结果另有冻结归档。

**开跑状态。** 数据 QA 通过，可以使用本次发布的数据开始开发和小规模试跑。当前 Python DR 的 RSF 整数截断边界待确认。E4 所需的 Censored Poisson 尚待实现和验收。正式 100 轮比较前，请在组内固定要采用的模型版本。详见 [正式数据质量报告](reports/data_quality/DATA_QUALITY_REPORT.md)。

## 1. 拿到数据后先验证

1. 使用 Python 3.11。在仓库根目录创建环境。
2. 从负责人取得同一份 release 2.0.0，将 `train/`、`test/`、`ground_truth/` 和 `metadata/` 放在仓库的 `data/` 下。这些扩展实验的 CSV 随数据包分发。
3. 校验发布文件。全组使用相同的发布 ID，`run_manifest.json` 会记录本次训练实际使用的 ID。

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
(cd data && shasum -a 256 -c metadata/checksums.sha256)
```

本次正式发布 ID 为 `a896f4bd54d582eb3a55c472fc2cba8597b0979bdc2e47cb7f054d78c2355dd3`。日常实验直接使用分发好的 CSV。统一生成器由 `generate_data.py` 调用 [pricing_sim/experiment_data.py](pricing_sim/experiment_data.py)。正式目录的发布溯源、固定参数和校准结果在 `data/metadata/`。

## 2. 找到自己实验的训练目录

每个场景有 `train_001.csv` 至 `train_100.csv`。每份训练集和共享测试集均为 **2,000 条，Low 和 High 各 1,000 条**。目录名里的 `Lxx_Hxx` 是两组的目标截断率百分数。例如 `S07_L30_H70` 为 Low 30%、High 70%。

| 分工 | 需要的场景 | 建议由该组新跑的现有基线 |
| --- | --- | --- |
| E1 总体截断率 | S01、S02、S03、S04、S05 | S01–S05 |
| E2 同一总体截断率下的组间分布 | S03、S06、S07、S08、S09 | S06–S09，复用 E1 的 S03 |
| E3 总体截断率与组间不平衡的交互 | S02/S10/S11；S03/S06/S07；S04/S12/S13 | S10–S13，复用 E1 的 S02/S03/S04 和 E2 的 S06/S07 |
| E4.1 Poisson 市场方法对比 | S01、S03、S05、S07、S09 | 同版本的 Naive 和 DR 可复用；在这些场景另跑 Censored Poisson |
| E4.2 负二项市场稳健性 | S14、S15、S16、S17、S18 | 在 NB 场景拟合各方法；CP 沿用 Poisson 模型假设，并用 NB 市场的观测数据拟合 |

| 场景 | 目标 Low / High | 训练目录 |
| --- | ---: | --- |
| S01 | 10% / 10% | `data/train/poisson/S01_L10_H10/` |
| S02 | 30% / 30% | `data/train/poisson/S02_L30_H30/` |
| S03 | 50% / 50% | `data/train/poisson/S03_L50_H50/` |
| S04 | 70% / 70% | `data/train/poisson/S04_L70_H70/` |
| S05 | 90% / 90% | `data/train/poisson/S05_L90_H90/` |
| S06 | 40% / 60% | `data/train/poisson/S06_L40_H60/` |
| S07 | 30% / 70% | `data/train/poisson/S07_L30_H70/` |
| S08 | 20% / 80% | `data/train/poisson/S08_L20_H80/` |
| S09 | 10% / 90% | `data/train/poisson/S09_L10_H90/` |
| S10 | 20% / 40% | `data/train/poisson/S10_L20_H40/` |
| S11 | 10% / 50% | `data/train/poisson/S11_L10_H50/` |
| S12 | 60% / 80% | `data/train/poisson/S12_L60_H80/` |
| S13 | 50% / 90% | `data/train/poisson/S13_L50_H90/` |
| S14 | 10% / 10% | `data/train/negative_binomial/S14_L10_H10/` |
| S15 | 50% / 50% | `data/train/negative_binomial/S15_L50_H50/` |
| S16 | 90% / 90% | `data/train/negative_binomial/S16_L90_H90/` |
| S17 | 30% / 70% | `data/train/negative_binomial/S17_L30_H70/` |
| S18 | 10% / 90% | `data/train/negative_binomial/S18_L10_H90/` |

**同轮共用测试。** 所有 18 个场景和所有方法在第 `NNN` 轮使用 `data/test/shared/test_NNN.csv` 的同一批商品特征与库存。第 1 轮用 `test_001.csv`，第 2 轮用 `test_002.csv`。因此只有 **100 份共享测试，共 20 万条**。Poisson 和 NB 市场按各自的真实需求分布计算利润与 Oracle。1800 份训练集共 360 万条，主训练和共享测试合计 380 万条。

| 辅助目录 | 用途 |
| --- | --- |
| `data/test/diagnostics/{poisson,negative_binomial}/` | 每市场每轮一份历史价、库存、销量和截断指标，用于需求恢复和 Q 诊断，与同轮共享测试逐行对齐 |
| `data/ground_truth/<市场>/<场景>/` | 训练真值，由 `sample_id` 和对应场景轮次对齐，仅供 QA 和诊断 |
| `data/metadata/` | `scenario_manifest.csv`、`experiment_mapping.json`、固定参数、校准结果和 SHA-256 |
| `data/e0_frozen/` 和 `data/v1/` | E0 的只读 CSV 镜像与原始归档，供历史核验和显式重跑 |

## 3. 用现有模型训练

先用 1 轮验证读数和输出，再用 5 轮检查自己的场景。下面以 E2 新增的四个场景为例。每次给一个新的结果目录。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python train.py --scenarios S06 S07 S08 S09 --limit 1 --output results/e2-smoke
.venv/bin/python train.py --scenarios S06 S07 S08 S09 --limit 5 --output results/e2-pilot
```

- E1 可用 `--experiment E1`。E3 只跑新场景时用 `--scenarios S10 S11 S12 S13`。E4.2 的现有对照可用 `--experiment E4.2`。按 `--experiment E2`、`E3` 或 `E4.1` 选择时，CLI 会跑对应实验的全部场景，包括可复用的场景。
- 统一的训练入口只拟合 `naive` 和 `censoring_aware`（当前 Python DR），并载入固定的 [训练配置](config/baseline.yaml)与[各轮拟合种子](config/fit_seeds.json)。Oracle 由评价函数计算。
- 正式训练省略 `--limit`，使用新的输出目录。中断后用原命令、相同数据和输出目录加 `--resume`。数据包放在其他位置时加 `--data /path/to/release`。
- 组内共享同版本的 `run_manifest.json` 和 `metrics_per_run.csv`。复用场景时核对数据、配置、模型版本和轮次，同一场景／方法／轮次只保留一份成绩。代码版本用运行时的 Git commit 记录。

## 4. 用自己模型调用公共评价

训练用 [data_loader.py](data_loader.py) 的 `open_dataset()` 或 `load_train()`。公共读取器的训练输出只有 `sample_id, x1, x2, price, inventory, sales, delta`。其中 `sample_id` 用于对齐，组别用于评价汇总。先完成拟合与调参，再读共享测试和诊断表。真值、Oracle、测试数据和诊断数据均不可用于拟合或选择超参数。

以下用常数 0.5 推荐价演示完整评分调用。接入自己的模型时，将 `price` 替换成拟合后对相同测试行的推荐价向量。

```python
import numpy as np
from data_loader import open_dataset
from pricing_sim.experiment_evaluation import Predictions, evaluate_round, summarize_metrics

data = open_dataset()
scenario_id, round_id = "S07", 1
train = data.training(scenario_id, round_id)  # 自有模型在这里用 train 拟合
test = data.test(round_id)
price = np.full(len(test), 0.5)  # 顺序与 test 一致；每行一个 [0, 1] 内的价格
oracle_cache = {}
scores = evaluate_round(
    data, scenario_id, round_id,
    {"example_policy": Predictions(price=price)},
    cache=oracle_cache,
)
print(scores[["group", "method", "profit", "profit_ratio", "regret"]])
```

- `Predictions` 可附加 `censored_mean`（市场诊断表 `delta=0` 行的顺序）及 `q_logged`（该诊断表全部行的顺序、历史价下的 Q）。如果使用了原来的 `PricingPipeline`，也可把已拟合的模型字典传给同模块的 `evaluate_fitted_models(data, scenario_id, round_id, models)`。
- 重复评价可复用同一个 `cache` 字典，Oracle 按发布版本、轮次和市场分别缓存。各方法的利润在推荐的新价格下求真实期望，测试诊断的需求真值为 E[D｜X,P,D≥Y]。
- `metrics_per_run.csv` 保存按场景、轮次、方法和 `Overall / Low / High` 划分的结果。`metrics_summary.csv` 保存跨轮估计、标准误和 95% CI。跨轮利润比是平均利润 / 平均 Oracle 利润，保存为小数，乘 100 得到百分比；两种方法的百分比相减得到百分点。
- 将相同发布版本的逐轮结果合并后可用 `summarize_metrics(combined)` 汇总。用 `paired_contrast` 对齐同轮比较利润、Regret 等非比值指标。E2、E3 保留 Low 和 High 的结果。

## 5. QA、冻结的 E0 与代码入口

正式数据核对见 [质量报告](reports/data_quality/DATA_QUALITY_REPORT.md)，包括 [逐轮统计](reports/data_quality/per_round_summary.csv)、[库存覆盖](reports/data_quality/coverage_summary.csv) 和 [实际运行日志](reports/data_quality/validation.log)。完整 QA 还需 E0 的 `data/e0_frozen/` CSV 镜像及仓库自带的 `data/v1/`。重新运行审核时，输出到新目录。默认包含 S05 和 S16 各 5 轮真实模型试跑，只检查数据时可加 `--skip-model-pilot`。逐字节复现检查使用发布时的 Python 3.11.7 与 `requirements.txt` 中固定的 NumPy、Pandas、SciPy。

```bash
.venv/bin/python -m pip install matplotlib==3.8.0
.venv/bin/python scripts/check_data_quality.py --data data --output results/qa-recheck
```

E0 保存在 [历史结果目录](results/baseline-2026-11-25/)，是 **100×100 Q 网络、3 折、100 次重复**的 Python 生存近似基线。Oracle 平均期望利润 2.126574，Naive 利润比 95.3535%，DR 95.9894%，相差 0.6359 pp。该版用解析网格 Oracle；论文 §5.2.2 用 MLP 近似 Oracle，生存森林用 R `ranger`。E0 核验用 `.venv/bin/python verify_baseline.py`。需要重跑时显式使用 `train.py --replay-e0 --limit 1 --output results/e0-replay`，预算与扩展实验分开。

| 工作 | 入口 |
| --- | --- |
| 统一发布和读取 | [generate_data.py](generate_data.py)、[data_loader.py](data_loader.py) |
| 现有模型的训练和结果输出 | [train.py](train.py)、[pricing_sim/learning.py](pricing_sim/learning.py) |
| 公共评价和跨轮统计 | [pricing_sim/experiment_evaluation.py](pricing_sim/experiment_evaluation.py) |
| 正式数据复核与 E0 核验 | [scripts/check_data_quality.py](scripts/check_data_quality.py)、[verify_baseline.py](verify_baseline.py) |
