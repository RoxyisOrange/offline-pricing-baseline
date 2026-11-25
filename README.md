# 离线定价仿真实验

仓库包含已冻结的 E0 复现和 E1–E4 的公共数据与评价接口。E0 有 100 次独立重复，每次训练和测试各 2,000 条记录，Q 网络为 100×100，交叉拟合为 3 折。对照 Oracle、Naive 和截断感知方法（DR）。Naive 奖励是 `价格 × 销量`；DR 使用 Python 的极随机树和叶节点 Nelson–Aalen 曲线近似生存森林。它不等同于论文所用 R `ranger`，100×100 也不是论文明确声明的主实验 Q 网络。

## 存档结果

| 方法 | 平均期望利润 | Oracle 利润比 |
| --- | ---: | ---: |
| Oracle | 2.126574 | 100.0000% |
| Naive | 2.027762 | 95.3535% |
| DR（Python 近似） | 2.041285 | 95.9894% |

DR 比 Naive 高 0.6359 个百分点。逐次配对利润比差的 95% 区间约为 −0.26 至 +1.53 个百分点；100 次中 DR 有 60 次更高。`results/baseline-2026-11-25/` 保存完整的逐次指标、20 万条测试预测和诊断。这些是从先前完成的 `q100_folds3_full` 实验中恢复的历史结果；本次打包没有重新训练全部 100 次。前 20 次实验曾复用相同设置和种子的检查点。

## 结构

```text
config/baseline.yaml                 固定实验配置与种子
pricing_sim/simulation.py            数据生成规则及真实期望利润
pricing_sim/data_store.py            冻结、读取和校验数据
pricing_sim/survival.py              Python 生存近似
pricing_sim/learning.py              密度、Q、策略和 Oracle 模型
pricing_sim/evaluation.py            E0 的冻结评价接口
pricing_sim/experiment_data.py       E1–E4 的场景表、公共生成器与数据读取器
pricing_sim/experiment_evaluation.py E1–E4 的公共评价函数
generate_data.py                     生成 E0 或选定的 E1–E4 数据
train.py                             读取冻结数据，训练 Naive 与 DR
verify_baseline.py                   核对数据与存档结果
data/v1/                             100 组数据、统计及 SHA-256 清单
results/baseline-2026-11-25/         100 次存档结果
tests/                                核心不变量测试
```

`data/v1/runs/run_XXX/` 下每组有 `observed_train.csv.gz`、`observed_test.csv.gz`、`truth_train.csv.gz`、`truth_test.csv.gz`。模型入口默认只加载观测数据，真实需求只供检查。`manifest.json` 记录种子和全部 400 个数据文件的 SHA-256；`statistics.csv` 给出各组截断率、平局率等。库存生成使用均值 6、方差 2 的正态变量向下取整；未截断定义固定为 `D < Y`。

## 运行

使用 Python 3.11；`requirements.txt` 固定依赖。在仓库根目录执行：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python verify_baseline.py
```

只跑第 0 组以检查训练流程：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python train.py --output results/local-smoke --limit 1
```

运行 100 组时省略 `--limit` 并指定新的输出目录；中断后以相同参数加 `--resume`。脚本拒绝覆盖存档结果。数据生成器已经用于生成 `data/v1`，平时训练不需要调用；再次运行默认生成命令会因目录存在而失败。未来修改数据规则时应建立 `data/v2`，保留旧版。

本地 Git 提交和标签 `baseline-v1` 标识本基线。GitHub CI 检查测试、存档结果和相对基线标签的数据与核心代码差异。若只拿到不含 `.git` 的源码 ZIP，可使用同时交付的 Git bundle 恢复完整 Git 历史。

## E1–E4 共用数据

全组由同一人导出一份数据，其他成员读取相同的目录和 `manifest.json`。生成器版本 `2.0.0` 按固定总体中位数 `x1²+x2² ≤ 2/π` 分组，训练和测试每轮均分层抽取 Low、High 各 1000 条。Poisson 与 NB 的条件需求均值都是 `1+λ`，NB 离散参数 `k=2`。同一市场和轮次下的场景共享特征、历史价格、需求抽样和库存分位数。Poisson 与 NB 共用特征、历史价格及库存分位数，各自抽样需求。

历史库存用独立 Sobol 样本校准组内平均截断率，只调整库存正态的均值参数。库存方差为 2，负的连续值重抽，保留零库存，`sales=min(D,Y)`，`delta=1(sales<Y)`。每轮只有一份共用的未来特征与库存，测试库存沿用 E0 的机制（均值参数 6、方差 2）。校准、分层抽样及 NB 市场是本组的扩展设定。

```bash
.venv/bin/python generate_data.py --self-test --rounds 100
.venv/bin/python generate_data.py --experiment all --rounds 5 --output datasets/group-pilot
.venv/bin/python generate_data.py --experiment all --rounds 100 --output datasets/group-v2
```

`--experiment` 接受 `E1`、`E2`、`E3`、`E4.1`、`E4.2` 及 `all`。小范围检查也可用 `--scenarios S01 S16`。正式的 `all` 包含 13 个 Poisson 场景、5 个 NB 场景和 100 份共享测试集。生成时指定新目录，导出完成后保存生成器版本、种子、场景和逐文件 SHA-256。

| 实验 | 使用的场景 |
| --- | --- |
| E1 | S01–S05 |
| E2 | 复用 S03，新增 S06–S09 |
| E3 | 复用 S02、S03、S04、S06、S07，新增 S10–S13 |
| E4.1 | 复用 S01、S03、S05、S07、S09 |
| E4.2 | S14–S18 |

例如 `S01` 对应 `poisson/10_10/train_001.csv`，`S16` 对应 `negative_binomial/90_90/train_001.csv`。两者共用 `shared_test/test_001.csv`。`actual_censoring_rates.csv` 保存每组及总体的实际截断率、库存均值参数和零库存比例。`evaluation_only/` 保存核验用训练真值，以及用于需求误差和 Q 误差的测试观测。两种市场的测试观测各自使用独立于训练的需求抽样，共享测试的特征与库存。

## E1–E4 共用训练与评价

用以下命令选取场景，读取导出的 5 轮数据并调用已有的 Naive 和 DR。训练只接触 `ExperimentData.training(scenario_id, round_id)` 返回的 7 个观测字段。训练配置与每轮拟合种子来自已冻结的 E0 配置和清单。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python train.py \
  --data datasets/group-pilot --scenarios S01 S16 --limit 5 --output results/group-pilot
```

脚本输出 `metrics_per_run.csv` 与 `metrics_summary.csv`，按 `scenario_id`、`market`、`round_id`、`method` 和 `group` 汇总。中断后加 `--resume` 复用同批输入的已完成轮次。

在 Python 里直接调用 `evaluate_fitted_models(data, scenario_id, round_id, models)` 可评价已拟合的 E0 模型。E4 组员可将自有模型的输出封装成 `Predictions`，调用 `evaluate_round(data, scenario_id, round_id, predictions)`，复用同一把尺子。两种函数接收 `cache={}`，把同一字典传给多场景调用可复用每轮、每个市场的 Oracle。

- 利润按推荐价下各市场的真实期望计算。Oracle 在 `0.001` 价格网格上最大化这个期望，沿用 E0 的解析网格约定。
- 利润比以小数保存，汇总采用论文式 (23) 的跨轮平均利润之比。分组 Oracle 平均利润非正时，利润比留空。价格误差、Regret 和相对 Naive 的修正价值也按总体与各组报告。
- 需求恢复 MAE 只在共享诊断表的 `delta=0` 行计算，以真实条件均值 `E[D | X,P,D≥Y]` 为目标。Q MAE 在同一诊断表的历史价上计算，E0 模型的 Q 预测取三个拟合折的平均。
- `summarize_metrics` 输出跨轮均值、标准误和 95% 区间。`paired_contrast` 支持同一轮的方法差及 E3 的场景差，通过场景和方法的权重定义对比。
