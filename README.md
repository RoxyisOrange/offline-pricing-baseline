# Python 生存近似定价基线

本仓库只保留一次已经完成的仿真实验：100 次独立重复，每次训练和测试各 2,000 条记录，Q 网络为 100×100，交叉拟合为 3 折。对照 Oracle、Naive 和截断感知方法（DR）。Naive 奖励是 `价格 × 销量`；DR 使用 Python 的极随机树和叶节点 Nelson–Aalen 曲线近似生存森林。它不等同于论文所用 R `ranger`，100×100 也不是论文明确声明的主实验 Q 网络。

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
pricing_sim/evaluation.py            唯一的评估指标入口
generate_data.py                     一次性生成 data/v1
train.py                             使用冻结数据训练，新输出目录
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
