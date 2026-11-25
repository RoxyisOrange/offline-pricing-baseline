# E1–E4 公用数据

本目录有 18 个训练场景和 100 轮，每份训练集、共享测试集均有 2000 行，Low 和 High 各 1000 行。

- `train/` 按市场及场景保存观测训练数据。同一场景和轮次供所有方法复用。
- `test/shared/` 每轮一份，共用商品特征和库存。
- `test/diagnostics/` 保存两个市场各自的评价附表。
- `ground_truth/` 保存仅用于评价和核验的训练真值。按场景、轮次和 `sample_id` 对齐。
- `metadata/` 保存参数、校准、逐轮数据质量和校验和。库存覆盖率按当轮训练样本的最大库存计算。

组员可调用仓库 `data_loader.py` 的 `load_train`、`load_test` 和 `get_scenarios`。
普通训练读取 `train/` 的观测列。共享测试、诊断和真值属于评价环节。
在本目录运行 `shasum -a 256 -c metadata/checksums.sha256` 核验本次发布文件。
E0 CSV 副本位于单独的 `e0_frozen/`。
