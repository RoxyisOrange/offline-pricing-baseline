# E0 baseline-v1 冻结 CSV 副本

数据来自已校验的 `data/v1` 历史 E0。复制时保留原始行顺序、样本 ID 和数值。
原来的 `run_000` 对应本目录第 1 轮，依次到第 100 轮。
Low 和 High 由原特征打标签，人数沿用历史 E0 的自然分组。
测试主表和 Poisson 诊断附表使用原来同一批 E0 测试记录。
本副本可由公共读取器读取。存档训练和结果仍由 `baseline-v1` 标识。

在本目录运行 `shasum -a 256 -c metadata/checksums.sha256` 核验副本。
