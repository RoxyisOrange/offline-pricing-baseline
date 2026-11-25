"""Render the Chinese QA report directly from the measured results."""

from qa_statistics import EXTREME


def write_report(output, summary, scenarios, distribution, coverage):
    ready = summary["dataset_release_readiness"] == "READY"
    conclusion = ("**数据发布 READY。** 正式数据通过验收，可以冻结并分发，无需重新生成。"
                  if ready else "**数据发布 NOT READY。** 请通过异常表定位阻塞项，处理后重新验收。")
    lines = [
        "# E1–E4 正式数据质量报告", "",
        conclusion, "",
        "**模型实验 " + summary["model_experiment_readiness"] + "。** "
        "当前 Naive／Python DR 可作为已标注的兼容性基线。正式 100 轮比较应先确定 RSF 整数边界的处理和模型版本。"
        "E4 比较另需实现并核验 Censored Poisson。", "",
        "## 逐项验收", "",
        "| 检查 | 状态 |", "| --- | --- |",
    ]
    for key, check in summary["checks"].items():
        lines.append(f"| {key} {check['label']} | **{check['status']}** |")
    lines.extend([
        "", "## 01–02 数据范围与逐行规则", "",
        f"- 输入为 `{summary['data_path']}`。正式发布 ID 为 `{summary['dataset_sha256']}`。",
        f"- 逐个读取 **{summary['official_files']:,} 个正式 CSV**，共 "
        f"**{summary['official_physical_rows']:,} 行含附表**。主训练与共享测试共 "
        f"**{summary['official_main_rows']:,} 行**。同一潜在需求或测试环境可能在多个场景被引用。",
        f"- 单独读取 E0 CSV 副本 {summary['files_by_type'].get('e0_csv_copy', 0):,} 个 / "
        f"{summary['rows_by_type'].get('e0_csv_copy', 0):,} 行，以及 E0 原始压缩数据 "
        f"{summary['files_by_type'].get('e0_original_gzip', 0):,} 个 / "
        f"{summary['rows_by_type'].get('e0_original_gzip', 0):,} 行。"
        "归档验证器的历史预测复算登记在运行日志。E0 使用自己的发布 ID 和固定训练种子。",
    ])
    for key, name in (("train", "正式训练"), ("shared_test", "共享测试"),
                      ("training_truth", "训练真值附表"), ("test_diagnostics", "市场测试诊断")):
        lines.append(f"- {name} {summary['files_by_type'].get(key, 0):,} 份 × 2,000 行。")
    details = summary["official_row_details"]
    if details:
        lines.append(f"- 正式训练共核验 {details['zero_stock']:,} 条零库存记录与 {details['ties']:,} 条 "
                     "D=Y 记录。所用规则为 S=min(D,Y)、Δ=1(S<Y)。D=Y 属于截断，零库存和零销量均合法。")
    lines.extend([
        "", "## 03 数据生成参数与实际分布", "",
        "| 维度 | 论文 §2–3、§5.1–5.2 | 本组正式发布 |",
        "| --- | --- | --- |",
        "| 特征 | 两个均匀特征 X₁、X₂ | 从独立 U[0,1] 总体按 X₁²+X₂²≤2/π 分层，每份 Low、High 各 1,000 条。组内是条件分布 |",
        "| 历史价 | N(0.5, 0.5²) 条件截于 [0,1] | 保留参数与价格支持，用截断正态逆 CDF |",
        "| 需求 | D=1+Poisson(λ)，λ=5+X₁²+X₂²−5P | 增加 D=1+NB(k=2,p=k/(k+λ))。条件均值均为 1+λ，条件方差分别为 λ 与 λ+λ²/2 |",
        "| 库存 | §5.1 给出 Y=⌊N(6,2)⌋ | 2 解释为方差。测试均值参数固定 6，训练按市场、组别和目标截断率校准。先条件于非负，再向下取整，允许 0 |",
        "| 截断 | §2.2 观测定义为 S<Y | Δ=1 表示完整需求，Δ=0 包含 D=Y |",
        "",
        "通过实际文件逐行核验 λ、1+λ、需求抽样和库存分位数。特征/价格统计每轮只取一份配对样本。"
        "需求按市场各取一份配对训练样本，分 λ 区间核对条件残差。CDF 距离作描述性统计；分组内不要求独立均匀。"
        "条件截断率使用独立 Gauss–Legendre 积分核对 24 个校准键。", "",
    ])
    if not distribution.empty:
        train = distribution[(distribution.kind == "marginal") & (distribution.scope == "train_shared_once")].set_index("field")
        if {"x1", "x2", "price"}.issubset(train.index):
            price = train.loc["price"]
            corr = distribution[(distribution.kind == "feature_correlation")
                                & (distribution.scope == "train_shared_once") & (distribution.group == "Overall")]
            lines.append(f"- 训练配对特征 X₁、X₂ 均值分别为 {train.loc['x1','mean']:.6f}、"
                         f"{train.loc['x2','mean']:.6f}，合并相关系数 {corr.iloc[0].correlation:.6f}。"
                         f"价格均值 {price['mean']:.6f}，标准差 {price['std']:.6f}，截断后理论标准差 "
                         f"{price['theory_std']:.6f}，0 和 1 端点计数分别为 {price.at_zero:.0f}、{price.at_one:.0f}。")
        for row in distribution[(distribution.kind == "conditional_demand") &
                                (distribution.group == "all")].itertuples():
            lines.append(f"- {row.scope} 的条件平均残差 {row.mean_residual:+.6f}，残差平方均值 / "
                         f"理论条件方差均值 {row.second_moment_ratio:.6f}，n={int(row.n):,}。")
        calibration = distribution[distribution.kind == "inventory_calibration"]
        if len(calibration):
            error = calibration.deviation_pp.abs().max()
            lines.append(f"- {len(calibration)} 个库存校准键独立积分的最大绝对偏差为 {error:.5f} pp。")
    lines.extend(["", "## 04 截断率", ""])
    maximum = summary["max_censoring_deviation"]
    if maximum:
        lines.append(f"100 轮汇总最大绝对偏差 **{abs(maximum['signed_pp']):.4f} pp**，出现在 "
                     f"{maximum['scenario_id']} {maximum['group'].title()}，"
                     f"目标 {maximum['target_rate']:.2%}，实际 {maximum['actual_rate']:.4%}。"
                     "验收上限为 1 pp，逐轮波动详见附表。")
        lines.extend(["", "| 场景 | Low 目标→实际 | High 目标→实际 | Overall 实际 |",
                      "| --- | ---: | ---: | ---: |"])
        for row in scenarios.itertuples():
            lines.append(f"| {row.scenario_id} | {row.target_low_rate:.0%} → {row.actual_low_mean:.3%} | "
                         f"{row.target_high_rate:.0%} → {row.actual_high_mean:.3%} | {row.actual_overall_mean:.3%} |")
    lines.extend([
        "", "## 05–06 配对、共享测试与拟合隔离", "",
        "- 核对同轮全部场景的 sample_id、X₁、X₂、历史价格和组别，同一市场额外逐行核对 D、λ。"
        "不同轮次及训练与测试使用分开的 SeedSequence 键。代码中的特征、价格、两市场需求和库存分别用流号 "
        "10、20、30、31、40。独立重放验证全部训练库存与冻结均值和随机分位数一致。",
        "- 生成器按市场、组别和目标率以独立 Sobol 辅助样本校准。正式库存读取组别、预先固定的均值及独立随机分位数。"
        "给定 X、P 后，库存机制与该条需求的随机实现分离。边际相关性可由组别和特征产生；条件独立性的判断依据是生成机制。",
        "- 每轮共享测试表只供预测，两个市场的诊断表与它逐行对齐，记录各自历史价下的销量和截断。"
        "真值附表有 sample_id、D、λ、1+λ。训练读取器和真实试跑核查七字段白名单，sample_id 用于对齐，"
        "group 用于评价分组。策略输入 X₁、X₂、Y/10，Q 额外读取历史价。公共 CLI 在拟合完成后才打开测试或诊断表。",
        "", "## 07 极端场景的库存覆盖", "",
        "完整需求记录占比及库存覆盖回答两个不同的问题。模型仍可读取合法的零库存行。由于本发布 D≥1，"
        "Y≤1 且截断只提供一个已经满足的需求下界。下表只展示这些场景中目标为 90% 的商品组。",
        "",
    ])
    if not coverage.empty:
        lines.extend([
            "| 场景 | 90% 组 | 零库存／零销量 | 库存 ≤1 | 完整需求 | 同组测试超过训练库存上限 |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ])
        for row in coverage[coverage.scenario_id.isin(EXTREME)].itertuples():
            if row.group == "Overall" or (row.group == "Low" and row.scenario_id not in ("S05", "S16")):
                continue
            lines.append(f"| {row.scenario_id} | {row.group} | {row.zero_inventory_rate:.2%} | "
                         f"{row.inventory_at_most_one_rate:.2%} | {row.complete_rate:.2%} | "
                         f"{row.test_above_group_train_max_rate:.2%} |")
    lines.extend([
        "", "覆盖表和逐轮表还记录同轮全训练最大库存的覆盖、市场组别、同值率、实际损失量、"
        "截断后的平均未满足需求、库存分位数以及库存分布的总变差距离。90% 场景的覆盖限制属于设计风险。",
        "", "## 08 公共评价", "",
        "评价按推荐价 P 重新计算需求分布下的 E[P·min(D,Y)−0.1·max(D−Y,0)]，按各市场的分布"
        "在 [0,1] 的 0.001 网格求解析期望的最大值。历史价诊断按 E[D|X,P,D≥Y] 做真实需求恢复对照。"
        "利润比汇总采用跨轮的平均利润比值，Regret 为 Oracle 与方法的利润差。", "",
        "E0 存档和扩展均使用解析网格 Oracle。E0 是 Poisson，扩展按各场景市场分别计算。"
        "原论文 §5.2.2 在相同间距的网格上使用一个基于 50,000 条辅助样本的 MLP 来近似期望。E0 历史结果保持冻结。",
    ])
    math = summary["numeric_evaluation"]
    if math:
        lines.append(f"独立求和覆盖 {math['edge_cases']} 个常规与边界输入、{math['impossible_conditions']} 个不可能的条件事件、"
                     f"{math['oracle_grid_rows']} 个扩展 Oracle 状态、{math['e0_oracle_grid_rows']} 个 E0 Oracle 状态和 "
                     f"{math['public_evaluation_calls']} 次完整评价。在常规精度通过的检查中，期望利润、条件期望需求、Oracle 利润的最大绝对误差分别为 "
                     f"{math['max_profit_error']:.2e}、{math['max_censored_mean_error']:.2e}、"
                     f"{math['max_oracle_profit_error']:.2e}。输入覆盖 Y=0、1、P=0、1、λ 极小或为零及高尾部。")
        for market, entry in math["release_conditional_targets"].items():
            lines.append(f"- {market} 正式诊断表全部 {entry['n_censored']:,} 条截断记录的条件需求经独立求和核对。"
                         f"最小 λ 为 {entry['min_lambda']:.6f}，最大 Y 为 {entry['max_inventory']}，"
                         f"最大绝对误差 {entry['max_abs_error']:.2e}。")
        if math["off_release_warnings"]:
            lines.append("- 合成边界 **WARNING**。λ≈10⁻⁸、Y=3 时，NB 条件需求有约 10⁻⁸ 的数值误差，"
                         "结果略低于理论下界 3。相同 λ、Y=60 时，两市场的极小正尾概率均下溢为 0，"
                         "触发条件事件错误。正式诊断数据的条件需求逐行通过。若扩展到这类边界，应使用稳定的尾概率或概率比。")
    lines.extend(["", "## 09 发布来源和复现", "",
                  f"校验通过 {summary['checksums_passed']:,}/{summary['checksums_verified']:,} 条 "
                  "正式发布和 E0 CSV 发布校验项，覆盖数据文件和各自的元数据。E0 原始压缩数据另由其清单验证。"])
    provenance = summary["provenance"]
    if provenance:
        versions = "，".join(f"{key} {value}" for key, value in provenance["versions"].items())
        lines.append(f"生成器版本 2.0.0，Git 提交 `{provenance['code_revision']}`，{versions}。"
                     "主种子 20261125，独立 Sobol 校准种子 20261126，使用 2¹⁷ 个辅助点。"
                     f"在临时目录对 {'、'.join(provenance['scenario_ids'])} 的第 1、2 轮重新导出，"
                     f"{provenance['reproduced_byte_identical_files']} 个 CSV 逐字节一致。")
    lines.extend(["", "## 模型兼容性和正式运行条件", ""])
    pilot = summary["model_pilot"]
    if pilot:
        lines.append(f"- S05、S16 各 5 轮，完成 {pilot['completed']} 个场景轮次、"
                     f"{pilot['verified_method_fits']} 次方法拟合。每次保留 2,000 条观测。"
                     f"S05 每轮保留 {pilot['zero_rows']['S05']['min']}–{pilot['zero_rows']['S05']['max']} 条零库存行，"
                     f"S16 保留 {pilot['zero_rows']['S16']['min']}–{pilot['zero_rows']['S16']['max']} 条。"
                     f"实际推荐价格范围 [{pilot['price_min']:.6f}, {pilot['price_max']:.6f}]。"
                     "中间 RSF/代理利润/Q/DR 和最终利润比、利润及 Regret 有限。")
        bound = pilot["rsf_discrete_boundary"]
        lines.append(f"- RSF 整数边界 **{bound['status']}**。给现有代理利润函数输入已知生存曲线，"
                     f"D=1+Poisson(1)、Y=2 时隐含需求为 {bound['current']:.8f}，"
                     f"独立的 E[D|D≥Y] 为 {bound['correct_conditional_mean']:.8f}。如果 H(t)=Pr(D>t)，"
                     "分母应为 H(Y−1)。拟合叶子时，当前代码把在 Y 截断的记录计入同值事件的风险集；"
                     "还需明确同值事件与截断的顺序，以及有限观测销售上限之外的尾部处理。"
                     "Python ExtraTrees/叶节点 Nelson–Aalen 与论文 R ranger 的树分割也有差异。")
    lines.extend([
        "- Censored Poisson **NOT TESTED**。仓库缺少拟合实现。验收截断似然时应使用 Pr(D≥Y)，"
        "在 Poisson 和 NB 市场分别用当地的观测训练。E4 完整实验等待 CP 和 RSF 模型版本确认。",
        "- 可并行开始编写各组模型和用少量轮次验证调用。建议保留当前 Naive／Python DR 的版本标识，"
        "先确认离散 RSF 的统计处理再运行正式 100 轮对比。数据质量通过意味着可先统一分发数据。",
        "- 论文 Assumption 3(b) 对需求使用有界条件，仿真的 Poisson 和 NB 有理论上的无界上尾。"
        "这是理论条件和仿真分布之间的差异，保留在结果说明中。",
        "", "## Double Check 和附件", "",
    ])
    verify = summary["double_check"]
    if verify:
        lines.append(f"重读落盘表，复算 {verify['round_rows']:,} 个场景轮次、{verify['scenario_rows']} 个场景、"
                     f"{verify['coverage_rows']} 个覆盖统计行和 {verify['checksum_rows']:,} 条校验记录。"
                     "截断率、组别计数和两个发布 ID 与本次实际核查一致。")
    lines.extend([
        f"共记录 {summary['warned_issues']} 个 WARNING、{summary['failed_issues']} 个 FAIL。",
        "",
        "- [结构化状态](validation_summary.json) · [执行日志](validation.log) · [异常定位](anomalies.csv)",
        "- [场景汇总](scenario_summary.csv) · [逐轮统计](per_round_summary.csv) · [分布核对](distribution_summary.csv)",
        "- [库存覆盖](coverage_summary.csv) · [文件哈希](checksum_verification.csv)",
    ])
    if pilot:
        lines.append("- [试跑日志](model_pilot/model_pilot.log) · [方法拟合](model_pilot/model_pilot_per_run.csv) "
                     "· [逐轮评价](model_pilot/model_pilot_metrics.csv)")
    figures = {"censoring_by_round.png": "各场景截断率波动", "inventory_distribution.png": "极端场景库存分布",
               "inventory_coverage.png": "测试库存超过训练上限"}
    for file in summary["figures"]:
        lines.append(f"- [{figures[file]}](figures/{file})")
    (output / "DATA_QUALITY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
