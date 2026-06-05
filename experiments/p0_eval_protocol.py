"""
P0：建立统一评测基准

目标：
  1. 在新 Rolling Evaluation Protocol 下重新评测所有历史实验（R/B/O/T/C 系列）
  2. 以 R02_ridge_legacy_plus_norm_core 为 baseline，输出跨 profile 的 leaderboard
  3. 可选加入 11月 review holdout 和 12月 final holdout

运行方式：
  # 日常筛选（默认）：short + long 快车道，~10-15 min
  cd MEOW--predict
  PYTHONPATH=src python experiments/p0_eval_protocol.py --suite daily

  # P1–P3 候选筛选：daily 快车道 + 指定候选（baseline 自动并入算 delta）
  PYTHONPATH=src python experiments/p0_eval_protocol.py --suite daily \\
    --spec-ids O1_R02_plus_ofi_raw O2_R02_plus_ofi_dynamic

  # 快速验证（每个 profile 只跑 2 个 fold，只跑 Ridge 系列）
  PYTHONPATH=src python experiments/p0_eval_protocol.py --suite quick

  # Ridge 全四 profile 重建基线
  PYTHONPATH=src python experiments/p0_eval_protocol.py --suite ridge

  # 全部历史实验重新评测（含 O/T/C 系列，耗时较长）
  PYTHONPATH=src python experiments/p0_eval_protocol.py --suite full

  # 指定特定 profile + 含 review holdout
  PYTHONPATH=src python experiments/p0_eval_protocol.py --suite ridge \\
    --profiles short_8d_2d medium_20d_5d --include-review-holdout

  # 完整协议含 final holdout（慎用，少跑 final）
  PYTHONPATH=src python experiments/p0_eval_protocol.py --suite full \\
    --include-review-holdout --include-final-holdout

输出目录结构：
  results/eval_protocol/<run_id>/
    config.json
    fold_manifest.csv
    fold_metrics.csv
    profile_summary.csv
    leaderboard.csv
    review_holdout.csv    （如果 --include-review-holdout）
    final_holdout.csv     （如果 --include-final-holdout）
"""

import argparse
import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT / "src"))

from eval_protocol import (
    EvaluationProtocolRunner,
    ROLLING_PROFILES,
    ALL_SPECS,
    RIDGE_SPECS,
    BASELINE_ID,
)
from experiment_runner import ExperimentRunner


# ================================================================== #
# 默认配置
# ================================================================== #

DATA_DIR = str(PROJ_ROOT / "data")
OUTPUT_DIR = str(PROJ_ROOT / "results" / "eval_protocol")
FEATURE_DIR = str(PROJ_ROOT / "data" / "features")

# 第一层 rolling 区间（内部选模型主依据）
ROLLING_START = 20230601
ROLLING_END = 20231031

# 第二层：11月 review holdout
REVIEW_TRAIN_START = 20230601
REVIEW_TRAIN_END = 20231031
REVIEW_HOLDOUT_START = 20231101
REVIEW_HOLDOUT_END = 20231130

# 第三层：12月 final holdout（尽量少跑）
FINAL_TRAIN_START = 20230601
FINAL_TRAIN_END = 20231130
FINAL_HOLDOUT_START = 20231201
FINAL_HOLDOUT_END = 20231229


# ================================================================== #
# CLI
# ================================================================== #

def build_arg_parser():
    parser = argparse.ArgumentParser(description="P0 Rolling Evaluation Protocol")
    parser.add_argument(
        "--suite",
        type=str,
        default="daily",
        choices=["quick", "daily", "gate", "ridge", "full"],
        help="daily=日常筛选(short+long 快车道, 默认); gate=关口(expanding 候选+基线 2 spec); "
             "quick=2折Ridge调试; ridge=Ridge 全四 profile 重建基线; full=全部历史实验",
    )
    parser.add_argument(
        "--profiles",
        nargs="*",
        default=None,
        help="指定 profile 名称列表覆盖 suite 默认（如单独 ad-hoc 跑 medium_20d_5d / expanding_40d_5d）",
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="限制每个 profile 的 fold 数（调试用，默认不限制）",
    )
    parser.add_argument(
        "--include-review-holdout",
        action="store_true",
        help="同时运行 11月 review holdout",
    )
    parser.add_argument(
        "--include-final-holdout",
        action="store_true",
        help="同时运行 12月 final holdout（谨慎使用）",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=4,
        help="并发 worker 进程数（默认 4，设为 1 退回串行模式；16 GB Mac 不建议超过 4）",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点续跑：跳过已完成的 job（需与上次相同的 --run-id）",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="固定 run_id（resume 时必须与上次一致）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help="输出目录（默认 results/eval_protocol）",
    )
    parser.add_argument(
        "--h5dir",
        type=str,
        default=DATA_DIR,
        help="数据目录",
    )
    parser.add_argument(
        "--feature-dir",
        type=str,
        default=FEATURE_DIR,
        help="特征缓存目录（PE1/M4 起评测主链路依赖该目录）",
    )
    parser.add_argument(
        "--candidate-spec-id",
        type=str,
        default=None,
        help="关口模式下要评估的候选 experiment_id；suite=gate 时必填",
    )
    parser.add_argument(
        "--spec-ids",
        nargs="*",
        default=None,
        help="daily 模式下显式指定本批候选 experiment_id（如 P1 的 O1..O6）；"
             "会自动并入 baseline 以在同表算 delta。不给则跑默认 RIDGE_SPECS。",
    )
    parser.add_argument(
        "--baseline-spec-id",
        type=str,
        default=BASELINE_ID,
        help="关口模式下对照基线 experiment_id（默认当前 baseline）",
    )
    parser.add_argument(
        "--target-winsorize",
        type=str,
        default="on",
        choices=["on", "off"],
        help="训练标签 winsorize 开关：on=开启（默认，当前工作口径），off=关闭做对照",
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=2.0,
        help="标准 ridge 主路径的 alpha（默认 2.0；P0.5 扫描时由外部显式覆写）",
    )
    parser.add_argument(
        "--target-winsor-lower-q",
        type=float,
        default=0.01,
        help="训练标签 winsorize 下分位（默认 0.01，对应 P1）",
    )
    parser.add_argument(
        "--target-winsor-upper-q",
        type=float,
        default=0.99,
        help="训练标签 winsorize 上分位（默认 0.99，对应 P99）",
    )
    parser.add_argument(
        "--train-subsample-frac",
        type=float,
        default=1.0,
        help="训练行降采样比例（默认 1.0=全量、不影响既有路径）。P4 树初筛提速用，"
             "如 0.33 表示每折只用 1/3 训练行；仅采训练，验证集全量不动、排名指标不受污染。",
    )
    parser.add_argument(
        "--dump-oof",
        action="store_true",
        help="落盘逐行 OOF 验证预测到 run 目录下 oof/（P5 加权融合/stacking 用）；"
             "仅串行（--n-workers 1）支持，并行模式会告警跳过。",
    )
    return parser


def build_target_winsorize_config(args):
    """
    把 CLI 参数收敛成一份可直接传给 ExperimentRunner 的配置。

    这里故意不做“自动猜测”：
    - `on/off` 明确表达用户是否要裁训练标签
    - 分位数显式透传给 runner / scheduler / worker，保证真实跑数全链路一致
    """
    return {
        "enabled": args.target_winsorize == "on",
        "lower_quantile": float(args.target_winsor_lower_q),
        "upper_quantile": float(args.target_winsor_upper_q),
    }


def resolve_specs_by_ids(spec_ids):
    """
    按 experiment_id 解析 spec 列表。

    这里优先从 ALL_SPECS 建索引，避免关口模式靠切片猜“第几个 spec 是谁”。
    """
    spec_map = {spec["experiment_id"]: spec for spec in ALL_SPECS}
    missing = [spec_id for spec_id in spec_ids if spec_id not in spec_map]
    if missing:
        raise ValueError(f"未找到 experiment_id: {missing}")
    return [spec_map[spec_id] for spec_id in spec_ids]


def build_daily_specs(spec_ids, baseline_id, default_specs=None):
    """
    daily 快车道的候选选择。

    - 未显式给 spec_ids（None / 空）：回退默认（RIDGE_SPECS），保持历史行为不变。
    - 显式给 spec_ids：把 baseline 放在首位后并入候选（去重、保序），交给
      resolve_specs_by_ids 解析；这样 P1–P3 的 O/T/C 候选才能在 short+long 快车道上
      与同口径 baseline 同批跑、同表算 delta（§4.4 / §5.1）。
    """
    if not spec_ids:
        return list(default_specs if default_specs is not None else RIDGE_SPECS)
    ids = [baseline_id] + [sid for sid in spec_ids if sid != baseline_id]
    return resolve_specs_by_ids(ids)


def main():
    args = build_arg_parser().parse_args()
    target_winsorize_config = build_target_winsorize_config(args)

    # profile 名 → 对象映射，供 daily 选取与 --profiles 覆盖
    profile_map = {p.profile_name: p for p in ROLLING_PROFILES}
    daily_profiles = [profile_map[n] for n in ("short_8d_2d", "long_40d_5d") if n in profile_map]

    # 选择要运行的 specs 与 profiles
    if args.suite == "quick":
        specs = RIDGE_SPECS[:3]           # R00/R01/R02，快速验证
        max_folds = args.max_folds or 2   # 默认只跑 2 折
        selected_profiles = ROLLING_PROFILES[:1]  # 只跑 short profile
        effective_n_workers = args.n_workers
        print("[P0] 快速调试模式：R00-R02，short_8d_2d profile，2 folds")
    elif args.suite == "daily":
        # 给了 --spec-ids 就跑指定候选 + baseline，否则默认 Ridge 系列
        specs = build_daily_specs(args.spec_ids, args.baseline_spec_id)
        max_folds = args.max_folds        # None = 全部 fold
        selected_profiles = daily_profiles  # 快车道：short + long（medium/expanding 不进日常）
        effective_n_workers = args.n_workers
        if args.spec_ids:
            print(f"[P0] 日常筛选 suite：指定候选 {args.spec_ids} + baseline {args.baseline_spec_id}，short + long 快车道")
        else:
            print("[P0] 日常筛选 suite：short + long 快车道（medium 移出日常、expanding 只在关口跑）")
    elif args.suite == "gate":
        if not args.candidate_spec_id:
            raise ValueError("suite=gate 时必须提供 --candidate-spec-id")
        gate_spec_ids = [args.baseline_spec_id, args.candidate_spec_id]
        specs = resolve_specs_by_ids(gate_spec_ids)
        max_folds = args.max_folds
        selected_profiles = [
            profile_map["expanding_40d_5d"]
        ] if "expanding_40d_5d" in profile_map else ROLLING_PROFILES
        # #17：关口模式默认压到 2 worker；若调用方显式给 1，则保留串行排障能力。
        effective_n_workers = min(args.n_workers, 2)
        print(
            "[P0] 关口 suite：只跑 baseline + candidate 两条 spec，默认 expanding profile，"
            f"candidate={args.candidate_spec_id}, baseline={args.baseline_spec_id}"
        )
    elif args.suite == "ridge":
        specs = RIDGE_SPECS               # R00-R04 完整 Ridge 系列
        max_folds = args.max_folds        # None = 全部 fold
        selected_profiles = ROLLING_PROFILES
        effective_n_workers = args.n_workers
        print("[P0] Ridge 全 profile 重建：R00-R04，四个 profiles，全量 folds")
    else:  # full
        specs = ALL_SPECS
        max_folds = args.max_folds
        selected_profiles = ROLLING_PROFILES
        effective_n_workers = args.n_workers
        print(f"[P0] 完整评测：{len(specs)} 个历史实验，四个 profiles")

    # 按用户指定过滤 profile
    if args.profiles:
        profile_map = {p.profile_name: p for p in ROLLING_PROFILES}
        selected_profiles = [profile_map[name] for name in args.profiles if name in profile_map]
        if not selected_profiles:
            print(f"[P0] 警告：指定的 profiles {args.profiles} 无效，使用全部 profiles")
            selected_profiles = ROLLING_PROFILES

    print(f"\n[P0] 数据目录: {args.h5dir}")
    print(f"[P0] 特征目录: {args.feature_dir}")
    print(f"[P0] 输出目录: {args.output_dir}")
    print(f"[P0] rolling 区间: {ROLLING_START} ~ {ROLLING_END}")
    print(f"[P0] 实验数: {len(specs)}, profiles: {[p.profile_name for p in selected_profiles]}")
    print(f"[P0] max_folds: {max_folds or '全量'}")
    print(
        f"[P0] n_workers: {effective_n_workers}"
        f"{'（串行）' if effective_n_workers == 1 else '（并行）'}"
    )
    print(
        "[P0] target winsorize: {enabled} q=({lower:.3f}, {upper:.3f})".format(
            enabled=target_winsorize_config["enabled"],
            lower=target_winsorize_config["lower_quantile"],
            upper=target_winsorize_config["upper_quantile"],
        )
    )
    print(f"[P0] ridge alpha: {args.ridge_alpha}")
    if args.resume:
        print(f"[P0] resume 模式：跳过已完成 job")

    runner = ExperimentRunner(
        args.h5dir,
        feature_dir=args.feature_dir,
        target_winsorize_config=target_winsorize_config,
        ridge_alpha=args.ridge_alpha,
        train_subsample_frac=args.train_subsample_frac,
    )
    protocol = EvaluationProtocolRunner(runner)

    result = protocol.run_full_protocol(
        rolling_start=ROLLING_START,
        rolling_end=ROLLING_END,
        specs=specs,
        profiles=selected_profiles,
        max_folds=max_folds,
        include_review_holdout=args.include_review_holdout,
        review_train_start=REVIEW_TRAIN_START,
        review_train_end=REVIEW_TRAIN_END,
        review_holdout_start=REVIEW_HOLDOUT_START,
        review_holdout_end=REVIEW_HOLDOUT_END,
        include_final_holdout=args.include_final_holdout,
        final_train_start=FINAL_TRAIN_START,
        final_train_end=FINAL_TRAIN_END,
        final_holdout_start=FINAL_HOLDOUT_START,
        final_holdout_end=FINAL_HOLDOUT_END,
        baseline_id=args.baseline_spec_id,
        n_workers=effective_n_workers,
        resume=args.resume,
        output_dir=args.output_dir,
        run_id=args.run_id,
        dump_oof=args.dump_oof,
    )

    # 打印关键结果
    lb = result["leaderboard"]
    if not lb.empty:
        display_cols = [
            "experiment_id", "protocol_corr_mean", "protocol_stability_score",
            "protocol_daily_ic_mean", "protocol_daily_ic_ir",
            "protocol_corr_min", "protocol_positive_fold_rate",
            "baseline_delta_corr", "decision",
        ]
        display_cols = [c for c in display_cols if c in lb.columns]
        print("\n" + "=" * 80)
        print("Leaderboard（按 protocol_corr_mean 降序；stability / 每日 IC-IR 为并排守门指标）")
        print("=" * 80)
        print(lb[display_cols].to_string(index=False))
    else:
        print("\n[P0] 警告：leaderboard 为空，请检查数据和配置")

    print(f"\n[P0] 完成。run_id={result['run_id']}")


if __name__ == "__main__":
    main()
