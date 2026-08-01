from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from .io.generate_data import generate_rules, save_rules_json
from .io.load_rules import load_rules, export_rules_json
from .io.export_results import save_json, ensure_dir
from .preprocessing.validators import validate_rules
from .preprocessing.anomaly_checks import rule_specificity
from .evaluation.traffic import generate_traffic
from .evaluation.policy_reference import make_reference_policy, apply_order
from .evaluation.metrics import compute_metrics
from .evaluation.fitness import FitnessConfig
from .ga.engine import run_ga, GAConfig
from .visualization.plots import plot_fitness, plot_before_after_bar, plot_multi_before_after
from .visualization.report_pdf import ReportInputs, generate_report_pdf


def _specificity_baseline(rules) -> np.ndarray:
    scores = np.array([rule_specificity(r) for r in rules], dtype=float)
    return np.argsort(-scores).astype(np.int32)


def run_pipeline(
    rules_path: Path,
    out_dir: Path,
    n_packets: int,
    seed: int,
    ga_cfg: GAConfig,
    fit_cfg: FitnessConfig,
    student_name: str,
) -> None:
    ensure_dir(out_dir)
    fig_dir = ensure_dir(out_dir / "figures")

    rules = load_rules(rules_path)
    validate_rules(rules)

    traffic = generate_traffic(n_packets=n_packets, seed=seed)
    ref = make_reference_policy(rules, traffic)
    n_rules = len(rules)

    original = np.arange(n_rules, dtype=np.int32)
    baseline = _specificity_baseline(rules)

    dec_o, pos_o, avg_o = apply_order(original, ref.match_matrix, ref.actions)
    m_o = compute_metrics(dec_o, ref.ref_decisions, pos_o, avg_o, n_rules)

    dec_b, pos_b, avg_b = apply_order(baseline, ref.match_matrix, ref.actions)
    m_b = compute_metrics(dec_b, ref.ref_decisions, pos_b, avg_b, n_rules)

    ga_res = run_ga(ref, ga_cfg=ga_cfg, fit_cfg=fit_cfg, seed=seed)
    dec_g, pos_g, avg_g = apply_order(ga_res.best_ordering, ref.match_matrix, ref.actions)
    m_g = compute_metrics(dec_g, ref.ref_decisions, pos_g, avg_g, n_rules)

    save_json(out_dir / "ordering_original.json", original.tolist())
    save_json(out_dir / "ordering_specificity_baseline.json", baseline.tolist())
    save_json(out_dir / "ordering_ga_best.json", ga_res.best_ordering.tolist())

    # Extra convenience outputs for demos: the full ordered rules (human-readable).
    # This makes it easy to inspect the final optimized ruleset, not just indices.
    export_rules_json([rules[i] for i in original], out_dir / "rules_original_order.json")
    export_rules_json([rules[i] for i in ga_res.best_ordering], out_dir / "rules_ga_ordered.json")

    metrics_payload = {
        "original": m_o.to_dict(),
        "specificity_baseline": m_b.to_dict(),
        "ga_best": m_g.to_dict(),
        "ga_best_fitness": ga_res.best_fitness,
    }
    save_json(out_dir / "metrics.json", metrics_payload)

    plot_fitness(ga_res.history_best, ga_res.history_mean, fig_dir / "fitness_evolution.png")
    plot_before_after_bar(
        m_o.to_dict(),
        m_g.to_dict(),
        key="avg_checks",
        title="Average Rules Checked per Packet",
        out_path=fig_dir / "avg_checks_before_after.png",
        ylabel="Avg Checks",
    )
    plot_before_after_bar(
        m_o.to_dict(),
        m_g.to_dict(),
        key="shadow_rate",
        title="Traffic-Based Shadow Rate",
        out_path=fig_dir / "shadow_rate_before_after.png",
        ylabel="Shadow Rate",
    )
    plot_multi_before_after(
        m_o.to_dict(),
        m_g.to_dict(),
        keys=["coverage", "shadow_rate"],
        title="Coverage & Shadow Rate",
        out_path=fig_dir / "coverage_shadow.png",
        ylabel="Value",
    )

    report = ReportInputs(
        title="Firewall Rule Optimizer using Genetic Algorithm (GA)",
        student_name=student_name,
        summary_points=[
            "GA reorders rules to reduce average inspection cost while preserving policy decisions (high coverage).",
            "Traffic-based shadowing decreases when more specific rules are prioritized.",
            "Fitness improves steadily over generations; early stopping prevents over-computation.",
        ],
        metrics_before=m_o.to_dict(),
        metrics_after=m_g.to_dict(),
        figures={
            "Fitness evolution": str(fig_dir / "fitness_evolution.png"),
            "Avg checks (before vs after)": str(fig_dir / "avg_checks_before_after.png"),
        },
        ga_config={
            "Population size": ga_cfg.population_size,
            "Generations (max)": ga_cfg.generations,
            "Tournament k": ga_cfg.tournament_k,
            "Crossover rate": ga_cfg.crossover_rate,
            "Swap mutation p": ga_cfg.p_swap,
            "Inversion mutation p": ga_cfg.p_inversion,
            "Elitism": ga_cfg.elitism,
            "Min coverage": fit_cfg.min_coverage,
        },
        notes=None,
    )
    generate_report_pdf(report, out_dir / "report.pdf")

    print("\nDone.")
    print(f"Outputs: {out_dir}")
    print("Key metrics:")
    print(f"  Original: coverage={m_o.coverage:.4f}, avg_checks={m_o.avg_checks:.2f}, shadow_rate={m_o.shadow_rate:.3f}")
    print(f"  GA Best : coverage={m_g.coverage:.4f}, avg_checks={m_g.avg_checks:.2f}, shadow_rate={m_g.shadow_rate:.3f}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Firewall Rule Optimizer using Genetic Algorithm")
    p.add_argument("--rules", type=str, default=str(Path("data") / "rules_sample.json"), help="Rules JSON/CSV path")
    p.add_argument("--generate_rules", type=int, default=300, help="If rules file doesn't exist, generate N rules")
    p.add_argument("--packets", type=int, default=4000, help="Number of traffic samples")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=str(Path("outputs")), help="Output directory")
    p.add_argument(
        "--student",
        type=str,
        default="Your Name",
        help="Student name to print in report (Latin characters recommended for PDF fonts)",
    )

    # GA
    p.add_argument("--pop", type=int, default=60)
    p.add_argument("--gens", type=int, default=60)
    p.add_argument("--crossover", type=float, default=0.90)
    p.add_argument("--p_swap", type=float, default=0.20)
    p.add_argument("--p_inv", type=float, default=0.10)
    p.add_argument("--min_coverage", type=float, default=0.990)
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)

    rules_path = Path(args.rules)
    if not rules_path.exists():
        rules_path.parent.mkdir(parents=True, exist_ok=True)
        rules = generate_rules(args.generate_rules, seed=args.seed)
        save_rules_json(rules, rules_path)

    ga_cfg = GAConfig(
        population_size=args.pop,
        generations=args.gens,
        crossover_rate=args.crossover,
        p_swap=args.p_swap,
        p_inversion=args.p_inv,
    )
    fit_cfg = FitnessConfig(min_coverage=args.min_coverage)

    run_pipeline(
        rules_path=rules_path,
        out_dir=Path(args.out),
        n_packets=args.packets,
        seed=args.seed,
        ga_cfg=ga_cfg,
        fit_cfg=fit_cfg,
        student_name=args.student,
    )


if __name__ == "__main__":
    main()
