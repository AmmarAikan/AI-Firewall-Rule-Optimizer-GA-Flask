# Firewall Rule Optimizer (Genetic Algorithm)

This project optimizes the ordering of a large firewall ruleset using a **Genetic Algorithm (GA)**.
The goal is to keep the firewall policy behavior (high **coverage**) while reducing:

- Average packet inspection work (fewer rules checked per packet)
- Traffic-based rule shadowing (rules never triggered by traffic samples)

## Key Idea
Many firewalls use **first-match semantics**: the first rule that matches a packet decides ALLOW/DENY.
Therefore, rule ordering strongly affects performance.

We model an ordering as a **permutation** of rule indices and evolve better permutations with GA.

## Folder Layout
```
firewall_rule_optimizer_ga/
├── data/            # input rules
├── src/             # implementation
├── outputs/         # generated outputs (metrics, figures, report)
├── report/          # (optional) final report location
├── tests/           # small sanity tests
└── requirements.txt
```

## Quick Start

1) Install dependencies:
```bash
pip install -r requirements.txt
```

2) Run (generates a synthetic ruleset if `data/rules_sample.json` does not exist):
```bash
python -m src.main --student "YOUR NAME" --generate_rules 300 --packets 4000 --seed 0
```

Note: the PDF report uses built-in fonts; Latin characters render best.

## Inputs
Rules can be **JSON** or **CSV** and each rule should include:

- `source_ip` (CIDR, single IP, or ANY)
- `destination_ip` (CIDR, single IP, or ANY)
- `port` (e.g., `443`, `1000-2000`, or `ANY`)
- `protocol` (TCP/UDP/ICMP/ANY)
- `action` (ALLOW/DENY)

## Outputs
The run creates an `outputs/` folder that contains:

- `ordering_original.json`
- `ordering_specificity_baseline.json`
- `ordering_ga_best.json`
- `metrics.json`
- `figures/` (plots)
- `report.pdf` (2-3 pages)

## Demo Tip
For a clean demo video:

```bash
python -m src.main --student "YOUR NAME" --generate_rules 500 --packets 6000 --seed 7 --pop 60 --gens 60
```

Then open `outputs/report.pdf` and show the before/after metrics + plots.

## Web Demo (Flask)

If you want a clean "click-and-run" demo in a browser:

```bash
pip install -r requirements.txt
python app.py
```

Then open: `http://127.0.0.1:5000`

The web UI lets you:
- Use the sample rules, generate a synthetic ruleset, or upload your own JSON/CSV
- Adjust GA parameters (population, generations, mutation rates...)
- View before/after metrics and plots
- Download the generated report/outputs for the run
