from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)


@dataclass(frozen=True)
class ReportInputs:
    title: str
    student_name: str
    summary_points: list[str]
    metrics_before: dict
    metrics_after: dict
    figures: dict[str, str]  # label -> path
    ga_config: dict
    notes: Optional[str] = None


def generate_report_pdf(inp: ReportInputs, out_pdf: str | Path) -> Path:
    """Generate a 2-3 page PDF report (aligned with the course requirements)."""

    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(out_pdf),
        pagesize=LETTER,
        rightMargin=0.8 * inch,
        leftMargin=0.8 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title=inp.title,
        author=inp.student_name,
    )

    styles = getSampleStyleSheet()
    h1 = styles["Heading1"]
    h2 = styles["Heading2"]
    h3 = styles["Heading3"]
    body = styles["BodyText"]
    body.spaceAfter = 8
    small = ParagraphStyle("small", parent=body, fontSize=9, leading=12)

    story = []

    # Title block
    story.append(Paragraph(inp.title, h1))
    story.append(Paragraph(f"Student: {inp.student_name}", body))
    story.append(Spacer(1, 10))

    story.append(Paragraph("1. Problem Description", h2))
    story.append(
        Paragraph(
            "Firewalls often apply a <b>first-match</b> policy: packets are checked against rules in order, "
            "and the first matching rule decides ALLOW/DENY. With large rule sets, poor ordering increases "
            "inspection time and creates rule anomalies (e.g., rules that never get used).",
            body,
        )
    )

    story.append(Paragraph("2. Chosen Technique", h2))
    story.append(
        Paragraph(
            "We solve the ordering problem using a <b>Genetic Algorithm (GA)</b>. Each chromosome is a complete "
            "permutation of the rule indices. The GA evolves orderings that preserve policy decisions (high coverage) "
            "while reducing average inspection cost and traffic-based shadowing.",
            body,
        )
    )

    story.append(Paragraph("3. Algorithm Design", h2))
    story.append(Paragraph("Chromosome: permutation of rule indices (ordering).", body))
    story.append(Paragraph("Selection: tournament selection.", body))
    story.append(Paragraph("Crossover: Order Crossover (OX).", body))
    story.append(Paragraph("Mutation: swap + inversion (segment reverse).", body))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Fitness Function", h3))
    story.append(
        Paragraph(
            "We evaluate an ordering using: <b>Coverage</b> (agreement with a reference policy on traffic samples), "
            "<b>Shadow rate</b> (rules never triggered by the traffic), and <b>Avg. checks</b> "
            "(average number of rules tested per packet). Coverage is treated as a near-hard constraint.",
            body,
        )
    )

    cfg_rows = [["GA Parameter", "Value"]] + [[k, str(v)] for k, v in inp.ga_config.items()]
    cfg_tbl = Table(cfg_rows, colWidths=[2.7 * inch, 3.5 * inch])
    cfg_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2d3d")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
            ]
        )
    )
    story.append(Spacer(1, 6))
    story.append(cfg_tbl)

    # Continue naturally (keeps the report within 2-3 pages for typical runs).
    story.append(Spacer(1, 12))
    story.append(Paragraph("4. Experiments & Results", h2))
    story.append(
        Paragraph(
            "We compare the original (input) ordering with the GA-optimized ordering. The reference policy is the "
            "original ordering applied on the same traffic samples.",
            body,
        )
    )

    keys = ["coverage", "avg_checks", "shadow_rate", "inactive_rules"]
    rows = [["Metric", "Before", "After"]]
    for k in keys:
        b = inp.metrics_before[k]
        a = inp.metrics_after[k]
        rows.append(
            [
                k,
                f"{b:.4f}" if isinstance(b, float) else str(b),
                f"{a:.4f}" if isinstance(a, float) else str(a),
            ]
        )
    tbl = Table(rows, colWidths=[2.2 * inch, 2.0 * inch, 2.0 * inch])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2d3d")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
            ]
        )
    )
    story.append(tbl)
    story.append(Spacer(1, 10))

    for label, img_path in inp.figures.items():
        p = Path(img_path)
        if p.exists():
            story.append(Paragraph(label, h3))
            # Smaller figure height to avoid exceeding 3 pages.
            story.append(Image(str(p), width=6.5 * inch, height=2.8 * inch))
            story.append(Spacer(1, 12))

    story.append(Paragraph("5. Key Takeaways", h2))
    bullets = "<br/>".join([f"• {x}" for x in inp.summary_points])
    story.append(Paragraph(bullets, body))

    if inp.notes:
        story.append(Spacer(1, 6))
        story.append(Paragraph("Notes", h3))
        story.append(Paragraph(inp.notes, small))

    doc.build(story)
    return out_pdf
