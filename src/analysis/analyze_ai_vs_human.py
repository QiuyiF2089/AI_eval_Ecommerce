#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
from collections import Counter
from pathlib import Path
from statistics import mean


MODEL_FILES = {
    "openai": "openai_eval_1000_gpt52.jsonl",
    "deepseek": "deepseek_eval_1000.jsonl",
    "gemini": "gemini3flash_eval_1000.jsonl",
    "kimi": "kimi_eval_1000.jsonl",
}


def load_model_scores(path: Path):
    scores = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pid = obj.get("parent_asin")
            score = obj.get("score")
            if pid and isinstance(score, int):
                scores[pid] = score
    return scores


def load_human_scores(
    csv_path: Path,
    key_col: str,
    human_score_col: str,
    fallback_average_rating: bool,
):
    out = {}
    meta = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = (row.get(key_col) or "").strip()
            if not pid:
                continue

            title = (row.get("title") or "").strip()
            avg_rating_text = (row.get("average_rating") or "").strip()
            meta[pid] = {
                "title": title,
                "average_rating": avg_rating_text,
            }

            raw_human = (row.get(human_score_col) or "").strip()
            score = None

            if raw_human:
                try:
                    score = int(round(float(raw_human)))
                except Exception:
                    score = None
            elif fallback_average_rating and avg_rating_text:
                try:
                    score = int(round(float(avg_rating_text)))
                except Exception:
                    score = None

            if score is not None:
                out[pid] = max(1, min(5, score))

    return out, meta


def pearson(x, y):
    if not x:
        return float("nan")
    mx = mean(x)
    my = mean(y)
    xs = [v - mx for v in x]
    ys = [v - my for v in y]
    sx = sum(v * v for v in xs)
    sy = sum(v * v for v in ys)
    if sx == 0 or sy == 0:
        return float("nan")
    return sum(a * b for a, b in zip(xs, ys)) / math.sqrt(sx * sy)


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = r
        i = j + 1
    return out


def spearman(x, y):
    return pearson(ranks(x), ranks(y))


def mae(x, y):
    return mean([abs(a - b) for a, b in zip(x, y)])


def safe_float(v):
    if isinstance(v, float) and math.isnan(v):
        return "nan"
    return f"{v:.3f}"


def confusion_5x5(y_true, y_pred):
    mat = [[0 for _ in range(5)] for _ in range(5)]
    for t, p in zip(y_true, y_pred):
        if 1 <= t <= 5 and 1 <= p <= 5:
            mat[t - 1][p - 1] += 1
    return mat


def make_figures(fig_dir: Path, human_vec, model_vectors, rows):
    os.environ.setdefault("MPLCONFIGDIR", str(fig_dir / ".mplconfig"))
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    fig_dir.mkdir(parents=True, exist_ok=True)

    models = list(model_vectors.keys())
    colors = {
        "human": "#1f2937",
        "openai": "#2563eb",
        "deepseek": "#16a34a",
        "gemini": "#f59e0b",
        "kimi": "#dc2626",
    }

    x = [1, 2, 3, 4, 5]

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.16
    human_counts = [Counter(human_vec).get(v, 0) for v in x]
    ax.bar([i - 2 * width for i in x], human_counts, width=width, label="human", color=colors["human"])
    for k, model in enumerate(models):
        counts = [Counter(model_vectors[model]).get(v, 0) for v in x]
        ax.bar([i + (k - 1) * width for i in x], counts, width=width, label=model, color=colors[model])
    ax.set_title("Score Distribution: Human vs 4 AI Models")
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "score_distribution.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = list(range(-4, 5))
    width = 0.18
    for i, model in enumerate(models):
        diffs = [p - t for p, t in zip(model_vectors[model], human_vec)]
        cnt = Counter(diffs)
        ys = [cnt.get(b, 0) for b in bins]
        ax.bar([b + (i - 1.5) * width for b in bins], ys, width=width, label=model, color=colors[model])
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Error Distribution (Model Score - Human Score)")
    ax.set_xlabel("Score Difference")
    ax.set_ylabel("Count")
    ax.set_xticks(bins)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "error_distribution.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for model in models:
        xs = []
        ys = []
        for h in [1, 2, 3, 4, 5]:
            vals = [model_vectors[model][i] for i, hv in enumerate(human_vec) if hv == h]
            if vals:
                xs.append(h)
                ys.append(mean(vals))
        ax.plot(xs, ys, marker="o", label=model, color=colors[model])
    ax.plot([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], linestyle="--", color="#6b7280", label="perfect match")
    ax.set_title("Average AI Score by Human Score Bucket")
    ax.set_xlabel("Human Score")
    ax.set_ylabel("Average AI Score")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "bucket_trend.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for idx, model in enumerate(models):
        r = idx // 2
        c = idx % 2
        ax = axes[r][c]
        cm = confusion_5x5(human_vec, model_vectors[model])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"Confusion: {model} vs human")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Human")
        ax.set_xticks(range(5), labels=[1, 2, 3, 4, 5])
        ax.set_yticks(range(5), labels=[1, 2, 3, 4, 5])
        for i in range(5):
            for j in range(5):
                val = cm[i][j]
                if val > 0:
                    ax.text(j, i, str(val), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    fig.tight_layout()
    fig.savefig(fig_dir / "confusion_matrices.png", dpi=180)
    plt.close(fig)

    gap_cases = sorted(rows, key=lambda r: abs(r["ai_human_gap"]), reverse=True)[:10]
    labels = [r["parent_asin"] for r in gap_cases]
    vals = [r["ai_human_gap"] for r in gap_cases]
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_colors = ["#dc2626" if v < 0 else "#16a34a" for v in vals]
    ax.barh(labels, vals, color=bar_colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Top 10 Largest Average AI-Human Gaps")
    ax.set_xlabel("AI Mean - Human")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "top_gap_cases.png", dpi=180)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description="Analyze 4 AI judge score files against human score.")
    parser.add_argument("--data-dir", default="data", help="Data directory path")
    parser.add_argument(
        "--human-csv",
        default="industrial_and_scientific_items_clean_sample_1000.csv",
        help="CSV file under data-dir containing human score or average_rating",
    )
    parser.add_argument("--human-key-col", default="parent_asin")
    parser.add_argument("--human-score-col", default="human_score")
    parser.add_argument(
        "--fallback-average-rating",
        action="store_true",
        help="Use rounded average_rating if human_score_col is missing/empty",
    )
    parser.add_argument(
        "--merged-csv",
        default="outputs/model_vs_human/analysis_4ai_vs_human_merged.csv",
        help="Merged output CSV path",
    )
    parser.add_argument(
        "--report-md",
        default="docs/model_vs_human/ANALYSIS_SUMMARY.md",
        help="Markdown report path",
    )
    parser.add_argument(
        "--fig-dir",
        default="outputs/model_vs_human",
        help="Output directory for generated figures",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    model_scores = {}
    for model, filename in MODEL_FILES.items():
        model_scores[model] = load_model_scores(data_dir / filename)

    human_scores, meta = load_human_scores(
        csv_path=data_dir / args.human_csv,
        key_col=args.human_key_col,
        human_score_col=args.human_score_col,
        fallback_average_rating=args.fallback_average_rating,
    )

    common = set(human_scores)
    for scores in model_scores.values():
        common &= set(scores)
    common = sorted(common)

    if not common:
        raise SystemExit("No common parent_asin across human source and 4 model outputs.")

    models = list(MODEL_FILES.keys())
    human_vec = [human_scores[pid] for pid in common]
    model_vectors = {m: [model_scores[m][pid] for pid in common] for m in models}

    vs_human_metrics = []
    for model in models:
        pred = model_vectors[model]
        exact = sum(int(a == b) for a, b in zip(pred, human_vec)) / len(common)
        within1 = sum(int(abs(a - b) <= 1) for a, b in zip(pred, human_vec)) / len(common)
        mean_diff = mean([a - b for a, b in zip(pred, human_vec)])
        over = sum(int(a > b) for a, b in zip(pred, human_vec)) / len(common)
        under = sum(int(a < b) for a, b in zip(pred, human_vec)) / len(common)
        vs_human_metrics.append(
            {
                "model": model,
                "mean_score": mean(pred),
                "pearson": pearson(pred, human_vec),
                "spearman": spearman(pred, human_vec),
                "mae": mae(pred, human_vec),
                "exact": exact,
                "within1": within1,
                "mean_diff": mean_diff,
                "over_rate": over,
                "under_rate": under,
            }
        )

    pairwise = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a = models[i]
            b = models[j]
            av = model_vectors[a]
            bv = model_vectors[b]
            pairwise.append(
                {
                    "pair": f"{a} vs {b}",
                    "exact": sum(int(x == y) for x, y in zip(av, bv)) / len(common),
                    "pearson": pearson(av, bv),
                    "spearman": spearman(av, bv),
                    "mae": mae(av, bv),
                }
            )

    rows = []
    for pid in common:
        row = {
            "parent_asin": pid,
            "title": meta.get(pid, {}).get("title", ""),
            "average_rating": meta.get(pid, {}).get("average_rating", ""),
            "human_score": human_scores[pid],
        }
        for model in models:
            row[model] = model_scores[model][pid]
        row["ai_mean"] = mean([row[m] for m in models])
        row["ai_human_gap"] = row["ai_mean"] - row["human_score"]
        row["ai_spread"] = max(row[m] for m in models) - min(row[m] for m in models)
        rows.append(row)

    rows_gap = sorted(rows, key=lambda r: abs(r["ai_human_gap"]), reverse=True)
    rows_spread = sorted(rows, key=lambda r: r["ai_spread"], reverse=True)

    fig_dir = Path(args.fig_dir)
    figures_generated = make_figures(
        fig_dir=fig_dir, human_vec=human_vec, model_vectors=model_vectors, rows=rows
    )

    merged_out = Path(args.merged_csv)
    merged_out.parent.mkdir(parents=True, exist_ok=True)
    with merged_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "parent_asin",
                "title",
                "average_rating",
                "human_score",
                "openai",
                "deepseek",
                "gemini",
                "kimi",
                "ai_mean",
                "ai_human_gap",
                "ai_spread",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    best_corr = sorted(vs_human_metrics, key=lambda x: x["pearson"], reverse=True)[0]
    most_conservative = sorted(vs_human_metrics, key=lambda x: x["mean_diff"])[0]
    most_over = sorted(vs_human_metrics, key=lambda x: x["over_rate"], reverse=True)[0]

    high_human_idx = [i for i, v in enumerate(human_vec) if v == 5]
    high_human_note = []
    for m in models:
        preds = [model_vectors[m][i] for i in high_human_idx]
        if preds:
            high_human_note.append((m, mean(preds)))
    high_human_note.sort(key=lambda x: x[1])

    report_lines = []
    report_lines.append("# 4 AI vs Human Score Analysis")
    report_lines.append("")
    report_lines.append(f"- Samples used: `{len(common)}`")
    report_lines.append(f"- Figures: `{fig_dir}`")
    if args.fallback_average_rating:
        report_lines.append(
            "- Human score source: rounded `average_rating` proxy (no explicit `human_score` column)."
        )
    else:
        report_lines.append(
            f"- Human score source column: `{args.human_score_col}` from `{args.human_csv}`."
        )

    report_lines.append("")
    report_lines.append("## Visual Summary")
    report_lines.append("")
    if figures_generated:
        report_lines.append("![](../../outputs/model_vs_human/score_distribution.png)")
        report_lines.append("")
        report_lines.append("![](../../outputs/model_vs_human/error_distribution.png)")
        report_lines.append("")
        report_lines.append("![](../../outputs/model_vs_human/bucket_trend.png)")
        report_lines.append("")
        report_lines.append("![](../../outputs/model_vs_human/confusion_matrices.png)")
        report_lines.append("")
        report_lines.append("![](../../outputs/model_vs_human/top_gap_cases.png)")
    else:
        report_lines.append(
            "- Figure generation skipped: `matplotlib` is not available in current environment."
        )

    report_lines.append("")
    report_lines.append("## Key Findings (Human-Centric)")
    report_lines.append("")
    report_lines.append(
        f"- Best linear alignment to human: **{best_corr['model']}** (`pearson={best_corr['pearson']:.3f}`)."
    )
    report_lines.append(
        f"- Most conservative model: **{most_conservative['model']}** (`mean_diff={most_conservative['mean_diff']:.3f}`), with strongest under-scoring tendency."
    )
    report_lines.append(
        f"- Most likely to score above human: **{most_over['model']}** (`over_rate={most_over['over_rate']:.3f}`)."
    )
    if high_human_note:
        report_lines.append(
            f"- For human=5 items, lowest AI average is **{high_human_note[0][0]}** (`avg_pred={high_human_note[0][1]:.3f}`), showing notable under-rating on high-quality samples."
        )
    if rows_spread:
        r0 = rows_spread[0]
        report_lines.append(
            f"- Largest cross-model disagreement: `{r0['parent_asin']}` (spread={r0['ai_spread']}, human={r0['human_score']}, openai/deepseek/gemini/kimi={r0['openai']}/{r0['deepseek']}/{r0['gemini']}/{r0['kimi']})."
        )

    report_lines.append("")
    report_lines.append("## Against Human")
    report_lines.append("")
    report_lines.append(
        "| model | mean_score | pearson | spearman | mae | exact | within1 | mean_diff | over_rate | under_rate |"
    )
    report_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m in vs_human_metrics:
        report_lines.append(
            f"| {m['model']} | {safe_float(m['mean_score'])} | {safe_float(m['pearson'])} | "
            f"{safe_float(m['spearman'])} | {safe_float(m['mae'])} | {safe_float(m['exact'])} | "
            f"{safe_float(m['within1'])} | {safe_float(m['mean_diff'])} | {safe_float(m['over_rate'])} | "
            f"{safe_float(m['under_rate'])} |"
        )

    report_lines.append("")
    report_lines.append("## AI Pairwise Agreement")
    report_lines.append("")
    report_lines.append("| pair | exact | pearson | spearman | mae |")
    report_lines.append("|---|---:|---:|---:|---:|")
    for p in pairwise:
        report_lines.append(
            f"| {p['pair']} | {safe_float(p['exact'])} | {safe_float(p['pearson'])} | "
            f"{safe_float(p['spearman'])} | {safe_float(p['mae'])} |"
        )

    report_lines.append("")
    report_lines.append("## Score Distribution")
    report_lines.append("")
    dist = {"human": Counter(human_vec)}
    for model in models:
        dist[model] = Counter(model_vectors[model])
    report_lines.append("| source | 1 | 2 | 3 | 4 | 5 |")
    report_lines.append("|---|---:|---:|---:|---:|---:|")
    for src in ["human"] + models:
        c = dist[src]
        report_lines.append(
            f"| {src} | {c.get(1, 0)} | {c.get(2, 0)} | {c.get(3, 0)} | {c.get(4, 0)} | {c.get(5, 0)} |"
        )

    report_lines.append("")
    report_lines.append("## Special Cases: Largest AI-Human Gaps (Top 10)")
    report_lines.append("")
    report_lines.append(
        "| parent_asin | human | openai | deepseek | gemini | kimi | ai_mean | gap | spread | title |"
    )
    report_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in rows_gap[:10]:
        report_lines.append(
            f"| {r['parent_asin']} | {r['human_score']} | {r['openai']} | {r['deepseek']} | {r['gemini']} | "
            f"{r['kimi']} | {r['ai_mean']:.2f} | {r['ai_human_gap']:+.2f} | {r['ai_spread']} | {r['title'][:80]} |"
        )

    report_lines.append("")
    report_lines.append("## Special Cases: Largest Model Disagreement (Top 10)")
    report_lines.append("")
    report_lines.append(
        "| parent_asin | spread | human | openai | deepseek | gemini | kimi | title |"
    )
    report_lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for r in rows_spread[:10]:
        report_lines.append(
            f"| {r['parent_asin']} | {r['ai_spread']} | {r['human_score']} | {r['openai']} | {r['deepseek']} | "
            f"{r['gemini']} | {r['kimi']} | {r['title'][:80]} |"
        )

    report_lines.append("")
    report_lines.append("## Notes")
    report_lines.append("")
    report_lines.append("- `mean_diff = model_score - human_score`; negative means model tends to score lower.")
    report_lines.append("- `exact` is exact match rate; `within1` is within +/-1 rate.")

    report_out = Path(args.report_md)
    report_out.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"done: merged -> {merged_out}")
    print(f"done: report -> {report_out}")
    if figures_generated:
        print(f"done: figures -> {fig_dir}")
    else:
        print("done: figures -> skipped (matplotlib unavailable)")


if __name__ == "__main__":
    main()
