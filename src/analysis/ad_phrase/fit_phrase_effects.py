#!/usr/bin/env python3
"""
Phrase effect model for LLM-specific scoring bias.

Model idea (dependency-light):
For each (model, phrase), we compare two groups of items:
- with_phrase: items where the phrase appears
- without_phrase: items where the phrase does not appear

Target:
- bias = model_score - human_score

Effect:
- effect = mean(bias | with_phrase) - mean(bias | without_phrase)
  > 0  => phrase tends to raise this model's score vs human
  < 0  => phrase tends to lower this model's score vs human

Significance:
- Welch-style z approximation based on group variances
- two-sided p-value approximated from normal tail
- Benjamini-Hochberg FDR correction per model

Why this approach:
- Runs in plain Python without pandas/statsmodels
- Gives interpretable phrase x model effect ranking quickly
- Suitable first pass before a full mixed-effects regression
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Estimate phrase effects on model bias.")
    p.add_argument(
        "--bias-csv",
        default="outputs/ad_phrase/baseline/model_bias_long.csv",
        help="Long table with columns: parent_asin, model, bias",
    )
    p.add_argument(
        "--phrase-matrix-csv",
        default="outputs/ad_phrase/baseline/sample_phrase_matrix_long.csv",
        help="Long sparse phrase matrix with columns: parent_asin, phrase, present",
    )
    p.add_argument(
        "--vocab-csv",
        default="outputs/ad_phrase/baseline/phrase_vocabulary_stats.csv",
        help="Phrase vocabulary stats (for doc_freq and filtering).",
    )
    p.add_argument(
        "--out-csv",
        default="outputs/ad_phrase/baseline/phrase_model_effects.csv",
        help="Output effect table.",
    )
    p.add_argument(
        "--out-md",
        default="docs/ad_phrase/PHRASE_EFFECT_REPORT.md",
        help="Output markdown summary.",
    )
    p.add_argument(
        "--min-support",
        type=int,
        default=25,
        help="Minimum with-phrase sample count per model for reporting.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="Top positive/negative phrases per model in markdown report.",
    )
    return p.parse_args()


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def variance(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def normal_two_sided_p(z):
    # p = 2 * (1 - Phi(|z|)) = erfc(|z| / sqrt(2))
    return math.erfc(abs(z) / math.sqrt(2.0))


def welch_z(x1, x0):
    # Returns (z, p). Uses normal approximation for large enough groups.
    n1 = len(x1)
    n0 = len(x0)
    if n1 < 2 or n0 < 2:
        return 0.0, 1.0

    m1 = mean(x1)
    m0 = mean(x0)
    v1 = variance(x1)
    v0 = variance(x0)
    se2 = (v1 / n1) + (v0 / n0)
    if se2 <= 0:
        return 0.0, 1.0

    z = (m1 - m0) / math.sqrt(se2)
    p = normal_two_sided_p(z)
    return z, p


def bh_fdr(pvals):
    """
    Benjamini-Hochberg correction.
    Input: list of (key, pval)
    Output: dict key -> qval
    """
    m = len(pvals)
    if m == 0:
        return {}

    ranked = sorted(pvals, key=lambda x: x[1])
    q = [0.0] * m
    for i, (_, p) in enumerate(ranked, start=1):
        q[i - 1] = (p * m) / i

    # Enforce monotonicity from tail.
    for i in range(m - 2, -1, -1):
        q[i] = min(q[i], q[i + 1])

    out = {}
    for (k, _), qv in zip(ranked, q):
        out[k] = min(1.0, qv)
    return out


def fmt(v):
    return f"{v:.4f}"


def main():
    args = parse_args()

    bias_rows = read_csv(Path(args.bias_csv))
    phrase_rows = read_csv(Path(args.phrase_matrix_csv))
    vocab_rows = read_csv(Path(args.vocab_csv))

    # Build map: phrase -> set(asin)
    phrase_to_asins = defaultdict(set)
    for r in phrase_rows:
        if str(r.get("present", "")).strip() != "1":
            continue
        asin = (r.get("parent_asin") or "").strip()
        phrase = (r.get("phrase") or "").strip()
        if asin and phrase:
            phrase_to_asins[phrase].add(asin)

    phrase_df = {}
    for r in vocab_rows:
        phrase = (r.get("phrase") or "").strip()
        if not phrase:
            continue
        try:
            df = int(float(r.get("doc_freq", "0")))
        except Exception:
            df = 0
        phrase_df[phrase] = df

    # Build model -> asin -> bias
    model_bias = defaultdict(dict)
    all_asins = set()
    for r in bias_rows:
        asin = (r.get("parent_asin") or "").strip()
        model = (r.get("model") or "").strip()
        if not asin or not model:
            continue
        try:
            b = float(r.get("bias", ""))
        except Exception:
            continue
        model_bias[model][asin] = b
        all_asins.add(asin)

    models = sorted(model_bias.keys())
    if not models:
        raise SystemExit("No model bias rows found.")

    results = []

    for model in models:
        bias_map = model_bias[model]
        model_asins = set(bias_map.keys())

        for phrase, with_set_all in phrase_to_asins.items():
            with_set = model_asins & with_set_all
            without_set = model_asins - with_set

            n_with = len(with_set)
            n_without = len(without_set)

            if n_with < args.min_support or n_without < args.min_support:
                continue

            x1 = [bias_map[a] for a in with_set]
            x0 = [bias_map[a] for a in without_set]

            m1 = mean(x1)
            m0 = mean(x0)
            effect = m1 - m0
            z, p = welch_z(x1, x0)

            results.append(
                {
                    "model": model,
                    "phrase": phrase,
                    "doc_freq": phrase_df.get(phrase, len(with_set_all)),
                    "n_with": n_with,
                    "n_without": n_without,
                    "mean_bias_with": m1,
                    "mean_bias_without": m0,
                    "effect": effect,
                    "z": z,
                    "p_value": p,
                }
            )

    # FDR correction per model.
    by_model = defaultdict(list)
    for i, r in enumerate(results):
        by_model[r["model"]].append((i, r["p_value"]))

    qvals = {}
    for m, arr in by_model.items():
        q = bh_fdr(arr)
        qvals.update(q)

    for i, r in enumerate(results):
        r["q_value"] = qvals.get(i, 1.0)

    # Sort overall for deterministic output.
    results.sort(key=lambda r: (r["model"], r["q_value"], -abs(r["effect"]), -r["doc_freq"]))

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "model",
            "phrase",
            "doc_freq",
            "n_with",
            "n_without",
            "mean_bias_with",
            "mean_bias_without",
            "effect",
            "z",
            "p_value",
            "q_value",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(
                {
                    **r,
                    "mean_bias_with": fmt(r["mean_bias_with"]),
                    "mean_bias_without": fmt(r["mean_bias_without"]),
                    "effect": fmt(r["effect"]),
                    "z": fmt(r["z"]),
                    "p_value": f"{r['p_value']:.6g}",
                    "q_value": f"{r['q_value']:.6g}",
                }
            )

    # Markdown report.
    lines = []
    lines.append("# Phrase Effects by LLM")
    lines.append("")
    lines.append("Target: `bias = model_score - human_score`")
    lines.append("Effect definition: `effect = mean_bias(with_phrase) - mean_bias(without_phrase)`")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- `effect > 0`: phrase tends to raise this model relative to human")
    lines.append("- `effect < 0`: phrase tends to lower this model relative to human")
    lines.append("")
    lines.append(f"Filters: `min_support={args.min_support}`, per-model FDR correction (BH)")
    lines.append("")

    for model in models:
        subset = [r for r in results if r["model"] == model]
        sig = [r for r in subset if r["q_value"] <= 0.10]

        lines.append(f"## {model}")
        lines.append("")
        lines.append(f"- Tested phrases: `{len(subset)}`")
        lines.append(f"- Significant phrases (q <= 0.10): `{len(sig)}`")
        lines.append("")

        pos = [r for r in sig if r["effect"] > 0]
        neg = [r for r in sig if r["effect"] < 0]
        pos.sort(key=lambda r: (r["q_value"], -r["effect"]))
        neg.sort(key=lambda r: (r["q_value"], r["effect"]))

        lines.append("Top phrases that **raise** score vs human")
        lines.append("")
        lines.append("| phrase | effect | q_value | n_with | doc_freq |")
        lines.append("|---|---:|---:|---:|---:|")
        for r in pos[: args.top_k]:
            lines.append(
                f"| {r['phrase']} | {r['effect']:.4f} | {r['q_value']:.4g} | {r['n_with']} | {r['doc_freq']} |"
            )
        if not pos[: args.top_k]:
            lines.append("| (none) |  |  |  |  |")

        lines.append("")
        lines.append("Top phrases that **lower** score vs human")
        lines.append("")
        lines.append("| phrase | effect | q_value | n_with | doc_freq |")
        lines.append("|---|---:|---:|---:|---:|")
        for r in neg[: args.top_k]:
            lines.append(
                f"| {r['phrase']} | {r['effect']:.4f} | {r['q_value']:.4g} | {r['n_with']} | {r['doc_freq']} |"
            )
        if not neg[: args.top_k]:
            lines.append("| (none) |  |  |  |  |")

        lines.append("")

    # Highlight "unique" phrase effects: significant for one model but not others.
    lines.append("## Cross-Model Unique Effects")
    lines.append("")
    by_phrase = defaultdict(list)
    for r in results:
        by_phrase[r["phrase"]].append(r)

    unique_rows = []
    for phrase, arr in by_phrase.items():
        sig_arr = [x for x in arr if x["q_value"] <= 0.10]
        if len(sig_arr) != 1:
            continue
        x = sig_arr[0]
        unique_rows.append(x)

    unique_rows.sort(key=lambda r: (r["q_value"], -abs(r["effect"])))

    lines.append("Phrases significant in exactly one model (q <= 0.10):")
    lines.append("")
    lines.append("| phrase | model | effect | q_value | n_with |")
    lines.append("|---|---|---:|---:|---:|")
    for r in unique_rows[:40]:
        lines.append(
            f"| {r['phrase']} | {r['model']} | {r['effect']:.4f} | {r['q_value']:.4g} | {r['n_with']} |"
        )
    if not unique_rows:
        lines.append("| (none) |  |  |  |  |")

    out_md = Path(args.out_md)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"done: rows={len(results)}")
    print(f"done: csv={out_csv}")
    print(f"done: md={out_md}")


if __name__ == "__main__":
    main()
