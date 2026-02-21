#!/usr/bin/env python3
"""
Feature engineering for ad-phrase analysis.

Goal:
1) Build a unified ad text per item from title/features/description.
2) Extract phrase candidates (1-3 gram) with frequency filtering.
3) Build model-ready features:
   - item-level style features (length, punctuation, hype lexicon counts, etc.)
   - phrase-level vocabulary stats
   - sparse sample-phrase matrix (long format) for regression.

This script is intentionally dependency-light (standard library only),
so it can run in minimal environments.
"""

import argparse
import ast
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path


# Minimal stopword list for marketing text phrase mining.
# We keep it compact to avoid removing domain-meaningful words.
STOPWORDS = {
    "a", "an", "the", "and", "or", "for", "to", "of", "in", "on", "with", "by",
    "from", "at", "as", "is", "are", "was", "were", "be", "been", "this", "that",
    "these", "those", "it", "its", "your", "you", "our", "their", "they", "we",
    "can", "may", "will", "not", "no", "than", "then", "also", "only", "very",
    "into", "about", "over", "under", "up", "down", "all", "any", "more", "most",
}

# Generic functional words that frequently appear in product descriptions
# but usually do not represent advertising intent.
GENERIC_FUNCTION_WORDS = {
    "use",
    "used",
    "using",
    "has",
    "have",
    "had",
    "made",
    "make",
    "which",
    "please",
    "if",
    "when",
    "where",
    "just",
    "also",
    "can",
    "may",
    "will",
    "would",
    "should",
}

# Curated lexicons used as high-level style signals.
# These are useful controls in downstream models and can reveal marketing tone.
HYPE_WORDS = {
    "best", "premium", "ultimate", "professional", "high-quality", "high quality",
    "top", "perfect", "excellent", "amazing", "powerful", "advanced", "new", "upgraded",
    "durable", "heavy-duty", "heavy duty", "reliable",
}

TRUST_WORDS = {
    "guarantee", "warranty", "certified", "tested", "approved", "compliant",
    "safe", "trusted", "authentic", "official",
}

URGENCY_WORDS = {
    "limited", "exclusive", "now", "today", "instant", "quick", "fast",
}

# Tokens that often indicate ad-like intent (promise, quality, urgency, usability).
AD_SIGNAL_TOKENS = {
    "premium", "professional", "durable", "warranty", "guarantee", "certified",
    "safe", "reliable", "easy", "quick", "fast", "exclusive", "new", "upgraded",
    "heavy", "duty", "waterproof", "industrial", "commercial", "high", "quality",
}

# Keep unigram only when it is likely to be an ad signal.
AD_UNIGRAM_ALLOWLIST = {
    "premium", "professional", "durable", "warranty", "guarantee", "certified",
    "safe", "reliable", "easy", "quick", "fast", "exclusive", "new", "upgraded",
    "heavy-duty", "waterproof", "commercial", "industrial", "quality",
}


def parse_args():
    p = argparse.ArgumentParser(description="Build ad-phrase feature engineering outputs.")
    p.add_argument(
        "--items-csv",
        default="data/industrial_and_scientific_items_clean_sample_1000.csv",
        help="Source item CSV with title/features/description columns.",
    )
    p.add_argument(
        "--scores-csv",
        default="data/analysis_4ai_vs_human_merged.csv",
        help="Merged score CSV (optional but recommended for alignment).",
    )
    p.add_argument(
        "--output-dir",
        default="outputs/ad_phrase/baseline",
        help="Directory for generated feature files.",
    )
    p.add_argument(
        "--min-phrase-df",
        type=int,
        default=20,
        help="Minimum document frequency for phrase retention.",
    )
    p.add_argument(
        "--top-k-phrases",
        type=int,
        default=600,
        help="Maximum number of retained phrases after filtering.",
    )
    p.add_argument(
        "--method",
        choices=["baseline", "ad_focus"],
        default="baseline",
        help="Feature extraction method. ad_focus suppresses generic functional words.",
    )
    p.add_argument(
        "--max-phrase-df-ratio",
        type=float,
        default=1.0,
        help="Drop phrases with doc_freq/doc_count above this ratio.",
    )
    return p.parse_args()


def read_csv_dict(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_listish(text: str):
    """
    Parse list-like strings from cleaned csv columns.

    The source data sometimes stores list fields as:
    - python-literal-ish: "['a', 'b']"
    - whitespace-separated quoted chunks: "['a' 'b']"

    We recover as many textual chunks as possible for feature extraction.
    """
    if not text:
        return []
    s = text.strip()
    if not s:
        return []

    # Try safe literal parsing first.
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list):
            return [normalize_space(str(x)) for x in obj if str(x).strip()]
        if isinstance(obj, str):
            return [normalize_space(obj)]
    except Exception:
        pass

    # Fallback: extract quoted segments.
    parts = re.findall(r"'([^']+)'|\"([^\"]+)\"", s)
    recovered = []
    for a, b in parts:
        val = (a or b).strip()
        if val:
            recovered.append(normalize_space(val))
    if recovered:
        return recovered

    # Last fallback: strip brackets and split by separators.
    s = s.strip("[]")
    rough = re.split(r"\s{2,}|\|", s)
    return [normalize_space(x) for x in rough if normalize_space(x)]


def build_ad_text(row):
    """
    Build a single ad text block per item.

    Why:
    - Phrase mining should use a consistent textual surface.
    - We concatenate title + features + description to represent marketing copy.
    """
    title = normalize_space(row.get("title", ""))
    features_list = parse_listish(row.get("features", ""))
    description_list = parse_listish(row.get("description", ""))

    chunks = []
    if title:
        chunks.append(title)
    if features_list:
        chunks.extend(features_list)
    if description_list:
        chunks.extend(description_list)

    return normalize_space(" ; ".join(chunks))


def normalize_text_for_tokens(text: str) -> str:
    t = text.lower()
    # Preserve %, +, -, and decimal points because these can carry ad meaning
    # (e.g., "99.9%", "heavy-duty", "+ bonus", etc.).
    t = re.sub(r"[^a-z0-9%+\-\.\s]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def tokenize(text: str):
    norm = normalize_text_for_tokens(text)
    raw = norm.split()
    tokens = []
    for tok in raw:
        # Drop pure punctuation-like residues and very short junk.
        if len(tok) <= 1:
            continue
        if tok in STOPWORDS:
            continue
        tokens.append(tok)
    return tokens


def make_ngrams(tokens, n):
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def has_ad_signal(phrase: str):
    tokens = phrase.split()
    if any(any(c.isdigit() for c in t) for t in tokens):
        return True
    if "%" in phrase or "-" in phrase:
        return True
    return any(t in AD_SIGNAL_TOKENS for t in tokens)


def is_phrase_kept(phrase: str, method: str):
    tokens = phrase.split()
    n = len(tokens)

    if method == "baseline":
        return True

    # ad_focus:
    # 1) suppress generic functional unigram tokens
    # 2) keep unigram only if it looks ad-intent
    # 3) for n>=2, require at least one ad-signal token/pattern
    if n == 1:
        t = tokens[0]
        if t in GENERIC_FUNCTION_WORDS:
            return False
        return t in AD_UNIGRAM_ALLOWLIST

    if all(t in GENERIC_FUNCTION_WORDS for t in tokens):
        return False

    return has_ad_signal(phrase)


def style_features(text: str):
    """
    Build high-level style signals used as controls in downstream analysis.

    These features help isolate phrase effects from writing-style effects.
    Example: if a model likes long texts generally, we don't want to misattribute
    that to one specific phrase.
    """
    lower = text.lower()
    words = re.findall(r"\b\w+\b", lower)

    def lexicon_count(lex):
        c = 0
        for w in lex:
            if " " in w:
                c += lower.count(w)
            else:
                c += sum(1 for x in words if x == w)
        return c

    # Number density can capture "spec-heavy" ad style.
    num_tokens = re.findall(r"\b\d+(?:\.\d+)?\b", lower)

    return {
        "char_len": len(text),
        "word_len": len(words),
        "semicolon_cnt": text.count(";"),
        "exclaim_cnt": text.count("!"),
        "all_caps_token_cnt": len(re.findall(r"\b[A-Z]{2,}\b", text)),
        "number_token_cnt": len(num_tokens),
        "hype_word_cnt": lexicon_count(HYPE_WORDS),
        "trust_word_cnt": lexicon_count(TRUST_WORDS),
        "urgency_word_cnt": lexicon_count(URGENCY_WORDS),
    }


def main():
    args = parse_args()
    items_path = Path(args.items_csv)
    scores_path = Path(args.scores_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = read_csv_dict(items_path)
    score_rows = read_csv_dict(scores_path) if scores_path.exists() else []

    # Keep only items present in score table to ensure downstream alignment.
    # This avoids building phrase features for rows that were never scored.
    score_by_asin = {r.get("parent_asin", ""): r for r in score_rows}

    docs = []
    for row in items:
        asin = (row.get("parent_asin") or "").strip()
        if not asin:
            continue
        if score_rows and asin not in score_by_asin:
            continue

        text = build_ad_text(row)
        if not text:
            continue

        tokens = tokenize(text)
        if not tokens:
            continue

        doc = {
            "parent_asin": asin,
            "ad_text": text,
            "tokens": tokens,
            "style": style_features(text),
            "scores": score_by_asin.get(asin, {}),
        }
        docs.append(doc)

    if not docs:
        raise SystemExit("No usable documents after alignment/filtering.")

    # Candidate phrase extraction strategy:
    # 1) Build document-frequency counters for 1/2/3-grams.
    # 2) Use DF (not TF) to reduce dominance from repeated phrases in single docs.
    # 3) Keep phrases above min_df and cap to top_k for manageable modeling size.
    phrase_df = Counter()
    per_doc_phrase_sets = {}

    for d in docs:
        pset = set()
        toks = d["tokens"]
        for n in (1, 2, 3):
            for g in make_ngrams(toks, n):
                pset.add(g)
        per_doc_phrase_sets[d["parent_asin"]] = pset
        phrase_df.update(pset)

    doc_count = len(docs)
    max_df = int(args.max_phrase_df_ratio * doc_count)
    if max_df < args.min_phrase_df:
        max_df = args.min_phrase_df

    kept = []
    for p, df in phrase_df.items():
        if df < args.min_phrase_df:
            continue
        if df > max_df:
            continue
        if not is_phrase_kept(p, args.method):
            continue
        kept.append(p)

    kept.sort(key=lambda p: (phrase_df[p], len(p.split())), reverse=True)
    kept = kept[: args.top_k_phrases]
    kept_set = set(kept)

    # Output 1: item-level feature table.
    # Contains scores + style metrics + label text, useful for direct regression joins.
    item_feature_path = out_dir / "item_style_features.csv"
    with item_feature_path.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "parent_asin",
            "human_score",
            "openai",
            "deepseek",
            "gemini",
            "kimi",
            "ai_mean",
            "ai_human_gap",
            "char_len",
            "word_len",
            "semicolon_cnt",
            "exclaim_cnt",
            "all_caps_token_cnt",
            "number_token_cnt",
            "hype_word_cnt",
            "trust_word_cnt",
            "urgency_word_cnt",
            "ad_text",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for d in docs:
            s = d["scores"]
            out = {
                "parent_asin": d["parent_asin"],
                "human_score": s.get("human_score", ""),
                "openai": s.get("openai", ""),
                "deepseek": s.get("deepseek", ""),
                "gemini": s.get("gemini", ""),
                "kimi": s.get("kimi", ""),
                "ai_mean": s.get("ai_mean", ""),
                "ai_human_gap": s.get("ai_human_gap", ""),
                "ad_text": d["ad_text"],
            }
            out.update(d["style"])
            w.writerow(out)

    # Output 2: phrase vocabulary stats.
    # This file helps you inspect what phrase candidates entered the model.
    phrase_vocab_path = out_dir / "phrase_vocabulary_stats.csv"
    with phrase_vocab_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["phrase", "ngram_n", "doc_freq"])
        w.writeheader()
        for p in kept:
            w.writerow(
                {
                    "phrase": p,
                    "ngram_n": len(p.split()),
                    "doc_freq": phrase_df[p],
                }
            )

    # Output 3: sparse sample-phrase matrix in long format.
    # Each row indicates phrase presence (1/0) for a sample.
    # Long format is easier for SQL / statsmodels / R pipelines than ultra-wide CSV.
    phrase_matrix_path = out_dir / "sample_phrase_matrix_long.csv"
    with phrase_matrix_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["parent_asin", "phrase", "present", "doc_freq"],
        )
        w.writeheader()
        for d in docs:
            asin = d["parent_asin"]
            pset = per_doc_phrase_sets.get(asin, set())
            for p in kept:
                if p in pset:
                    w.writerow(
                        {
                            "parent_asin": asin,
                            "phrase": p,
                            "present": 1,
                            "doc_freq": phrase_df[p],
                        }
                    )

    # Bonus output: model-specific bias targets in long format.
    # This simplifies phrase x model interaction modeling.
    # bias = model_score - human_score
    model_bias_path = out_dir / "model_bias_long.csv"
    with model_bias_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["parent_asin", "model", "model_score", "human_score", "bias"],
        )
        w.writeheader()
        for d in docs:
            s = d["scores"]
            try:
                h = float(s.get("human_score", ""))
            except Exception:
                continue
            for m in ("openai", "deepseek", "gemini", "kimi"):
                try:
                    ms = float(s.get(m, ""))
                except Exception:
                    continue
                w.writerow(
                    {
                        "parent_asin": d["parent_asin"],
                        "model": m,
                        "model_score": ms,
                        "human_score": h,
                        "bias": ms - h,
                    }
                )

    readme_path = out_dir / "README.md"
    readme_path.write_text(
        "\n".join(
            [
                "# Ad Phrase Feature Engineering Outputs",
                "",
                "Generated files:",
                "- item_style_features.csv: sample-level style features + scores + ad_text.",
                "- phrase_vocabulary_stats.csv: retained phrase candidates and document frequency.",
                "- sample_phrase_matrix_long.csv: sparse sample-phrase presence matrix.",
                "- model_bias_long.csv: per-sample per-model bias target (model_score - human_score).",
                "",
                "Suggested next regression design:",
                "bias ~ phrase + model + phrase:model + style_controls + category_controls",
                "",
                "Default phrase filtering:",
                f"- method = {args.method}",
                f"- min_phrase_df = {args.min_phrase_df}",
                f"- max_phrase_df_ratio = {args.max_phrase_df_ratio}",
                f"- top_k_phrases = {args.top_k_phrases}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"done: docs={len(docs)}")
    print(f"done: kept_phrases={len(kept)}")
    print(f"done: {item_feature_path}")
    print(f"done: {phrase_vocab_path}")
    print(f"done: {phrase_matrix_path}")
    print(f"done: {model_bias_path}")


if __name__ == "__main__":
    main()
