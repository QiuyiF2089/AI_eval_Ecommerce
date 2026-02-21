# LLM-as-Judge for E-Commerce: Evaluation & Bias Exploration

## Background

As LLMs are increasingly used as automated evaluators ("judges") for product quality, a critical question arises: **how reliable and unbiased are these AI judges compared to human ratings?**

This project uses Amazon Industrial & Scientific product data to systematically investigate two questions:

1. **AI vs Human alignment** — Do LLM judges agree with human ratings? Where do they diverge?
2. **Textual bias in LLM scoring** — Are certain words or advertising phrases in product descriptions systematically inflating (or deflating) AI scores?

## Models Evaluated

We benchmark four LLMs as product-quality judges, each scoring products on a 1–5 scale:

| Model | Provider |
|---|---|
| GPT-5.2 | OpenAI |
| DeepSeek | DeepSeek |
| Gemini 3 Flash | Google |
| Kimi K2 Turbo | Moonshot |

Human scores are proxied by the rounded `average_rating` from real customer reviews.

## Key Findings

### 1. AI–Human Score Alignment

- **Best aligned with human**: OpenAI (Pearson r = 0.379, exact match 46.1%)
- **Most conservative**: Kimi (mean diff = −0.71), consistently under-scoring relative to humans
- **Most generous**: Gemini (28% over-rate), most likely to score above human
- All four models show moderate correlation with human ratings but systematic negative bias (scoring lower than humans on average)

### 2. AI Pairwise Agreement

Models agree with each other more than they agree with humans. OpenAI–DeepSeek show the highest pairwise alignment (exact match 68.5%, Pearson 0.785), suggesting shared scoring heuristics across LLMs.

### 3. Advertising Phrase Bias

Certain persuasive or marketing-oriented phrases **systematically inflate** AI scores relative to human ratings across all four models:

| Phrase | Effect Direction | Cross-Model? |
|---|---|---|
| "easy" | ↑ raises score | All 4 models |
| "quality" / "high quality" | ↑ raises score | All 4 models |
| "professional" | ↑ raises score | All 4 models |
| "heavy-duty" | ↑ raises score | All 4 models |
| "durable" | ↑ raises score | All 4 models |
| "warranty" / "guarantee" | ↑ raises score | All 4 models |

No phrase was found to significantly **lower** any model's score relative to human — the bias is uniformly upward for persuasive language.

Some model-specific sensitivities were also identified (e.g., Kimi responds to "304 stainless steel"; Gemini to "brand new").

## Data Pipeline

1. **Raw data**: Amazon product metadata from the Industrial & Scientific category (HuggingFace)
2. **Cleaning**: Field selection, deduplication, normalization (see `docs/DATA_CLEANING.md`)
3. **LLM evaluation**: Each model scores 1,000 sampled products via structured prompt → JSON output
4. **Analysis**: Statistical comparison of AI vs human scores; phrase-level bias regression with FDR correction

## Project Structure

```
├── src/
│   ├── evaluation/          # LLM scoring scripts
│   └── analysis/            # AI-vs-human & ad-phrase bias analysis
├── configs/prompts/         # Scoring prompt templates (YAML)
├── notebooks/               # Data import, processing & cleaning
├── docs/                    # Reports and documentation
│   ├── DATA_CLEANING.md
│   ├── model_vs_human/      # AI vs human analysis report
│   └── ad_phrase/           # Phrase bias reports
├── data/                    # Datasets and evaluation results
└── outputs/                 # Generated figures and intermediate tables
```

## Detailed Reports

- [AI vs Human Score Analysis](docs/model_vs_human/ANALYSIS_SUMMARY.md)
- [Phrase Bias Effects (baseline)](docs/ad_phrase/PHRASE_EFFECT_REPORT.md)
- [Phrase Bias Effects (ad-focused)](docs/ad_phrase/PHRASE_EFFECT_REPORT_ad_focus.md)
- [Top 10 Preferred Phrases per Model](docs/ad_phrase/TOP10_PHRASES_PER_MODEL.md)
- [Data Cleaning Documentation](docs/DATA_CLEANING.md)
