# 4 AI vs Human Score Analysis

- Samples used: `996`
- Figures: `outputs/model_vs_human`
- Human score source: rounded `average_rating` proxy (no explicit `human_score` column).

## Visual Summary

- Figure generation skipped: `matplotlib` is not available in current environment.

## Key Findings (Human-Centric)

- Best linear alignment to human: **openai** (`pearson=0.379`).
- Most conservative model: **kimi** (`mean_diff=-0.713`), with strongest under-scoring tendency.
- Most likely to score above human: **gemini** (`over_rate=0.280`).
- For human=5 items, lowest AI average is **kimi** (`avg_pred=3.612`), showing notable under-rating on high-quality samples.
- Largest cross-model disagreement: `B000CSZ74Y` (spread=3, human=1, openai/deepseek/gemini/kimi=4/3/5/2).

## Against Human

| model | mean_score | pearson | spearman | mae | exact | within1 | mean_diff | over_rate | under_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| openai | 4.005 | 0.379 | 0.301 | 0.656 | 0.461 | 0.908 | -0.206 | 0.203 | 0.336 |
| deepseek | 3.932 | 0.318 | 0.276 | 0.759 | 0.419 | 0.854 | -0.279 | 0.215 | 0.366 |
| gemini | 4.022 | 0.175 | 0.148 | 0.876 | 0.360 | 0.813 | -0.189 | 0.280 | 0.359 |
| kimi | 3.498 | 0.304 | 0.223 | 0.898 | 0.365 | 0.787 | -0.713 | 0.084 | 0.550 |

## AI Pairwise Agreement

| pair | exact | pearson | spearman | mae |
|---|---:|---:|---:|---:|
| openai vs deepseek | 0.685 | 0.785 | 0.768 | 0.328 |
| openai vs gemini | 0.591 | 0.727 | 0.699 | 0.433 |
| openai vs kimi | 0.488 | 0.742 | 0.716 | 0.551 |
| deepseek vs gemini | 0.676 | 0.805 | 0.781 | 0.341 |
| deepseek vs kimi | 0.543 | 0.813 | 0.801 | 0.470 |
| gemini vs kimi | 0.455 | 0.774 | 0.751 | 0.588 |

## Score Distribution

| source | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| human | 25 | 19 | 74 | 481 | 397 |
| openai | 1 | 71 | 133 | 508 | 283 |
| deepseek | 0 | 96 | 187 | 402 | 311 |
| gemini | 3 | 89 | 196 | 303 | 405 |
| kimi | 3 | 149 | 275 | 487 | 82 |

## Special Cases: Largest AI-Human Gaps (Top 10)

| parent_asin | human | openai | deepseek | gemini | kimi | ai_mean | gap | spread | title |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| B001I9WNJS | 5 | 1 | 2 | 1 | 2 | 1.50 | -3.50 | 1 | Miller Dowels, Oak, size 1x, 40 Pack |
| B01AB7Y4EE | 5 | 2 | 2 | 1 | 1 | 1.50 | -3.50 | 1 | GRQV-0591960496 |
| B0044UKK02 | 5 | 2 | 2 | 2 | 2 | 2.00 | -3.00 | 0 | Commercial Vacuum Cleaner Bag |
| B00HRY96MA | 5 | 2 | 2 | 2 | 2 | 2.00 | -3.00 | 0 | Sunex RSSCPP1 |
| B06XFSLR6B | 5 | 2 | 2 | 2 | 2 | 2.00 | -3.00 | 0 | ICOCO Android Cable,PowerLine Micro USB (4ft)-Dual sides Durable Charging Cable  |
| B075BKPNXX | 5 | 2 | 2 | 2 | 2 | 2.00 | -3.00 | 0 | Res-Med Air-Fit P10 (Blue) |
| B07L1TYS1V | 5 | 2 | 2 | 2 | 2 | 2.00 | -3.00 | 0 | Motor Flexible Coupling Coupler Connector |
| B07LG75HGY | 5 | 2 | 2 | 2 | 2 | 2.00 | -3.00 | 0 | ZP4510 Liquid Water Level Sensor Vertical Float Switches 5 Pieces |
| B085KZWQYQ | 5 | 2 | 2 | 2 | 2 | 2.00 | -3.00 | 0 | SZAT PRO Over The Door Coat Hanger Hook Rack Antique Copper Brass Bronze Dark Br |
| B001GBMN5C | 5 | 2 | 2 | 3 | 2 | 2.25 | -2.75 | 1 | Dayton 1DMR2 Web Sling, Reverse Eye, L4Ft, 4600Lb, W2In |

## Special Cases: Largest Model Disagreement (Top 10)

| parent_asin | spread | human | openai | deepseek | gemini | kimi | title |
|---|---:|---:|---:|---:|---:|---:|---|
| B000CSZ74Y | 3 | 1 | 4 | 3 | 5 | 2 | Dorman 803-207 Hex Head Cap Screw |
| B08PKF8G7F | 3 | 1 | 4 | 3 | 5 | 2 | MYERZI Linear Rail MGN12H Miniature Linear Rail Guide 450mm Length 12mm Width +  |
| B0000DCZL8 | 2 | 5 | 4 | 3 | 2 | 2 | Grizzly G8989 Toggle Safety Switch |
| B000AYHDYW | 2 | 5 | 4 | 3 | 3 | 2 | Safety 36-20924 2Pc Fiber Safety Flag 9"X12" 9Ft Straight Mount |
| B000EDPQU2 | 2 | 4 | 4 | 4 | 2 | 4 | Bostitch RH-S12D120EP 3-1/4-in x 0.120-in 21 Degree Plastic Collated Smooth Shan |
| B000FN2LQI | 2 | 5 | 5 | 5 | 3 | 4 | AmericanTool 6959 IRWIN SEPTLS5856959 - hanson High Carbon Steel Metric Hexagon  |
| B000HHE32C | 2 | 4 | 4 | 4 | 5 | 3 | Hillman Tacks 7/16" No. 4 Copper 1/2 Oz |
| B000NDL7VA | 2 | 5 | 5 | 4 | 3 | 3 | Grote 94100 2 5/16" Hole Grommet (Open Grommet (92120 + 67050) Kit) |
| B0028AL4WI | 2 | 3 | 4 | 4 | 5 | 3 | Stanley National N180-042 Stanley Flat Rod, 1-1/4 in W X 36 in L X 1/8 in T, Ste |
| B002JH0PI4 | 2 | 2 | 2 | 3 | 4 | 3 | Dixie Poly D-18-20 Plastic Drum Dolly for 20 gallon Drum, 600 lbs Capacity, 18.5 |

## Notes

- `mean_diff = model_score - human_score`; negative means model tends to score lower.
- `exact` is exact match rate; `within1` is within +/-1 rate.
