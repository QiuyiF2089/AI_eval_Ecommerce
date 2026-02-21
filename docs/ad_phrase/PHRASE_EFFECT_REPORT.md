# Phrase Effects by LLM

Target: `bias = model_score - human_score`
Effect definition: `effect = mean_bias(with_phrase) - mean_bias(without_phrase)`

Interpretation:
- `effect > 0`: phrase tends to raise this model relative to human
- `effect < 0`: phrase tends to lower this model relative to human

Filters: `min_support=25`, per-model FDR correction (BH)

## deepseek

- Tested phrases: `474`
- Significant phrases (q <= 0.10): `405`

Top phrases that **raise** score vs human

| phrase | effect | q_value | n_with | doc_freq |
|---|---:|---:|---:|---:|
| use | 0.6367 | 3.048e-24 | 280 | 280 |
| has | 0.6367 | 8.215e-22 | 175 | 175 |
| made | 0.6297 | 1.003e-21 | 227 | 227 |
| size | 0.6053 | 8.715e-19 | 233 | 233 |
| used | 0.5653 | 5.686e-18 | 241 | 241 |
| material | 0.5785 | 1.4e-16 | 191 | 191 |
| diameter | 0.6071 | 1.422e-16 | 161 | 161 |
| length | 0.6421 | 1.719e-15 | 139 | 139 |
| high | 0.5272 | 6.041e-15 | 246 | 246 |
| convenient | 0.9045 | 4.017e-14 | 50 | 50 |
| easily | 0.6399 | 4.017e-14 | 93 | 93 |
| have | 0.5312 | 1.776e-13 | 152 | 152 |

Top phrases that **lower** score vs human

| phrase | effect | q_value | n_with | doc_freq |
|---|---:|---:|---:|---:|
| (none) |  |  |  |  |

## gemini

- Tested phrases: `474`
- Significant phrases (q <= 0.10): `376`

Top phrases that **raise** score vs human

| phrase | effect | q_value | n_with | doc_freq |
|---|---:|---:|---:|---:|
| use | 0.6849 | 2.02e-19 | 280 | 280 |
| diameter | 0.7067 | 7.858e-17 | 161 | 161 |
| has | 0.7004 | 2.049e-16 | 175 | 175 |
| material | 0.6870 | 2.049e-16 | 191 | 191 |
| made | 0.6724 | 3.795e-16 | 227 | 227 |
| size | 0.6274 | 3.103e-13 | 233 | 233 |
| used | 0.5993 | 2.185e-12 | 241 | 241 |
| easy | 0.5554 | 5.721e-11 | 207 | 207 |
| length | 0.6374 | 1.222e-10 | 139 | 139 |
| high | 0.5098 | 8.775e-10 | 246 | 246 |
| please | 0.6839 | 1.306e-09 | 90 | 90 |
| mounting | 0.9699 | 1.685e-09 | 32 | 32 |

Top phrases that **lower** score vs human

| phrase | effect | q_value | n_with | doc_freq |
|---|---:|---:|---:|---:|
| (none) |  |  |  |  |

## kimi

- Tested phrases: `474`
- Significant phrases (q <= 0.10): `384`

Top phrases that **raise** score vs human

| phrase | effect | q_value | n_with | doc_freq |
|---|---:|---:|---:|---:|
| use | 0.5693 | 8.331e-20 | 280 | 280 |
| made | 0.5752 | 2.601e-17 | 227 | 227 |
| has | 0.5736 | 3.351e-15 | 175 | 175 |
| material | 0.5192 | 5.684e-14 | 191 | 191 |
| which | 0.5495 | 9.076e-14 | 141 | 141 |
| please | 0.5882 | 2.253e-13 | 90 | 90 |
| size | 0.4935 | 1.589e-12 | 233 | 233 |
| diameter | 0.5465 | 3.711e-12 | 161 | 161 |
| used | 0.4696 | 8.971e-12 | 241 | 241 |
| easily | 0.5846 | 2.091e-11 | 93 | 93 |
| if | 0.5289 | 5.831e-11 | 101 | 101 |
| convenient | 0.7716 | 8.834e-11 | 50 | 50 |

Top phrases that **lower** score vs human

| phrase | effect | q_value | n_with | doc_freq |
|---|---:|---:|---:|---:|
| (none) |  |  |  |  |

## openai

- Tested phrases: `474`
- Significant phrases (q <= 0.10): `318`

Top phrases that **raise** score vs human

| phrase | effect | q_value | n_with | doc_freq |
|---|---:|---:|---:|---:|
| made | 0.5119 | 1.202e-14 | 227 | 227 |
| use | 0.4453 | 3.108e-12 | 280 | 280 |
| has | 0.5062 | 1.1e-11 | 175 | 175 |
| used | 0.4193 | 2.284e-09 | 241 | 241 |
| diameter | 0.4307 | 1.085e-08 | 161 | 161 |
| material | 0.4036 | 1.085e-08 | 191 | 191 |
| length | 0.4733 | 3.115e-08 | 139 | 139 |
| easy | 0.3818 | 4.399e-08 | 207 | 207 |
| size | 0.3975 | 5.69e-08 | 233 | 233 |
| package | 0.4702 | 4.265e-07 | 114 | 114 |
| convenient | 0.5326 | 1.389e-06 | 50 | 50 |
| includes | 0.4848 | 1.389e-06 | 84 | 84 |

Top phrases that **lower** score vs human

| phrase | effect | q_value | n_with | doc_freq |
|---|---:|---:|---:|---:|
| (none) |  |  |  |  |

## Cross-Model Unique Effects

Phrases significant in exactly one model (q <= 0.10):

| phrase | model | effect | q_value | n_with |
|---|---|---:|---:|---:|
| 304 stainless | kimi | 0.4030 | 0.001252 | 25 |
| available | deepseek | 0.3148 | 0.007343 | 51 |
| cutting | deepseek | 0.3261 | 0.01087 | 26 |
| just | deepseek | 0.2714 | 0.01915 | 48 |
| tubing | deepseek | 0.3607 | 0.02051 | 28 |
| look | deepseek | 0.3273 | 0.02471 | 25 |
| 30 | kimi | 0.4028 | 0.04008 | 31 |
| duty | kimi | 0.2823 | 0.04206 | 71 |
| what | deepseek | 0.3684 | 0.04556 | 25 |
| outdoor | deepseek | 0.2518 | 0.05005 | 50 |
| lock | kimi | 0.2760 | 0.05046 | 38 |
| ball | deepseek | 0.3499 | 0.05207 | 34 |
| wear | gemini | 0.4008 | 0.06 | 30 |
| 32 | deepseek | 0.3214 | 0.06622 | 31 |
| extra | kimi | 0.3539 | 0.06665 | 35 |
| inner | kimi | 0.3029 | 0.06665 | 31 |
| voltage | gemini | 0.2986 | 0.06702 | 41 |
| 15 | deepseek | 0.2893 | 0.06793 | 35 |
| cover | gemini | 0.2823 | 0.06976 | 36 |
| connector | kimi | 0.3370 | 0.07558 | 26 |
| safety | deepseek | 0.2426 | 0.07668 | 59 |
| 18 | gemini | 0.4075 | 0.07716 | 29 |
| carbon | deepseek | 0.2597 | 0.08194 | 35 |
| red | gemini | 0.2768 | 0.08224 | 55 |
| aluminum | gemini | 0.2188 | 0.08378 | 80 |
| waterproof | gemini | 0.2892 | 0.08544 | 33 |
| pin | kimi | 0.3619 | 0.09047 | 25 |
| support | deepseek | 0.2585 | 0.09243 | 34 |
| commercial | kimi | 0.3307 | 0.09297 | 38 |
