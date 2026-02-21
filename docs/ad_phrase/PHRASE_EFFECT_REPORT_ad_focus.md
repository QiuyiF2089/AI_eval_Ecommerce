# Phrase Effects by LLM

Target: `bias = model_score - human_score`
Effect definition: `effect = mean_bias(with_phrase) - mean_bias(without_phrase)`

Interpretation:
- `effect > 0`: phrase tends to raise this model relative to human
- `effect < 0`: phrase tends to lower this model relative to human

Filters: `min_support=20`, per-model FDR correction (BH)

## deepseek

- Tested phrases: `25`
- Significant phrases (q <= 0.10): `17`

Top phrases that **raise** score vs human

| phrase | effect | q_value | n_with | doc_freq |
|---|---:|---:|---:|---:|
| easy | 0.5048 | 7.043e-13 | 207 | 207 |
| quality | 0.4687 | 4.853e-10 | 187 | 187 |
| industrial | 0.4996 | 1.846e-07 | 77 | 77 |
| professional | 0.5339 | 3.364e-07 | 53 | 53 |
| heavy-duty | 0.8461 | 3.607e-07 | 20 | 20 |
| durable | 0.3648 | 2.397e-06 | 122 | 122 |
| high quality | 0.4530 | 1.032e-05 | 80 | 80 |
| safe | 0.4595 | 0.00012 | 51 | 51 |
| premium | 0.3705 | 0.0005411 | 57 | 57 |
| reliable | 0.4454 | 0.0005832 | 33 | 33 |

Top phrases that **lower** score vs human

| phrase | effect | q_value | n_with | doc_freq |
|---|---:|---:|---:|---:|
| (none) |  |  |  |  |

## gemini

- Tested phrases: `25`
- Significant phrases (q <= 0.10): `19`

Top phrases that **raise** score vs human

| phrase | effect | q_value | n_with | doc_freq |
|---|---:|---:|---:|---:|
| easy | 0.5554 | 2.414e-11 | 207 | 207 |
| quality | 0.4431 | 4.923e-06 | 187 | 187 |
| durable | 0.4113 | 2.099e-05 | 122 | 122 |
| industrial | 0.5283 | 7.541e-05 | 77 | 77 |
| professional | 0.4983 | 0.0009263 | 53 | 53 |
| warranty | 0.5944 | 0.001117 | 31 | 31 |
| guarantee | 0.4856 | 0.003916 | 32 | 32 |
| premium | 0.3863 | 0.004648 | 57 | 57 |
| high quality | 0.3547 | 0.008862 | 80 | 80 |
| safe | 0.4263 | 0.01221 | 51 | 51 |

Top phrases that **lower** score vs human

| phrase | effect | q_value | n_with | doc_freq |
|---|---:|---:|---:|---:|
| (none) |  |  |  |  |

## kimi

- Tested phrases: `25`
- Significant phrases (q <= 0.10): `17`

Top phrases that **raise** score vs human

| phrase | effect | q_value | n_with | doc_freq |
|---|---:|---:|---:|---:|
| easy | 0.4425 | 4.342e-10 | 207 | 207 |
| reliable | 0.6433 | 1.898e-06 | 33 | 33 |
| quality | 0.3773 | 2.802e-06 | 187 | 187 |
| guarantee | 0.6074 | 3.334e-06 | 32 | 32 |
| industrial | 0.4489 | 1.134e-05 | 77 | 77 |
| professional | 0.4739 | 1.813e-05 | 53 | 53 |
| quick | 0.4534 | 2.861e-05 | 43 | 43 |
| durable | 0.3360 | 4.092e-05 | 122 | 122 |
| warranty | 0.4694 | 4.154e-05 | 31 | 31 |
| high quality | 0.3945 | 0.0003293 | 80 | 80 |

Top phrases that **lower** score vs human

| phrase | effect | q_value | n_with | doc_freq |
|---|---:|---:|---:|---:|
| (none) |  |  |  |  |

## openai

- Tested phrases: `25`
- Significant phrases (q <= 0.10): `12`

Top phrases that **raise** score vs human

| phrase | effect | q_value | n_with | doc_freq |
|---|---:|---:|---:|---:|
| easy | 0.3818 | 1.856e-08 | 207 | 207 |
| quality | 0.3324 | 8.456e-06 | 187 | 187 |
| professional | 0.4964 | 2.339e-05 | 53 | 53 |
| industrial | 0.3920 | 0.0003786 | 77 | 77 |
| durable | 0.2813 | 0.0005851 | 122 | 122 |
| high quality | 0.3189 | 0.001546 | 80 | 80 |
| reliable | 0.3382 | 0.01397 | 33 | 33 |
| premium | 0.2741 | 0.01921 | 57 | 57 |
| heavy-duty | 0.5162 | 0.03595 | 20 | 20 |
| safe | 0.2583 | 0.03595 | 51 | 51 |

Top phrases that **lower** score vs human

| phrase | effect | q_value | n_with | doc_freq |
|---|---:|---:|---:|---:|
| (none) |  |  |  |  |

## Cross-Model Unique Effects

Phrases significant in exactly one model (q <= 0.10):

| phrase | model | effect | q_value | n_with |
|---|---|---:|---:|---:|
| 304 stainless | kimi | 0.4030 | 0.001171 | 25 |
| 304 stainless steel | kimi | 0.4036 | 0.002473 | 22 |
| brand new | gemini | 0.6793 | 0.02786 | 21 |
| waterproof | gemini | 0.2892 | 0.08799 | 33 |
