# Ad Phrase Feature Engineering Outputs

Generated files:
- item_style_features.csv: sample-level style features + scores + ad_text.
- phrase_vocabulary_stats.csv: retained phrase candidates and document frequency.
- sample_phrase_matrix_long.csv: sparse sample-phrase presence matrix.
- model_bias_long.csv: per-sample per-model bias target (model_score - human_score).

Suggested next regression design:
bias ~ phrase + model + phrase:model + style_controls + category_controls

Default phrase filtering:
- method = ad_focus
- min_phrase_df = 20
- max_phrase_df_ratio = 0.35
- top_k_phrases = 600
