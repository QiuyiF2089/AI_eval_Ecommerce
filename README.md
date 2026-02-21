E-Commerce LLM-as-Judge Evaluation System
📌 Project Overview

This project aims to build an evaluation system for comparing the performance of multiple Large Language Models (LLMs) in e-commerce–related tasks, including explicit feature analysis, scoring, and recommendation.

The ultimate goals are:

Use multiple large language models (e.g., GPT, DeepSeek, Qwen, Gemini) as “judges”
to score and infer preferences from e-commerce reviews, ratings, and advertising copy;

Use BERT-based models to identify and quantify LLM preference biases toward different numerical and textual evaluation criteria;

Quantitatively analyze preference trends and score differences across models under different tasks and advertising scenarios;

Build an automated code generation and testing pipeline.

🎯 Research Objectives

This project seeks to answer the following key questions:

Under explicit features (review text, ratings, ad copy), what are the differences between LLM-based and traditional models in scoring and preference prediction?

Can we apply automated evaluation metrics to achieve interpretable analysis of LLM scoring behavior?

Is it possible to construct a general-purpose LLM evaluation framework that can serve as a foundation for future e-commerce recommendation and evaluation systems?

Can we predict LLM preferences in advance and intentionally craft advertising copy that receives higher scores and stronger recommendations from LLM judges?

📦 Dataset Sources

The following dataset can be considered as a primary data source
(publicly available and up-to-date as of 2025):

https://www.kaggle.com/datasets/abhayayare/e-commerce-dataset/data

🗂️ Project Structure

AI_eval_Ecommerce/
  README.md           # Project overview and documentation
  LICENSE             # License file
  data/               # Raw dataset files
    events.csv        # User events
    order_items.csv   # Order-item details
    orders.csv        # Order records
    products.csv      # Product catalog
    reviews.csv       # Reviews and ratings
    users.csv         # User information
  data_check.ipynb    # Data inspection and sanity checks

🔧 Module Architecture
Module	Description


📊 LLM Judge Task Definition
Recommendation Rating Prediction

Given user review content, predict a 1–5 star rating.

The desired unified output format is:

{
  "model": "gpt-4.1",
  "stars": 4
}





https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023