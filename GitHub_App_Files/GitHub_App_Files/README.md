# Starbucks Review Intelligence System

This Streamlit app implements the two required Hugging Face pipelines for the ISOM5240 project.

- Pipeline 1: `text-classification` to classify each Starbucks review as positive or negative.
- Pipeline 2: `token-classification` to extract coffee names mentioned in the review.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Model loading

The app first looks for fine-tuned local model folders:

- `../../Group01_Dataset_files/Fine-tuned_Model_files/sentiment_model`
- `../../Group01_Dataset_files/Fine-tuned_Model_files/coffee_ner_model`

If those folders do not exist yet, it falls back to public Hugging Face models so the app can still be demonstrated. You can also set environment variables:

- `COFFEE_SENTIMENT_MODEL`
- `COFFEE_NER_MODEL`
