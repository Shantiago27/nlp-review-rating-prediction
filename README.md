# Amazon Review Star Rating Prediction

A text classification model that predicts a product review's star rating (1-5) from its free-text content alone, trained on Amazon Home & Kitchen reviews.

In practice, this kind of model can flag problem products before their rating visibly drops, prioritize responses to negative reviews, and track satisfaction at scale without reading every review by hand.

## Key finding

The star rating distribution is heavily skewed: 63.4% of reviews are 5 stars. That means a trivial classifier that always predicts "5 stars" already reaches **63.4% accuracy** — and the trained Logistic Regression model reaches **63%**, essentially tied with that baseline. Reporting accuracy alone would make the model look far weaker than it actually is.

The metric that shows the real signal is **macro-F1**: **0.47** for the trained model vs **0.16** for the trivial baseline, with per-class F1 of **0.50 / 0.26 / 0.36** on the 1★ / 2★ / 3★ classes — where the trivial baseline scores **0.0** on every one of them, since it never predicts anything but 5★. That gap is what `class_weight='balanced'` buys: real ability to recognize minority-class reviews that a plain accuracy number would hide entirely.

## Results

| Model | CV accuracy (5-fold) | Test accuracy | Macro F1 (test) |
|---|---|---|---|
| Logistic Regression | 0.6204 | 0.63 | 0.47 |
| Random Forest | 0.5960 | 0.60 | 0.40 |
| Majority-class baseline (always predict 5★) | — | 0.634 | 0.16 |

Per-class F1 (test set), for context on where the models actually struggle:

| Class | Logistic Regression | Random Forest |
|---|---|---|
| 1★ | 0.50 | 0.41 |
| 2★ | 0.26 | 0.16 |
| 3★ | 0.36 | 0.28 |
| 4★ | 0.43 | 0.38 |
| 5★ | 0.80 | 0.77 |

## Approach

- **Data**: a stratified sample of **399,838 reviews**, drawn from the full 551,682-review Home & Kitchen dataset (stratified by star rating, then filtered to drop rows that preprocessed to empty text).
- **Preprocessing**: lowercase, strip punctuation and special characters, tokenize, remove English stopwords, lemmatize (WordNet).
- **Vectorization**: TF-IDF over unigrams and bigrams, `max_features=50_000`, `min_df=5`, `max_df=0.7`.
- **Models**: Logistic Regression and Random Forest, both with `class_weight='balanced'`, each wrapped in its own scikit-learn `Pipeline` (vectorizer + classifier) so the vectorizer is refit from scratch per model and per cross-validation fold — there's no leakage between folds.
- **Split**: 75/25 train/test, stratified by rating.
- **Validation**: 5-fold cross-validation on the training set, plus a final evaluation on the held-out test set.

Everything runs on CPU — this is a classic scikit-learn TF-IDF pipeline, no GPU involved at any point.

## Reproduction

1. Get the dataset — see [`data/README.md`](data/README.md) for the source, download link, and required citation.
2. Install dependencies: `pip install -r requirements.txt`
3. Open `review_rating_prediction.ipynb` and run the cells top to bottom (Jupyter, or Google Colab with `USE_COLAB_DRIVE = True` in the first code cell).

The preprocessed `.pkl` file is not distributed with this repo — it's generated automatically the first time the notebook runs, and reused on subsequent runs so the preprocessing step isn't repeated.

## Limitations & next steps

- Class imbalance is not resolved, just accounted for at training time. 2★ reviews remain the hardest class by a wide margin (F1 of 0.26 / 0.16).
- Next iteration ideas: class-balancing techniques (SMOTE or oversampling) to improve recall on negative reviews, pretrained embeddings (BERT/transformers) instead of classic TF-IDF vectorization, and deploying the model as a simple API for real-time review scoring.
- Output text labels were translated to English after the notebook's original run; word cloud plot titles remain in Spanish since they're rasterized into the saved images.

## Dataset & citation

Source: [Amazon Product Reviews Dataset](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html) (Julian McAuley, UCSD), Home & Kitchen category, 551,682 reviews. See [`data/README.md`](data/README.md) for download instructions.

Please cite one or both of the following if you use this data:

> R. He, J. McAuley. "Ups and downs: Modeling the visual evolution of fashion trends with one-class collaborative filtering." WWW, 2016.

> J. McAuley, C. Targett, J. Shi, A. van den Hengel. "Image-based recommendations on styles and substitutes." SIGIR, 2015.

## License

MIT — see [`LICENSE`](LICENSE). The license covers the code in this repository only; it does not extend to the dataset (see [`data/README.md`](data/README.md)).

---
**Author:** Santiago Cisneros — [github.com/Shantiago27](https://github.com/Shantiago27)
