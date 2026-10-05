# AI Customer Support Ticket Triage

An NLP system that reads a customer support ticket, predicts its **category** (billing, technical,
account, product) and **urgency** (low, medium, high), routes it to the right team, and sends
low-confidence predictions to a human reviewer.

Built as the Final Capstone Project for the Zeravia Training & Internship Program (Artificial Intelligence).

## Approach
1. Collect tickets (CSV with `text`, `category`, `urgency`) - or generate the built-in synthetic dataset
2. Clean text (lowercase, strip URLs/emails/digits/punctuation)
3. TF-IDF features (unigrams + bigrams)
4. Train two Logistic Regression classifiers (category + urgency); compared against Naive Bayes and Linear SVM with 5-fold CV
5. Evaluate with F1-score, confusion matrix and per-class precision/recall
6. Confidence thresholds send uncertain tickets to a manual-review log
7. Streamlit app for live demo

## Results (held-out test set, 1,080 tickets)
| Model | Accuracy | Macro F1 |
|---|---|---|
| Category | 0.972 | 0.972 |
| Urgency | 0.752 | 0.743 |

About 90% of tickets are auto-routed; the rest go to manual review. Full numbers are in `results.json`.
The bundled dataset is synthetic, so absolute scores will differ on real tickets.

## Run it
```bash
pip install -r requirements.txt
python ticket_triage.py        # or open ticket_triage.ipynb in Jupyter / Google Colab
streamlit run app.py
```
To use your own data, set `DATA_CSV = "your_file.csv"` in the notebook (columns: `text`, `category`, `urgency`).

## Files
| File | Purpose |
|---|---|
| `ticket_triage.ipynb` / `ticket_triage.py` | Full pipeline: data, training, evaluation, saving models |
| `app.py` | Streamlit demo |
| `models/` | Saved category and urgency models |
| `figures/` | Confusion matrices, class distribution, threshold trade-off |
| `results.json` | All metrics |
