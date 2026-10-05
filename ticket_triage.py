# %% [markdown]
# # AI Customer Support Ticket Triage
# **Zeravia Final Capstone Project - Artificial Intelligence**
#
# This notebook builds an AI system that reads an incoming support ticket, predicts its
# **category** (billing, technical, account, product) and its **urgency** (low, medium, high),
# routes it to the right queue, and sends low-confidence predictions to a human reviewer.
#
# Pipeline: data -> cleaning -> TF-IDF features -> two classifiers -> evaluation (F1, confusion
# matrix) -> confidence threshold + human review log -> saved models used by the Streamlit app.

# %% [markdown]
# ## 1. Setup

# %%
import os, re, json, random, warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score, accuracy_score)

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.makedirs("figures", exist_ok=True)
os.makedirs("models", exist_ok=True)
print("Setup complete")

# %% [markdown]
# ## 2. Dataset
# The notebook works with **any ticket CSV** that has the columns `text`, `category`, `urgency`.
# Set `DATA_CSV` to your file (for example a public support-ticket dataset after renaming its
# columns). If `DATA_CSV` is `None`, a reproducible **synthetic ticket dataset** is generated so the
# whole project runs offline. The generator mixes in realistic difficulty: typos, mixed-topic
# tickets, misleading urgency words, and 3% label noise.

# %%
DATA_CSV = None          # e.g. "tickets.csv"  (columns: text, category, urgency)
N_TICKETS = 6000

CATEGORIES = ["billing", "technical", "account", "product"]
URGENCY = ["low", "medium", "high"]

CORE = {
    "billing": [
        "I was charged twice for my last invoice", "my refund has not arrived yet",
        "the amount on my bill looks wrong", "I need a copy of my payment receipt",
        "my subscription fee went up without notice", "the discount code did not apply at checkout",
        "I want to cancel my plan and get a prorated refund",
        "there is an unknown charge on my credit card statement"],
    "technical": [
        "the app crashes every time I open the dashboard", "I keep getting error 500 when uploading a file",
        "the website is loading very slowly", "the API returns a timeout error",
        "sync between my phone and laptop is not working", "I cannot connect to the server from my network",
        "the export to CSV feature gives a blank file", "notifications are not being delivered"],
    "account": [
        "I forgot my password and the reset email never arrives", "I want to change the email address on my profile",
        "my account was locked after several login attempts", "please delete my account and personal data",
        "I cannot enable two factor authentication", "how do I add a new team member to my workspace",
        "my username shows the wrong name", "I need to update my phone number on file"],
    "product": [
        "does the premium plan include custom reports", "I would like to request a dark mode feature",
        "what is the difference between the basic and pro versions", "is there an integration with Slack",
        "can you tell me when the new release will be available", "how many users can I add to the team plan",
        "I love the product but I wish it had bulk editing", "do you offer a student discount for the pro version"],
}
CUES = {
    "high": ["this is urgent", "need this fixed asap", "our whole team is blocked", "production is down",
             "please respond immediately", "this is a critical issue for us", "we are losing customers because of this"],
    "medium": ["this has been going on since yesterday", "please look into it soon",
               "it is affecting my work", "I would appreciate a quick reply"],
    "low": ["no rush", "whenever you get a chance", "just curious", "not urgent at all", "just a small question"],
}
GREETINGS = ["Hi team,", "Hello,", "Dear support,", "Good morning,", ""]
SIGNOFFS = ["Thanks", "Regards", "Thank you", "Best", ""]


def add_typos(text, rng, p=0.04):
    words = text.split()
    for i, w in enumerate(words):
        if len(w) > 4 and rng.random() < p:
            j = rng.randrange(1, len(w) - 1)
            words[i] = w[:j] + w[j + 1:]
    return " ".join(words)


def make_ticket(rng):
    cat = rng.choice(CATEGORIES)
    urg = rng.choices(URGENCY, weights=[0.30, 0.45, 0.25])[0]
    parts = [rng.choice(GREETINGS), rng.choice(CORE[cat]) + "."]
    if rng.random() < 0.15:                                   # mixed-topic ticket (hard case)
        other = rng.choice([c for c in CATEGORIES if c != cat])
        parts.append("Also, " + rng.choice(CORE[other]) + ".")
    r = rng.random()
    if r < 0.70:
        cue_level = urg                                       # correct urgency wording
    elif r < 0.85:
        cue_level = rng.choice([u for u in URGENCY if u != urg])   # misleading wording
    else:
        cue_level = None                                      # no urgency wording
    if cue_level:
        parts.append(rng.choice(CUES[cue_level]).capitalize() + ".")
    parts.append(rng.choice(SIGNOFFS))
    text = add_typos(" ".join(p for p in parts if p), rng)
    label = cat if rng.random() > 0.03 else rng.choice(CATEGORIES)   # 3% annotation noise
    return text, label, urg


def load_data():
    if DATA_CSV and os.path.exists(DATA_CSV):
        d = pd.read_csv(DATA_CSV)[["text", "category", "urgency"]].dropna()
    else:
        rng = random.Random(SEED)
        d = pd.DataFrame([make_ticket(rng) for _ in range(N_TICKETS)],
                         columns=["text", "category", "urgency"])
    return d.drop_duplicates(subset="text").reset_index(drop=True)


df = load_data()
print("Tickets:", len(df))
print(df.head(5).to_string(index=False))

# %% [markdown]
# ## 3. Exploratory analysis

# %%
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
df["category"].value_counts().reindex(CATEGORIES).plot.bar(ax=axes[0], color="#2F6DB5", rot=0)
axes[0].set_title("Tickets per category")
df["urgency"].value_counts().reindex(URGENCY).plot.bar(ax=axes[1], color="#E07B39", rot=0)
axes[1].set_title("Tickets per urgency level")
plt.tight_layout()
plt.savefig("figures/class_distribution.png", dpi=200)
plt.show()
print(df["category"].value_counts().to_dict())
print(df["urgency"].value_counts().to_dict())

# %% [markdown]
# ## 4. Text cleaning

# %%
def clean_text(t):
    t = str(t).lower()
    t = re.sub(r"http\S+|www\.\S+", " url ", t)
    t = re.sub(r"\S+@\S+", " email ", t)
    t = re.sub(r"[^a-z\s]", " ", t)          # drop digits and punctuation
    return re.sub(r"\s+", " ", t).strip()


df["clean"] = df["text"].apply(clean_text)
print(df[["text", "clean"]].head(3).to_string(index=False))

# %% [markdown]
# ## 5. Train / test split (80 / 20, stratified by category)

# %%
train, test = train_test_split(df, test_size=0.2, stratify=df["category"], random_state=SEED)
print("Train:", len(train), " Test:", len(test))

# %% [markdown]
# ## 6. Model comparison (5-fold cross-validation on the training set)
# Three standard text classifiers on TF-IDF features (unigrams + bigrams). The final model is
# Logistic Regression because it outputs calibrated-style probabilities, which the confidence
# threshold in Section 9 needs.

# %%
def tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)


candidates = {
    "Naive Bayes": MultinomialNB(alpha=0.3),
    "Linear SVM": LinearSVC(C=1.0, class_weight="balanced"),
    "Logistic Regression": LogisticRegression(C=5, max_iter=1000, class_weight="balanced"),
}
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
cv_rows = []
for name, clf in candidates.items():
    for target in ["category", "urgency"]:
        pipe = Pipeline([("tfidf", tfidf()), ("clf", clf)])
        scores = cross_val_score(pipe, train["clean"], train[target], cv=cv, scoring="f1_macro")
        cv_rows.append({"model": name, "target": target, "cv_macro_f1": round(scores.mean(), 4)})
cv_table = pd.DataFrame(cv_rows).pivot(index="model", columns="target", values="cv_macro_f1")
print(cv_table)

# %% [markdown]
# ## 7. Train the final category and urgency models

# %%
def build_model():
    return Pipeline([("tfidf", tfidf()),
                     ("clf", LogisticRegression(C=5, max_iter=1000, class_weight="balanced"))])


cat_model = build_model().fit(train["clean"], train["category"])
urg_model = build_model().fit(train["clean"], train["urgency"])
print("Both models trained")

# %% [markdown]
# ## 8. Evaluation on the held-out test set (F1-score, confusion matrix, precision/recall)

# %%
def plot_cm(y_true, y_pred, labels, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap="Blues", colorbar=False,
                                                           values_format="d")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.show()


cat_pred = cat_model.predict(test["clean"])
urg_pred = urg_model.predict(test["clean"])

print("=== CATEGORY MODEL ===")
print(classification_report(test["category"], cat_pred, labels=CATEGORIES, digits=3))
plot_cm(test["category"], cat_pred, CATEGORIES, "Category model - confusion matrix",
        "figures/confusion_category.png")

print("=== URGENCY MODEL ===")
print(classification_report(test["urgency"], urg_pred, labels=URGENCY, digits=3))
plot_cm(test["urgency"], urg_pred, URGENCY, "Urgency model - confusion matrix",
        "figures/confusion_urgency.png")

metrics = {
    "category": {"accuracy": accuracy_score(test["category"], cat_pred),
                 "macro_f1": f1_score(test["category"], cat_pred, average="macro"),
                 "weighted_f1": f1_score(test["category"], cat_pred, average="weighted"),
                 "report": classification_report(test["category"], cat_pred, labels=CATEGORIES,
                                                 output_dict=True)},
    "urgency": {"accuracy": accuracy_score(test["urgency"], urg_pred),
                "macro_f1": f1_score(test["urgency"], urg_pred, average="macro"),
                "weighted_f1": f1_score(test["urgency"], urg_pred, average="weighted"),
                "report": classification_report(test["urgency"], urg_pred, labels=URGENCY,
                                                output_dict=True)},
}
for k in metrics:
    print(f"{k}: accuracy={metrics[k]['accuracy']:.3f}  macro-F1={metrics[k]['macro_f1']:.3f}")

# %% [markdown]
# ## 9. Explainability - most influential words per class

# %%
def top_terms(model, k=6):
    """Highest-weight single words (unigrams) for each class."""
    names = np.array(model.named_steps["tfidf"].get_feature_names_out())
    clf = model.named_steps["clf"]
    uni = np.array([" " not in n for n in names])
    out = {}
    for i, lab in enumerate(clf.classes_):
        w = np.where(uni, clf.coef_[i], -np.inf)
        out[lab] = names[np.argsort(w)[-k:][::-1]].tolist()
    return out


terms_cat, terms_urg = top_terms(cat_model), top_terms(urg_model)
for lab, words in {**terms_cat, **terms_urg}.items():
    print(f"{lab:10s} -> {', '.join(words)}")

# %% [markdown]
# ## 10. Confidence threshold and human review
# A ticket is sent to **manual review** when the category confidence is below `T_CAT` or the
# urgency confidence is below `T_URG`. The sweep shows the trade-off: a higher threshold means
# fewer tickets are automated but those that are automated are more accurate.

# %%
T_CAT, T_URG = 0.60, 0.50

cat_proba = cat_model.predict_proba(test["clean"])
urg_proba = urg_model.predict_proba(test["clean"])
cat_conf, urg_conf = cat_proba.max(axis=1), urg_proba.max(axis=1)
cat_ok = (cat_pred == test["category"].values)
urg_ok = (urg_pred == test["urgency"].values)

auto = (cat_conf >= T_CAT) & (urg_conf >= T_URG)
review_stats = {
    "T_CAT": T_CAT, "T_URG": T_URG,
    "auto_routed_pct": float(auto.mean() * 100),
    "sent_to_review_pct": float((~auto).mean() * 100),
    "category_acc_auto": float(cat_ok[auto].mean()),
    "category_acc_review": float(cat_ok[~auto].mean()) if (~auto).any() else None,
    "category_acc_all": float(cat_ok.mean()),
}
print(json.dumps(review_stats, indent=2))

ths = np.arange(0.30, 0.91, 0.05)
cov = [(cat_conf >= t).mean() * 100 for t in ths]
acc = [cat_ok[cat_conf >= t].mean() * 100 if (cat_conf >= t).any() else np.nan for t in ths]
fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.plot(ths, cov, marker="o", label="Tickets automated (%)", color="#2F6DB5")
ax.plot(ths, acc, marker="s", label="Accuracy on automated (%)", color="#E07B39")
ax.axvline(T_CAT, ls="--", color="grey")
ax.set_xlabel("Category confidence threshold")
ax.set_ylabel("Percent")
ax.set_title("Automation vs accuracy trade-off")
ax.legend()
plt.tight_layout()
plt.savefig("figures/threshold_tradeoff.png", dpi=200)
plt.show()

# %% [markdown]
# ## 11. Routing function with review log
# `predict_ticket` is the function used by the Streamlit app.

# %%
ROUTES = {"billing": "Billing & Payments Team", "technical": "Technical Support Team",
          "account": "Account Services Team", "product": "Product & Sales Team"}
PRIORITY = {"high": "P1 - respond within 1 hour", "medium": "P2 - respond within 8 hours",
            "low": "P3 - respond within 48 hours"}      # illustrative service levels


def predict_ticket(text, log_path="review_log.csv"):
    clean = clean_text(text)
    pc, pu = cat_model.predict_proba([clean])[0], urg_model.predict_proba([clean])[0]
    cat, urg = cat_model.classes_[pc.argmax()], urg_model.classes_[pu.argmax()]
    cc, uc = float(pc.max()), float(pu.max())
    needs_review = cc < T_CAT or uc < T_URG
    result = {"category": cat, "category_confidence": round(cc, 3),
              "urgency": urg, "urgency_confidence": round(uc, 3),
              "queue": "Manual review" if needs_review else ROUTES[cat],
              "priority": PRIORITY[urg], "needs_review": needs_review}
    if needs_review:                                       # log low-confidence predictions
        row = pd.DataFrame([{"time": datetime.now().isoformat(timespec="seconds"), "text": text, **result}])
        row.to_csv(log_path, mode="a", header=not os.path.exists(log_path), index=False)
    return result


samples = [
    "Hi team, I was charged twice for my invoice and this is urgent. Thanks",
    "The API returns a timeout error and our whole team is blocked, need this fixed asap",
    "Just curious, is there an integration with Slack? No rush.",
    "I forgot my password and the reset email never arrives. Please look into it soon.",
    "hello my thing is broken",
]
sample_rows = []
for s in samples:
    r = predict_ticket(s)
    sample_rows.append({"ticket": s, **r})
    print(f"{s[:70]!r}\n   -> {r['category']} ({r['category_confidence']}) | {r['urgency']} "
          f"({r['urgency_confidence']}) | {r['queue']} | {r['priority']}\n")

# %% [markdown]
# ## 12. Save models and results (used by `app.py` and the report)

# %%
joblib.dump(cat_model, "models/category_model.joblib")
joblib.dump(urg_model, "models/urgency_model.joblib")

results = {
    "n_total": len(df), "n_train": len(train), "n_test": len(test),
    "class_counts_category": df["category"].value_counts().to_dict(),
    "class_counts_urgency": df["urgency"].value_counts().to_dict(),
    "cv_table": cv_table.round(4).to_dict(),
    "metrics": metrics, "review": review_stats,
    "top_terms_category": terms_cat, "top_terms_urgency": terms_urg,
    "samples": sample_rows,
}
with open("results.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
print("Saved models/ and results.json")
