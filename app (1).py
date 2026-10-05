"""Streamlit demo for AI Customer Support Ticket Triage.

Run the notebook (or `python ticket_triage.py`) first so that models/ exists, then:
    streamlit run app.py
"""
import os
import re
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

T_CAT, T_URG = 0.60, 0.50
ROUTES = {"billing": "Billing & Payments Team", "technical": "Technical Support Team",
          "account": "Account Services Team", "product": "Product & Sales Team"}
PRIORITY = {"high": "P1 - respond within 1 hour", "medium": "P2 - respond within 8 hours",
            "low": "P3 - respond within 48 hours"}
EXAMPLES = {
    "Billing - urgent": "Hi team, I was charged twice for my invoice and this is urgent. Thanks",
    "Technical - blocked": "The API returns a timeout error and our whole team is blocked, need this fixed asap",
    "Product - low priority": "Just curious, is there an integration with Slack? No rush.",
    "Vague ticket (goes to review)": "hello my thing is broken",
}


def clean_text(t):
    t = str(t).lower()
    t = re.sub(r"http\S+|www\.\S+", " url ", t)
    t = re.sub(r"\S+@\S+", " email ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


@st.cache_resource
def load_models():
    return (joblib.load("models/category_model.joblib"),
            joblib.load("models/urgency_model.joblib"))


st.set_page_config(page_title="AI Ticket Triage", page_icon="🎫", layout="centered")
st.title("🎫 AI Customer Support Ticket Triage")
st.caption("Predicts ticket category and urgency, routes it to a queue, and flags uncertain tickets for human review.")

if not os.path.exists("models/category_model.joblib"):
    st.error("Models not found. Run the notebook or `python ticket_triage.py` first.")
    st.stop()
cat_model, urg_model = load_models()

choice = st.selectbox("Load an example ticket (optional)", ["-"] + list(EXAMPLES))
text = st.text_area("Ticket text", value=EXAMPLES.get(choice, ""), height=140,
                    placeholder="Paste a customer support ticket here...")

if st.button("Triage ticket", type="primary") and text.strip():
    clean = clean_text(text)
    pc, pu = cat_model.predict_proba([clean])[0], urg_model.predict_proba([clean])[0]
    cat, urg = cat_model.classes_[pc.argmax()], urg_model.classes_[pu.argmax()]
    cc, uc = float(pc.max()), float(pu.max())
    review = cc < T_CAT or uc < T_URG

    c1, c2 = st.columns(2)
    c1.metric("Category", cat.title(), f"{cc:.0%} confidence", delta_color="off")
    c2.metric("Urgency", urg.title(), f"{uc:.0%} confidence", delta_color="off")

    if review:
        st.warning("Low confidence - sent to **Manual review**. Logged in review_log.csv.")
        row = pd.DataFrame([{"time": datetime.now().isoformat(timespec="seconds"), "text": text,
                             "category": cat, "category_confidence": round(cc, 3),
                             "urgency": urg, "urgency_confidence": round(uc, 3)}])
        row.to_csv("review_log.csv", mode="a", header=not os.path.exists("review_log.csv"), index=False)
    else:
        st.success(f"Routed to **{ROUTES[cat]}**  |  {PRIORITY[urg]}")

    with st.expander("Class probabilities"):
        st.bar_chart(pd.DataFrame({"category": pd.Series(pc, index=cat_model.classes_)}))
        st.bar_chart(pd.DataFrame({"urgency": pd.Series(pu, index=urg_model.classes_)}))
