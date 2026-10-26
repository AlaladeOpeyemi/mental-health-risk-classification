# Author: Opeyemi Alalade
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Mental Health Risk Classifier", page_icon="🧠", layout="wide")


# ---------- 1. Load the saved model and preprocessing objects ----------
@st.cache_resource
def load_artifacts():
    return joblib.load("OpeyemiAlalade_mental_health_artifacts.joblib")


A = load_artifacts()
model = A["model"]
scaler = A["scaler"]
fill_values = A["fill_values"]          # training medians for the 85 encoded columns
all_columns = A["all_columns"]          # the 94 columns the scaler was fitted on
selected = A["selected_features"]       # the 42 columns the model uses
class_order = A["class_order"]          # ['Healthy', 'Needs Attention', 'Critical']


# ---------- 2. The same feature-engineering recipe used in the notebook ----------
def engineer(d):
    d = d.copy()
    d["Sleep_Deficit"] = (8 - d["Sleep_Hours"]).clip(lower=0)
    d["Leisure_Screen_Hours"] = d["Social_Media_Hours"] + d["Gaming_Hours"]
    d["Psych_Distress_Index"] = (d["Anxiety_Score"] + d["Depression_Score"]
                                 + (100 - d["Mood_Score"]) + (100 - d["Emotional_Stability"])) / 4
    d["Work_Strain"] = d["Work_Hours_Per_Week"] * (11 - d["Work_Life_Balance"])
    d["Recovery_Index"] = d["Physical_Activity_Hours"] + d["Meditation_Minutes"] / 60 + d["Sleep_Hours"]
    d["Unhealthy_Habits_Count"] = d["Alcohol_Consumption"] + d["Smoking"] + (1 - d["Healthy_Diet"])
    d["Income_Per_Work_Hour"] = d["Monthly_Income_USD"] / (d["Work_Hours_Per_Week"] * 4.33)
    d["Satisfaction_Index"] = (d["Job_Satisfaction"] + d["Life_Satisfaction"]) / 2
    d["Protective_Factors"] = (d["Support_System"] + d["Exercise_Frequency"]
                               + d["Therapy_Attendance"] + d["Healthy_Diet"])
    return d


# ---------- 3. Page header ----------
st.title("🧠 Mental Health Risk Classifier")
st.caption(
    "Prototype built for the DataLab Analytics final project. It estimates the probability that a person is "
    "**Healthy**, **Needs Attention** or **Critical**. It is a screening aid trained on synthetic data and "
    "**does not replace a clinical diagnosis or professional assessment.**"
)

# ---------- 4. Input form ----------
def default(col, fallback):
    try:
        return fill_values[col]
    except KeyError:
        return fallback


def slider(label, col, lo, hi, is_int=True, step=None):
    d = float(np.clip(default(col, (lo + hi) / 2), lo, hi))
    if is_int:
        return st.slider(label, int(lo), int(hi), int(round(d)), key=col)
    return st.slider(label, float(lo), float(hi), round(d, 1), step=step or 0.1, key=col)


def yes_no(label, col):
    idx = 1 if default(col, 0) >= 0.5 else 0
    return 1 if st.selectbox(label, ["No", "Yes"], index=idx, key=col) == "Yes" else 0


def ordinal(label, col, options):
    d = int(round(default(col, 0)))
    d = min(max(d, 0), len(options) - 1)
    return options.index(st.selectbox(label, options, index=d, key=col))


st.subheader("Enter the person's details")
vals = {}

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**Psychological scores (0-100)**")
    vals["Anxiety_Score"] = slider("Anxiety score", "Anxiety_Score", 0, 100)
    vals["Depression_Score"] = slider("Depression score", "Depression_Score", 0, 100)
    vals["Mood_Score"] = slider("Mood score (higher = better)", "Mood_Score", 0, 100)
    vals["Emotional_Stability"] = slider("Emotional stability (higher = better)", "Emotional_Stability", 0, 100)
    vals["Productivity_Score"] = slider("Productivity score", "Productivity_Score", 0, 100)
with c2:
    st.markdown("**Stress and work**")
    vals["Stress_Level"] = ordinal("Daily stress level", "Stress_Level", ["Low", "Moderate", "High"])
    vals["Chronic_Stress"] = yes_no("Chronic (long-term) stress?", "Chronic_Stress")
    vals["Work_Hours_Per_Week"] = slider("Work hours per week", "Work_Hours_Per_Week", 20, 70)
    vals["Work_Life_Balance"] = slider("Work-life balance (1-10)", "Work_Life_Balance", 1, 10)
    vals["Job_Satisfaction"] = slider("Job satisfaction (1-10)", "Job_Satisfaction", 1, 10)
with c3:
    st.markdown("**Sleep and lifestyle**")
    vals["Sleep_Hours"] = slider("Sleep hours per night", "Sleep_Hours", 4.0, 10.0, is_int=False, step=0.1)
    vals["Sleep_Quality"] = ordinal("Sleep quality", "Sleep_Quality", ["Poor", "Average", "Good", "Excellent"])
    vals["Screen_Time_Hours"] = slider("Screen time (hours/day)", "Screen_Time_Hours", 2.0, 14.0, is_int=False)
    vals["Coffee_Cups_Per_Day"] = slider("Coffee cups per day", "Coffee_Cups_Per_Day", 0, 8)
    vals["Physical_Activity_Hours"] = slider("Physical activity (hours/week)", "Physical_Activity_Hours", 0.0, 10.0, is_int=False)
    vals["Meditation_Minutes"] = slider("Meditation (minutes/week)", "Meditation_Minutes", 0, 60)

with st.expander("More details (optional, defaults are the training-set typical values)"):
    d1, d2, d3 = st.columns(3)
    with d1:
        vals["Age"] = slider("Age", "Age", 18, 65)
        vals["Monthly_Income_USD"] = slider("Monthly income (USD)", "Monthly_Income_USD", 500, 12000)
        vals["Education_Level"] = ordinal("Education level", "Education_Level",
                                          ["High School", "Diploma", "Bachelor", "Master", "PhD"])
        vals["Absenteeism_Days"] = slider("Absenteeism days", "Absenteeism_Days", 0, 30)
    with d2:
        vals["Life_Satisfaction"] = slider("Life satisfaction (1-10)", "Life_Satisfaction", 1, 10)
        vals["Social_Media_Hours"] = slider("Social media (hours/day)", "Social_Media_Hours", 0.0, 8.0, is_int=False)
        vals["Gaming_Hours"] = slider("Gaming (hours/day)", "Gaming_Hours", 0.0, 6.0, is_int=False)
        vals["Support_System"] = ordinal("Support system", "Support_System", ["Poor", "Average", "Good", "Excellent"])
        vals["Exercise_Frequency"] = ordinal("Exercise frequency", "Exercise_Frequency",
                                             ["Never", "Rarely", "Weekly", "Daily"])
    with d3:
        vals["Alcohol_Consumption"] = yes_no("Regular alcohol use?", "Alcohol_Consumption")
        vals["Smoking"] = yes_no("Regular smoking?", "Smoking")
        vals["Healthy_Diet"] = yes_no("Healthy diet?", "Healthy_Diet")
        vals["Family_History_Mental_Illness"] = yes_no("Family history of mental illness?",
                                                       "Family_History_Mental_Illness")
        vals["Therapy_Attendance"] = yes_no("Currently attending therapy?", "Therapy_Attendance")
        remote = st.selectbox("Remote work", ["No", "Hybrid", "Yes"], index=0)

predict = st.button("Predict mental health status", type="primary")


# ---------- 5. Turn the form into the exact row the model expects ----------
def build_row():
    row = fill_values.to_dict()                  # start from training medians (85 columns)
    row.update(vals)                             # overwrite with what the user entered
    for opt in ["No", "Hybrid", "Yes"]:          # one-hot Remote_Work
        key = "Remote_Work_" + opt
        if key in row:
            row[key] = 1 if remote == opt else 0
    base = pd.DataFrame([row])[list(fill_values.index)]
    full = engineer(base).reindex(columns=all_columns)       # the 94 columns, same order as training
    scaled = pd.DataFrame(scaler.transform(full), columns=all_columns)
    return scaled[selected]                                  # the 42 columns the model uses


# ---------- 6. Predict and explain ----------
def explain(x, class_idx):
    """Contribution of every feature to the probability of the chosen class."""
    if hasattr(model, "coef_"):                                  # Logistic Regression
        return pd.Series(model.coef_[class_idx] * x.iloc[0].values, index=x.columns)
    import shap                                                  # tree models
    sv = shap.TreeExplainer(model).shap_values(x)
    if isinstance(sv, list):
        sv = np.stack(sv, axis=-1)
    sv = np.array(sv)
    if sv.ndim == 3:
        contrib = sv[0, :, class_idx]
    else:
        contrib = sv[0]
    return pd.Series(contrib, index=x.columns)


if predict:
    X_row = build_row()
    proba = model.predict_proba(X_row)[0]
    pred_idx = int(np.argmax(proba))
    pred_name = class_order[pred_idx]

    st.divider()
    st.subheader("Prediction")
    icon = {"Healthy": "🟢", "Needs Attention": "🟠", "Critical": "🔴"}[pred_name]
    st.markdown(f"### {icon} The model predicts: **{pred_name}**")

    m1, m2, m3 = st.columns(3)
    for col, name, p in zip([m1, m2, m3], class_order, proba):
        col.metric(name, f"{p * 100:.2f}%")
        col.progress(float(p))

    st.write(
        f"The model predicts a **{proba[2] * 100:.2f}% probability** that this person is in the "
        f"**Critical** group, **{proba[1] * 100:.2f}%** for **Needs Attention** and "
        f"**{proba[0] * 100:.2f}%** for **Healthy**."
    )

    st.subheader(f"Why this prediction? (what pushed towards '{pred_name}')")
    contrib = explain(X_row, pred_idx)
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(8)[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(top.index, top.values, color=["#e76f51" if v > 0 else "#2a9d8f" for v in top.values])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(f"Push towards (+) or away from (-) '{pred_name}'")
    ax.set_title("Top 8 factors for this person")
    plt.tight_layout()
    st.pyplot(fig)
    st.caption(
        "Red bars push the prediction towards the predicted class and green bars push away from it. "
        "Longer bars mean a stronger influence for this person."
    )

    up = [f for f, v in top[::-1].items() if v > 0][:3]
    down = [f for f, v in top[::-1].items() if v < 0][:3]
    if up:
        st.write("**Main factors supporting this prediction:** " + ", ".join(up).replace("_", " "))
    if down:
        st.write("**Main factors pulling against it:** " + ", ".join(down).replace("_", " "))
else:
    st.info("Adjust the values above and click **Predict mental health status**.")
