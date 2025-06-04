import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

st.set_page_config(page_title="Lead Scoring Tool", layout="wide")

@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df

def auto_detect_target_column(df):
    candidates = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if not candidates:
        candidates = [col for col in df.columns if df[col].nunique() < 10 and df[col].nunique() > 1]
    if not candidates:
        raise ValueError("No suitable target column found.")
    target = min(candidates, key=lambda col: df[col].nunique())
    return target

def preprocess_data(df, target_column):
    df = df.copy()
    df[target_column] = df[target_column].astype(str)
    df = pd.get_dummies(df)
    return df

def score_leads(model, df, target_column, feature_columns):
    df_processed = preprocess_data(df, target_column)
    X = df_processed[feature_columns]
    proba = model.predict_proba(X)
    lead_score = proba.max(axis=1)
    df_scored = df.copy()
    df_scored['Lead Score'] = lead_score
    df_scored['Lead Category'] = pd.cut(df_scored['Lead Score'], bins=[0, 0.33, 0.66, 1], labels=['Low', 'Medium', 'High'])
    return df_scored

def highlight_scores(val):
    if val == 'High':
        return 'background-color: #a1e3a1'
    elif val == 'Medium':
        return 'background-color: #f9f79f'
    else:
        return 'background-color: #f4a7a7'

def download_csv(df):
    output = BytesIO()
    df.to_csv(output, index=False)
    return output.getvalue()

st.sidebar.title("Upload Your Lead Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)
    st.write("### Uploaded Data Preview")
    st.dataframe(df.head())

    try:
        target_column = auto_detect_target_column(df)
        st.success(f"Auto-detected target column: **{target_column}**")

        if st.sidebar.button("Train & Score Leads"):
            with st.spinner("Preprocessing and splitting data..."):
                df_processed = preprocess_data(df, target_column)
                target_cols = [col for col in df_processed.columns if col.startswith(target_column + '_')]
                X = df_processed.drop(columns=target_cols)
                y = df_processed[target_cols]
                y_labels = y.idxmax(axis=1).apply(lambda x: x[len(target_column)+1:])
                X_train, X_test, y_train, y_test = train_test_split(X, y_labels, test_size=0.3, random_state=42)

            with st.spinner("Training model..."):
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)

            with st.spinner("Evaluating model and scoring leads..."):
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)
                acc = accuracy_score(y_test, y_pred)

                try:
                    auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
                except ValueError:
                    auc = None

                df_scored = score_leads(model, df, target_column, X.columns)

            st.success("Lead Scoring Complete!")

            st.write("### Scored Leads (with Categories)")
            st.dataframe(df_scored.style.applymap(highlight_scores, subset=['Lead Category']))

            st.write("### Model Performance")
            st.metric("Accuracy", f"{acc:.2f}")
            if auc is not None:
                st.metric("AUC Score", f"{auc:.2f}")
            else:
                st.info("AUC Score not available for this dataset.")

            st.download_button(
                label="Download Scored Leads CSV",
                data=download_csv(df_scored),
                file_name='scored_leads.csv',
                mime='text/csv'
            )

            st.markdown("---")
            st.info("**Note:** This app automatically detects the target column and trains on 70% of the data.")

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.warning("Please upload a CSV file to begin.")
