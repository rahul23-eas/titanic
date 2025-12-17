import os
from typing import Any, Dict

import requests
import streamlit as st

# =============================================================================
# Page config (MUST be first)
# =============================================================================
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="",
    layout="centered",
)

# =============================================================================
# Config
# =============================================================================
API_BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
PREDICT_ENDPOINT = f"{API_BASE_URL}/predict"

# =============================================================================
# Header
# =============================================================================
st.title(" Titanic Survival Prediction")
st.markdown(
    """
Predict whether a passenger **survived the Titanic disaster** using a
machine-learning model served via **FastAPI**.
"""
)

st.info(f" Connected API: `{PREDICT_ENDPOINT}`")

st.markdown("---")

# =============================================================================
# Input Section
# =============================================================================
st.header("留 Passenger Information")

user_input: Dict[str, Any] = {}

col1, col2 = st.columns(2)

with col1:
    user_input["Pclass"] = st.selectbox(
        "Passenger Class",
        options=[1, 2, 3],
        index=2,
        help="1 = First class, 2 = Second class, 3 = Third class",
    )

    user_input["Age"] = st.slider(
        "Age",
        min_value=0,
        max_value=80,
        value=28,
        step=1,
    )

    user_input["SibSp"] = st.slider(
        "Siblings / Spouses aboard",
        min_value=0,
        max_value=8,
        value=0,
    )

with col2:
    user_input["sex"] = st.selectbox(
        "Sex",
        options=["male", "female"],
    )

    user_input["Fare"] = st.number_input(
        "Fare (£)",
        min_value=0.0,
        max_value=600.0,
        value=32.0,
        step=1.0,
    )

    user_input["Parch"] = st.slider(
        "Parents / Children aboard",
        min_value=0,
        max_value=6,
        value=0,
    )

st.markdown("---")

# =============================================================================
# Predict Button
# =============================================================================
if st.button(" Predict Survival", use_container_width=True):
    payload = {"instances": [user_input]}

    with st.spinner("Calling FastAPI for prediction..."):
        try:
            response = requests.post(
                PREDICT_ENDPOINT,
                json=payload,
                timeout=30,
            )
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Failed to connect to API:\n\n{e}")
        else:
            if response.status_code != 200:
                st.error(
                    f"❌ API Error (HTTP {response.status_code})\n\n{response.text}"
                )
            else:
                result = response.json()
                predictions = result.get("predictions", [])

                if not predictions:
                    st.warning("⚠️ No prediction returned by the API.")
                else:
                    prediction = predictions[0]

                    st.success("✅ Prediction successful")

                    st.markdown("###  Result")

                    if prediction == 1:
                        st.metric(
                            label="Survival Outcome",
                            value=" Survived",
                        )
                    else:
                        st.metric(
                            label="Survival Outcome",
                            value=" Did NOT Survive",
                        )

                    with st.expander(" Submitted Passenger Data"):
                        st.json(user_input)

st.markdown("---")
st.caption("⚙️ Powered by FastAPI + Scikit-Learn | Titanic ML Classification")
