import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Fretting Regime Predictor",
    page_icon="⚙️",
)

st.title("Fretting Regime Classification Predictor")

st.markdown(
    "Enter the ball-on-flat fretting contact parameters to predict the slip regime."
)

# Input parameters
P = st.number_input(
    "Normal Load, P (N)",
    min_value=0.01,
    value=50.0,
    step=1.0,
)

D = st.number_input(
    "Displacement Amplitude, D (mm)",
    min_value=0.0001,
    value=0.025,
    format="%.4f",
)

mu = st.number_input(
    "Coefficient of Friction, μ",
    min_value=0.01,
    max_value=1.0,
    value=0.5,
    step=0.05,
)

a = st.number_input(
    "Contact Radius, a (mm)",
    min_value=0.0001,
    value=0.11804,
    format="%.5f",
)

if st.button("Predict Fretting Regime"):
    input_data = {
        "P": P,
        "D": D,
        "μ": mu,
        "a": a,
    }

    try:
        response = requests.post(
            API_URL,
            json=input_data,
            timeout=10,
        )

        result = response.json()

        if response.status_code == 200:
            predicted_regime = result["predicted_regime"]

            st.success(f"Predicted Fretting Regime: **{predicted_regime}**")

            if "derived_features" in result:
                st.subheader("Derived Features")
                st.json(result["derived_features"])

        else:
            st.error(f"API Error: {response.status_code}")
            st.json(result)

    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to the FastAPI server. "
            "Make sure the FastAPI application is running."
        )

    except requests.exceptions.Timeout:
        st.error("The API request timed out.")

    except requests.exceptions.RequestException as error:
        st.error(f"API request failed: {error}")
