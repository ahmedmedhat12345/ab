import streamlit as st
import pandas as pd
import joblib
import numpy as np
import xgboost as xgb
import os

# --- LLM / Manual Fallback Function ---
def get_explanation(features, predicted_price, median_price):
    # Manual explanation logic (Baseline)
    diff_pct = ((predicted_price - median_price) / median_price) * 100
    
    manual_text = ""
    if diff_pct > 10:
        manual_text = f"This property is valued {diff_pct:.1f}% above the market median. This premium is likely driven by superior features."
    elif diff_pct < -10:
        manual_text = f"This property is valued {abs(diff_pct):.1f}% below the market median, potentially offering good value."
    else:
        manual_text = "This property is valued close to the market median, reflecting standard market conditions."

    # Try Connecting to OpenAI
    try:
        # Check for API ID in Secrets (Streamlit Cloud) or Env Var
        api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("No API Key found")

        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        prompt = f"Explain why a house with {features} is valued at ${predicted_price:,.0f} when the median is ${median_price:,.0f}. Keep it under 50 words."
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a real estate expert."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=60
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        # SILENT FALLBACK: If ANY error occurs (no key, quota, net), return manual text
        # print(f"LLM Error: {e}") # Debugging only
        return manual_text

# --- Main App ---
st.set_page_config(page_title="Real Estate AI", layout="wide")

# Load Models
try:
    clf = joblib.load('category_classifier.pkl')
    reg = joblib.load('price_regressor.pkl')
    cls_feats = joblib.load('cls_features.pkl')
    reg_feats = joblib.load('reg_features.pkl')
except:
    st.error("Models not found. Please run the notebook first to generate .pkl files.")
    st.stop()

st.title("Real Estate Valuation AI 🏡")
st.markdown("Predicts house price and category, with AI-driven market insights.")

# Sidebar Inputs
st.sidebar.header("Property Details")
input_data = {}

# Raw inputs
sqft_living = st.sidebar.number_input("SqFt Living", 500, 10000, 2000)
grade = st.sidebar.slider("Grade (1-13)", 1, 13, 7)
yr_built = st.sidebar.number_input("Year Built", 1900, 2025, 2000)
bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 5, 2)
yr_renovated = st.sidebar.number_input("Year Renovated (0 if none)", 0, 2025, 0)

# Feature Engineering
current_year = 2025
input_dict = {
    'sqft_living': sqft_living,
    'grade': grade,
    'yr_built': yr_built,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'yr_renovated': yr_renovated,
    # Engineered
    'house_age': current_year - yr_built,
    'has_renovated': 1 if yr_renovated > 0 else 0,
    'grade_sqft': grade * sqft_living
}

# Fill others with 0 (zipcode, etc. not in UI)
for feat in reg_feats:
    if feat not in input_dict:
        input_dict[feat] = 0

input_df = pd.DataFrame([input_dict])

if st.sidebar.button("Values & Insights"):
    col1, col2 = st.columns(2)
    
    # Classification
    try:
        cat = clf.predict(input_df[cls_feats])[0]
        col1.subheader(f"Category: {cat}")
    except:
        col1.warn("Classification unavailable")
    
    # Regression
    try:
        price_log = reg.predict(input_df[reg_feats])[0]
        price = np.expm1(price_log)
        col2.metric("Estimated Price", f"${price:,.0f}")
        
        # Insight
        st.divider()
        st.subheader("Market Insight")
        
        median_price = 450000 # Could be loaded dynamically
        
        # Pass features for LLM context
        feat_summary = f"{sqft_living}sqft, Grade {grade}, Built {yr_built}"
        
        with st.spinner("Generating expert analysis..."):
            explanation = get_explanation(feat_summary, price, median_price)
            st.success(explanation)
            
    except Exception as e:
       st.error(f"Prediction Error: {e}")
