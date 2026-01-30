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

# --- Validation Logic ---
def validate_inputs(data):
    """
    Checks for impossible or highly unlikely feature combinations.
    Returns: (is_valid, message, corrected_data)
    """
    warnings = []
    
    # 1. Grade vs SqFt checks
    # Grade 11+ (Excellent) usually implies significant size (>1500 sqft)
    if data['grade'] >= 11 and data['sqft_living'] < 1000:
        warnings.append("⚠️ High Grade (11+) is inconsistent with small size (<1000 sqft). Grade adjusted to 8.")
        data['grade'] = 8 # Auto-correct to a good but not luxury grade
        
    # Grade < 5 (Low) usually implies older/smaller or huge dilapidated. 
    # If huge (>4000) and low grade, it's possible but rare. Let's warn.
    if data['grade'] <= 5 and data['sqft_living'] > 4000:
        warnings.append("⚠️ Large property (>4000 sqft) with low Grade (<=5). Please confirm condition.")

    # 2. Bedrooms vs SqFt
    # Tiny rooms check
    if data['bedrooms'] > 0 and (data['sqft_living'] / data['bedrooms'] < 150):
        warnings.append(f"⚠️ {data['bedrooms']} bedrooms in {data['sqft_living']} sqft is unrealistic. Beds adjusted to reasonable count.")
        data['bedrooms'] = max(1, int(data['sqft_living'] / 200))

    # 3. Bathrooms
    if data['bathrooms'] > data['bedrooms'] + 2:
         warnings.append("⚠️ More bathrooms than bedrooms + 2 is unusual. Please verify.")

    return warnings, data

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

# Raw inputs with cleaner limits
sqft_living = st.sidebar.number_input("SqFt Living", 300, 10000, 2000) # Enforced Min 300
grade = st.sidebar.slider("Grade (1-13)", 1, 13, 7)
yr_built = st.sidebar.number_input("Year Built", 1900, 2025, 2000)
bedrooms = st.sidebar.slider("Bedrooms", 0, 10, 3) # Cap at 10 for UI
bathrooms = st.sidebar.slider("Bathrooms", 0.0, 8.0, 2.0, 0.5) # Cap at 8
yr_renovated = st.sidebar.number_input("Year Renovated (0 if none)", 0, 2025, 0)

if st.sidebar.button("Values & Insights"):
    # 1. Prepare Initial Data
    raw_data = {
        'sqft_living': sqft_living,
        'grade': grade,
        'yr_built': yr_built,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'yr_renovated': yr_renovated
    }
    
    # 2. Validate & Correct
    warnings, validated_data = validate_inputs(raw_data.copy())
    
    for w in warnings:
        st.warning(w)
        
    # 3. Feature Engineering on VALIDATED data
    current_year = 2025
    input_dict = {
        **validated_data, # Unpack validated values
        'house_age': current_year - validated_data['yr_built'],
        'has_renovated': 1 if validated_data['yr_renovated'] > 0 else 0,
        'grade_sqft': validated_data['grade'] * validated_data['sqft_living']
    }
    
    # Fill others
    for feat in reg_feats:
        if feat not in input_dict:
            input_dict[feat] = 0

    input_df = pd.DataFrame([input_dict])

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
        
        # Safeguard: Extreme Price Warning
        # If price is suspiciously high given the size (e.g. > $1000/sqft for non-luxury)
        price_sqft = price / validated_data['sqft_living']
        if price > 3000000 or price_sqft > 1500:
             st.warning(f"⚠️ High Valuation Detected (${price:,.0f}). This is unusual for the input features. Please verify inputs.")

        col2.metric("Estimated Price", f"${price:,.0f}")
        
        # Insight
        st.divider()
        st.subheader("Market Insight")
        
        median_price = 450000 
        
        # Pass VALIDATED features for LLM context
        feat_summary = f"{validated_data['sqft_living']}sqft, Grade {validated_data['grade']}, Built {validated_data['yr_built']}"
        
        with st.spinner("Generating expert analysis..."):
            explanation = get_explanation(feat_summary, price, median_price)
            st.success(explanation)
            
    except Exception as e:
       st.error(f"Prediction Error: {e}")
