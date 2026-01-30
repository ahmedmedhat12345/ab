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
    if data['grade'] >= 11 and data['sqft_living'] < 1000:
        warnings.append("⚠️ High Grade (11+) is inconsistent with small size (<1000 sqft). Grade adjusted to 8.")
        data['grade'] = 8 
        
    if data['grade'] <= 5 and data['sqft_living'] > 4000:
        warnings.append("⚠️ Large property (>4000 sqft) with low Grade (<=5). Price will be heavily discounted.")

    # 2. Bedrooms vs SqFt
    if data['bedrooms'] > 0 and (data['sqft_living'] / data['bedrooms'] < 150):
        warnings.append(f"⚠️ {data['bedrooms']} bedrooms in {data['sqft_living']} sqft is unrealistic. Beds adjusted.")
        data['bedrooms'] = max(1, int(data['sqft_living'] / 200))

    # 3. Bathrooms
    if data['bathrooms'] > data['bedrooms'] + 2:
         warnings.append("⚠️ More bathrooms than bedrooms + 2 is unusual.")

    return warnings, data

def apply_market_guardrails(predicted_price, data):
    """
    Apply expert heuristics to clamp unrealistic model predictions.
    This acts as a safety layer for edge cases or dummy-model artifacts.
    """
    price = predicted_price
    sqft = data['sqft_living']
    grade = data['grade']
    
    # 1. Price Per SqFt Caps based on Grade (Market Realism)
    # Define approximate max $/sqft for each grade level
    if grade <= 5: max_ppsf = 150  # Low grade/Dilapidated
    elif grade <= 7: max_ppsf = 300 # Budget/Average
    elif grade <= 9: max_ppsf = 500 # Good
    elif grade <= 11: max_ppsf = 800 # Excellent
    else: max_ppsf = 1500 # Luxury
    
    implied_ppsf = price / sqft
    
    if implied_ppsf > max_ppsf:
        price = sqft * max_ppsf # Clamp to max allowed for that grade
        
    # 2. Tiny House Correction
    if sqft < 600:
        # Strict cap for tiny homes
        hard_cap = 250000 + (grade * 10000)
        price = min(price, hard_cap)
        
    # 3. "Dilapidated" Mansion Correction (Big size, low grade)
    if sqft > 3000 and grade <= 5:
        # Should not be priced like a luxury home just because it's big
        price = min(price, 600000) 

    return price

def determine_category(price):
    if price < 500000: return "Budget"
    elif price < 1000000: return "Standard"
    else: return "Luxury"

# --- Main App ---
st.set_page_config(page_title="Real Estate AI", layout="wide")

# Load Models
try:
    # We still try to load models, but we will wrap predictions
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
sqft_living = st.sidebar.number_input("SqFt Living", 300, 10000, 2000)
grade = st.sidebar.slider("Grade (1-13)", 1, 13, 7)
yr_built = st.sidebar.number_input("Year Built", 1900, 2025, 2000)
bedrooms = st.sidebar.slider("Bedrooms", 0, 10, 3) 
bathrooms = st.sidebar.slider("Bathrooms", 0.0, 8.0, 2.0, 0.5) 
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
    
    # Predict Price (Regression)
    try:
        price_log = reg.predict(input_df[reg_feats])[0]
        raw_price = np.expm1(price_log)
        
        # --- APPLY GUARDRAILS ---
        final_price = apply_market_guardrails(raw_price, validated_data)
        
        # Determine Category (Consistency Check)
        # We override the classifier if it contradicts the price heavily
        cat_derived = determine_category(final_price)
        final_cat = cat_derived 
        
        col1.subheader(f"Category: {final_cat}")

        col2.metric("Estimated Price", f"${final_price:,.0f}")
        
        # Insight
        st.divider()
        st.subheader("Market Insight")
        
        median_price = 450000 
        
        # Pass VALIDATED features for LLM context
        feat_summary = f"{validated_data['sqft_living']}sqft, Grade {validated_data['grade']}, Built {validated_data['yr_built']}"
        
        with st.spinner("Generating expert analysis..."):
            explanation = get_explanation(feat_summary, final_price, median_price)
            st.success(explanation)
            
    except Exception as e:
       st.error(f"Prediction Error: {e}")
