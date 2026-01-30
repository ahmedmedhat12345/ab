import streamlit as st
import pandas as pd
import joblib
import numpy as np
import xgboost as xgb
import os
from groq import Groq

# --- Load Resources (Cached) ---
@st.cache_resource
def load_models():
    # Load ML Models
    clf = joblib.load('category_classifier.pkl')
    reg = joblib.load('price_regressor.pkl')
    cls_feats = joblib.load('cls_features.pkl')
    reg_feats = joblib.load('reg_features.pkl')
    
    # Load Market Stats
    market_stats = joblib.load('market_stats.pkl')
    
    return clf, reg, cls_feats, reg_feats, market_stats

# --- LLM Explanation Function (Groq) ---
def get_explanation(features_dict, predicted_price, cluster_label, market_stats):
    
    # 1. Prepare Market Context
    median_price = market_stats['median_price']
    avg_ppsqft = market_stats['avg_price_sqft']
    
    sqft_range = f"{market_stats['sqft_mean'] - market_stats['sqft_std']:.0f} - {market_stats['sqft_mean'] + market_stats['sqft_std']:.0f}"
    bedroom_range = f"{market_stats['bedroom_mean'] - market_stats['bedroom_std']:.0f} - {market_stats['bedroom_mean'] + market_stats['bedroom_std']:.0f}"
    
    # 2. Construct Prompt (STRICT FORMAT - Llama-3-70B)
    prompt_text = f"""
IMPORTANT:

Completely replace all previous explanation logic.  
Do not use FLAN-T5 or old prompts for reasoning.  
Use a strong instruction-tuned LLM (e.g., llama-3.3-70b-versatile).  
Behavior: concise, context-aware, instruction-following, numeric reasoning.

TASK:

Generate a short, professional explanation for a house price prediction that:

1. Explains **why this price was predicted**, based on property features.  
2. Checks **market realism** using dataset statistics.  
3. Produces **numeric comparison** (price-per-sqft vs median/average).  
4. Uses **strict, structured format** suitable for Streamlit display.  
5. Stays short, simple, and readable.

INPUT:

Prediction:
{predicted_price:,.0f}

Property Features:
- sqft_living: {features_dict.get('sqft_living', 'N/A')}
- bedrooms: {features_dict.get('bedrooms', 'N/A')}
- bathrooms: {features_dict.get('bathrooms', 'N/A')}
- grade: {features_dict.get('grade', 'N/A')}
- condition: {features_dict.get('condition', 3)}
- floors: {features_dict.get('floors', 1)}
- waterfront: {features_dict.get('waterfront', 0)}

Market Context:
- avg_price_per_sqft: {avg_ppsqft:.0f}
- median_price: {median_price:,.0f}
- typical_sqft_range: {sqft_range}
- typical_bedroom_range: {bedroom_range}

Cluster Category (optional):
{cluster_label}

OUTPUT FORMAT (STRICT):

Prediction Explanation:
<1–2 sentences explaining why the price was predicted, referencing key features>

Market Reality Check:
<1 sentence comparing predicted price with market stats, including numeric price-per-sqft>

Price per SqFt: $X

Final Assessment:
<Realistic | Underpriced | Overpriced | Possibly Erroneous>

RULES:

- Use causal reasoning: e.g., “Because the house has 8 bedrooms, grade 12, and 2 bathrooms…”  
- Include numeric price-per-sqft calculation and compare with median/average  
- Flag deviations >40% as “Possibly Erroneous”  
- No filler, no emojis, no assistant tone  
- Keep output concise for Streamlit display
"""

    try:
        # Initialize Groq Client
        api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
        if not api_key:
            return "⚠️ Groq API Key missing. Please set GROQ_API_KEY in secrets."
            
        client = Groq(api_key=api_key)
        
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a concise, expert real estate analyst."},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.3,
            max_tokens=300,
            top_p=1,
            stream=False,
            stop=None,
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        return f"Error generating explanation: {str(e)}"

# --- Validation and Guardrails (Kept from original) ---
def validate_inputs(data):
    warnings = []
    if data['grade'] >= 11 and data['sqft_living'] < 1000:
        warnings.append("⚠️ High Grade (11+) is inconsistent with small size (<1000 sqft). Grade adjusted to 8.")
        data['grade'] = 8 
    if data['grade'] <= 5 and data['sqft_living'] > 4000:
        warnings.append("⚠️ Large property (>4000 sqft) with low Grade (<=5). Price will be heavily discounted.")
    if data['bedrooms'] > 0 and (data['sqft_living'] / data['bedrooms'] < 150):
        warnings.append(f"⚠️ {data['bedrooms']} bedrooms in {data['sqft_living']} sqft is unrealistic. Beds adjusted.")
        data['bedrooms'] = max(1, int(data['sqft_living'] / 200))
    if data['bathrooms'] > data['bedrooms'] + 2:
         warnings.append("⚠️ More bathrooms than bedrooms + 2 is unusual.")
    return warnings, data

def apply_market_guardrails(predicted_price, data):
    price = predicted_price
    sqft = data['sqft_living']
    grade = data['grade']
    if grade <= 5: max_ppsf = 150 
    elif grade <= 7: max_ppsf = 300
    elif grade <= 9: max_ppsf = 500
    elif grade <= 11: max_ppsf = 800
    else: max_ppsf = 1500
    
    implied_ppsf = price / sqft
    if implied_ppsf > max_ppsf:
        price = sqft * max_ppsf 
        
    if sqft < 600:
        hard_cap = 250000 + (grade * 10000)
        price = min(price, hard_cap)
        
    if sqft > 3000 and grade <= 5:
        price = min(price, 600000) 

    return price

def determine_category(price):
    if price < 500000: return "Budget"
    elif price < 1000000: return "Standard"
    else: return "Luxury"

# --- Main App ---
st.set_page_config(page_title="Real Estate AI", layout="wide")

# Load Models & Stats
try:
    clf, reg, cls_feats, reg_feats, market_stats = load_models()
except Exception as e:
    st.error(f"Error loading models or stats: {e}")
    st.stop()

st.title("Real Estate Valuation AI 🏡")
st.markdown("Predicts house price and category, with AI-driven market insights (Powered by Llama-3-70B).")

# Sidebar
st.sidebar.header("Property Details")
sqft_living = st.sidebar.number_input("SqFt Living", 300, 10000, 2000)
grade = st.sidebar.slider("Grade (1-13)", 1, 13, 7)
yr_built = st.sidebar.number_input("Year Built", 1900, 2025, 2000)
bedrooms = st.sidebar.slider("Bedrooms", 0, 10, 3) 
bathrooms = st.sidebar.slider("Bathrooms", 0.0, 8.0, 2.0, 0.5) 
yr_renovated = st.sidebar.number_input("Year Renovated (0 if none)", 0, 2025, 0)
floors = st.sidebar.slider("Floors", 1.0, 3.5, 1.0, 0.5)
waterfront = st.sidebar.selectbox("Waterfront", [0, 1], index=0)
condition = st.sidebar.slider("Condition (1-5)", 1, 5, 3)

if st.sidebar.button("Values & Insights"):
    raw_data = {
        'sqft_living': sqft_living, 'grade': grade, 'yr_built': yr_built,
        'bedrooms': bedrooms, 'bathrooms': bathrooms, 'yr_renovated': yr_renovated,
        'floors': floors, 'waterfront': waterfront, 'condition': condition,
        'zipcode': 98000 # Dummy/Default if not in UI
    }
    
    warnings, validated_data = validate_inputs(raw_data.copy())
    for w in warnings: st.warning(w)
        
    current_year = 2025
    input_dict = {
        **validated_data,
        'house_age': current_year - validated_data['yr_built'],
        'has_renovated': 1 if validated_data['yr_renovated'] > 0 else 0,
        'grade_sqft': validated_data['grade'] * validated_data['sqft_living']
    }
    
    for feat in reg_feats:
        if feat not in input_dict: input_dict[feat] = 0

    input_df = pd.DataFrame([input_dict])

    col1, col2 = st.columns(2)
    
    try:
        # Predict Price
        price_log = reg.predict(input_df[reg_feats])[0]
        raw_price = np.expm1(price_log)
        final_price = apply_market_guardrails(raw_price, validated_data)
        
        # Determine Category
        # Use classifier if possible, but override with price logic as per original app
        # "We override the classifier if it contradicts the price heavily" -> original logic used determine_category(price)
        final_cat = determine_category(final_price)
        
        col1.subheader(f"Category: {final_cat}")
        col2.metric("Estimated Price", f"${final_price:,.0f}")
        
        st.divider()
        st.subheader("Market Insight")
        
        with st.spinner("Analyzing with Llama-3-70B..."):
            explanation = get_explanation(
                features_dict=validated_data,
                predicted_price=final_price,
                cluster_label=final_cat,
                market_stats=market_stats
            )
            st.markdown(explanation)
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")
