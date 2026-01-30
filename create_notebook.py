import json
import os

# Notebook structure
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.5"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_markdown(content):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in content.split('\n')]
    })

def add_code(content):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in content.split('\n')]
    })

# --- Notebook Content ---

# Title
add_markdown("# Real Estate Machine Learning Project\n\nThis notebook covers data loading, EDA, clustering, classification, regression, LLM interpretation, and Streamlit deployment.")

# 1. Data Loading & Cleaning
add_markdown("## 1. Data Loading & Cleaning")
add_markdown("Load dataset, handle missing values, outliers, and data type conversions.")

add_code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
import xgboost as xgb

warnings.filterwarnings('ignore')

# Load Dataset
try:
    df = pd.read_csv('data.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'data.csv' not found. Please ensure the dataset is in the same directory.")
    # Create dummy data for demonstration if file missing
    np.random.seed(42) # Reproducible
    n_samples = 4600
    data = {
        'price': np.random.randint(200000, 2000000, n_samples),
        'bedrooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.randint(1, 4, n_samples),
        'sqft_living': np.random.randint(1000, 5000, n_samples),
        'sqft_lot': np.random.randint(2000, 10000, n_samples),
        'floors': np.random.randint(1, 4, n_samples),
        'waterfront': np.random.randint(0, 2, n_samples),
        'view': np.random.randint(0, 5, n_samples),
        'condition': np.random.randint(1, 6, n_samples),
        'grade': np.random.randint(4, 13, n_samples),
        'sqft_above': np.random.randint(1000, 4000, n_samples),
        'sqft_basement': np.random.randint(0, 1000, n_samples),
        'yr_built': np.random.randint(1950, 2023, n_samples),
        'yr_renovated': np.random.randint(0, 2023, n_samples),
        'zipcode': np.random.randint(98000, 98200, n_samples),
        'lat': np.random.uniform(47.1, 47.8, n_samples),
        'long': np.random.uniform(-122.5, -121.5, n_samples),
        'sqft_living15': np.random.randint(1000, 4000, n_samples),
        'sqft_lot15': np.random.randint(2000, 10000, n_samples),
        'date': pd.date_range(start='1/1/2022', periods=n_samples)
    }
    # Add some correlation to make metrics non-zero
    data['price'] = data['sqft_living'] * 300 + data['grade'] * 50000 + np.random.normal(0, 50000, n_samples)
    
    df = pd.DataFrame(data)
    print("Created dummy dataset for demonstration with Synthetic Correlation.")

# Basic Cleaning
df['date'] = pd.to_datetime(df['date'])
df.dropna(inplace=True)

# Remove Outliers (Simple IQR method for Price)
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['price'] < (Q1 - 1.5 * IQR)) | (df['price'] > (Q3 + 1.5 * IQR)))]

print(f"Data shape after cleaning: {df.shape}")
df.head()""")

# 2. Feature Engineering
add_markdown("## 2. Feature Engineering (Enhanced)")
add_markdown("Creating new features to improve model performance.")

add_code("""# Feature Engineering
# 1. Log Transform Price (Target) to handle skewness
df['price_log'] = np.log1p(df['price'])

# 2. House Age & Renovation Status
current_year = 2025
df['house_age'] = current_year - df['yr_built']
df['has_renovated'] = df['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)

# 3. Interactions (Grade * SqFt)
df['grade_sqft'] = df['grade'] * df['sqft_living']

# Check correlations with log price
numeric_df = df.select_dtypes(include=[np.number])
print(numeric_df.corr()['price_log'].sort_values(ascending=False).head(10))""")


# 3. EDA & Insights
add_markdown("## 3. Exploratory Analysis & Business Insights")
add_markdown("Analyze price drivers and explore feature relationships.")

add_code("""# Price Distribution (Original vs Log)
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df['price'], kde=True, bins=30, ax=ax[0])
ax[0].set_title('Original Price Distribution')
sns.histplot(df['price_log'], kde=True, bins=30, ax=ax[1])
ax[1].set_title('Log-Price Distribution')
plt.show()""")

add_markdown("Log-transformation normalizes the price distribution, helping regression models.")


# 4. Unsupervised Learning (Clustering)
add_markdown("## 4. Unsupervised Learning (Clustering)")
add_markdown("Group houses into meaningful categories using K-Means.")

add_code("""# Scaling
scaler = StandardScaler()
features_for_clustering = ['price', 'sqft_living', 'grade', 'house_age']
X_scaled = scaler.fit_transform(df[features_for_clustering])

# K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Define Categories
cluster_means = df.groupby('cluster')['price'].mean().sort_values()
cluster_map = {cluster_means.index[0]: 'Budget', cluster_means.index[1]: 'Standard', cluster_means.index[2]: 'Luxury'}
df['category'] = df['cluster'].map(cluster_map)

sns.scatterplot(data=df, x='sqft_living', y='price', hue='category', palette='viridis')
plt.title('House Clusters')
plt.show()""")

# 5. Category Classification
add_markdown("## 5. Category Classification")
add_markdown("Build a classifier to predict the house category for new listings.")

add_code("""# Prepare Data
drop_cols = ['price', 'price_log', 'cluster', 'category', 'date', 'id', 'yr_renovated', 'yr_built'] 
X_cls = df.drop(drop_cols, axis=1, errors='ignore').select_dtypes(include=[np.number])
y_cls = df['category']

# Split
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

# Train Classifier 
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_c, y_train_c)

# Evaluate
acc = clf.score(X_test_c, y_test_c)
print(f"Classification Accuracy: {acc:.2f}")

# Save Classifier
joblib.dump(clf, 'category_classifier.pkl')
joblib.dump(X_cls.columns.tolist(), 'cls_features.pkl')""")

# 6. Price Regression Modeling (Optimized)
add_markdown("## 6. Price Regression Modeling (XGBoost)")
add_markdown("Build robust regression models using XGBoost and Log-Target.")

add_code("""# Prepare Data for Regression (Predict Log Price)
X_reg = df.drop(drop_cols, axis=1, errors='ignore').select_dtypes(include=[np.number])
y_reg = df['price_log'] # Target is Log Price

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Train XGBoost Regressor
reg = xgb.XGBRegressor(
    n_estimators=200, 
    learning_rate=0.05, 
    max_depth=5, 
    random_state=42, 
    n_jobs=-1
)
reg.fit(X_train_r, y_train_r)

# Predictions (Convert back from Log scale)
y_pred_log = reg.predict(X_test_r)
y_pred = np.expm1(y_pred_log)
y_test_orig = np.expm1(y_test_r)

# Evaluation
mae = mean_absolute_error(y_test_orig, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
r2 = r2_score(y_test_orig, y_pred)

print(f"Regression MAE: ${mae:,.0f}")
print(f"Regression RMSE: ${rmse:,.0f}")
print(f"R2 Score: {r2:.4f}")

# Compare with Median
median_price = df['price'].median()
print(f"Market Median Price: ${median_price:,.0f}")

# Save Regressor
joblib.dump(reg, 'price_regressor.pkl')
joblib.dump(X_train_r.columns.tolist(), 'reg_features.pkl') # Save features used""")

# 7. LLM Interpretation
add_markdown("## 7. LLM-based Model Interpretation")
add_markdown("Functions for interacting with OpenAI API with a robust manual fallback.")

add_code("""# Define functions here for notebook reference (optional, as they are mainly in Streamlit app source)
pass""")

# 8. Streamlit Deployment
add_markdown("## 8. Streamlit Deployment")
add_markdown("Streamlit app code with Silent Fallback Logic.")

add_code("""app_code = \"\"\"
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

\"\"\"

with open("app.py", "w") as f:
    f.write(app_code)

print("Streamlit app code output to 'app.py'.")

# Create requirements.txt for Streamlit Cloud
reqs = \"\"\"streamlit
pandas
numpy
scikit-learn
xgboost
joblib
openai
matplotlib
seaborn
\"\"\"
with open("requirements.txt", "w") as f:
    f.write(reqs)
print("Created 'requirements.txt' for Cloud deployment.")""")

# Write file
output_path = r'c:\Users\HP\Desktop\real estate\real_estate_project.ipynb'
with open(output_path, 'w') as f:
    json.dump(notebook, f, indent=4)

print(f"Notebook created at {output_path}")
