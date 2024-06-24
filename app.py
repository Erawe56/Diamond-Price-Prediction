import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Load the model
model = joblib.load('D:/diamond/diamond_price_model.pkl')

# Load training data to fit the preprocessor
X_train = pd.read_csv('D:/diamond/diamonds.csv').drop(columns=['price'])  # Assuming you have the original training data
y_train = pd.read_csv('D:/diamond/diamonds.csv')['price']  # Assuming you have the target variable 'price'

# Preprocessing
numeric_features = ['carat', 'depth', 'table', 'x', 'y', 'z']
categorical_features = ['cut', 'color', 'clarity']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='drop'  # Drop other columns that are not specified
)

# Combine preprocessing and model pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Fit the preprocessing pipeline with the training data
pipeline.fit(X_train, y_train)

# Define the user input function
def user_input_features():
    carat = st.number_input('Carat Weight', min_value=0.0, step=0.01)
    cut = st.selectbox('Cut', ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
    color = st.selectbox('Color', ['J', 'I', 'H', 'G', 'F', 'E', 'D'])
    clarity = st.selectbox('Clarity', ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])
    depth = st.number_input('Depth Percentage', min_value=0.0, step=0.01)
    table = st.number_input('Table Percentage', min_value=0.0, step=0.01)
    x = st.number_input('Length (mm)', min_value=0.0, step=0.01)
    y = st.number_input('Width (mm)', min_value=0.0, step=0.01)
    z = st.number_input('Depth (mm)', min_value=0.0, step=0.01)
    
    data = {
        'carat': carat,
        'cut': cut,
        'color': color,
        'clarity': clarity,
        'depth': depth,
        'table': table,
        'x': x,
        'y': y,
        'z': z
    }
    return pd.DataFrame(data, index=[0])

# Ensure all categories from categorical features are present in the input data
def validate_input(input_df, X_train):
    categorical_features = ['cut', 'color', 'clarity']
    for cat_feature in categorical_features:
        if input_df[cat_feature].iloc[0] not in X_train[cat_feature].unique():
            input_df[cat_feature] = X_train[cat_feature].mode().iloc[0]
    return input_df



# Function to convert USD to INR (you can replace with actual conversion rate)
def usd_to_inr(price_usd):
    # Assuming 1 USD = 75 INR
    conversion_rate = 75.0
    return price_usd * conversion_rate


# Main Streamlit app
st.title('Diamond Price Prediction')

# Get user input
input_df = user_input_features()

# Validate input
input_df = validate_input(input_df, X_train)

# # Predict the price
# if st.button('Predict'):
#     prediction = pipeline.predict(input_df)
#     st.write(f"### Predicted Price: ${prediction[0]:,.2f}")

# Predict the price
if st.button('Predict'):
    try:
        prediction_usd = pipeline.predict(input_df)[0]
        prediction_inr = usd_to_inr(prediction_usd)
        st.write(f"### Predicted Price: ₹{prediction_inr:,.2f}")  # ₹ is the symbol for Indian Rupee
    except Exception as e:
        st.write("### Error:", e)

        
