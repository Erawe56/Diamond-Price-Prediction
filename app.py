import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Load the model
model = joblib.load('diamond_price_model.pkl')

# Input features
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
    data = {'carat': carat,
            'cut': cut,
            'color': color,
            'clarity': clarity,
            'depth': depth,
            'table': table,
            'x': x,
            'y': y,
            'z': z}
    return pd.DataFrame(data, index=[0])


# Preprocess the input features
#input_df = user_input_features()

# # Ensure all categories from categorical features are present in the input data
# categorical_features = ['cut', 'color', 'clarity']
# X_train = pd.read_csv('diamonds.csv').drop(columns=['price'])  # Assuming you have the original training data
# for cat_feature in categorical_features:
#     if input_df[cat_feature].iloc[0] not in X_train[cat_feature].unique():
#         input_df[cat_feature] = X_train[cat_feature].mode().iloc[0]

# Preprocessing
numeric_features = ['carat', 'depth', 'table', 'x', 'y', 'z']
categorical_features = ['cut', 'color', 'clarity']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ],
    remainder='passthrough'  # Ensure that any extra columns are passed through without transformation
)

# # Combine preprocessing and model pipeline
# full_pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('model', model)  # Use the pre-trained model
# ])

# # Load the model and fit the preprocessor
# x_train = pd.read_csv('diamonds.csv').drop(columns=['price'])  # Assuming you have the original training data
# full_pipeline.fit(x_train)  # Fit the preprocessing pipeline with the training data


# # Predict the price
# if st.button('Predict'):
#     input_df = user_input_features()
#   # Convert user input features to DataFrame
#     prediction = full_pipeline.predict(input_df)
#     st.write(f"### Predicted Price: ${prediction[0]:,.2f}")

    # Model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Load the model and fit the preprocessor
X_train = pd.read_csv('diamonds.csv').drop(columns=['price'])  # Assuming you have the original training data
y_train = pd.read_csv('diamonds.csv')['price']  # Assuming the target variable is 'price'
model.fit(X_train, y_train)  # Fit the model pipeline with the training data

# Predict the price
if st.button('Predict'):
    input_df = user_input_features()
    prediction = model.predict(input_df)
    st.write(f"### Predicted Price: ${prediction[0]:,.2f}")


