import pandas as pd
import numpy 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

d= pd.read_csv("D:/diamond/diamonds.csv")
d.head()

d.info()

# Check for missing values in the entire dataset
missing_values = d.isnull().sum()

print(missing_values)



# plot
 
color_order = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
colors = sns.color_palette("coolwarm", len(color_order))  # Use a Seaborn palette 

# Create a dictionary mapping from color to the custom color
palette = dict(zip(color_order, colors))

# Plot the countplot with the custom palette
sns.countplot(x='color', data=d, order=color_order, palette=palette)
# Add the label "J (worst) to D (best)"
plt.text(x=0.5, y=max(d['color'].value_counts()) + 1000, s='J (worst) to D (best)', ha='center', va='bottom', fontsize=12, color='black', weight='bold')
plt.show()

sns.countplot(x=d["cut"],data=d , palette=["red","orange","yellow","blue","green"])
plt.show()


color_order = ['SI2', 'SI1', 'VS1', 'VS2', 'VVS2', 'VVS1', 'I1',"IF"] 
sns.countplot(x=d["clarity"],data=d , order=color_order, palette=palette)
plt.show()

# Features and target variable
x=d.drop(columns=["price"])
y=d["price"]

print(f"Features shape: {x.shape}")
print(f"Target shape: {y.shape}")

# Train-test split
x_train ,x_test , y_train ,y_test = train_test_split(x,y,test_size=0.2, random_state=32)

# Preprocessing
numeric_features = ['carat', 'depth', 'table', 'x', 'y', 'z']
categorical_features = ['cut', 'color', 'clarity']

# ColumnTransformer is a utility in the scikit-learn library that allows you to apply different preprocessing techniques to different columns (features) in your dataset
# StandardScaler standardizes the numerical features by removing the mean and scaling to unit variance. This is often necessary for many machine learning algorithms to perform well.
# OneHotEncoder converts categorical variables into a format that can be provided to ML algorithms to do a better job in prediction. It creates binary (0 or 1) columns for each category level.

preprocessor= ColumnTransformer(
    transformers=[
     ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])


# Model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train model
model.fit(x_train, y_train)

# Predict and evaluate
y_pred = model.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

# Calculate basic statistics
price_mean = d['price'].mean()
price_std = d['price'].std()
price_min = d['price'].min()
price_max = d['price'].max()


print(f'Price Mean: {price_mean}')
print(f'Price Std Dev: {price_std}')
print(f'Price Min: {price_min}')
print(f'Price Max: {price_max}')

# Plot actual vs. predicted prices
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs. Predicted Prices')
plt.show()

# Plot residuals
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.xlabel('Residuals')
plt.title('Distribution of Residuals')
plt.show()

# Summary statistics of residuals
residuals_summary = residuals.describe()
print(residuals_summary)


import joblib

# Save the model
joblib.dump(model, 'diamond_price_model.pkl')

# Load the model
loaded_model = joblib.load('diamond_price_model.pkl')

# Predict using the loaded model
y_loaded_pred = loaded_model.predict(x_test)
loaded_mae = mean_absolute_error(y_test, y_loaded_pred)
print(f'Mean Absolute Error of the loaded model: {loaded_mae}')
