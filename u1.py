import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Load dataset 
df=pd.read_excel('C:\\Users\\hp\Desktop\\Crime_Rate Prediction\\StoresData.xlsx')
print("Initial Data Shape:", df.shape)
print("\nColumns:", df.columns.tolist())

# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# Handle missing data
df.fillna({
    'Wages $m': df['Wages $m'].median(),
    'No. Staff': df['No. Staff'].median(),
    'Mng-Exp': df['Mng-Exp'].mean()
}, inplace=True)

# Drop columns with too many missing values or identifiers
df.drop(columns=['Store No.'], inplace=True, errors='ignore')

# Convert categorical columns
df['Location'] = df['Location'].astype('category')
df['State'] = df['State'].astype('category')
df['Mng-Sex'] = df['Mng-Sex'].astype('category')

# Verify cleaning
print("\nAfter Cleaning - Missing Values:\n", df.isnull().sum())

# Encode categorical variables
label_encoders = {}
categorical_cols = ['Location', 'State', 'Mng-Sex']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Normalize numerical features
scaler = StandardScaler()
numerical_cols = ['Sales $m', 'Wages $m', 'No. Staff', 'GrossProfit',  "Adv.$'000'"]
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Create new features
df['Profit_per_Staff'] = df['GrossProfit'] / df['No. Staff']
df['Wage_to_Sales_Ratio'] = df['Wages $m'] / df['Sales $m']

print("\nSample Preprocessed Data:")
print(df.head())

# Define target variable (example: predicting Sales)
X = df.drop(columns=['Sales $m'])  # Features
y = df['Sales $m']  # Target

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

print("\nShapes after splitting:")
print(f"Training data: {X_train.shape}, {y_train.shape}")
print(f"Testing data: {X_test.shape}, {y_test.shape}")