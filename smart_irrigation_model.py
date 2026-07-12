import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# Update dataset path to point directly to your data folder
DATASET_PATH = "data/irrigation_machine.csv"
MODEL_FILENAME = "Farm_Irrigation_System.pkl"
SCALER_FILENAME = "MinMaxScaler.pkl"

print("--- Starting Data Loading and Preprocessing ---")

if not os.path.exists(DATASET_PATH):
    print(f"ERROR: Dataset not found at path: {DATASET_PATH}")
    exit()

try:
    df = pd.read_csv(DATASET_PATH)
except Exception as e:
    print(f"ERROR: Could not read CSV file. Error: {e}")
    exit()

if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

print(f"Data loaded successfully. Total rows: {len(df)}")

feature_cols = [col for col in df.columns if col.startswith('sensor')]
target_cols = [col for col in df.columns if col.startswith('parcel')]

X = df[feature_cols]
y = df[target_cols]

if not feature_cols or not target_cols:
    print("ERROR: Could not identify feature or target columns.")
    exit()

# Apply feature scaling (MinMaxScaler)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

# Split data into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"Training set size: {len(X_train)} samples")
print("Data splitting and scaling complete.")

print("\n--- Starting Model Training ---")

# Define the base classifier with optimized hyperparameters
rf = RandomForestClassifier(
    n_estimators=200,      
    max_depth=10,          
    min_samples_split=4,   
    min_samples_leaf=2,    
    max_features='sqrt',   
    random_state=42
)

# Use MultiOutputClassifier to handle simultaneous multiple targets
model = MultiOutputClassifier(rf)
model.fit(X_train, y_train)

print("Model training complete.")

print("\n--- Starting Model Evaluation ---")

y_pred = model.predict(X_test)

print("Classification Report on Test Data:")
print(classification_report(y_test, y_pred, target_names=y.columns))

print("\n--- Saving Model and Scaler ---")

joblib.dump(model, MODEL_FILENAME)
joblib.dump(scaler, SCALER_FILENAME)

print(f"Trained model saved to: {MODEL_FILENAME}")
print(f"Data scaler saved to: {SCALER_FILENAME}")
print("--- Training Script Finished Successfully ---")