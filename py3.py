import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Define file paths
transactions_path = r"C:\Users\abdul\OneDrive\Desktop\wallet_risk_score\transactions.parquet"
train_addresses_path = r"C:\Users\abdul\OneDrive\Desktop\wallet_risk_score\train_addresses.parquet"
test_addresses_path = r"C:\Users\abdul\OneDrive\Desktop\wallet_risk_score\test_addresses.parquet"

# Load datasets
print("Loading datasets...")
transactions_df = pd.read_parquet(transactions_path)
train_addresses_df = pd.read_parquet(train_addresses_path)
test_addresses_df = pd.read_parquet(test_addresses_path)
print("Datasets loaded successfully!")

# Feature engineering
print("Engineering features...")
wallet_features = transactions_df.groupby("FROM_ADDRESS").agg(
    total_transactions=("TX_HASH", "count"),
    total_value=("VALUE", "sum"),
    avg_transaction_value=("VALUE", "mean"),
    median_transaction_value=("VALUE", "median"),
    std_transaction_value=("VALUE", "std"),
    unique_receivers=("TO_ADDRESS", "nunique"),
    max_transaction_value=("VALUE", "max"),
    min_transaction_value=("VALUE", "min")
).reset_index()

# Replace NaNs with zeros for new features
wallet_features.fillna(0, inplace=True)

# Merge with train addresses
wallet_features = wallet_features.merge(
    train_addresses_df.rename(columns={"ADDRESS": "FROM_ADDRESS", "LABEL": "Malicious_Label"}),
    on="FROM_ADDRESS",
    how="left"
)

# Prepare data for training
X = wallet_features[[
    "total_transactions", "total_value", "avg_transaction_value",
    "median_transaction_value", "std_transaction_value", "unique_receivers",
    "max_transaction_value", "min_transaction_value"
]]
X.fillna(0, inplace=True)  # Handle any remaining NaNs
y = wallet_features["Malicious_Label"].fillna(0).astype(int)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Define LightGBM model with adjusted parameters
lgbm = LGBMClassifier(
    random_state=42, 
    class_weight='balanced', 
    n_estimators=200,  # Lower number of estimators
    max_depth=15,      # Reduce max_depth
    num_leaves=50,     # Reduce number of leaves
    learning_rate=0.05,  # Adjust learning rate
    min_child_samples=10,  # Reduce min_child_samples
    lambda_l1=0.1,     # Adjust L1 regularization
    lambda_l2=0.1,     # Adjust L2 regularization
)

# Define the parameter grid for RandomizedSearchCV
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 15, 20],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [31, 50, 100],
    'min_child_samples': [5, 10, 15],
    'lambda_l1': [0.1, 1.0],
    'lambda_l2': [0.1, 0.5],
}

# Perform RandomizedSearchCV to find the best hyperparameters
random_search = RandomizedSearchCV(
    estimator=lgbm, 
    param_distributions=param_grid, 
    n_iter=50, 
    scoring='f1', 
    cv=5, 
    n_jobs=-1, 
    verbose=2, 
    random_state=42
)

# Perform the hyperparameter tuning (without early stopping)
print("Performing hyperparameter tuning...")
random_search.fit(X_train, y_train)

# Best model from RandomizedSearchCV
best_lgbm = random_search.best_estimator_
print(f"Best Parameters: {random_search.best_params_}")

# Define early stopping parameters
eval_set = [(X_val, y_val)]  # Use validation data for early stopping
eval_metric = 'binary_error'  # Choose the evaluation metric, e.g., binary error

# Train the best model with early stopping
print("Training the best model with early stopping...")
best_lgbm.fit(X_train, y_train, eval_set=eval_set, eval_metric=eval_metric, early_stopping_rounds=50)

# Evaluate on validation set
y_val_pred = best_lgbm.predict(X_val)
print("Validation Results:")
print(classification_report(y_val, y_val_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))

# Feature importance using LightGBM
print("Plotting feature importance...")
plt.figure(figsize=(10, 6))
best_lgbm.plot_importance(best_lgbm, max_num_features=10, importance_type='gain', figsize=(10, 6))
plt.title("Feature Importance")
plt.show()

# Predict for test addresses
print("Predicting for test addresses...")
test_wallet_features = test_addresses_df.rename(columns={"ADDRESS": "FROM_ADDRESS"}).merge(
    wallet_features, on="FROM_ADDRESS", how="left"
).fillna(0)

X_test = test_wallet_features[[
    "total_transactions", "total_value", "avg_transaction_value",
    "median_transaction_value", "std_transaction_value", "unique_receivers",
    "max_transaction_value", "min_transaction_value"
]]
X_test_scaled = scaler.transform(X_test)  # Apply scaling to test data
test_addresses_df["PRED"] = best_lgbm.predict(X_test_scaled)

# Save predictions
submission_file = r"C:\Users\abdul\OneDrive\Desktop\wallet_risk_score\submission5.csv"
test_addresses_df[["ADDRESS", "PRED"]].to_csv(submission_file, index=False)
print(f"Predictions saved to {submission_file}")
