import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('14.banking_loan_dataset_1000.csv')

print("=" * 80)
print("BANK LOAN CLASSIFICATION PIPELINE")
print("=" * 80)

# Encode categorical features
le = LabelEncoder()
df['EmploymentType'] = le.fit_transform(df['EmploymentType'])

# Separate features and target
X = df.drop('LoanApproved', axis=1)
y = df['LoanApproved']

print("\n1. FEATURE SKEWNESS ANALYSIS")
print("-" * 80)

# Create a copy for preprocessing
X_processed = X.copy()

# Detect and handle skewness for each numeric feature
for column in X.columns:
    skewness = X[column].skew()
    print(f"\n{column}:")
    print(f"  Skewness: {skewness:.4f}")
    
    if abs(skewness) > 0.5:
        # Feature is skewed - use PowerTransformer + StandardScaler
        skew_type = "Left skewed" if skewness < 0 else "Right skewed"
        print(f"  Type: {skew_type}")
        print(f"  Preprocessing: PowerTransformer + StandardScaler")
        
        pt = PowerTransformer(method='yeo-johnson')
        X_processed[column] = pt.fit_transform(X_processed[[column]])
    else:
        # Feature is normal - use StandardScaler only
        print(f"  Type: Normal/Symmetric")
        print(f"  Preprocessing: StandardScaler")

# Apply StandardScaler to all features
scaler = StandardScaler()
X_processed = scaler.fit_transform(X_processed)
X_processed = pd.DataFrame(X_processed, columns=X.columns)

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

print(f"\n\n2. DATA SPLIT")
print("-" * 80)
print(f"Training set size: {len(X_train)} ({80}%)")
print(f"Test set size: {len(X_test)} ({20}%)")

# Initial Logistic Regression
print(f"\n\n3. LOGISTIC REGRESSION (Initial Model)")
print("-" * 80)
lr_initial = LogisticRegression(random_state=42)
lr_initial.fit(X_train, y_train)
y_pred_lr = lr_initial.predict(X_test)
print(classification_report(y_test, y_pred_lr))

# Train multiple classifiers
print(f"\n\n4. MULTIPLE CLASSIFIER COMPARISON")
print("-" * 80)

classifiers = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'KNN Classifier (k=5)': KNeighborsClassifier(n_neighbors=5),
    'SGD Classifier': SGDClassifier(random_state=42, max_iter=1000),
    'SVC': SVC(kernel='rbf', random_state=42),
    'Gaussian NB Classifier': GaussianNB(),
    'Random Forest Classifier': RandomForestClassifier(random_state=42, n_estimators=100)
}

results = []

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results.append({'Model': name, 'Accuracy Score': accuracy})
    print(f"\n{name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

# Create results table
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Accuracy Score', ascending=False)

print(f"\n\n5. MODEL COMPARISON TABLE")
print("-" * 80)
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]['Model']
best_accuracy = results_df.iloc[0]['Accuracy Score']

print(f"\n{'='*80}")
print(f"BEST MODEL: {best_model_name}")
print(f"ACCURACY: {best_accuracy:.4f}")
print(f"{'='*80}")

# Save the best model and scalers for Streamlit
import pickle

best_clf = classifiers[best_model_name]
best_clf.fit(X_train, y_train)

with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_clf, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("\n✓ Model artifacts saved successfully!")
