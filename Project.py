import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score

# Define the ANFIS model class for diabetes prediction
class DiabetesANFIS:
    def __init__(self, n_mf=3, lr=0.005, epochs=100, n_rules=8):
        # n_mf: Number of membership functions per feature
        # lr: Learning rate
        # epochs: Number of training iterations
        # n_rules: Number of fuzzy rules
        self.n_mf = n_mf
        self.lr = lr
        self.epochs = epochs
        self.n_rules = n_rules
        self.centers = None
        self.widths = None
        self.consequents = None
        self.rules_idx = None

    # Triangular membership function based on [a, b, c] parameters
    def _triangular(self, x, a, b, c):
        term1 = (x - a) / (b - a) if b > a else 0
        term2 = (c - x) / (c - b) if c > b else 0
        return np.maximum(np.minimum(term1, term2), 0)

    # Layer 1: Apply membership functions
    def _layer1(self, X):
        mf = np.zeros((X.shape[0], X.shape[1], self.n_mf))    # (samples × features × membership functions)
        for i in range(X.shape[1]):
            for j in range(self.n_mf):
                a = self.centers[i, j] - self.widths[i, j]
                b = self.centers[i, j]
                c = self.centers[i, j] + self.widths[i, j]
                mf[:, i, j] = self._triangular(X[:, i], a, b, c)
        return np.clip(mf, 1e-6, 1.0)

    # Layer 2: Calculate normalized firing strength of rules
    def _layer2(self, mf):
        if self.rules_idx is None:
            self.rules_idx = [np.random.choice(mf.shape[1], 3, replace=False) for _ in range(self.n_rules)]
        firing = np.zeros((mf.shape[0], self.n_rules))
        for i, idx in enumerate(self.rules_idx):
            selected = mf[:, idx, :].mean(axis=2)
            firing[:, i] = selected.mean(axis=1)
        firing_sum = firing.sum(axis=1, keepdims=True) + 1e-6
        return firing / firing_sum

    # Sigmoid activation for output layer
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

    # Forward pass: from input to final prediction
    def _forward(self, X):
        mf = self._layer1(X)
        norm_firing = self._layer2(mf)
        X_aug = np.column_stack([X, np.ones(X.shape[0])])  # Add bias term
        rule_outputs = X_aug @ self.consequents.T
        final_output = np.sum(norm_firing * rule_outputs, axis=1)
        return self._sigmoid(final_output)

    # Initialize membership function centers, widths, and rule consequents
    def _init_params(self, X):
        n_feat = X.shape[1]
        self.centers = np.zeros((n_feat, self.n_mf))
        self.widths = np.zeros((n_feat, self.n_mf))
        for i in range(n_feat):
            min_, max_ = X[:, i].min(), X[:, i].max()
            self.centers[i] = np.linspace(min_, max_, self.n_mf)
            self.widths[i] = np.ones(self.n_mf) * ((max_ - min_) / (self.n_mf + 1))
        self.consequents = np.random.randn(self.n_rules, X.shape[1] + 1) * 0.01

    # Training the model using gradient descent
    def fit(self, X, y, X_test, y_test):  # أضفنا X_test و y_test كمدخلات
        self._init_params(X)
        best_acc = 0

        for ep in range(self.epochs):
            out = self._forward(X)
            err = out - y
            X_aug = np.column_stack([X, np.ones(X.shape[0])])
            mf = self._layer1(X)
            norm_firing = self._layer2(mf)

            # Update rule consequents based on error gradient
            for i in range(self.n_rules):
                grad = ((err * norm_firing[:, i]) * out * (1 - out)) @ X_aug
                self.consequents[i] -= self.lr * grad

            # Print train and test accuracy every 10 epochs
            if ep % 10 == 0 or ep == self.epochs - 1:
                train_acc = accuracy_score(y, (out > 0.5).astype(int))
                test_out = self._forward(X_test)  # حساب مخرجات بيانات الاختبار
                test_acc = accuracy_score(y_test, (test_out > 0.5).astype(int))
                print(f"Epoch {ep}: Train Accuracy = {train_acc:.4f}, Test Accuracy = {test_acc:.4f}")
                if train_acc > best_acc:
                    best_acc = train_acc

    # Predict binary outcomes (0 or 1)
    def predict(self, X):
        return (self._forward(X) > 0.5).astype(int)

    # Predict probabilities (before thresholding)
    def predict_proba(self, X):
        return self._forward(X)

# ========== Data Loading and Preprocessing ==========
# Load the dataset
file_path = "diabetes_dataset_with_notes.csv"
df = pd.read_csv(file_path)

# Drop irrelevant columns
df = df.drop(columns=["location", "clinical_notes"])

# Encode categorical columns
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
df['smoking_history'] = df['smoking_history'].astype('category').cat.codes

# Remove any rows with missing values
df = df.dropna()

# Separate features and target label
X = df.drop(columns=['diabetes'])
y = df['diabetes']

# Normalize feature values between 0 and 1
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ========== Model Training ==========
model = DiabetesANFIS(n_mf=3, lr=0.005, epochs=200, n_rules=8)
model.fit(X_train, y_train.values, X_test, y_test.values)  # أضفنا X_test و y_test

# ========== Evaluation ==========
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

# Final evaluation results
print("\n=== Final Evaluation ===")
print(f"Accuracy: {acc * 100:.2f}%")
print(f"AUC-ROC Score: {auc:.4f}")