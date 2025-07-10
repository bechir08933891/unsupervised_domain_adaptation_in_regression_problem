"""
Unsupervised Domain Adaptation (UDA) for Real Estate Price Prediction
=====================================================================

- Source Domain: Boston Housing Dataset (with house prices)
- Target Domain: California Housing Dataset (used as proxy for LA area, no labels used)

Goal:
-----
Train a regression model that learns from labeled data (Boston) and generalizes to an unlabeled domain (LA), using Domain Adversarial Neural Network (DANN).

Libraries Required:
-------------------
- pandas: for data handling
- numpy: for numerical computation
- scikit-learn: for preprocessing, baseline model, metrics
- matplotlib: for visualization
- torch (PyTorch): for deep learning model and training
"""

# =======================
# SECTION 1: Import Libraries
# =======================
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function
# =======================
# SECTION 2: Load and Preview Data
# =======================

# Load Boston Housing dataset as the source (labeled)
source_df = pd.read_csv("data/source_boston.csv")

# Load California Housing dataset as the target (unlabeled)
target_df = pd.read_csv("data/target_la.csv")

# Show a few rows from each
print("Source (Boston) sample:")
print(source_df.head())

print("\nTarget (LA) sample:")
print(target_df.head())

# Print number of records in each dataset
print(f"\nRecords in source (Boston): {len(source_df)}")
print(f"Records in target (LA): {len(target_df)}")
# =======================
# SECTION 3: Feature Selection and Alignment
# =======================

# Convert source columns to lowercase to match across datasets
source_df.columns = [col.lower() for col in source_df.columns]

# Select features from Boston dataset
source_features = ["rm", "age", "dis", "tax", "ptratio", "lstat"]
source_label = "medv"  # Median value of owner-occupied homes

X_source = source_df[source_features].copy()
y_source = source_df[source_label].copy()

# Drop rows with missing values in California data
target_df = target_df.dropna()

# Manually create equivalent features for LA dataset
X_target = pd.DataFrame()
X_target["rm"] = target_df["total_rooms"] / (target_df["households"] + 1)
X_target["age"] = target_df["housing_median_age"]
X_target["dis"] = 1 / (target_df["longitude"] + 123 + 1e-6)  # inverse of longitude distance
X_target["tax"] = target_df["median_income"] * 1000
X_target["ptratio"] = 20  # approximate value
X_target["lstat"] = 100 - (target_df["median_income"] * 10)  # proxy for lower status

# Check for infinite or NaN values before scaling
print("Any inf in X_target:", np.isinf(X_target).any().any())
print("Any NaN in X_target:", X_target.isna().any().any())
# Ensure column order matches source before scaling
X_target = X_target[source_features]
# =======================
# SECTION 4: Normalize Features
# =======================

# Normalize all features using StandardScaler
scaler = StandardScaler()
X_source_scaled = scaler.fit_transform(X_source)
X_target_scaled = scaler.transform(X_target)

# =======================
# SECTION 5: Split Source Data
# =======================

# Split labeled source data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_source_scaled, y_source, test_size=0.2, random_state=42
)
# =======================
# SECTION 6: Baseline Model (Random Forest)
# =======================

from sklearn.ensemble import RandomForestRegressor

# Train a Random Forest Regressor on source data only
baseline_model = RandomForestRegressor(n_estimators=100, random_state=42)
baseline_model.fit(X_train, y_train)

# Predict and evaluate on validation set
val_preds = baseline_model.predict(X_val)
baseline_rmse = mean_squared_error(y_val, val_preds, squared=False) if hasattr(mean_squared_error, 'squared') else np.sqrt(mean_squared_error(y_val, val_preds))
print(f"\nBaseline RMSE on source validation set: {baseline_rmse:.4f}")
# =======================
# SECTION 7: Visualize Domain Shift (t-SNE)
# =======================

# Combine both domains for visualization
X_combined = np.vstack([X_source_scaled, X_target_scaled])
domain_labels = np.array([0] * len(X_source_scaled) + [1] * len(X_target_scaled))

# Reduce dimensions to 2D using t-SNE for plotting
tsne = TSNE(n_components=2, perplexity=30, random_state=0)
X_tsne = tsne.fit_transform(X_combined)

# Plot the domain shift
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[domain_labels == 0, 0], X_tsne[domain_labels == 0, 1], label="Source (Boston)", alpha=0.5)
plt.scatter(X_tsne[domain_labels == 1, 0], X_tsne[domain_labels == 1, 1], label="Target (LA)", alpha=0.5)
plt.title("t-SNE Visualization of Domain Shift")
plt.legend()
plt.show()
# =======================
# SECTION 8: Define Gradient Reversal Layer
# =======================

# Custom function that reverses gradients (used in domain adaptation)
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha=1.0):
    return GradientReversalFunction.apply(x, alpha)
# =======================
# SECTION 9: Build DANN Network
# =======================

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x)

class Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.net(x)

class DomainClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # 0 = source, 1 = target
        )
    def forward(self, x, alpha):
        x = grad_reverse(x, alpha)
        return self.net(x)

# =======================
# SECTION 10: Prepare Dataloaders
# =======================

# Define dataset classes
class SourceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class TargetDataset(Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx]

# Instantiate dataloaders
source_loader = DataLoader(SourceDataset(X_train, y_train), batch_size=64, shuffle=True)
target_loader = DataLoader(TargetDataset(X_target_scaled), batch_size=64, shuffle=True)

# =======================
# SECTION 11: Initialize Models and Optimizers
# =======================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

feature_extractor = FeatureExtractor().to(device)
regressor = Regressor().to(device)
domain_classifier = DomainClassifier().to(device)

# Define loss functions
mse_loss = nn.MSELoss()
ce_loss = nn.CrossEntropyLoss()

# Define optimizers
optimizer_F = optim.Adam(feature_extractor.parameters(), lr=1e-3)
optimizer_R = optim.Adam(regressor.parameters(), lr=1e-3)
optimizer_D = optim.Adam(domain_classifier.parameters(), lr=1e-3)

# =======================
# SECTION 12: Train DANN
# =======================

num_epochs = 50
len_dataloader = min(len(source_loader), len(target_loader))

for epoch in range(num_epochs):
    feature_extractor.train()
    regressor.train()
    domain_classifier.train()

    total_reg_loss = 0
    total_domain_loss = 0
    total_correct = 0
    total_samples = 0

    source_iter = iter(source_loader)
    target_iter = iter(target_loader)

    for _ in range(len_dataloader):
        xs, ys = next(source_iter)
        xt = next(target_iter)

        xs, ys, xt = xs.to(device), ys.to(device), xt.to(device)

        optimizer_F.zero_grad()
        optimizer_R.zero_grad()
        optimizer_D.zero_grad()

        feat_s = feature_extractor(xs)
        feat_t = feature_extractor(xt)

        preds = regressor(feat_s)
        reg_loss = mse_loss(preds, ys)

        domain_s = torch.zeros(xs.size(0), dtype=torch.long).to(device)
        domain_t = torch.ones(xt.size(0), dtype=torch.long).to(device)

        alpha = 1.0
        domain_preds_s = domain_classifier(feat_s, alpha)
        domain_preds_t = domain_classifier(feat_t, alpha)

        domain_loss = ce_loss(domain_preds_s, domain_s) + ce_loss(domain_preds_t, domain_t)

        loss = reg_loss + domain_loss
        loss.backward()

        optimizer_F.step()
        optimizer_R.step()
        optimizer_D.step()

        total_reg_loss += reg_loss.item()
        total_domain_loss += domain_loss.item()
        total_correct += (torch.argmax(domain_preds_s, 1) == domain_s).sum().item()
        total_correct += (torch.argmax(domain_preds_t, 1) == domain_t).sum().item()
        total_samples += len(domain_s) + len(domain_t)

    acc = total_correct / total_samples
    print(f"Epoch {epoch+1}/{num_epochs} - Reg Loss: {total_reg_loss:.4f}, Domain Loss: {total_domain_loss:.4f}, Domain Acc: {acc:.4f}")

# =======================
# SECTION 13: Evaluation
# =======================

feature_extractor.eval()
regressor.eval()

X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1).to(device)

with torch.no_grad():
    feat_val = feature_extractor(X_val_tensor)
    preds_val = regressor(feat_val)
    val_rmse = torch.sqrt(torch.mean((preds_val - y_val_tensor) ** 2)).item()

print(f"\nDANN Validation RMSE (Boston): {val_rmse:.4f}")

# =======================
# SECTION 13.5: Additional Evaluation Metrics
# =======================

from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error, mean_absolute_percentage_error


from sklearn.metrics import mean_absolute_error, r2_score

# Convert tensors to numpy for metrics
true_vals = y_val_tensor.cpu().numpy()
pred_vals = preds_val.cpu().numpy()

mae = mean_absolute_error(true_vals, pred_vals)
r2 = r2_score(true_vals, pred_vals)
medae = median_absolute_error(true_vals, pred_vals)
mape = mean_absolute_percentage_error(true_vals, pred_vals)

print(f"Mean Absolute Error (MAE) on Boston: {mae:.4f}")
print(f"R² Score on Boston: {r2:.4f}")
print(f"Median Absolute Error (MedAE) on Boston: {medae:.4f}")
print(f"Mean Absolute Percentage Error (MAPE) on Boston: {mape:.4f}")

X_target_tensor = torch.tensor(X_target_scaled, dtype=torch.float32).to(device)

with torch.no_grad():
    feat_target = feature_extractor(X_target_tensor)
    preds_target = regressor(feat_target).cpu().numpy()

print("\nSample Predicted Prices on Target (LA):")
print(preds_target[:10])

import os
# Histogram of predictions on LA

# Create directory for plots if not exists
os.makedirs("plots", exist_ok=True)

# Additional Visualization: Actual vs Predicted Scatter Plot
plt.figure(figsize=(6, 6))
plt.scatter(true_vals, pred_vals, alpha=0.6)
plt.plot([true_vals.min(), true_vals.max()], [true_vals.min(), true_vals.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices (Boston)")
plt.grid(True)
plt.savefig("plots/actual_vs_predicted.png")
plt.show()
plt.plot([true_vals.min(), true_vals.max()], [true_vals.min(), true_vals.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices (Boston)")
plt.grid(True)
plt.show()

# Residual Plot
residuals = true_vals - pred_vals
plt.figure(figsize=(6, 4))
plt.scatter(pred_vals, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Price")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residual Plot (Boston)")
plt.grid(True)
plt.savefig("plots/residual_plot.png")
plt.show()
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Price")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residual Plot (Boston)")
plt.grid(True)
plt.show()

# Bar Chart of Metrics
metrics = [baseline_rmse, val_rmse, mae, medae, mape]
labels = ["Baseline RMSE", "DANN RMSE", "MAE", "MedAE", "MAPE"]
plt.figure(figsize=(8, 4))
plt.bar(labels, metrics, color='skyblue')
plt.title("Comparison of Regression Metrics")
plt.ylabel("Error")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/metrics_comparison.png")
plt.show()
plt.title("Comparison of Regression Metrics")
plt.ylabel("Error")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.hist(preds_target, bins=50, alpha=0.7, color='purple')
plt.title("Predicted Prices on Target Domain (LA)")
plt.xlabel("Predicted Price")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("plots/predicted_prices_target.png")
plt.show()
plt.title("Predicted Prices on Target Domain (LA)")
plt.xlabel("Predicted Price")
plt.ylabel("Frequency")
plt.show()