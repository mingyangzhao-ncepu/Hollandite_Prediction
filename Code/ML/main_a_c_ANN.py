import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


np.random.seed(42)
torch.manual_seed(42)


infile = r"E:\Hollandite_ML\Hollandite_data.xlsx"
outdir = r"E:\Hollandite_ML\Results"
os.makedirs(outdir, exist_ok=True)


df = pd.read_excel(infile)


X = df[['rO+rB', 'deltaA', 'deltaB', 'ZA', 'ZB', 'ENA', 'ENB', 'Occ']].values
y = df[["a' (Å)", "c' (Å)"]].values


y_a = y[:, 0]
y_bin = None
for q_try in (10, 8, 6, 5, 4):
    try:
        tmp = pd.qcut(y_a, q=q_try, labels=False, duplicates='drop')
        if pd.Series(tmp).nunique() >= 2:
            y_bin = tmp.astype(int)
            break
    except Exception:
        pass

if y_bin is None:

    print('[Stratify] qcut')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
else:

    (unique_bins, counts) = np.unique(y_bin, return_counts=True)
    print(f'[Stratify] = {len(unique_bins)}, Sample = {counts.tolist()}')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,   
        stratify=y_bin     
    )




scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)


X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train_scaled)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test_scaled)


train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)


class HollanditeNN(nn.Module):
    def __init__(self):
        super(HollanditeNN, self).__init__()
        self.hidden = nn.Linear(8, 4)   
        self.output = nn.Linear(4, 2)   
        self.activation = nn.Sigmoid()  

    def forward(self, x):
        x = self.activation(self.hidden(x))
        x = self.output(x)
        return x


model = HollanditeNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


num_epochs = 1000
train_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, targets in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')


model.eval()
with torch.no_grad():
    
    y_train_pred_scaled = model(X_train_tensor)
    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.numpy())

    
    y_test_pred_scaled = model(X_test_tensor)
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.numpy())

    
    X_all_tensor = torch.FloatTensor(scaler_X.transform(X))
    y_all_pred_scaled = model(X_all_tensor)
    y_all_pred = scaler_y.inverse_transform(y_all_pred_scaled.numpy())


def calculate_r(y_true, y_pred, column_idx):
    y_true_col = y_true[:, column_idx]
    y_pred_col = y_pred[:, column_idx]
    mean_y_true = np.mean(y_true_col)
    mean_y_pred = np.mean(y_pred_col)
    numerator = np.sum((y_true_col - mean_y_true) * (y_pred_col - mean_y_pred))
    denominator = np.sqrt(np.sum((y_true_col - mean_y_true)**2) *
                          np.sum((y_pred_col - mean_y_pred)**2))
    return numerator / denominator


r_train_a = calculate_r(y_train, y_train_pred, 0)
r_test_a  = calculate_r(y_test,  y_test_pred,  0)
r_all_a   = calculate_r(y,       y_all_pred,   0)

R2_train_a = r_train_a**2
R2_test_a  = r_test_a**2
R2_all_a   = r_all_a**2


r_train_c = calculate_r(y_train, y_train_pred, 1)
r_test_c  = calculate_r(y_test,  y_test_pred,  1)
r_all_c   = calculate_r(y,       y_all_pred,   1)

R2_train_c = r_train_c**2
R2_test_c  = r_test_c**2
R2_all_c   = r_all_c**2

print(f"Correlation coefficient (R^2) for a' - Training: {R2_train_a:.5f}")
print(f"Correlation coefficient (R^2) for a' - Testing: {R2_test_a:.5f}")
print(f"Correlation coefficient (R^2) for a' - All data: {R2_all_a:.5f}")
print(f"Correlation coefficient (R^2) for c' - Training: {R2_train_c:.5f}")
print(f"Correlation coefficient (R^2) for c' - Testing: {R2_test_c:.5f}")
print(f"Correlation coefficient (R^2) for c' - All data: {R2_all_c:.5f}")



def calc_rmse_mae(y_true, y_pred, idx):
    """idx=0 for a', idx=1 for c'; y_* (Å)"""
    err = y_pred[:, idx] - y_true[:, idx]
    rmse = float(np.sqrt(np.mean(err**2)))
    mae  = float(np.mean(np.abs(err)))
    return rmse, mae


rmse_train_a, mae_train_a = calc_rmse_mae(y_train,    y_train_pred, 0)
rmse_test_a,  mae_test_a  = calc_rmse_mae(y_test,     y_test_pred,  0)
rmse_all_a,   mae_all_a   = calc_rmse_mae(y,          y_all_pred,   0)


rmse_train_c, mae_train_c = calc_rmse_mae(y_train,    y_train_pred, 1)
rmse_test_c,  mae_test_c  = calc_rmse_mae(y_test,     y_test_pred,  1)
rmse_all_c,   mae_all_c   = calc_rmse_mae(y,          y_all_pred,   1)


print("\n[a′] RMSE / MAE (Å)")
print(f"  Training : RMSE = {rmse_train_a:.5f}, MAE = {mae_train_a:.5f}")
print(f"  Testing  : RMSE = {rmse_test_a:.5f}, MAE = {mae_test_a:.5f}")
print(f"  All data : RMSE = {rmse_all_a:.5f}, MAE = {mae_all_a:.5f}")

print("\n[c′] RMSE / MAE (Å)")
print(f"  Training : RMSE = {rmse_train_c:.5f}, MAE = {mae_train_c:.5f}")
print(f"  Testing  : RMSE = {rmse_test_c:.5f}, MAE = {mae_test_c:.5f}")
print(f"  All data : RMSE = {rmse_all_c:.5f}, MAE = {mae_all_c:.5f}")





def create_correlation_plot(y_true, y_pred, title, xlabel, ylabel, R2_value):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, color='blue', alpha=0.7, label='Data')

    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1')

    m, b = np.polyfit(y_true, y_pred, 1)
    plt.plot(y_true, m*y_true + b, 'magenta', label='Fit')

    plt.title(f"{title} (R\u00b2={R2_value:.5f})")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)

    
    errors = y_pred - y_true
    ax_inset = plt.axes([0.6, 0.2, 0.25, 0.2])
    ax_inset.hist(errors, bins=20, color='green', alpha=0.7)
    ax_inset.axvline(x=0, color='red', linestyle='-', linewidth=1)
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
    # return current figure so we can save with high DPI
    return plt.gcf()


fig = create_correlation_plot(
    y_train[:, 0], y_train_pred[:, 0],
    "ANN Training", "Observed a (Å)", "Predicted a (Å)",
    R2_train_a
)
fig.savefig(os.path.join(outdir, 'a_parameter_training.svg'), format='svg', dpi=600, bbox_inches='tight', pad_inches=0.05)
plt.close(fig)

fig = create_correlation_plot(
    y_test[:, 0], y_test_pred[:, 0],
    "ANN Testing", "Observed a (Å)", "Predicted a (Å)",
    R2_test_a
)
fig.savefig(os.path.join(outdir, 'a_parameter_testing.svg'), format='svg', dpi=600, bbox_inches='tight', pad_inches=0.05)
plt.close(fig)

fig = create_correlation_plot(
    y[:, 0], y_all_pred[:, 0],
    "ANN All", "Observed a (Å)", "Predicted a (Å)",
    R2_all_a
)
fig.savefig(os.path.join(outdir, 'a_parameter_all.svg'), format='svg', dpi=600, bbox_inches='tight', pad_inches=0.05)
plt.close(fig)

fig = create_correlation_plot(
    y_train[:, 1], y_train_pred[:, 1],
    "ANN Training", "Observed c (Å)", "Predicted c (Å)",
    R2_train_c
)
fig.savefig(os.path.join(outdir, 'c_parameter_training.svg'), format='svg', dpi=600, bbox_inches='tight', pad_inches=0.05)
plt.close(fig)

fig = create_correlation_plot(
    y_test[:, 1], y_test_pred[:, 1],
    "ANN Testing", "Observed c (Å)", "Predicted c (Å)",
    R2_test_c
)
fig.savefig(os.path.join(outdir, 'c_parameter_testing.svg'), format='svg', dpi=600, bbox_inches='tight', pad_inches=0.05)
plt.close(fig)

fig = create_correlation_plot(
    y[:, 1], y_all_pred[:, 1],
    "ANN All", "Observed c (Å)", "Predicted c (Å)",
    R2_all_c
)
fig.savefig(os.path.join(outdir, 'c_parameter_all.svg'), format='svg', dpi=600, bbox_inches='tight', pad_inches=0.05)
plt.close(fig)


torch.save(model.state_dict(), os.path.join(outdir, 'hollandite_nn_model.pth'))

print("Neural network training and evaluation completed. SVG figures saved, model saved.")
