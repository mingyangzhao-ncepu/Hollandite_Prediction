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


def pick_col(candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"NotFound, tried：{candidates}")

col_rOrB = pick_col(['rO+rB', '(rO+rB)'])
col_dd   = pick_col(['d-d (Å)', 'd–d (Å)', 'd-d (Å)', 'd_d (Å)', 'd-d', 'd–d', 'd-d', 'd_d'])


X = df[[col_rOrB, 'deltaA', 'deltaB', 'ZA', 'ZB', 'ENA', 'ENB', 'Occ']].values
y = df[[col_dd]].values  


y_flat = y.ravel()
y_bin = None
for q_try in (10, 8, 6, 5, 4):
    try:
        tmp = pd.qcut(y_flat, q=q_try, labels=False, duplicates='drop')
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
        X, y, test_size=0.3, random_state=42, stratify=y_bin
    )


scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled  = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)  
y_test_scaled  = scaler_y.transform(y_test)


X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train_scaled)  
X_test_tensor  = torch.FloatTensor(X_test_scaled)
y_test_tensor  = torch.FloatTensor(y_test_scaled)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor),
                          batch_size=16, shuffle=True)


class HollanditeNN_DD(nn.Module):
    def __init__(self):
        super(HollanditeNN_DD, self).__init__()
        self.hidden = nn.Linear(8, 4)    
        self.output = nn.Linear(4, 1)    
        self.activation = nn.Sigmoid()   

    def forward(self, x):
        x = self.activation(self.hidden(x))
        x = self.output(x)
        return x

model = HollanditeNN_DD()
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
    avg_loss = running_loss / max(1, len(train_loader))
    train_losses.append(avg_loss)
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')


model.eval()
with torch.no_grad():
    
    y_train_pred_scaled = model(X_train_tensor).numpy()   
    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled)

    
    y_test_pred_scaled = model(X_test_tensor).numpy()
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)

    
    X_all_tensor = torch.FloatTensor(scaler_X.transform(X))
    y_all_pred_scaled = model(X_all_tensor).numpy()
    y_all_pred = scaler_y.inverse_transform(y_all_pred_scaled)


def R_value(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    num = np.sum((y_true - y_true.mean()) * (y_pred - y_pred.mean()))
    den = np.sqrt(np.sum((y_true - y_true.mean())**2) * np.sum((y_pred - y_pred.mean())**2))
    return num / den

def rmse(y_true, y_pred):
    y_true = y_true.reshape(-1); y_pred = y_pred.reshape(-1)
    return np.sqrt(np.mean((y_true - y_pred)**2))

def mae(y_true, y_pred):
    y_true = y_true.reshape(-1); y_pred = y_pred.reshape(-1)
    return np.mean(np.abs(y_true - y_pred))

R_train = R_value(y_train, y_train_pred); R2_train = R_train**2
R_test  = R_value(y_test,  y_test_pred ); R2_test  = R_test**2
R_all   = R_value(y,       y_all_pred  ); R2_all   = R_all**2

print(f"R\u00b2 (train/test/all) for d–d: {R2_train:.5f} / {R2_test:.5f} / {R2_all:.5f}")
print(f"RMSE (train/test/all): {rmse(y_train, y_train_pred):.5f} / {rmse(y_test, y_test_pred):.5f} / {rmse(y, y_all_pred):.5f}")
print(f"MAE  (train/test/all): {mae (y_train, y_train_pred):.5f} / {mae (y_test, y_test_pred):.5f} / {mae (y, y_all_pred):.5f}")


def create_corr_fig(y_true, y_pred, title, xlabel, ylabel, R2_value):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, color='blue', alpha=0.7, label='Data')
    mn = min(np.min(y_true), np.min(y_pred))
    mx = max(np.max(y_true), np.max(y_pred))
    plt.plot([mn, mx], [mn, mx], 'r--', label='1:1')
    m, b = np.polyfit(y_true.reshape(-1), y_pred.reshape(-1), 1)
    plt.plot([mn, mx], [m*mn + b, m*mx + b], 'm-', label='Fit')
    plt.title(f"{title} (R\u00b2={R2_value:.5f})")
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.legend(); plt.grid(True, alpha=0.3)
    inset
    errors = (y_pred - y_true).reshape(-1)
    ax_inset = plt.axes([0.6, 0.2, 0.25, 0.2])
    ax_inset.hist(errors, bins=20, color='green', alpha=0.7)
    ax_inset.axvline(x=0, color='red', linestyle='-', linewidth=1)
    ax_inset.set_xticks([]); ax_inset.set_yticks([])
    return plt.gcf()

fig = create_corr_fig(y_train, y_train_pred,
                      "ANN Training", "Observed d–d (Å)", "Predicted d–d (Å)", R2_train)
fig.savefig(os.path.join(outdir, 'dd_training.svg'), format='svg', dpi=600, bbox_inches='tight', pad_inches=0.05); plt.close(fig)

fig = create_corr_fig(y_test,  y_test_pred,
                      "ANN Testing",  "Observed d–d (Å)", "Predicted d–d (Å)", R2_test)
fig.savefig(os.path.join(outdir, 'dd_testing.svg'), format='svg', dpi=600, bbox_inches='tight', pad_inches=0.05); plt.close(fig)

fig = create_corr_fig(y,       y_all_pred,
                      "ANN All",     "Observed d–d (Å)", "Predicted d–d (Å)", R2_all)
fig.savefig(os.path.join(outdir, 'dd_all.svg'), format='svg', dpi=600, bbox_inches='tight', pad_inches=0.05); plt.close(fig)


torch.save(model.state_dict(), os.path.join(outdir, 'hollandite_nn_model_dd.pth'))

print("ANN(d–d) training/evaluation done. SVG figures (600 dpi) and model saved.")