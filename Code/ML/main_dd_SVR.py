import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance


infile = r"E:\Hollandite_ML\Hollandite_data.xlsx"
outdir = r"E:\Hollandite_ML\Results"
os.makedirs(outdir, exist_ok=True)


df = pd.read_excel(infile)


X = df.iloc[:, 3:11].values  
try:
    y_d = df["d-d (Å)"].values
except KeyError:
    
    y_d = df.iloc[:, 1].values
    print("[Info] NotFound 'd-d (Å)'")


X_train, X_test, y_train, y_test = train_test_split(
    X, y_d, test_size=0.30, random_state=42
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


def evaluate_model(y_true, y_pred, model_name, target_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} for {target_name}:")
    print(f"  MSE  : {mse:.6f}")
    print(f"  RMSE : {rmse:.6f}")
    print(f"  R\u00b2 : {r2:.6f}")
    print("-" * 50)
    return mse, rmse, r2

def plot_predictions(y_true, y_pred, model_name, target_key, target_title):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    mn = float(min(min(y_true), min(y_pred)))
    mx = float(max(max(y_true), max(y_pred)))
    plt.plot([mn, mx], [mn, mx], 'r--')
    plt.xlabel(f'Actual {target_title}')
    plt.ylabel(f'Predicted {target_title}')
    plt.title(f'{model_name} - Actual vs Predicted {target_title}')
    plt.grid(True, alpha=0.3)
    fname = f'{model_name}_{target_key}_predictions.svg'
    plt.savefig(os.path.join(outdir, fname), dpi=600, bbox_inches='tight', pad_inches=0.05)
    plt.close()


print("Training SVR model for d-d ...")
param_grid_svr = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1, 'scale', 'auto'],
    'epsilon': [0.01, 0.1, 0.2]
}
svr = SVR(kernel='rbf')
grid_search_svr = GridSearchCV(
    svr, param_grid_svr, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
)
grid_search_svr.fit(X_train_scaled, y_train)
svr_best = grid_search_svr.best_estimator_
y_pred_svr = svr_best.predict(X_test_scaled)

print(f"Best SVR parameters for d-d (Å): {grid_search_svr.best_params_}")
svr_metrics = evaluate_model(y_test, y_pred_svr, "SVR", "d-d (Å)")
plot_predictions(y_test, y_pred_svr, "SVR", "d_d", "d-d (Å)")




print("Model Comparison Summary (d-d):")
print("-" * 50)
print(f"SVR - RMSE: {svr_metrics[1]:.6f}, R\u00b2: {svr_metrics[2]:.6f}")
print("-" * 50)


feature_names = df.iloc[:, 3:11].columns.tolist()

result = permutation_importance(
    svr_best, X_test_scaled, y_test, n_repeats=10, random_state=42, n_jobs=-1
)
imp = result.importances_mean
idx = np.argsort(imp)[::-1]

print("Feature importance for d-d (Å):")
for i in idx:
    print(f"{feature_names[i]}: {imp[i]:.6f}")
print("-" * 50)

plt.figure(figsize=(10, 6))
plt.bar(range(len(imp)), imp[idx])
plt.xticks(range(len(imp)), [feature_names[i] for i in idx], rotation=45, ha='right')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance for d-d (Å) [SVR Permutation]')
plt.tight_layout()
plt.savefig(os.path.join(outdir, 'feature_importance_d_d.svg'), dpi=600, bbox_inches='tight', pad_inches=0.05)
plt.close()

print(f"Done. Results saved to: {outdir}")






from sklearn.metrics import r2_score


X_all_scaled = scaler.transform(X)


def create_correlation_plot(y_true, y_pred, title, xlabel, ylabel, R2_value, fname):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, color='blue', alpha=0.7, label='Data')

    min_val = float(min(np.min(y_true), np.min(y_pred)))
    max_val = float(max(np.max(y_true), np.max(y_pred)))
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

    plt.savefig(os.path.join(outdir, fname), dpi=600, bbox_inches='tight', pad_inches=0.05)
    plt.close()


y_train_pred_svr = svr_best.predict(X_train_scaled)
y_all_pred_svr   = svr_best.predict(X_all_scaled)

R2_svr_train = r2_score(y_train, y_train_pred_svr)
R2_svr_test  = r2_score(y_test,  y_pred_svr)        
R2_svr_all   = r2_score(y_d,     y_all_pred_svr)

print("\n[SVR - d-d] R^2 summary:")
print(f"  Train R^2 = {R2_svr_train:.5f}")
print(f"  Test  R^2 = {R2_svr_test:.5f}")
print(f"  All   R^2 = {R2_svr_all:.5f}")

create_correlation_plot(y_train, y_train_pred_svr,
                        "SVR Training", "Observed d-d (Å)", "Predicted d-d (Å)",
                        R2_svr_train, "dd_SVR_training.svg")
create_correlation_plot(y_test, y_pred_svr,
                        "SVR Testing",  "Observed d-d (Å)", "Predicted d-d (Å)",
                        R2_svr_test,  "dd_SVR_testing.svg")
create_correlation_plot(y_d, y_all_pred_svr,
                        "SVR All",      "Observed d-d (Å)", "Predicted d-d (Å)",
                        R2_svr_all,   "dd_SVR_all.svg")