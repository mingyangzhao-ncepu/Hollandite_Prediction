import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt


infile = r"E:\Hollandite_ML\Hollandite_data.xlsx"
outdir = r'E:\Hollandite_ML\Results'
os.makedirs(outdir, exist_ok=True)


df = pd.read_excel(infile)


X = df.iloc[:, 3:11].values

def _get_col_by_candidates(_df, candidates, fallback_iloc):
    for name in candidates:
        if name in _df.columns:
            return _df[name].values, name
    print(f"[Info] NotFound {candidates}，{fallback_iloc+1}")
    return _df.iloc[:, fallback_iloc].values, f"col_{fallback_iloc+1}"


y_a, a_colname = _get_col_by_candidates(
    df, ["a' (Å)", "a (Å)", "a", "a_prime"], 1
)
y_c, c_colname = _get_col_by_candidates(
    df, ["c' (Å)", "c (Å)", "c", "c_prime"], 2
)


X_train, X_test, y_a_train, y_a_test, y_c_train, y_c_test = train_test_split(
    X, y_a, y_c, test_size=0.30, random_state=42
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
X_all_scaled   = scaler.transform(X) 


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

def plot_predictions(y_true, y_pred, model_name, target_key, target_title=None):
    
    if target_title is None:
        pretty = {
            'a_prime': 'a (Å)', 'c_prime': 'c (Å)',
            'a': 'a (Å)', 'c': 'c (Å)'
        }
        target_title = pretty.get(target_key, target_key)

    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    mn = float(min(min(y_true), min(y_pred)))
    mx = float(max(max(y_true), max(y_pred)))
    plt.plot([mn, mx], [mn, mx], 'r--')
    plt.xlabel(f'Observed {target_title}')
    plt.ylabel(f'Predicted {target_title}')
    plt.title(f'{model_name} - Observed vs Predicted {target_title}')
    plt.grid(True, alpha=0.3)
    fname = f'{model_name}_{target_key}_predictions.svg'
    plt.savefig(os.path.join(outdir, fname), dpi=600, bbox_inches='tight', pad_inches=0.05)
    plt.close()


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

def analyze_feature_importance(model, X_test_scaled_local, y_test_local, feature_names, target_key):

    result = permutation_importance(
        model, X_test_scaled_local, y_test_local, n_repeats=10, random_state=42, n_jobs=-1
    )
    imp = result.importances_mean
    idx = np.argsort(imp)[::-1]

    print(f"Feature importance for {target_key}:")
    for i in idx:
        print(f"{feature_names[i]}: {imp[i]:.6f}")
    print("-" * 50)

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(imp)), imp[idx])
    plt.xticks(range(len(imp)), [feature_names[i] for i in idx], rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title(f'Feature Importance for {target_key} [SVR Permutation]')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f'feature_importance_{target_key}.svg'), dpi=600, bbox_inches='tight', pad_inches=0.05)
    plt.close()

feature_names = df.iloc[:, 3:11].columns.tolist()


print("Training SVR models...")

param_grid_svr = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1, 'scale', 'auto'],
    'epsilon': [0.01, 0.1, 0.2]
}


svr_a = SVR(kernel='rbf')
grid_search_svr_a = GridSearchCV(svr_a, param_grid_svr, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_svr_a.fit(X_train_scaled, y_a_train)
svr_a_best = grid_search_svr_a.best_estimator_
y_a_pred_svr_test = svr_a_best.predict(X_test_scaled)
print(f"Best SVR parameters for a (Å): {grid_search_svr_a.best_params_}")
svr_a_metrics = evaluate_model(y_a_test, y_a_pred_svr_test, "SVR", "a (Å)")
plot_predictions(y_a_test, y_a_pred_svr_test, "SVR", "a_prime", "a (Å)")


svr_c = SVR(kernel='rbf')
grid_search_svr_c = GridSearchCV(svr_c, param_grid_svr, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_svr_c.fit(X_train_scaled, y_c_train)
svr_c_best = grid_search_svr_c.best_estimator_
y_c_pred_svr_test = svr_c_best.predict(X_test_scaled)
print(f"Best SVR parameters for c (Å): {grid_search_svr_c.best_params_}")
svr_c_metrics = evaluate_model(y_c_test, y_c_pred_svr_test, "SVR", "c (Å)")
plot_predictions(y_c_test, y_c_pred_svr_test, "SVR", "c_prime", "c (Å)")


y_a_pred_svr_train = svr_a_best.predict(X_train_scaled)
y_a_pred_svr_all   = svr_a_best.predict(X_all_scaled)
R2_a_svr_train = r2_score(y_a_train, y_a_pred_svr_train)
R2_a_svr_test  = r2_score(y_a_test,  y_a_pred_svr_test)
R2_a_svr_all   = r2_score(y_a,       y_a_pred_svr_all)
print("\n[SVR - a] R^2 summary:")
print(f"  Train R^2 = {R2_a_svr_train:.5f}")
print(f"  Test  R^2 = {R2_a_svr_test:.5f}")
print(f"  All   R^2 = {R2_a_svr_all:.5f}")
create_correlation_plot(y_a_train, y_a_pred_svr_train,
                        "SVR Training", "Observed a (Å)", "Predicted a (Å)",
                        R2_a_svr_train, "a_SVR_training.svg")
create_correlation_plot(y_a_test, y_a_pred_svr_test,
                        "SVR Testing",  "Observed a (Å)", "Predicted a (Å)",
                        R2_a_svr_test,  "a_SVR_testing.svg")
create_correlation_plot(y_a, y_a_pred_svr_all,
                        "SVR All",      "Observed a (Å)", "Predicted a (Å)",
                        R2_a_svr_all,   "a_SVR_all.svg")


y_c_pred_svr_train = svr_c_best.predict(X_train_scaled)
y_c_pred_svr_all   = svr_c_best.predict(X_all_scaled)
R2_c_svr_train = r2_score(y_c_train, y_c_pred_svr_train)
R2_c_svr_test  = r2_score(y_c_test,  y_c_pred_svr_test)
R2_c_svr_all   = r2_score(y_c,       y_c_pred_svr_all)
print("\n[SVR - c] R^2 summary:")
print(f"  Train R^2 = {R2_c_svr_train:.5f}")
print(f"  Test  R^2 = {R2_c_svr_test:.5f}")
print(f"  All   R^2 = {R2_c_svr_all:.5f}")
create_correlation_plot(y_c_train, y_c_pred_svr_train,
                        "SVR Training (c)", "Actual c (Å)", "Predicted c (Å)",
                        R2_c_svr_train, "c_SVR_training.svg")
create_correlation_plot(y_c_test, y_c_pred_svr_test,
                        "SVR Testing (c)",  "Actual c (Å)", "Predicted c (Å)",
                        R2_c_svr_test,  "c_SVR_testing.svg")
create_correlation_plot(y_c, y_c_pred_svr_all,
                        "SVR All (c)",      "Actual c (Å)", "Predicted c (Å)",
                        R2_c_svr_all,   "c_SVR_all.svg")



print("\nModel Comparison Summary (a):")
print("-" * 50)
print(f"SVR - RMSE: {svr_a_metrics[1]:.6f}, R\u00b2: {svr_a_metrics[2]:.6f}")

print("-" * 50)
print("Model Comparison Summary (c):")
print("-" * 50)
print(f"SVR - RMSE: {svr_c_metrics[1]:.6f}, R\u00b2: {svr_c_metrics[2]:.6f}")

print("-" * 50)

analyze_feature_importance(svr_a_best, X_test_scaled, y_a_test, feature_names, "a (Å)")
analyze_feature_importance(svr_c_best, X_test_scaled, y_c_test, feature_names, "c (Å)")

print(f"Done. Results saved to: {outdir}")