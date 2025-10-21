filePath = 'E:\Hollandite\Hollandite_data.xlsx';
data = readmatrix(filePath, 'Range', 'B2:K196');


data(any(isnan(data), 2), :) = [];

a_values = data(:, 1);       
c_values = data(:, 2);      
rO_rB    = data(:, 3);     
deltaA   = data(:, 4);   
deltaB   = data(:, 5);      
ZA       = data(:, 6);    
ZB       = data(:, 7);    
ENA      = data(:, 8);      
ENB      = data(:, 9);       
Occ      = data(:,10);   

rO_rB_part_a = 5.13 * rO_rB;


X_a = [ ...
    deltaA, deltaA.^2, ...
    Occ, Occ.^2, ...
    deltaA .* Occ, ...
];


Y_a = a_values - rO_rB_part_a;


coefficients_a = X_a \ Y_a;


a_pred = rO_rB_part_a + X_a * coefficients_a;


residuals_a = a_values - a_pred;
MSE_a = mean(residuals_a.^2);
RMSE_a = sqrt(MSE_a);
MAE_a = mean(abs(residuals_a));
R2_a = 1 - sum(residuals_a.^2) / sum((a_values - mean(a_values)).^2);


fprintf('===== 3: a = 5.13*(rO+rB) + w1*deltaA + w2*(deltaA^2) + w3*Occ + w4*(Occ^2) + w5*(deltaA*Occ) =====\n');
for i = 1:length(coefficients_a)
    fprintf('w%d = %.6f\n', i, coefficients_a(i));
end
fprintf('R² = %.6f\n', R2_a);
fprintf('MSE = %.6f, RMSE = %.6f, MAE = %.6f\n\n', MSE_a, RMSE_a, MAE_a);

% ========== 计算并输出调整后 R² ==========
n = length(a_values);     % 样本数量
p = size(X_a, 2);         % 自变量数量（不含固定项 5.13*(rO+rB)）

R2_adj_a = 1 - (1 - R2_a) * (n - 1) / (n - p - 1);

fprintf('调整后 R² = %.6f\n', R2_adj_a);

% ========== 计算 AIC / BIC ==========
% AIC = n * log(RSS/n) + 2 * p
% BIC = n * log(RSS/n) + p * log(n)
RSS = sum(residuals_a .^ 2);

AIC_a = n * log(RSS / n) + 2 * p;
BIC_a = n * log(RSS / n) + p * log(n);

fprintf('AIC = %.6f\n', AIC_a);
fprintf('BIC = %.6f\n', BIC_a);
fprintf('n = %.6f\n', n)
fprintf('p = %.6f\n', p)