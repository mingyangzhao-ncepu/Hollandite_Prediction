
filePath = 'E:\Hollandite\Hollandite_data_dd.xlsx';
data = readmatrix(filePath, 'Range', 'B2:L200');


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
dd_values = data(:,11);


rO_rB_part_dd = sqrt(6) * rO_rB;


X_dd = [ ...
    deltaA, ...
    deltaB, ...
    ZB, ...
    ENA, ...
    ENB, ...
    Occ, ...
];


Y_dd = dd_values - rO_rB_part_dd;


coefficients_dd = X_dd \ Y_dd;


dd_pred = rO_rB_part_dd + X_dd * coefficients_dd;


contributions_dd = abs(coefficients_dd);
total_contribution = sum(contributions_dd);


normalized_contribution = contributions_dd / total_contribution * 100;


fprintf('deltaA: %.6f%%\n', normalized_contribution(1));  
fprintf('deltaB: %.6f%%\n', normalized_contribution(2));  
fprintf('ZB: %.6f%%\n', normalized_contribution(3));     
fprintf('ENA: %.6f%%\n', normalized_contribution(4));    
fprintf('ENB: %.6f%%\n', normalized_contribution(5));    
fprintf('Occ: %.6f%%\n', normalized_contribution(6));    



residuals_dd = dd_values - dd_pred;
MSE_dd = mean(residuals_dd.^2);
RMSE_dd = sqrt(MSE_dd);
MAE_dd = mean(abs(residuals_dd));
R2_dd = 1 - sum(residuals_dd.^2) / sum((dd_values - mean(dd_values)).^2);


fprintf('===== 3: dd = sqrt(6)*(rO+rB) + w1*deltaA + w2*deltaB + w3*ZB + w4*ENA + w5*ENB + w6*Occ =====\n');
for i = 1:length(coefficients_dd)
    fprintf('w%d = %.6f\n', i, coefficients_dd(i));
end
fprintf('R² = %.6f\n', R2_dd);
fprintf('MSE = %.6f, RMSE = %.6f, MAE = %.6f\n\n', MSE_dd, RMSE_dd, MAE_dd);


n = length(dd_values);     
p = size(X_dd, 2);         

R2_adj_dd = 1 - (1 - R2_dd) * (n - 1) / (n - p - 1);

fprintf('R² = %.6f\n', R2_adj_dd);


RSS = sum(residuals_dd .^ 2);

AIC_dd = n * log(RSS / n) + 2 * p;
BIC_dd = n * log(RSS / n) + p * log(n);

fprintf('AIC = %.6f\n', AIC_dd);
fprintf('BIC = %.6f\n', BIC_dd);
fprintf('n = %.6f\n', n)
fprintf('p = %.6f\n', p)


features_matrix = [deltaA, deltaB, ZB, ENA, ENB, Occ];


correlation_matrix = corr(features_matrix);


figure;
h = heatmap(correlation_matrix, 'XDisplayLabels', {'\delta_{A}', '\delta_{B}', 'z_{B}', 'EN_{A}', 'EN_{B}', 'Occ'}, 'YDisplayLabels', {'\delta_{A}', '\delta_{B}', 'z_{B}', 'EN_{A}', 'EN_{B}', 'Occ'});
title('Correlation Heatmap for Features');
xlabel('Features');
ylabel('Features');
colorbar;


colormap(h, 'coolwarm');  % 'coolwarm' has red at high (+1) and blue at low (-1)
caxis([-1 1]);  % Set the color scale to match correlation range from -1 to 1


X_dd_with_intercept = [ones(size(X_dd, 1), 1), X_dd];


[b_full, bint, r, rint, stats] = regress(Y_dd, X_dd_with_intercept);


fprintf('\n===== (p) =====\n');
for i = 2:length(b_full)  
    t_value = b_full(i) / ((bint(i,2) - bint(i,1)) / (2*1.96));  
    p_value = 2 * (1 - tcdf(abs(t_value), n - p - 1));  
    fprintf('%d p: %.6f\n', i-1, p_value);
end


fprintf('\n===== F =====\n');
fprintf('F = %.6f, p = %.6f\n', stats(2), stats(3));
if stats(3) < 0.05
    fprintf('→ Significant（p < 0.05）\n');
else
    fprintf('→ Insignificant（p ≥ 0.05）\n');
end


figure;
scatter(dd_pred, residuals_dd, 40, 'filled');
xlabel('dd_{pred}', 'FontName', 'Arial');
ylabel('', 'FontName', 'Arial');
title(' (Residual Plot)', 'FontName', 'Arial');
grid on;
refline(0, 0);  


figure;
bar(normalized_contribution, 'FaceColor', [0.2 0.4 0.6]);
xticklabels({'\delta_A', '\delta_B', 'Z_B', 'EN_A', 'EN_B', 'Occ'});
ylabel('(%)', 'FontName', 'Arial');
title('dd', 'FontName', 'Arial');
grid on;


T = table(deltaA, deltaB, ZB, ENA, ENB, Occ, dd_values, ...
    'VariableNames', {'deltaA', 'deltaB', 'ZB', 'ENA', 'ENB', 'Occ', 'dd'});


step_model = stepwiselm(T, 'dd ~ 1', 'upper', 'linear', 'Verbose', 1);


disp('===== （stepwiselm） =====');
disp(step_model.Formula);
disp(step_model.Coefficients);


fprintf('\n===== VIF =====\n');
X_for_vif = X_dd;
var_names = {'deltaA', 'deltaB', 'ZB', 'ENA', 'ENB', 'Occ'};
VIF = zeros(1, size(X_for_vif,2));

for i = 1:size(X_for_vif,2)
    y_i = X_for_vif(:, i);
    X_others = X_for_vif(:, [1:i-1, i+1:end]);
    mdl = fitlm(X_others, y_i);
    R2_i = mdl.Rsquared.Ordinary;
    VIF(i) = 1 / (1 - R2_i);
    fprintf('%s VIF = %.3f\n', var_names{i}, VIF(i));
end
