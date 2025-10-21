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


rO_rB_part_c = sqrt(2) * rO_rB;


X_c = [ ...
    deltaA, ...
    deltaB, ...
    ZA, ...
    ENA, ...
    ENB, ...
    Occ, ...
];


Y_c = c_values - rO_rB_part_c;


coefficients_c = X_c \ Y_c;


c_pred = rO_rB_part_c + X_c * coefficients_c;


contributions_c = abs(coefficients_c);
total_contribution = sum(contributions_c);


normalized_contribution = contributions_c / total_contribution * 100;



fprintf('deltaA: %.6f%%\n', normalized_contribution(1));  
fprintf('deltaB: %.6f%%\n', normalized_contribution(2));  
fprintf('ZA: %.6f%%\n', normalized_contribution(3));     
fprintf('ENA: %.6f%%\n', normalized_contribution(4));    
fprintf('ENB: %.6f%%\n', normalized_contribution(5));  
fprintf('Occ: %.6f%%\n', normalized_contribution(6));    



residuals_c = c_values - c_pred;
MSE_c = mean(residuals_c.^2);
RMSE_c = sqrt(MSE_c);
MAE_c = mean(abs(residuals_c));
R2_c = 1 - sum(residuals_c.^2) / sum((c_values - mean(c_values)).^2);


fprintf('===== 2: c = sqrt(2)*(rO+rB) + w1*deltaA + w2*deltaB + w3*ZA + w4*ENA + w5*ENB + w6*Occ =====\n');
for i = 1:length(coefficients_c)
    fprintf('w%d = %.6f\n', i, coefficients_c(i));
end
fprintf('R² = %.6f\n', R2_c);
fprintf('MSE = %.6f, RMSE = %.6f, MAE = %.6f\n\n', MSE_c, RMSE_c, MAE_c);


n = length(c_values);     
p = size(X_c, 2);         

R2_adj_c = 1 - (1 - R2_c) * (n - 1) / (n - p - 1);

fprintf('R² = %.6f\n', R2_adj_c);


RSS = sum(residuals_c .^ 2);

AIC_c = n * log(RSS / n) + 2 * p;
BIC_c = n * log(RSS / n) + p * log(n);

fprintf('AIC = %.6f\n', AIC_c);
fprintf('BIC = %.6f\n', BIC_c);
fprintf('n = %.6f\n', n)
fprintf('p = %.6f\n', p)



features_matrix_a = [ZA, deltaA, deltaB, ENA, ENB, Occ];


correlation_matrix = corr(features_matrix_a);


figure;
h = heatmap(correlation_matrix, 'XDisplayLabels', {'z_{A}', '\delta_{A}', '\delta_{B}', 'EN_{A}', 'EN_{B}', 'Occ'}, 'YDisplayLabels', {'z_{A}', '\delta_{A}', '\delta_{B}', 'EN_{A}', 'EN_{B}', 'Occ'});
title('a_linear_5');
xlabel('Feature');
ylabel('Feature');
colorbar;


colormap(h, 'coolwarm');  % 'coolwarm' has red at high (+1) and blue at low (-1)
caxis([-1 1]);  % Set the color scale to match correlation range from -1 to 1




X_c_with_intercept = [ones(size(X_c, 1), 1), X_c];


[b_full, bint, r, rint, stats] = regress(Y_c, X_c_with_intercept);


fprintf('\n===== (p) =====\n');
for i = 2:length(b_full)  
    t_value = b_full(i) / ((bint(i,2) - bint(i,1)) / (2*1.96)); 
    p_value = 2 * (1 - tcdf(abs(t_value), n - p - 1)); 
    fprintf(' %d  p : %.6f\n', i-1, p_value);
end

fprintf('\n F\n');
fprintf('F = %.6f, p = %.6f\n', stats(2), stats(3));
if stats(3) < 0.05
    fprintf('→ Significant（p < 0.05）\n');
else
    fprintf('→ Insignificant（p ≥ 0.05）\n');
end




figure;
scatter(c_pred, residuals_c, 40, 'filled');
xlabel(' c_{pred}', 'FontName', 'Arial');
ylabel('', 'FontName', 'Arial');
title('Residual Plot', 'FontName', 'Arial');
grid on;
refline(0, 0);


figure;
bar(normalized_contribution, 'FaceColor', [0.2 0.4 0.6]);
xticklabels({'\delta_A', '\delta_B', 'Z_A', 'EN_A', 'EN_B', 'Occ'});
ylabel('', 'FontName', 'Arial');
title('', 'FontName', 'Arial');
grid on;




T = table(deltaA, deltaB, ZA, ENA, ENB, Occ, c_values, ...
    'VariableNames', {'\delta_A', '\delta_B', 'Z_A', 'EN_A', 'EN_B', 'Occ', 'c'});


step_model = stepwiselm(T, 'c ~ 1', 'upper', 'linear', 'Verbose', 1);


disp('===== stepwiselm =====');
disp(step_model.Formula);
disp(step_model.Coefficients);




fprintf('\n===== VIF=====\n');
X_for_vif = X_c;
var_names = {'\delta_A', '\delta_B', 'Z_A', 'EN_A', 'EN_B', 'Occ'};
VIF = zeros(1, size(X_for_vif,2));

for i = 1:size(X_for_vif,2)
    y_i = X_for_vif(:, i);
    X_others = X_for_vif(:, [1:i-1, i+1:end]);
    mdl = fitlm(X_others, y_i);
    R2_i = mdl.Rsquared.Ordinary;
    VIF(i) = 1 / (1 - R2_i);
    fprintf('%s VIF = %.3f\n', var_names{i}, VIF(i));
end

