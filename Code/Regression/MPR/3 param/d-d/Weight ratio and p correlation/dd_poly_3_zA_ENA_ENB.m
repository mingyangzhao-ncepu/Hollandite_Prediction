
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
    ZA, ZA.^2, ...
    ENA, ENA.^2, ...
    ZA .* ENA, ...
    ENB, ENB.^2, ...
    ZA .* ENB, ...
    ENA .* ENB ...
];


Y_dd = dd_values - rO_rB_part_dd;


coefficients_dd = X_dd \ Y_dd;


dd_pred = rO_rB_part_dd + X_dd * coefficients_dd;



contributions_dd = abs(coefficients_dd);
total_contribution = sum(contributions_dd);


normalized_contribution = contributions_dd / total_contribution * 100;



for i = 1:length(contributions_dd)
    fprintf('%d: %.6f%%\n', i, normalized_contribution(i)); 
end




residuals_dd = dd_values - dd_pred;
MSE_dd = mean(residuals_dd.^2);
RMSE_dd = sqrt(MSE_dd);
MAE_dd = mean(abs(residuals_dd));
R2_dd = 1 - sum(residuals_dd.^2) / sum((dd_values - mean(dd_values)).^2);


fprintf('===== 3: dd = sqrt(6)*(rO+rB) + w1*ZA + w2*(ZA^2) + w3*ENA + w4*(ENA^2) + w5*(ZA*ENA) + w6*ENB + w7*(ENB^2) + w8*(ZA*ENB) + w9*(ENA*ENB) =====\n');
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


features_matrix = [ZA, ZA.^2, ENA, ENA.^2, ENB, ENB.^2, ...
    ZA .* ENA, ZA .* ENB, ...
    ENA .* ENB];


correlation_matrix = corr(features_matrix);


figure;
h = heatmap(correlation_matrix, 'XDisplayLabels', {'ZA', 'ZA^2', 'ENA', 'ENA^2', 'ENB', 'ENB^2', 'ZA*ENA', 'ZA*ENB', 'ENA*ENB'}, 'YDisplayLabels', {'ZA', 'ZA^2', 'ENA', 'ENA^2', 'ENB', 'ENB^2', 'ZA*ENA', 'ZA*ENB', 'ENA*ENB'});
title('Correlation Heatmap for Features');
xlabel('Features');
ylabel('Features');
colorbar;


colormap(h, 'coolwarm');  % 'coolwarm' has red at high (+1) and blue at low (-1)
caxis([-1 1]);  % Set the color scale to match correlation range from -1 to 1


X_dd_with_intercept = [ones(size(X_dd,1),1), X_dd];
[b_full_dd, bint_dd, r_dd, rint_dd, stats_dd] = regress(Y_dd, X_dd_with_intercept);

fprintf('\n===== (p) =====\n');
for i = 2:length(b_full_dd)
    t_val = b_full_dd(i) / ((bint_dd(i,2) - bint_dd(i,1)) / (2*1.96));
    p_val = 2 * (1 - tcdf(abs(t_val), n - p - 1));
    fprintf('%d p: %.6f\n', i-1, p_val);
end

fprintf('\n===== F =====\n');
fprintf('F = %.6f, p = %.6f\n', stats_dd(2), stats_dd(3));
if stats_dd(3) < 0.05
    fprintf('→ Significant（p < 0.05）\n');
else
    fprintf('→ Insignificant（p ≥ 0.05）\n');
end


figure;
scatter(dd_pred, residuals_dd, 40, 'filled');
xlabel('dd_{pred}', 'FontName', 'Arial');
ylabel('', 'FontName', 'Arial');
title('(Residual Plot)', 'FontName', 'Arial');
grid on;
refline(0,0);


figure;
bar(normalized_contribution, 'FaceColor', [0.2 0.5 0.7]);
xticks(1:35);
xticklabels({ ...
    'ZA','ZA^2', ...
    'ENA','ENA^2','ENB','ENB^2', ...
    'ZA*ENA','ZA*ENB', ...
    'ENA*ENB'});
ylabel(' (%)', 'FontName', 'Arial');
title('dd', 'FontName', 'Arial');
grid on;



X = [ZA, ENA, ENB, ...
     ZA.^2, ENA.^2, ENB.^2, ...
     ZA.*ENA, ZA.*ENB,...
     ENA.*ENB];


varNames = {'ZA','ENA','ENB', ...
            'ZA2','ENA2','ENB2', ...
            'ZA_ENA','ZA_ENB', ...
            'ENA_ENB'};


[B, FitInfo] = lasso(X, dd_values, 'CV', 5);  
idxLambda1SE = FitInfo.Index1SE;
selectedCoeffs = B(:, idxLambda1SE);
selectedVars = varNames(selectedCoeffs ~= 0);

fprintf('===== LASSO=====\n');
disp(selectedVars');


X_selected = X(:, selectedCoeffs ~= 0);
mdl_final = fitlm(X_selected, dd_values);
disp(mdl_final);



fprintf('\n===== VIF =====\n');
X_for_vif_dd = X_dd;
var_names_dd = { ...
    'ZA','ZA^2', ...
    'ENA','ENA^2','ENB','ENB^2', ...
    'ZA*ENA','ZA*ENB', ...
    'ENA*ENB'};

VIF_dd = zeros(1, size(X_for_vif_dd,2));
for i = 1:size(X_for_vif_dd,2)
    y_i = X_for_vif_dd(:, i);
    X_others = X_for_vif_dd(:, [1:i-1, i+1:end]);
    mdl = fitlm(X_others, y_i);
    R2_i = mdl.Rsquared.Ordinary;
    VIF_dd(i) = 1 / (1 - R2_i);
    fprintf('%s 的 VIF = %.3f\n', var_names_dd{i}, VIF_dd(i));
end