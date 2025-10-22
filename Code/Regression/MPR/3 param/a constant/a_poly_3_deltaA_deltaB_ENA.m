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
    deltaB, deltaB.^2, ...
    deltaA .* deltaB, ...
    ENA, ENA.^2, ...
    deltaA .* ENA, ...
    deltaB .* ENA ...
];


Y_a = a_values - rO_rB_part_a;


coefficients_a = X_a \ Y_a;


a_pred = rO_rB_part_a + X_a * coefficients_a;



contributions_a = abs(coefficients_a);
total_contribution = sum(contributions_a);


normalized_contribution = contributions_a / total_contribution * 100;



fprintf('deltaA: %.6f%%\n', normalized_contribution(1));  
fprintf('deltaA^2: %.6f%%\n', normalized_contribution(2));  
fprintf('deltaB: %.6f%%\n', normalized_contribution(3)); 
fprintf('deltaB^2: %.6f%%\n', normalized_contribution(4)); 
fprintf('deltaA*deltaB: %.6f%%\n', normalized_contribution(5));  
fprintf('ENA: %.6f%%\n', normalized_contribution(6));    
fprintf('ENA^2: %.6f%%\n', normalized_contribution(7));  
fprintf('deltaA*ENA: %.6f%%\n', normalized_contribution(8));  
fprintf('deltaB*ENA: %.6f%%\n', normalized_contribution(9));  



residuals_a = a_values - a_pred;
MSE_a = mean(residuals_a.^2);
RMSE_a = sqrt(MSE_a);
MAE_a = mean(abs(residuals_a));
R2_a = 1 - sum(residuals_a.^2) / sum((a_values - mean(a_values)).^2);


fprintf('===== 3: a = 5.13*(rO+rB) + w1*deltaA + w2*(deltaA^2) + w3*deltaB + w4*(deltaB^2) + w5*(deltaA*deltaB) + w6*ENA + w7*(ENA^2) + w8*(deltaA*ENA) + w9*(deltaB*ENA) =====\n');
for i = 1:length(coefficients_a)
    fprintf('w%d = %.6f\n', i, coefficients_a(i));
end
fprintf('R² = %.6f\n', R2_a);
fprintf('MSE = %.6f, RMSE = %.6f, MAE = %.6f\n\n', MSE_a, RMSE_a, MAE_a);


n = length(a_values);   
p = size(X_a, 2);         

R2_adj_a = 1 - (1 - R2_a) * (n - 1) / (n - p - 1);

fprintf('R² = %.6f\n', R2_adj_a);


RSS = sum(residuals_a .^ 2);

AIC_a = n * log(RSS / n) + 2 * p;
BIC_a = n * log(RSS / n) + p * log(n);

fprintf('AIC = %.6f\n', AIC_a);
fprintf('BIC = %.6f\n', BIC_a);
fprintf('n = %.6f\n', n)
fprintf('p = %.6f\n', p)


X_poly_with_intercept = [ones(size(X_a, 1), 1), X_a];

[b_full, bint, r, rint, stats] = regress(Y_a, X_poly_with_intercept);


fprintf('\n=====  (p) =====\n');
for i = 2:length(b_full)  
    
    t_value = b_full(i) / ((bint(i,2) - bint(i,1)) / (2 * 1.96));
    
    p_value = 2 * (1 - tcdf(abs(t_value), n - p - 1));
    fprintf('%2d p : %.6f\n', i-1, p_value);
end


fprintf('\n===== F =====\n');
fprintf('F = %.6f, p = %.6f\n', stats(2), stats(3));
if stats(3) < 0.05
    fprintf('→ Significant（p < 0.05）\n');
else
    fprintf('→ Insignificant（p ≥ 0.05）\n');
end