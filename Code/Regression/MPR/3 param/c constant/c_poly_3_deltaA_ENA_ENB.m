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
    deltaA, deltaA.^2, ...
    ENA, ENA.^2, ...
    deltaA .* ENA, ...
    ENB, ENB.^2, ...
    deltaA .* ENB, ...
    ENA .* ENB ...
];


Y_c = c_values - rO_rB_part_c;


coefficients_c = X_c \ Y_c;


c_pred = rO_rB_part_c + X_c * coefficients_c;



contributions_c = abs(coefficients_c);
total_contribution = sum(contributions_c);


normalized_contribution = contributions_c / total_contribution * 100;


fprintf('deltaA: %.6f%%\n', normalized_contribution(1));  
fprintf('deltaA^2: %.6f%%\n', normalized_contribution(2));  
fprintf('ENA: %.6f%%\n', normalized_contribution(3));  
fprintf('ENA^2: %.6f%%\n', normalized_contribution(4));  
fprintf('deltaA*ENA: %.6f%%\n', normalized_contribution(5));  
fprintf('ENB: %.6f%%\n', normalized_contribution(6));    
fprintf('ENB^2: %.6f%%\n', normalized_contribution(7));  
fprintf('deltaA*ENB: %.6f%%\n', normalized_contribution(8));  
fprintf('ENA*ENB: %.6f%%\n', normalized_contribution(9));  



residuals_c = c_values - c_pred;
MSE_c = mean(residuals_c.^2);
RMSE_c = sqrt(MSE_c);
MAE_c = mean(abs(residuals_c));
R2_c = 1 - sum(residuals_c.^2) / sum((c_values - mean(c_values)).^2);

fprintf('===== 3: c = sqrt(2)*(rO+rB) + w1*deltaA + w2*(deltaA^2) + w3*ENA + w4*(ENA^2) + w5*(deltaA*ENA) + w6*ENB + w7*(ENB^2) + w8*(deltaA*ENB) + w9*(ENA*ENB) =====\n');
for i = 1:length(coefficients_c)
    fprintf(' w%d = %.6f\n', i, coefficients_c(i));
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


X_poly_with_intercept = [ones(size(X_c, 1), 1), X_c];


[b_full, bint, r, rint, stats] = regress(Y_c, X_poly_with_intercept);


fprintf('\n===== (p) =====\n');
for i = 2:length(b_full)  
    
    t_value = b_full(i) / ((bint(i,2) - bint(i,1)) / (2 * 1.96));
    
    p_value = 2 * (1 - tcdf(abs(t_value), n - p - 1));
    fprintf('%2d p: %.6f\n', i-1, p_value);
end


fprintf('\n F\n');
fprintf('F = %.6f, p = %.6f\n', stats(2), stats(3));
if stats(3) < 0.05
    fprintf('→ Significant（p < 0.05）\n');
else
    fprintf('→ Insignificant（p ≥ 0.05）\n');
end