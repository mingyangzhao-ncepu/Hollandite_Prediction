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
    ZB, ZB.^2, ...
    ENA, ENA.^2, ...
    deltaA .* deltaB, ...
    deltaA .* ZB, ...
    deltaA .* ENA, ...
    deltaB .* ZB, ...
    deltaB .* ENA, ...
    ZB .* ENA ...    
];


Y_a = a_values - rO_rB_part_a;


coefficients_a = X_a \ Y_a;


a_pred = rO_rB_part_a + X_a * coefficients_a;


residuals_a = a_values - a_pred;
MSE_a = mean(residuals_a.^2);
RMSE_a = sqrt(MSE_a);
MAE_a = mean(abs(residuals_a));
R2_a = 1 - sum(residuals_a.^2) / sum((a_values - mean(a_values)).^2);


fprintf('===== 1: a = 5.13*(rO+rB) + w1*deltaA + w2*(deltaA^2) + w3*deltaB + w4*(deltaB^2) + w5*ZB + w6*(ZB^2) + w7*ENA + w8*(ENA^2) + w9*(deltaA*deltaB) + w10*(deltaA*ZB) + w11*(deltaA*ENA) + w12*(deltaB*ZB) + w13*(deltaB*ENA) + w14*(ZB*ENA) =====\n');
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