
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
    deltaA, deltaA.^2, ...
    deltaB, deltaB.^2, ...
    ZA, ZA.^2, ...
    ENA, ENA.^2, ...
    ENB, ENB.^2, ...
    deltaA .* deltaB, ...
    deltaA .* ZA, ...
    deltaA .* ENA, ...
    deltaA .* ENB, ...
    deltaB .* ZA, ...
    deltaB .* ENA, ...
    deltaB .* ENB, ...
    ZA .* ENA, ...
    ZA .* ENB, ...
    ENA .* ENB ...    
];


Y_dd = dd_values - rO_rB_part_dd;


coefficients_dd = X_dd \ Y_dd;


dd_pred = rO_rB_part_dd + X_dd * coefficients_dd;


residuals_dd = dd_values - dd_pred;
MSE_dd = mean(residuals_dd.^2);
RMSE_dd = sqrt(MSE_dd);
MAE_dd = mean(abs(residuals_dd));
R2_dd = 1 - sum(residuals_dd.^2) / sum((dd_values - mean(dd_values)).^2);


fprintf('===== 3: dd = sqrt(6)*(rO+rB) + w1*deltaA + w2*(deltaA^2) + w3*deltaB + w4*(deltaB^2) + w5*ZA + w6*(ZA^2) + w7*ENA + w8*(ENA^2) + w9*ENB + w10*(ENB^2) + w11*(deltaA*deltaB) + w12*(deltaA*ZA) + w13*(deltaA*ENA) + w14*(deltaA*ENB) + w15*(deltaB*ZA) +  w16*(deltaB*ENA) + w17*(deltaB*ENB) + w18*(ZA*ENA) + w19*(ZA*ENB) + w20*(ENA*ENB) =====\n');
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

