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
    ZA, ZA.^2, ...
    ZB, ZB.^2, ...
    ENA, ENA.^2, ...
    ENB, ENB.^2, ...
    deltaA .* ZA, ...
    deltaA .* ZB, ...
    deltaA .* ENA, ...
    deltaA .* ENB, ...
    ZA .* ZB, ...
    ZA .* ENA, ...
    ZA .* ENB, ...
    ZB .* ENA, ...
    ZB .* ENB, ...
    ENA .* ENB ...    
];


Y_c = c_values - rO_rB_part_c;


coefficients_c = X_c \ Y_c;


c_pred = rO_rB_part_c + X_c * coefficients_c;


residuals_c = c_values - c_pred;
MSE_c = mean(residuals_c.^2);
RMSE_c = sqrt(MSE_c);
MAE_c = mean(abs(residuals_c));
R2_c = 1 - sum(residuals_c.^2) / sum((c_values - mean(c_values)).^2);


fprintf('===== 2: c = sqrt(2)*(rO+rB) + w1*deltaA + w2*(deltaA^2) + w3*ZA + w4*(ZA^2) + w5*ZB + w6*(ZB^2) + w7*ENA + w8*(ENA^2) + w9*ENB + w10*(ENB^2) + w11*(deltaA*ZA) + w12*(deltaA*ZB) + w13*(deltaA*ENA) + w14*(deltaA*ENB) + w15*(ZA*ZB) +  w16*(ZA*ENA) + w17*(ZA*ENB) + w18*(ZB*ENA) + w19*(ZB*ENB) + w20*(ENA*ENB) =====\n');
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