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
    deltaB, deltaB.^2, ...
    deltaA .* deltaB, ...
    ENA, ENA.^2, ...
    deltaA .* ENA, ...
    deltaB .* ENA ...
];

Y_c = c_values - rO_rB_part_c;


coefficients_c = X_c \ Y_c;


c_pred = rO_rB_part_c + X_c * coefficients_c;


contributions_c = abs(coefficients_c);
total_contribution_c = sum(contributions_c);

% Calculate the normalized contributions (percentage) for each feature
normalized_contribution_c = contributions_c / total_contribution_c * 100;



fprintf('deltaA: %.6f%%\n', normalized_contribution_c(1));  
fprintf('deltaA^2: %.6f%%\n', normalized_contribution_c(2));  
fprintf('deltaB: %.6f%%\n', normalized_contribution_c(3));  
fprintf('deltaB^2: %.6f%%\n', normalized_contribution_c(4));  
fprintf('deltaA*deltaB: %.6f%%\n', normalized_contribution_c(5));  
fprintf('ENA: %.6f%%\n', normalized_contribution_c(6));     
fprintf('ENA^2: %.6f%%\n', normalized_contribution_c(7));    
fprintf('deltaA*ENA: %.6f%%\n', normalized_contribution_c(8));    
fprintf('deltaB*ENA: %.6f%%\n', normalized_contribution_c(9));    



residuals_c = c_values - c_pred;
MSE_c = mean(residuals_c.^2);
RMSE_c = sqrt(MSE_c);
MAE_c = mean(abs(residuals_c));
R2_c = 1 - sum(residuals_c.^2) / sum((c_values - mean(c_values)).^2);


fprintf('===== 3: c = sqrt(2)*(rO+rB) + w1*deltaA + w2*(deltaA^2) + w3*deltaB + w4*(deltaB^2) + w5*(deltaA*deltaB) + w6*ENA + w7*(ENA^2) + w8*(deltaA*ENA) + w9*(deltaB*ENA) =====\n');
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


features_matrix = [deltaA, deltaA.^2, deltaB, deltaB.^2, deltaA.*deltaB, ENA, ENA.^2, deltaA.*ENA, deltaB.*ENA];


correlation_matrix = corr(features_matrix);


figure;
h = heatmap(correlation_matrix, 'XDisplayLabels', {'\delta_{A}', '\delta_{A}^2', '\delta_{B}', '\delta_{B}^2', '\delta_{A}*\delta_{B}', 'EN_{A}', 'EN_{A}^2', '\delta_{A}*EN_{A}', '\delta_{B}*EN_{A}'}, 'YDisplayLabels', {'\delta_{A}', '\delta_{A}^2', '\delta_{B}', '\delta_{B}^2', '\delta_{A}*\delta_{B}', 'EN_{A}', 'EN_{A}^2', '\delta_{A}*EN_{A}', '\delta_{B}*EN_{A}'});
title('Correlation Heatmap for c_poly_3');
xlabel('Features');
ylabel('Features');
colorbar;


colormap(h, 'coolwarm');  % 'coolwarm' has red at high (+1) and blue at low (-1)
caxis([-1 1]);  % Set the color scale to match correlation range from -1 to 1