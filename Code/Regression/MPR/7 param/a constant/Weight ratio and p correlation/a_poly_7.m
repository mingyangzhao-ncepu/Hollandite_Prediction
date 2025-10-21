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
    ZA, ZA.^2, ...
    ZB, ZB.^2, ...
    ENA, ENA.^2, ...
    ENB, ENB.^2, ...
    Occ, Occ.^2, ...
    deltaA .* deltaB, ...
    deltaA .* ZA, ...
    deltaA .* ZB, ...
    deltaA .* ENA, ...
    deltaA .* ENB, ...
    deltaA .* Occ, ...
    deltaB .* ZA, ...
    deltaB .* ZB, ...
    deltaB .* ENA, ...
    deltaB .* ENB, ...
    deltaB .* Occ, ...
    ZA .* ZB, ...
    ZA .* ENA, ...
    ZA .* ENB ...
    ZA .* Occ ...
    ZB .* ENA ...
    ZB .* ENB ...
    ZB .* Occ ...
    ENA .* ENB ...
    ENA .* Occ ...
    ENB .* Occ ...    
];


Y_a = a_values - rO_rB_part_a;


coefficients_a = X_a \ Y_a;


a_pred = rO_rB_part_a + X_a * coefficients_a;


contributions_a = abs(coefficients_a);
total_contribution = sum(contributions_a);


normalized_contribution = contributions_a / total_contribution * 100;



for i = 1:length(contributions_a)
    fprintf('%d: %.6f%%\n', i, normalized_contribution(i)); 
end




residuals_a = a_values - a_pred;
MSE_a = mean(residuals_a.^2);
RMSE_a = sqrt(MSE_a);
MAE_a = mean(abs(residuals_a));
R2_a = 1 - sum(residuals_a.^2) / sum((a_values - mean(a_values)).^2);


fprintf('===== 1: a = 5.13*(rO+rB) + w1*deltaA + w2*(deltaA^2) + w3*deltaB + w4*(deltaB^2) + w5*ZA + w6*(ZA^2) + w7*ZB + w8*(ZB^2) + w9*ENA + w10*(ENA^2)+ w11*ENB + w12*(ENB^2) + w13*Occ + w14*(Occ^2) + + w15*(deltaA*deltaB) + w16*(deltaA*ZA) + w17*(deltaA*ZB) + w18*(deltaA*ENA) + w19*(deltaA*ENB) + w20*(deltaA*Occ) + w21*(deltaB*ZA) +  w22*(deltaB*ZB) + w23*(deltaB*ENA) + w24*(deltaB*ENB) + w25*(deltaB*Occ) + w26*(ZA*ZB) + w27*(ZA*ENA) + w28*(ZA*ENB) + w29*(ZA*Occ) + w30*(ZB*ENA) + w31*(ZB*ENB) + w32*(ZB*Occ) + w33*(ENA*ENB) + w34*(ENA*Occ) + w35*(ENB*Occ) =====\n');
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


features_matrix = [deltaA, deltaA.^2, deltaB, deltaB.^2, ZA, ZA.^2, ZB, ZB.^2, ENA, ENA.^2, ENB, ENB.^2, Occ, Occ.^2, ...
    deltaA .* deltaB, deltaA .* ZA, deltaA .* ZB, deltaA .* ENA, deltaA .* ENB, deltaA .* Occ, ...
    deltaB .* ZA, deltaB .* ZB, deltaB .* ENA, deltaB .* ENB, deltaB .* Occ, ...
    ZA .* ZB, ZA .* ENA, ZA .* ENB, ZA .* Occ, ...
    ZB .* ENA, ZB .* ENB, ZB .* Occ, ENA .* ENB, ENA .* Occ, ENB .* Occ];


correlation_matrix = corr(features_matrix);


figure;
h = heatmap(correlation_matrix, 'XDisplayLabels', {'deltaA', 'deltaA^2', 'deltaB', 'deltaB^2', 'ZA', 'ZA^2', 'ZB', 'ZB^2', 'ENA', 'ENA^2', 'ENB', 'ENB^2', 'Occ', 'Occ^2', 'deltaA*deltaB', 'deltaA*ZA', 'deltaA*ZB', 'deltaA*ENA', 'deltaA*ENB', 'deltaA*Occ', 'deltaB*ZA', 'deltaB*ZB', 'deltaB*ENA', 'deltaB*ENB', 'deltaB*Occ', 'ZA*ZB', 'ZA*ENA', 'ZA*ENB', 'ZA*Occ', 'ZB*ENA', 'ZB*ENB', 'ZB*Occ', 'ENA*ENB', 'ENA*Occ', 'ENB*Occ'}, 'YDisplayLabels', {'deltaA', 'deltaA^2', 'deltaB', 'deltaB^2', 'ZA', 'ZA^2', 'ZB', 'ZB^2', 'ENA', 'ENA^2', 'ENB', 'ENB^2', 'Occ', 'Occ^2', 'deltaA*deltaB', 'deltaA*ZA', 'deltaA*ZB', 'deltaA*ENA', 'deltaA*ENB', 'deltaA*Occ', 'deltaB*ZA', 'deltaB*ZB', 'deltaB*ENA', 'deltaB*ENB', 'deltaB*Occ', 'ZA*ZB', 'ZA*ENA', 'ZA*ENB', 'ZA*Occ', 'ZB*ENA', 'ZB*ENB', 'ZB*Occ', 'ENA*ENB', 'ENA*Occ', 'ENB*Occ'});
title('Correlation Heatmap for Features');
xlabel('Features');
ylabel('Features');
colorbar;


colormap(h, 'coolwarm');  % 'coolwarm' has red at high (+1) and blue at low (-1)
caxis([-1 1]);  % Set the color scale to match correlation range from -1 to 1