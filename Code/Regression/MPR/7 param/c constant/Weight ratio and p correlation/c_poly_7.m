% ========== 数据读取部分 ==========
filePath = 'E:\华电入职\论文发表\Hollandite\All hollandite\All base(revised)\Polynomial\All based hollandite fit data with Name ENBA Occ ZA-04092025.xlsx';
data = readmatrix(filePath, 'Range', 'B2:K196');

% 删除包含 NaN 的行
data(any(isnan(data), 2), :) = [];

% ========== 提取变量 ==========
a_values = data(:, 1);       % a' (Å)
c_values = data(:, 2);       % c' (Å)
rO_rB    = data(:, 3);       % (rO + rB)
deltaA   = data(:, 4);       % deltaA
deltaB   = data(:, 5);       % deltaB
ZA       = data(:, 6);       % ZA
ZB       = data(:, 7);       % ZB
ENA      = data(:, 8);       % ENA
ENB      = data(:, 9);       % ENB
Occ      = data(:, 10);       % Occ

% ========== 构造拟合公式项 ==========
rO_rB_part_c = sqrt(2) * rO_rB;

% 构造特征矩阵 X（35 项）
X_c = [ ...
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

% 拟合 y = c - sqrt(2)*(rO+rB)
Y_c = c_values - rO_rB_part_c;

% 线性回归
coefficients_c = X_c \ Y_c;

% 计算预测值
c_pred = rO_rB_part_c + X_c * coefficients_c;

% 计算贡献值 (系数的绝对值表示贡献的大小)
contributions_c = abs(coefficients_c);
total_contribution = sum(contributions_c);

% 计算贡献值的归一化百分比
normalized_contribution = contributions_c / total_contribution * 100;

% 输出贡献值分析
fprintf('===== 贡献值分析：===== \n');
for i = 1:length(contributions_c)
    fprintf('特征 %d 的贡献: %.6f%%\n', i, normalized_contribution(i)); 
end

fprintf('总贡献 (百分比之和): %.6f%%\n\n', sum(normalized_contribution));  % Sum of percentages should be 100%

% ========== 拟合评估指标 ==========
residuals_c = c_values - c_pred;
MSE_c = mean(residuals_c.^2);
RMSE_c = sqrt(MSE_c);
MAE_c = mean(abs(residuals_c));
R2_c = 1 - sum(residuals_c.^2) / sum((c_values - mean(c_values)).^2);

% ========== 输出结果 ==========
fprintf('===== 公式 2: c = sqrt(2)*(rO+rB) + w1*deltaA + w2*(deltaA^2) + w3*deltaB + w4*(deltaB^2) + w5*ZA + w6*(ZA^2) + w7*ZB + w8*(ZB^2) + w9*ENA + w10*(ENA^2)+ w11*ENB + w12*(ENB^2) + w13*Occ + w14*(Occ^2) + + w15*(deltaA*deltaB) + w16*(deltaA*ZA) + w17*(deltaA*ZB) + w18*(deltaA*ENA) + w19*(deltaA*ENB) + w20*(deltaA*Occ) + w21*(deltaB*ZA) +  w22*(deltaB*ZB) + w23*(deltaB*ENA) + w24*(deltaB*ENB) + w25*(deltaB*Occ) + w26*(ZA*ZB) + w27*(ZA*ENA) + w28*(ZA*ENB) + w29*(ZA*Occ) + w30*(ZB*ENA) + w31*(ZB*ENB) + w32*(ZB*Occ) + w33*(ENA*ENB) + w34*(ENA*Occ) + w35*(ENB*Occ) =====\n');
for i = 1:length(coefficients_c)
    fprintf('拟合系数 w%d = %.6f\n', i, coefficients_c(i));
end
fprintf('R² = %.6f\n', R2_c);
fprintf('误差指标: MSE = %.6f, RMSE = %.6f, MAE = %.6f\n\n', MSE_c, RMSE_c, MAE_c);

% ========== 计算并输出调整后 R² ==========
n = length(c_values);     % 样本数量
p = size(X_c, 2);         % 自变量数量（不含固定项 sqrt(2)*(rO+rB)）

R2_adj_c = 1 - (1 - R2_c) * (n - 1) / (n - p - 1);

fprintf('调整后 R² = %.6f\n', R2_adj_c);

% ========== 计算 AIC / BIC ==========
% AIC = n * log(RSS/n) + 2 * p
% BIC = n * log(RSS/n) + p * log(n)
RSS = sum(residuals_c .^ 2);

AIC_c = n * log(RSS / n) + 2 * p;
BIC_c = n * log(RSS / n) + p * log(n);

fprintf('AIC = %.6f\n', AIC_c);
fprintf('BIC = %.6f\n', BIC_c);
fprintf('n = %.6f\n', n)
fprintf('p = %.6f\n', p)

% ========== 计算相关性热图 ==========
% 创建特征矩阵
features_matrix = [deltaA, deltaA.^2, deltaB, deltaB.^2, ZA, ZA.^2, ZB, ZB.^2, ENA, ENA.^2, ENB, ENB.^2, Occ, Occ.^2, ...
    deltaA .* deltaB, deltaA .* ZA, deltaA .* ZB, deltaA .* ENA, deltaA .* ENB, deltaA .* Occ, ...
    deltaB .* ZA, deltaB .* ZB, deltaB .* ENA, deltaB .* ENB, deltaB .* Occ, ...
    ZA .* ZB, ZA .* ENA, ZA .* ENB, ZA .* Occ, ...
    ZB .* ENA, ZB .* ENB, ZB .* Occ, ENA .* ENB, ENA .* Occ, ENB .* Occ];

% 计算特征之间的相关性矩阵
correlation_matrix = corr(features_matrix);

% 绘制相关性热图
figure;
h = heatmap(correlation_matrix, 'XDisplayLabels', {'deltaA', 'deltaA^2', 'deltaB', 'deltaB^2', 'ZA', 'ZA^2', 'ZB', 'ZB^2', 'ENA', 'ENA^2', 'ENB', 'ENB^2', 'Occ', 'Occ^2', 'deltaA*deltaB', 'deltaA*ZA', 'deltaA*ZB', 'deltaA*ENA', 'deltaA*ENB', 'deltaA*Occ', 'deltaB*ZA', 'deltaB*ZB', 'deltaB*ENA', 'deltaB*ENB', 'deltaB*Occ', 'ZA*ZB', 'ZA*ENA', 'ZA*ENB', 'ZA*Occ', 'ZB*ENA', 'ZB*ENB', 'ZB*Occ', 'ENA*ENB', 'ENA*Occ', 'ENB*Occ'}, 'YDisplayLabels', {'deltaA', 'deltaA^2', 'deltaB', 'deltaB^2', 'ZA', 'ZA^2', 'ZB', 'ZB^2', 'ENA', 'ENA^2', 'ENB', 'ENB^2', 'Occ', 'Occ^2', 'deltaA*deltaB', 'deltaA*ZA', 'deltaA*ZB', 'deltaA*ENA', 'deltaA*ENB', 'deltaA*Occ', 'deltaB*ZA', 'deltaB*ZB', 'deltaB*ENA', 'deltaB*ENB', 'deltaB*Occ', 'ZA*ZB', 'ZA*ENA', 'ZA*ENB', 'ZA*Occ', 'ZB*ENA', 'ZB*ENB', 'ZB*Occ', 'ENA*ENB', 'ENA*Occ', 'ENB*Occ'});
title('Correlation Heatmap for Features');
xlabel('Features');
ylabel('Features');
colorbar;

% 设置颜色映射，蓝色对应-1，红色对应+1
colormap(h, 'coolwarm');  % 'coolwarm' has red at high (+1) and blue at low (-1)
caxis([-1 1]);  % Set the color scale to match correlation range from -1 to 1