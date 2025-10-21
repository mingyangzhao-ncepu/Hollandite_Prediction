
filePath = 'E:\Hollandite\Hollandite_data_Symmetry.xlsx';
data = readmatrix(filePath, 'Range', 'B2:N196');


rA      = data(:,12);    
rO_rB   = data(:,3);     
deltaA  = data(:,4);     
deltaB  = data(:,5);     
ZA      = data(:,6);     
ZB      = data(:,7);     
ENA     = data(:,8);     
ENB     = data(:,9);     
Occ     = data(:,10);   

[~, txtData] = xlsread(filePath, 'N2:N196');
Symmetry = txtData(:,1);   


best_loss = Inf;
best_y = [];

for trial = 1:20   
    initial_y = randn(7,1) * 0.1;  
    options = optimset('Display', 'off', 'MaxIter', 1000, 'TolX', 1e-7);
    [y_trial, loss_trial] = fminsearch(@(y) compute_loss(y, rO_rB, rA, ZB, deltaA, deltaB, ZA, ENB, ENA, Occ, Symmetry), initial_y, options);
    
    if loss_trial < best_loss
        best_loss = loss_trial;
        best_y = y_trial;
    end
end



for i = 1:length(best_y)
    fprintf('y%d = %.6f\n', i, best_y(i));
end


[final_acc, total, correct] = compute_loss(best_y, rO_rB, rA, ZB, deltaA, deltaB, ZA, ENB, ENA, Occ, Symmetry, true);
fprintf('%.2f%% (%d/%d)\n', final_acc*100, correct, total);


function [loss, total, correct] = compute_loss(y, rO_rB, rA, ZB, deltaA, deltaB, ZA, ENB, ENA, Occ, Symmetry, verbose)
    if nargin < 12
        verbose = false;
    end
    rc_pred = sqrt(2) .* rO_rB + ...
              y(1) .* ZB + y(2) .* deltaA + y(3) .* deltaB + y(4) .* ZA + ...
              y(5) .* ENB + y(6) .* ENA + y(7) .* Occ - 1.4;
    delta = rA - rc_pred;
    predict_label = strings(length(delta),1);
    predict_label(delta < 0) = "I 2/m";
    predict_label(delta >= 0) = "I 4/m";
    
    correct = sum(predict_label == string(Symmetry));
    total = length(Symmetry);
    loss = total - correct;  
    
    
    loss = loss + 1e-4 * sum(y.^2);  
    
    if verbose
        acc = correct / total;
        loss = acc;
    end
end