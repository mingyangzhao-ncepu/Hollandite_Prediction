
filePath = 'E:\Hollandite\Hollandite_data_Symmetry.xlsx';
data = readmatrix(filePath, 'Range', 'B2:N196');


rO_rB   = data(:,3);     
deltaA  = data(:,4);     
deltaB  = data(:,5);     
ZA      = data(:,6);     
ZB      = data(:,7);     
ENA     = data(:,8);     
ENB     = data(:,9);     
Occ     = data(:,10);    
rA      = data(:,12);    


[~, txtData] = xlsread(filePath, 'B2:N196');
Symmetry = txtData(:,1);   


best_loss = Inf;
best_y = [];

for trial = 1:20  
    initial_y = randn(35,1) * 0.1;  
    options = optimset('Display', 'off', 'MaxIter', 1000, 'TolX', 1e-6);
    
    [y_trial, loss_trial] = fminsearch(@(y) compute_loss( ...
        y, rO_rB, deltaA, deltaB, ZA, ZB, ENA, ENB, Occ, rA, Symmetry), initial_y, options);

    if loss_trial < best_loss
        best_loss = loss_trial;
        best_y = y_trial;
    end
end



for i = 1:length(best_y)
    fprintf('y%d = %.4f\n', i, best_y(i));
end


[final_acc, total, correct] = compute_loss(best_y, rO_rB, deltaA, deltaB, ZA, ZB, ENA, ENB, Occ, rA, Symmetry, true);
fprintf('%.2f%% (%d/%d)\n', final_acc*100, correct, total);



function [loss, total, correct] = compute_loss(y, rO_rB, deltaA, deltaB, ZA, ZB, ENA, ENB, Occ, rA, Symmetry, verbose)
    if nargin < 12
        verbose = false;
    end
    
    A = deltaA;  B = deltaB;  C = ZA;  D = ZB;  E = ENA;  F = ENB;  G = Occ;

    
    
    
    rc_pred = sqrt(2) .* rO_rB ...
            + y(1).*A  + y(2).*B  + y(3).*C  + y(4).*D  + y(5).*E  + y(6).*F  + y(7).*G ...
            + y(8).*(A.^2) + y(9).*(B.^2) + y(10).*(C.^2) + y(11).*(D.^2) + y(12).*(E.^2) + y(13).*(F.^2) + y(14).*(G.^2) ...
            + y(15).*(A.*B) + y(16).*(A.*C) + y(17).*(A.*D) + y(18).*(A.*E) + y(19).*(A.*F) + y(20).*(A.*G) ...
            + y(21).*(B.*C) + y(22).*(B.*D) + y(23).*(B.*E) + y(24).*(B.*F) + y(25).*(B.*G) ...
            + y(26).*(C.*D) + y(27).*(C.*E) + y(28).*(C.*F) + y(29).*(C.*G) ...
            + y(30).*(D.*E) + y(31).*(D.*F) + y(32).*(D.*G) ...
            + y(33).*(E.*F) + y(34).*(E.*G) ...
            + y(35).*(F.*G) ...
            - 1.4;

    delta = rA - rc_pred;
    predict_label = strings(length(delta),1);
    predict_label(delta < 0)  = "I 2/m";
    predict_label(delta >= 0) = "I 4/m";

    
    pred = predict_label(:);
    trueLabel = string(Symmetry(:));

    correct = sum(pred == trueLabel);
    total   = numel(trueLabel);
    loss    = total - correct;           
    loss    = loss + 1e-4 * sum(y.^2);   

    if verbose
        loss = correct / total;          
    end
end