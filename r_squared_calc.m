% Step 1: Load the data from the CSV files
predictions = readmatrix('unscaled_predictions.csv');
targets = readmatrix('unscaled_targets.csv');

% Step 2: Compute R^2 (Coefficient of Determination)
% R^2 = 1 - (sum((y_true - y_pred).^2) / sum((y_true - mean(y_true)).^2))
% Where y_true is the targets, and y_pred is the predictions

% Ensure that the predictions and targets are column vectors
if isrow(predictions)
    predictions = predictions';
end

if isrow(targets)
    targets = targets';
end

% Ensure both matrices have the same number of rows and columns
if size(predictions) ~= size(targets)
    error('Predictions and targets matrices must have the same size!');
end

% Compute residual sum of squares
ss_res = sum((targets - predictions).^2);

% Compute total sum of squares
ss_tot = sum((targets - mean(targets)).^2);

% Compute R^2 score
r2 = 1 - (ss_res / ss_tot);

% Display the R^2 score
disp(['R^2 Score: ', num2str(r2)]);

%====================================================================
%COMPUTE THE R^2 for each column and visualize it
for col = 1:size(predictions, 2)
    pred_col = predictions(:, col);  % Extract the prediction values for the current column
    target_col = targets(:, col);    % Extract the target values for the current column
    
    % Compute residual sum of squares (SS_res)
    ss_res = sum((target_col - pred_col).^2);
    
    % Compute total sum of squares (SS_tot)
    ss_tot = sum((target_col - mean(target_col)).^2);
    
    % Compute R^2 for the current column pair
    r2_scores(col) = 1 - (ss_res / ss_tot);
end

% Step 5: Display R^2 scores for each column
disp('R^2 Scores for each column pair:');
disp(r2_scores);

% Step 6: Visualize the R^2 scores using a bar chart
figure; 
bar(r2_scores, 'FaceColor', [0.6, 0.8, 1], 'EdgeColor', 'black');
xlabel('Column Pair Index', 'FontSize', 12);
ylabel('R^2 Score', 'FontSize', 12);
title('R^2 Scores for Each Column Pair (Prediction vs Target)', 'FontSize', 14);
grid on;

