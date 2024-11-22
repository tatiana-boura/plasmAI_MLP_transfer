% Step 1: Load the data from the CSV files
predictions = readmatrix('unscaled_predictions.csv');
targets = readmatrix('test_data_no_head_outer_corner.csv');

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
%ss_res = sum((targets - predictions).^2);

% Compute total sum of squares
%ss_tot = sum((targets - mean(targets)).^2);

% Compute R^2 score
%r2 = 1 - (ss_res / ss_tot);

% Display the R^2 score
%disp(['R^2 Score: ', num2str(r2)]);

%====================================================================
%COMPUTE THE R^2 for each column and visualize it
for col = 1:size(predictions, 2)
    pred_col = predictions(:, col);  % Extract the prediction values for the current column
    target_col = targets(:, (col+2));    % Extract the target values for the current column
    
    % Compute residual sum of squares (SS_res)
    ss_res = sum((target_col - pred_col).^2);
    
    % Compute total sum of squares (SS_tot)
    ss_tot = sum((target_col - mean(target_col)).^2);
    
    % Compute R^2 for the current column pair
    r2_scores(col) = 1 - (ss_res / ss_tot);
    
    % Compute MSE for the current column pair
    mse_scores(col) = mean((target_col - pred_col).^2)./(max(target_col)-min(target_col));

    % Plot the target vs predicted values for each column
    subplot(5, 5, col);  % Create a subplot (5 rows, 5 columns)
    plot(target_col, 'ob', 'LineWidth', 1.5); % Plot the target values (blue)
    hold on;
    plot(pred_col, 'or', 'LineWidth', 1.5);   % Plot the predicted values (red)
    title(['Column ', num2str(col)], 'FontSize', 10);
    xlabel('Index', 'FontSize', 8);
    ylabel('Value', 'FontSize', 8);
    legend('Target', 'Prediction', 'Location', 'Best');
    grid on;
end

% Display R^2 scores for each column
% disp('R^2 Scores for each column pair:');
% disp(r2_scores);

% Visualize the R^2 scores using a bar chart
figure(2); 
bar(r2_scores, 'FaceColor', [0.6, 0.8, 1], 'EdgeColor', 'black');
xlabel('Column Pair Index', 'FontSize', 12);
ylabel('R^2 Score', 'FontSize', 12);
title('R^2 Scores for Each Column Pair (Prediction vs Target)', 'FontSize', 14);
grid on;

% Visualize the MSE scores using a bar chart
% figure(2)
% scaled_mse = ((mse_scores)-min(mse_scores))./(max(mse_scores)-min(mse_scores));
% subplot(1, 2, 2);
% bar(scaled_mse, 'FaceColor', [1, 0.6, 0.6], 'EdgeColor', 'black');
% xlabel('Column Pair Index', 'FontSize', 12);
% ylabel('MSE', 'FontSize', 12);
% title('MSE for Each Column Pair (Prediction vs Target)', 'FontSize', 14);
% grid on;

