%% Practical Assignment
clearvars
close all
clc

% loading the dataset
data = readtable("NHANES_dataset.csv");

data = rmmissing(data);
head(data)

%% Preprocessing 

data(:, 1) = []; % ID column removed

target_name = 'Age';
feature_names = data.Properties.VariableNames;
feature_names(strcmp(feature_names, target_name)) = [];

% Standardize features
data{:, feature_names} = zscore(data{:, feature_names});

% Prepare for GPstuff, as GPstuff requires numeric matrices, not tables

% Extract features (X) and target (y)
X = data{:, feature_names};
y = data{:, target_name};

% Center the target variable
y_mean = mean(y);
y = (y - y_mean);

% Splitting for testing and training, 80/20
rng(42); 

n = size(X, 1);
cv = cvpartition(n, 'HoldOut', 0.2);
idxTrain = training(cv);
idxTest = test(cv);

% Creating subsets
X_train = X(idxTrain, :);
y_train = y(idxTrain, :);
X_test  = X(idxTest, :);
y_test  = y(idxTest, :);

fprintf('Data split: %d training samples, %d testing samples.\n', ...
    sum(idxTrain), sum(idxTest));
fprintf('Data preprocessing complete.\n');

%% Linear model

% Just simple linear model as something to compare
mdl = fitlm(X_train, y_train);
y_pred_linear = predict(mdl, X_test);

% Calculating the performance metrics
rmse_linear = sqrt(mean((y_test - y_pred_linear).^2))
mae_linear = mean(abs(y_test - y_pred_linear))
r2_linear = mdl.Rsquared.Ordinary

%% Matern 3/2
lik = lik_gaussian('sigma2', var(y_train)*0.1);

% Model setup
cf = gpcf_matern32(); 
cf.l = 1;
cf.magnSigma2 = var(y_train);
gp = gp_set('lik', lik, 'cf', cf);

% Training
gp = gp_optim(gp, X_train, y_train);

% Predict on test set
[mu_matern_32, s2_m32, lpd] = gp_pred(gp, X_train, y_train, X_test, 'yt', y_test);
[mu_train, s2_train] = gp_pred(gp, X_train, y_train, X_train);

% Performance metrics
nlpd = -mean(lpd);
TSS = sum((y_test - mean(y_test)).^2);
RSS = sum((y_test - mu_matern_32).^2);
r2_gp = 1 - (RSS / TSS);

RMSE = sqrt(mean((y_test - mu_matern_32).^2));
RMSE_train = sqrt(mean((y_train - mu_train).^2));

fprintf('Matern 3/2:\n')
fprintf('Test Set NLPD: %.4f\n', nlpd);
fprintf('Train Set RMSE: %.2f years\n', RMSE_train);
fprintf('Test Set RMSE: %.2f years\n', RMSE);
fprintf('R2: %.3f\n', r2_gp);


%% Matern 5/2
% Model setup
cf = gpcf_matern52(); 
cf.l = 1;
cf.magnSigma2 = var(y_train);
gp = gp_set('lik', lik, 'cf', cf);

% Training
gp = gp_optim(gp, X_train, y_train);

% Predict on test set
[mu_matern52, s2, lpd] = gp_pred(gp, X_train, y_train, X_test, 'yt',y_test);
[mu_train, s2_train] = gp_pred(gp, X_train, y_train, X_train);

% Performance metrics
nlpd = -mean(lpd);
TSS = sum((y_test - mean(y_test)).^2);
RSS = sum((y_test - mu_matern52).^2);
r2_gp = 1 - (RSS / TSS);

RMSE = sqrt(mean((y_test - mu_matern52).^2));
RMSE_train = sqrt(mean((y_train - mu_train).^2));

fprintf('Matern 5/2:\n')
fprintf('Test Set NLPD: %.4f\n', nlpd);
fprintf('Train Set RMSE: %.2f years\n', RMSE_train);
fprintf('Test Set RMSE: %.2f years\n', RMSE);
fprintf('R2: %.3f\n', r2_gp);

%% Squared exponential
% Model setup
cf = gpcf_sexp();
cf.l = 1;
cf.magnSigma2 = var(y_train);
gp = gp_set('lik', lik, 'cf', cf);

% Training
gp = gp_optim(gp, X_train, y_train);

% Predict on test set
[mu_sexp, s2, lpd] = gp_pred(gp, X_train, y_train, X_test, 'yt', y_test);
[mu_train, s2_train] = gp_pred(gp, X_train, y_train, X_train);

% Performance metrics
nlpd = -mean(lpd);
TSS = sum((y_test - mean(y_test)).^2);
RSS = sum((y_test - mu_sexp).^2);
r2_gp = 1 - (RSS / TSS);

RMSE = sqrt(mean((y_test - mu_sexp).^2));
RMSE_train = sqrt(mean((y_train - mu_train).^2));

fprintf('Squared exponential:\n')
fprintf('Test Set NLPD: %.4f\n', nlpd);
fprintf('Train Set RMSE: %.2f years\n', RMSE_train);
fprintf('Test Set RMSE: %.2f years\n', RMSE);
fprintf('R2: %.3f\n', r2_gp);

%% ARD - Automatic Relevance Determination
% !NOT INCLUDED IN THE REPORT AS IT DOES NOT WORK

% Counting features
num_features = size(X_train, 2);

% Initialize the kernel -> separate length-scale for each column
cf_ard = gpcf_matern32();
cf_ard.l = ones(1, num_features);
cf_ard.magnSigma2 = var(y_train);

% Set up and optimize
gp_ard = gp_set('lik', lik_gaussian(), 'cf', cf_ard);
fprintf('Optimizing ARD... ');
gp_ard = gp_optim(gp_ard, X_train, y_train);

% Extract relevance
% Relevance defined as the inverse of the length-scale
relevance = 1 ./ gp_ard.cf{1}.l;

%% Visualization
close all

true_age = y_test + y_mean;
preds = {y_pred_linear + y_mean, mu_matern_32 + y_mean};
names = {'Linear Model', 'Matern 3/2'};
colors = [1,0,0; 0,0,1];

figure('Color', 'w', 'Units', 'inches', 'Position', [2, 2, 6, 5]);
hold on
grid on
box on

min_val = min(true_age) - 5;
max_val = max(true_age) + 5;
line([min_val, max_val], [min_val, max_val], 'Color', [0.5 0.5 0.5], 'LineStyle', '--', 'LineWidth', 1.5, 'HandleVisibility', 'off');

for i = 1:length(preds)
    % Calculate R-squared for the legend
    res = true_age - preds{i};
    r2 = 1 - sum(res.^2)/sum((true_age - mean(true_age)).^2);
    
    scatter(true_age, preds{i}, 30, colors(i,:), 'filled', 'MarkerFaceAlpha', 0.4, ...
        'DisplayName', sprintf('%s (R^2: %.2f)', names{i}, r2));
end

xlabel('Actual Age (Years)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Predicted Age (Years)', 'FontSize', 12, 'FontWeight', 'bold');
title('NHANES Age Prediction: GP Kernel Comparison', 'FontSize', 14);
set(gca, 'TickDir', 'out', 'FontName', 'Helvetica', 'FontSize', 10);
legend('Location', 'northwest', 'FontSize', 9);

axis equal;
xlim([min_val, max_val]);
ylim([min_val, max_val]);


% Calculate residuals (Actual - Predicted)
res_linear = true_age - (y_pred_linear + y_mean);
res_m32    = true_age - (mu_matern_32 + y_mean);
res_m52    = true_age - (mu_matern52 + y_mean);

residuals = {res_linear, res_m32, res_m52};
model_names = {'Linear Regression', 'GP (Matern 3/2)', 'GP (Matern 5/2)'};
node_colors = [0.2 0.2 0.2; 1 0 0; 0 0 1];

figure('Color', 'w', 'Units', 'inches', 'Position', [1, 1, 10, 4]);

for i = 1:3
    subplot(1, 3, i);
    hold on; grid on; box on;
    
    % Plot residuals vs actual age
    scatter(true_age, residuals{i}, 20, node_colors(i,:), 'filled', 'MarkerFaceAlpha', 0.3);
    
    % Add a zero-error reference line
    line([min(true_age) max(true_age)], [0 0], 'Color', 'k', 'LineWidth', 1.5);
    
    % Statistics for titles
    rmse = sqrt(mean(residuals{i}.^2));
    title(sprintf('%s\nRMSE: %.2f years', model_names{i}, rmse));
    xlabel('Actual Age');
    if i == 1, ylabel('Error (Years)'); end
    ylim([-25 25]); % Keep scale consistent for fair comparison
end

% Confidence in predictions
pred_m32 = mu_matern_32 + y_mean;
uncertainty = sqrt(s2_m32);

figure('Color', 'w', 'Units', 'inches', 'Position', [2, 2, 7, 6]);
hold on; 
grid on; 
box on;
all_vals = [true_age; pred_m32];
lims = [min(all_vals)-5, max(all_vals)+5];
line(lims, lims, 'Color', [0.5 0.5 0.5], 'LineStyle', '--', 'LineWidth', 2, 'HandleVisibility', 'off');
% scatter(X, Y, Size, Color, 'filled')
scatter(true_age, pred_m32, 40, uncertainty, 'filled', 'MarkerFaceAlpha', 0.7);
cb = colorbar;
ylabel(cb, 'Predictive Uncertainty', 'FontSize', 11)
colormap(jet)

xlabel('Actual Age', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Predicted Age', 'FontSize', 12, 'FontWeight', 'bold');
title('GP Age Prediction: Uncertainty Mapping', 'FontSize', 14);

set(gca, 'TickDir', 'out', 'FontSize', 10);
axis equal;
xlim(lims); ylim(lims);

% ARD Plot
% Bar chart labels
% feature_names = {'Gender', 'Physical activity', 'BMI', 'Glucose', 'DiabeticStatus', 'OralHealth', 'Insulin', 'BPSY1', 'BPDI1', 'LDL', 'GripHand1', 'GripHand2'}; % Adjust to your actual list
% 
% figure('Color', 'w');
% bar(relevance, 'FaceColor', [0.2 0.5 0.8]);
% set(gca, 'XTick', 1:num_features, 'XTickLabel', feature_names, 'XTickLabelRotation', 45);
% ylabel('Relevance (1/Length-scale)');
% title('Feature Importance via ARD');
% grid on;
% 
% [~, best_idx] = max(relevance);
% fprintf('The most important feature for age prediction is: %s\n', feature_names{best_idx});