clc
clear
close all

% Add path for required toolbox for Time-Frequency Toolbox (TFTB)
addpath("/tftb-0.2/mfiles");

% Add path for EEGLAB toolbox
addpath '/eeglab2024.0'; 

% Define parameters and load EEG data
filename= ['/BCICIV_2b_gdf 2/B0101T.gdf']; % File path
fl = 9; fh = 14; % Frequency range for bandpass filter
order = 4; % Order of the Butterworth filter
ti = 3.5; % Time interval for epoch duration in seconds
latency = 0.6; % Latency for epoch start in seconds
cspn = 1; % Number of Common Spatial Patterns
% Classification Accuracy: 95.83%

[s,h] = sload(filename); % Load EEG data using biosig library

% Load empty EEG structure for further use
load("/emptyEEG.mat")

% Select the first three channels of the signal
s = s(:,1:3);

% Normalize the signal (z-score normalization)
s = (s - mean(s)) ./ std(s);

% Sampling rate from the header
fs = h.SampleRate;

% Duration of an epoch in samples
epochDu = ti*fs;

% Remove missing values from the signal
indx     = find( isnan(s)); % Find NaN indices
s(indx)  = 0; % Replace NaNs with 0

% Extract event types and positions from the header
group    = h.EVENT.TYP;
pos      = h.EVENT.POS;

% Bandpass filter to extract mu and beta rhythms from the EEG signal
wn    = [fl fh] / (fs/2); % Normalized cutoff frequency vector
type  = 'bandpass'; % Filter type
[b,a] = butter(order, wn, type); % Design Butterworth filter
s     = filtfilt(b, a, s); % Apply filter to EEG data

% Initialize counters for classes
c1 = 0;
c2 = 0;

% Calculate latency in samples
lat = floor(latency*fs);

% Data segmentation based on event type
for i = 1:length(group)
    ind   = pos(i)+lat: pos(i) + epochDu-1; % Index range for each epoch
    trial = s(ind, :); % Extract trial data
    if     group(i) == 769 % Class 1
        c1 = c1+1;
        data1(:, :, c1) = trial; % Store trial data in data1

    elseif group(i) == 770 % Class 2
        c2 = c2+1;
        data2(:, :, c2) = trial; % Store trial data in data2

    end
end

% Compute Entropy Features for Each Epoch and Channel
window_size_long = 1 * fs; % Long window size for entropy calculation
window_size = 0.5 * fs; % Short window size for entropy calculation
step_size = .05 * fs; % Step size for window sliding
alpha = 2; % Parameter for entropy calculation

% Calculate the number of windows for each epoch
num_windows = floor((length(data1) - window_size) / step_size) + 1;

num_channels = size(data1, 2); % Number of EEG channels
num_trials1 = size(data1, 3); % Number of trials for class 1
num_trials2 = size(data2, 3); % Number of trials for class 2

% Initialize feature and label variables
labels = [];
feat_all1 =[];
feat_all2 =[];
feat_all3 =[];
featT_all1 = [];
featT_all2 = [];
features=[];

% Assign labels for each class
labels = [ones(1,size(data1,3)) 2*ones(1,size(data2,3))];

% Set random number generator seed for reproducibility
rng("default")

% Classification with K-Fold Cross-Validation
k = 5; % Number of folds
cv = cvpartition(labels, 'KFold', k); % Create cross-validation partition
accuracy = []; % Initialize accuracy array
data = []; % Initialize data array
data = cat(3, data1, data2); % Concatenate data from both classes

% Loop through each fold
for iter = 1:k
    tic % Start timing the iteration
    E1_all  = [];
    E2_all  = [];
    Et1_all = [];
    Et2_all = [];

    train_idx = training(cv, iter); % Get training indices for this fold
    test_idx = test(cv, iter); % Get test indices for this fold
  
    train_labels = labels(train_idx); % Training labels
    test_labels = labels(test_idx); % Test labels

    % Separate data into classes for training
    d1=0;
    d2=0;
    for i = 1:length(train_labels)
        if train_labels(i) == 1
            d1=d1+1;
            parfor channel = 1:size(data,2)         
                epoch1(:,channel,d1) = data(:, channel, i); % Store data for class 1
            end
        elseif train_labels(i) == 2
            d2=d2+1;
            parfor channel = 1:size(data,2)  
                epoch2(:,channel,d2) = data(:, channel, i); % Store data for class 2
            end
        end
     end
       
    num_channels = cspn*2; % Number of channels after CSP
    [w] = myCSP(epoch1, epoch2, cspn); % Compute CSP weights
    
    train_data_CSP = [];
    train_data =[];
    d1=0;
    d2=0;
    E1_all = [];
    E2_all = [];
    train_data = data(:,:,train_idx); % Extract training data
    for i = 1:length(train_labels)
        if train_labels(i) == 1
            d1=d1+1;
            train_data_CSP = train_data(:,:,i)';
            for channel = 1:num_channels          
                epoch1  = train_data_CSP(channel, :)';
        
                t = 1:length(epoch1); % Define the time vector
                [tfr1, rtfr1] = tfrrpwv(epoch1, t ,1024); % Compute time-frequency representation
                E1 = energydist(log10((tfr1)), window_size, step_size); % Compute entropy
                E1_all(:, d1,channel) = (E1'); % Store entropy features
            end
        elseif train_labels(i) == 2
            d2=d2+1;
            train_data_CSP = train_data(:,:,i)';
            for channel = 1:num_channels
                epoch2 = train_data_CSP(channel, :)';
        
                t = 1:length(epoch2); % Define the time vector
                [tfr2, rtfr2] = tfrrpwv(epoch2, t ,1024); % Compute time-frequency representation
                E2 = energydist(log10((tfr2)), window_size, step_size); % Compute entropy
                E2_all(:, d2,channel) = (E2'); % Store entropy features
            end
        end
    end

    % Prepare test data
    test_data =[];
    test_data_CSP =[];
    t1=0;
    t2=0;
    test_data = data(:,:,test_idx); % Extract test data
    for i = 1:length(test_labels)
        if test_labels(i) == 1
            t1=t1+1;
            test_data_CSP = test_data(:,:,i)';
            for channel = 1:num_channels          
                epocht1 = test_data_CSP(channel, :)';
        
                t = 1:length(epocht1); % Define the time vector
                [tfr1, rtfr1] = tfrrpwv(epocht1, t ,1024); % Compute time-frequency representation
                Et1 = energydist(log10((tfr1)), window_size, step_size); % Compute entropy
                Et1_all(:, t1,channel) = (Et1'); % Store entropy features
            end
        elseif test_labels(i) == 2
            t2=t2+1;
            test_data_CSP = test_data(:,:,i)';
            for channel = 1:num_channels
                epocht2 = test_data_CSP(channel, :)';
        
                t = 1:length(epocht2); % Define the time vector
                [tfr2, rtfr2] = tfrrpwv(epocht2, t ,1024); % Compute time-frequency representation
                Et2 = energydist(log10((tfr2)), window_size, step_size); % Compute entropy
                Et2_all(:, t2,channel) = (Et2'); % Store entropy features
            end
        end
    end

    toc % End timing the iteration

    % Initialize feature arrays
    feat_all1  = [];
    feat_all2  = [];
    feat_all3  = [];
    featT_all1 = [];
    featT_all2 = [];

    % Compute mean entropy features across channels
    mE1_all  = mean(E1_all,3);
    mE2_all  = mean(E2_all,3);
    mEt1_all = mean(Et1_all,3);
    mEt2_all = mean(Et2_all,3);

    % Concatenate features across channels
    for i=1:num_channels
        feat_all1  = [feat_all1;  E1_all( :,:,i)];
        feat_all2  = [feat_all2;  E2_all( :,:,i)];
        featT_all1 = [featT_all1; Et1_all(:,:,i)];
        featT_all2 = [featT_all2; Et2_all(:,:,i)];
    end
    
    % Combine features for training and testing
    features_train = [feat_all1, feat_all2];
    features_test  = [featT_all1, featT_all2];
    
    % Feature Selection using fscmrmr
    selected_features_idx = fscmrmr(features_train', train_labels); % Rank features using MRMR
    N = 50; % Number of top features to select
    selected_features_train = features_train(selected_features_idx(1:N), :); % Select top features for training
    selected_features_test  = features_test(selected_features_idx(1:N), :); % Select top features for testing
    rng("default");

    % Normalize selected features
    [selected_features_train, mu, sigma] = zscore(selected_features_train, 0, 2); % Z-score normalization
    selected_features_test  = (selected_features_test - mu) ./ sigma; % Apply normalization to test data

    [mdl] = fitcnet(selected_features_train', train_labels); % Train classification model using neural network
    % [mdl] = fitcsvm(selected_features_train',train_labels); % Alternative: SVM model
    % [mdl] = fitcauto(selected_features_train',train_labels); % Alternative: Auto classifier
    predictions = predict(mdl, selected_features_test'); % Predict on test data
    
    accuracy(iter) = sum(predictions == test_labels') / length(test_labels); % Calculate accuracy for this fold
    mean_accuracy = mean(accuracy) * 100; % Mean accuracy across folds

    fprintf('Classification Accuracy: %.2f%%\n', mean_accuracy); % Display accuracy
end

mean_accuracy = mean(accuracy) * 100; % Final mean accuracy
std_accuracy = std(accuracy) * 100; % Standard deviation of accuracy
fprintf('Mean Accuracy: %.2f%%\n', mean_accuracy); % Display mean accuracy
fprintf('Standard Deviation of Accuracy: %.2f%%\n', std_accuracy); % Display standard deviation

% ERP calculation and Time-Frequency Representation (TFR)
ERP_data1 = mean(data1(:,1,:), 3); % Average over trials for class 1
ERP_data2 = mean(data2(:,1,:), 3); % Average over trials for class 2
t = 1:length(ERP_data1);
[etfr1, ~, ~] = tfrrpwv(ERP_data1, t, 1024); % Compute TFR for class 1 ERP
ERPE13 = energydist((abs(etfr1)), window_size, step_size); % Compute entropy of TFR

[etfr2, ~, ~] = tfrrpwv(ERP_data2, t, 1024); % Compute TFR for class 2 ERP
ERPE23 = energydist((abs(etfr2)), window_size, step_size); % Compute entropy of TFR

% Plotting
t = linspace(0, 4, size(etfr1, 2)); % Define time vector for plotting
hz = linspace(0, fs/2, size(etfr1, 1)); % Define frequency vector for plotting
figure;
set(gcf, 'Color', 'w'); % Set figure background color to white

subplot(3, 2, 1)
imagesc(t, hz, log10(abs(etfr1))); % Plot TFR for class 1 ERP
axis xy;
title('TFR of ERP - Left Hand (C3)', 'Color', 'k'); % Set plot title
xlabel('Time (s)', 'Color', 'k'); % Set x-axis label
ylabel('Frequency (Hz)', 'Color', 'k'); % Set y-axis label
colormap('turbo'); % Set color map
hcb = colorbar; % Add color bar
caxis([0 .25]); % Set color axis limits
ylabel(hcb, 'Power (dB)', 'Color', 'k'); % Set color bar label
ylim([0 50]); % Set y-axis limits
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k'); % Set plot colors
set(hcb, 'Color', 'k'); % Set color bar ticks color to black

subplot(3, 2, 2)
imagesc(t, hz, log10(abs(etfr2))); % Plot TFR for class 2 ERP
axis xy;
title('TFR of ERP - Right Hand (C4)', 'Color', 'k');
xlabel('Time (s)', 'Color', 'k');
ylabel('Frequency (Hz)', 'Color', 'k');
colormap('turbo');
hcb = colorbar;
caxis([0 .25]);
ylabel(hcb, 'Power (dB)', 'Color', 'k');
ylim([0 50]);
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k');
set(hcb, 'Color', 'k'); % Set color bar ticks color to black

t = 1:length(ERP_data1); % Define time vector
ERP_data1 = mean(data1(:,3,:), 3); % Average over trials for class 1 (channel 3)
ERP_data2 = mean(data2(:,3,:), 3); % Average over trials for class 2 (channel 3)

[etfr1, ~, ~] = tfrrpwv(ERP_data1, t, 1024); % Compute TFR for class 1 ERP
ERPE1 = energydist((abs(etfr1)), window_size, step_size); % Compute entropy of TFR

[etfr2, ~, ~] = tfrrpwv(ERP_data2, t, 1024); % Compute TFR for class 2 ERP
ERPE2 = energydist((abs(etfr2)), window_size, step_size); % Compute entropy of TFR

t = linspace(0, 4, size(etfr1, 2)); % Define time vector for plotting
subplot(3, 2, 4)
imagesc(t, hz, log10(abs(etfr1))); % Plot TFR for class 1 ERP (channel 3)
axis xy;
title('TFR of ERP - Left Hand (C4)', 'Color', 'k');
xlabel('Time (s)', 'Color', 'k');
ylabel('Frequency (Hz)', 'Color', 'k');
colormap('turbo');
hcb = colorbar;
caxis([0 .25]);
ylabel(hcb, 'Power (dB)', 'Color', 'k');
ylim([0 50]);
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k');
set(hcb, 'Color', 'k'); % Set color bar ticks color to black

subplot(3, 2, 3)
imagesc(t, hz, log10(abs(etfr2))); % Plot TFR for class 2 ERP (channel 3)
axis xy;
title('TFR of ERP - Right Hand (C3)', 'Color', 'k');
xlabel('Time (s)', 'Color', 'k');
ylabel('Frequency (Hz)', 'Color', 'k');
colormap('turbo');
hcb = colorbar;
caxis([0 .25]);
ylabel(hcb, 'Power (dB)', 'Color', 'k');
ylim([0 50]);
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k');
set(hcb, 'Color', 'k'); % Set color bar ticks color to black

t = linspace(0,4,length(ERPE2)); % Define time vector for entropy plot
subplot(3, 2, 5)
plot(t, ERPE13, 'b') % Plot entropy of TFR for class 1 (C3)
hold on
plot(t, ERPE23, 'r') % Plot entropy of TFR for class 2 (C3)
title('Entropy of TFRs (C3)', 'Color', 'k');
xlabel('Time (s)', 'Color', 'k');
ylabel('Entropy', 'Color', 'k');
legend({'Left Hand (Class 769)', 'Right Hand (Class 770)'}, 'TextColor', 'k', 'Color', 'w');
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k');

subplot(3, 2, 6)
plot(t, ERPE1, 'b') % Plot entropy of TFR for class 1 (C4)
hold on
plot(t, ERPE2, 'r') % Plot entropy of TFR for class 2 (C4)
title('Entropy of TFRs (C4)', 'Color', 'k');
xlabel('Time (s)', 'Color', 'k');
ylabel('Entropy', 'Color', 'k');
legend({'Left Hand (Class 769)', 'Right Hand (Class 770)'}, 'TextColor', 'k', 'Color', 'w');
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k');
