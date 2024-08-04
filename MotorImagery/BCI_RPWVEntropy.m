clc
clear
close all

% Add path for the Time-Frequency Toolbox (TFTB)
addpath("/tftb-0.2/mfiles");

% Add path for the EEGLAB toolbox
addpath '/eeglab2024.0';

% Uncomment the appropriate filename line to select the EEG data file and parameters
% Each line contains the file path and parameters specific to the subject/session
% filename= ['/BCICIV_2b_gdf 2/B0101T.gdf']; fl = 9; fh = 14; order = 4; ti = 3.5; latency = 0.6; cspn = 1; % Classification Accuracy: 95.83%
% filename= ['/BCICIV_2b_gdf 2/B0201T.gdf']; fl = 4; fh = 12; order = 4; ti = 2; latency = .2; cspn = 1; % Classification Accuracy: 68.33%
% filename= ['/BCICIV_2b_gdf 2/B0301T.gdf']; fl = 9; fh = 20; order = 4; ti = 3.5; latency = 0.6; cspn = 1; % Classification Accuracy: 68.33%
% filename= ['/BCICIV_2b_gdf 2/B0401T.gdf']; fl = 9; fh = 24; order = 4; ti = 2; latency = 0.6; cspn = 1; % Classification Accuracy: 93.33%
% filename= ['/BCICIV_2b_gdf 2/B0501T.gdf']; fl = 11; fh = 33; order = 4; ti = 2.5; latency = 0.6; cspn = 1; % Classification Accuracy: 75.83%
% filename= ['/BCICIV_2b_gdf 2/B0601T.gdf']; fl = 9; fh = 15; order = 4; ti = 4; latency = 0.4; cspn = 1; % Classification Accuracy: 84.17%
% filename= ['/BCICIV_2b_gdf 2/B0701T.gdf']; fl = 12; fh = 16; order = 4; ti = 4; latency = 0.6; cspn = 1; % Classification Accuracy: 78.33%
% filename= ['/BCICIV_2b_gdf 2/B0801T.gdf']; fl = 6; fh = 12; order = 4; ti = 4; latency = 0.2; cspn = 1; % Classification Accuracy: 65.00%
filename= ['/BCICIV_2b_gdf 2/B0901T.gdf']; fl = 12; fh = 16; order = 4; ti = 4.1; latency = 0.5; cspn = 1; % Classification Accuracy: 80.00%

[s,h] = sload(filename); % Load the EEG data using the Biosig library

%% Load EEG Data
load("/emptyEEG.mat") % Load an empty EEG structure
s = s(:,1:3); % Select the first three channels

% Normalize the signal (z-score normalization)
s = (s - mean(s)) ./ std(s);

% Sampling rate and epoch duration in samples
fs = h.SampleRate;
epochDu = ti*fs;

% Remove missing values by setting NaNs to zero
indx     = find( isnan(s));
s(indx)  = 0;

% Extract event types and positions from the header
group    = h.EVENT.TYP;
pos      = h.EVENT.POS;

% Bandpass filtering to extract mu and beta rhythms from the EEG signal
wn    = [fl fh] / (fs/2); % Normalized cutoff frequency vector
type  = 'bandpass';
[b,a] = butter(order, wn, type); % Design a Butterworth filter
s     = filtfilt(b, a, s); % Apply zero-phase digital filtering

%%
c1 = 0; % Counter for class 1 trials
c2 = 0; % Counter for class 2 trials
lat = floor(latency*fs); % Calculate latency in samples

% Segment data into epochs based on event markers
for i = 1:length(group) % Iterate over all events
    ind   = pos(i)+lat: pos(i) + epochDu-1; % Determine indices for the current epoch
    trial = s(ind, :); % Extract trial data
    if     group(i) == 769 % Check if the event is for class 1
        c1 = c1+1;
        data1(:, :, c1) = trial; % Store trial data in data1

    elseif group(i) == 770 % Check if the event is for class 2
        c2 = c2+1;
        data2(:, :, c2) = trial; % Store trial data in data2

    end
end

% data1 = data1(:, :, 1:min(c1, 75)); % Uncomment to limit the number of trials
% data2 = data2(:, :, 1:min(c2, 75)); % Uncomment to limit the number of trials

%% Compute Entropy Features for Each Epoch and Channel
window_size_long = 1 * fs; % Long window size for entropy calculation
window_size = 0.5 * fs; % Short window size for entropy calculation
step_size = .05 * fs; % Step size for window sliding
alpha = 2; % Parameter for Renyi entropy calculation

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

%%
labels = [ones(1,size(data1,3)) 2*ones(1,size(data2,3))]; % Create label vector
rng("default") % Set random number generator for reproducibility

%% Classification with K-Fold Cross-Validation
k = 5; % Number of folds for cross-validation
cv = cvpartition(labels, 'KFold', k); % Create cross-validation partition
accuracy = []; % Initialize accuracy array
data = [];
data = cat(3, data1, data2); % Concatenate data from both classes

% Loop through each fold
for iter = 1:k
% tic
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
            train_data_CSP = w'*train_data(:,:,i)'; % Apply CSP to training data
            parfor channel = 1:num_channels          
                epoch1  = train_data_CSP(channel, :)'; % Extract CSP component
   
                t = 1:length(epoch1); % Define the time vector
                [tfr1, rtfr1, hat1] = tfrrpwv(epoch1, t ,1024); % Compute TFR using pseudo Wigner-Ville distribution
                E1 = compute_entropy(log10(abs(tfr1)), window_size, step_size,'shannon',alpha); % Compute entropy
                E1_all(:, d1,channel) = (E1'); % Store entropy features
            end
        elseif train_labels(i) == 2
            d2=d2+1;
            train_data_CSP = w'*train_data(:,:,i)'; % Apply CSP to training data
            parfor channel = 1:num_channels
                epoch2 = train_data_CSP(channel, :)'; % Extract CSP component
        
                t = 1:length(epoch2); % Define the time vector
                [tfr2, rtfr2, hat2] = tfrrpwv(epoch2, t ,1024); % Compute TFR using pseudo Wigner-Ville distribution
                E2 = compute_entropy(log10(abs(tfr2)), window_size, step_size,'shannon',alpha); % Compute entropy
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
    Et1_all = [];
    % Et2_all = zeros(num_windows, length(train_data), num_channels);
    Et2_all = [];
    for i = 1:length(test_labels)
% toc
% tic
        if test_labels(i) == 1
            t1=t1+1;
            test_data_CSP = w'*test_data(:,:,i)'; % Apply CSP to test data
            parfor channel = 1:num_channels          
                epocht1 = test_data_CSP(channel, :)'; % Extract CSP component
        
                t = 1:length(epocht1); % Define the time vector
                [tfr1, rtfr1, hat1] = tfrrpwv(epocht1, t ,1024); % Compute TFR using pseudo Wigner-Ville distribution
                Et1 = compute_entropy(log10(abs(tfr1)), window_size, step_size,'shannon',alpha); % Compute entropy
                Et1_all(:, t1,channel) = (Et1'); % Store entropy features
            end
        elseif test_labels(i) == 2
            t2=t2+1;
            test_data_CSP = w'*test_data(:,:,i)'; % Apply CSP to test data
            parfor channel = 1:num_channels
                epocht2 = test_data_CSP(channel, :)'; % Extract CSP component
        
                t = 1:length(epocht2); % Define the time vector
                [tfr2, rtfr2, hat2] = tfrrpwv(epocht2, t ,1024); % Compute TFR using pseudo Wigner-Ville distribution
                Et2 = compute_entropy(log10(abs(tfr2)), window_size, step_size,'shannon',alpha); % Compute entropy
                Et2_all(:, t2,channel) = (Et2'); % Store entropy features
            end
        end
    end
%%
% toc
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

    parfor i=1:num_channels
        feat_all1  = [feat_all1;  E1_all( :,:,i)];
        feat_all2  = [feat_all2;  E2_all( :,:,i)];
        featT_all1 = [featT_all1; Et1_all(:,:,i)];
        featT_all2 = [featT_all2; Et2_all(:,:,i)];
    end
    %%
    % Combine features for training and testing
    features_train = [feat_all1, feat_all2];
    features_test  = [featT_all1, featT_all2];
    
    %% Feature Selection using fscmrmr

    selected_features_idx = fscmrmr(features_train', train_labels); % Rank features using MRMR
    % Select the top N features (e.g., top 50 features)
    N = 20; % Number of top features to select
    selected_features_train = features_train(selected_features_idx(1:N), :); % Select top features for training
    selected_features_test  = features_test(selected_features_idx(1:N), :); % Select top features for testing
    rng("default");

    % Normalize selected features
    [selected_features_train, mu, sigma] = zscore(selected_features_train, 0, 2); % Z-score normalization
    selected_features_test  = (selected_features_test - mu) ./ sigma; % Apply normalization to test data

    [mdl] = fitcknn(selected_features_train', train_labels,"NumNeighbors",8); % Train a k-NN classifier
    % [mdl] = fitcnet(selected_features_train', train_labels,"Activations","tanh","Standardize",true,"LayerSizes",[246 14 16],"Lambda",0.00972); % Alternative: Neural network model
    predictions = predict(mdl, selected_features_test'); % Predict on test data
    
    accuracy(iter) = sum(predictions == test_labels') / length(test_labels); % Calculate accuracy for this fold
    mean_accuracy = mean(accuracy) * 100; % Mean accuracy across folds

    % fprintf('Classification Accuracy: %.2f%%\n', mean_accuracy);
end

mean_accuracy = mean(accuracy) * 100; % Final mean accuracy
std_accuracy = std(accuracy) * 100; % Standard deviation of accuracy
fprintf('Mean Accuracy: %.2f%%\n', mean_accuracy); % Display mean accuracy
fprintf('Standard Deviation of Accuracy: %.2f%%\n', std_accuracy); % Display standard deviation

%% ERP calculation and TFR
ERP_data1 = mean(data1(:,1,:), 3); % Average over trials for class 1 (channel 1)
ERP_data2 = mean(data2(:,1,:), 3); % Average over trials for class 2 (channel 1)
t = 1:length(ERP_data1); % Define time vector
[etfr1, ~, ~] = tfrrpwv(ERP_data1, t, 1024); % Compute TFR for class 1 ERP
ERPE13 = compute_entropy(log10(abs(etfr1)), window_size, step_size, 'shannon', alpha); % Compute entropy

[etfr2, ~, ~] = tfrrpwv(ERP_data2, t, 1024); % Compute TFR for class 2 ERP
ERPE23 = compute_entropy(log10(abs(etfr2)), window_size, step_size, 'shannon', alpha); % Compute entropy

% Plotting
t = linspace(0, 4, size(etfr1, 2)); % Define time vector for plotting
hz = linspace(0, fs/2, size(etfr1, 1)); % Define frequency vector for plotting
figure;
set(gcf, 'Color', 'w'); % Set figure background color to white

subplot(3, 2, 1)
imagesc(t, hz, log10(abs(etfr1))); % Plot TFR for class 1 ERP (channel 1)
axis xy;
title('TFR of ERP - Left Hand (C3)', 'Color', 'k');
xlabel('Time (s)', 'Color', 'k');
ylabel('Frequency (Hz)', 'Color', 'k');
colormap('turbo');
hcb = colorbar;
caxis([0 .25]);
ylabel(hcb, 'Power (dB)', 'Color', 'k');
ylim([0 50]);
set(gca, 'Color', 'w', 'XColor', 'k', 'YColor', 'k');
set(hcb, 'Color', 'k'); % Set colorbar ticks color to black

subplot(3, 2, 2)
imagesc(t, hz, log10(abs(etfr2))); % Plot TFR for class 2 ERP (channel 1)
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
set(hcb, 'Color', 'k'); % Set colorbar ticks color to black

t = 1:length(ERP_data1);
ERP_data1 = mean(data1(:,3,:), 3); % Average over trials for class 1 (channel 3)
ERP_data2 = mean(data2(:,3,:), 3); % Average over trials for class 2 (channel 3)

[etfr1, ~, ~] = tfrrpwv(ERP_data1, t, 1024); % Compute TFR for class 1 ERP
ERPE1 = compute_entropy(log10(abs(etfr1)), window_size, step_size, 'shannon', alpha); % Compute entropy

[etfr2, ~, ~] = tfrrpwv(ERP_data2, t, 1024); % Compute TFR for class 2 ERP
ERPE2 = compute_entropy(log10(abs(etfr2)), window_size, step_size, 'shannon', alpha); % Compute entropy

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
set(hcb, 'Color', 'k'); % Set colorbar ticks color to black

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
set(hcb, 'Color', 'k'); % Set colorbar ticks color to black

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

%% Function to calculate entropy
function entropy_features = compute_entropy(tfr, window_size, step_size, entropy_type, alpha)
    [num_frequencies, num_time_points] = size(tfr);
    window_size = round(window_size); % Round window size to nearest integer
    step_size = round(step_size); % Round step size to nearest integer
    num_windows = floor((num_time_points - window_size) / step_size) + 1; % Calculate number of windows
    entropy_features = zeros(1, num_windows); % Initialize entropy feature vector
    
    for i = 1:num_windows
        start_idx = (i - 1) * step_size + 1; % Start index for the current window
        end_idx = start_idx + window_size - 1; % End index for the current window
        
        if end_idx <= num_time_points
            window_data = abs(tfr(:, start_idx:end_idx)); % Extract windowed data
            window_data = window_data / sum(window_data(:)); % Normalize window data
            
            if strcmp(entropy_type, 'shannon') % Calculate Shannon entropy
                entropy_features(i) = -sum(window_data(:) .* log2(window_data(:) + eps));
            elseif strcmp(entropy_type, 'renyi') % Calculate Renyi entropy
                entropy_features(i) = 1 / (1 - alpha) * log2(sum(window_data(:).^alpha) + eps);
            end
        end
    end
end
