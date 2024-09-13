function [E] = energydist(tfr, window_size, window_step)

F = 1:700;
P = tfr; % Replace rtfr1 with your actual TFR matrix variable
T = 1:700; % Replace with the actual time vector if it differs

% Parameters

% Assuming T is in seconds, calculate time bin size
total_time = max(T) - min(T);
num_time_bins = length(T);
time_bin_size = total_time / num_time_bins; % Time difference between adjacent T values

% Convert window size and step size to bins
window_bins = round(window_size / time_bin_size);
step_bins = max(round(window_step / time_bin_size), 1); % Ensure step_bins is at least 1

% Preallocate energy array
num_windows = floor((num_time_bins - window_bins) / step_bins) + 1;
energy_values = zeros(1, num_windows);

% Loop over each window
for i = 1:num_windows
    start_idx = (i-1) * step_bins + 1;
    end_idx = start_idx + window_bins - 1;
    
    if end_idx > num_time_bins
        end_idx = num_time_bins;
    end
    
    % Extract windowed TFR
    P_window = P(:, start_idx:end_idx);
    
    % Compute the energy distribution
    energy = sum(abs(P_window(:)).^2);
    
    % Store the energy value
    E(i) = energy;
end

end