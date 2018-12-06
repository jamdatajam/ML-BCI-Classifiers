% This script is meant to pull in EEG data from a past project and then
% classify left and right decision making points into commands

EEG_LR_data = load('EEG_LR_data.mat');
stim_1 = EEG_LR_data.StimulusCode_Collection1;
EEG_1_rows = EEG_LR_data.eeg_data_Collection1;
EEG_1 = EEG_1_rows';

%Need to pull out channels FC5-1, FC1-3, FC2-5, FC6-7, C3-9, Cz-11, C4-13, 
%CP5-15, CP1-17, CP2-19, CP6-21, P3-49, Pz-51, P4-53, PO3-57, and PO4-59

y = 1:length(stim_1);

%length of data = m
m = length(y);

%Keep all SCPs above 0.1 Hz; Butterworth 0.1 to 2 Hz (bandpass)
EEG_1 = EEG_1(:,[1 3 5 7 9 11 13 15 17 19 21 49 53 57 59]);

%filtfilt with frequency domain
EEG_fft = fft(EEG_1);
EEG_fft(EEG_fft <0.1) = 0;
EEG_fft(EEG_fft > 2) = 0;
EEG_1 = ifft(EEG_fft)

%mean normalization
%[EEG_1 mu sigma] = normalizer(EEG_1);

%add the extra column
EEG_1 = [ones(m, 1) EEG_1];

%compute the grad desc
alpha = 0.1; %some number to set the grad desc rate
iterations = 25; %the number of cycles
theta = zeros(16, 1); %init fitting param
[theta, J_history] = GradDescenter(EEG_1, y, theta, alpha, iterations);

%plot the cost function plot
figure(1)
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Iteration number');
ylabel('Max cost (J)');
title('EEG channel data grad descent cost');


%next step is to multiply new data by theta to predict with new data
%ie, prediction = [matrix values] *theta; 