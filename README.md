# Time-Frequency_MotorImagery_BCI
This MATLAB project implements an EEG-based Brain-Computer Interface (BCI) system using time-frequency analysis with the pseudo Wigner-Ville distribution to classify motor imagery tasks through energy distribution features.

# EEG Signal Processing and Classification

This repository contains a MATLAB script for EEG signal processing and classification using Common Spatial Patterns (CSP) and entropy-based features. The code implements data loading, preprocessing, feature extraction, and classification using a k-fold cross-validation approach.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Data](#data)
- [Dependencies](#dependencies)
- [Results](#results)
- [License](#license)

## Introduction

### What is a Brain-Computer Interface (BCI)?

A Brain-Computer Interface (BCI) is a system that enables direct communication between the brain and an external device, typically a computer or a robotic system. BCIs are designed to decode neural activity patterns associated with specific mental tasks or intentions and translate them into commands for controlling external devices. These systems are particularly beneficial for individuals with motor disabilities, allowing them to interact with their environment through thought alone.

BCIs commonly use electroencephalography (EEG) signals, which are non-invasive brain signals captured from the scalp. EEG signals contain valuable information about brain activity, including responses to motor imagery tasks where a person imagines performing a physical movement, such as moving a limb, without actual movement. These imagined movements generate characteristic patterns in the EEG signal that can be detected and classified by a BCI system.

### How This Code Works

This MATLAB script is designed to process and classify EEG signals recorded during motor imagery tasks. The goal is to distinguish between different imagined movements using a feature known as **energy distribution**, which is derived from the time-frequency representations (TFRs) of the EEG signal. The code is structured to perform several key functions:

1. **Data Loading and Preprocessing:**
   - Load EEG data from a GDF file format using the Biosig toolbox.
   - Preprocess the data by normalizing and filtering to focus on specific frequency bands relevant to motor imagery, typically the mu (8-12 Hz) and beta (13-30 Hz) rhythms.

2. **Signal Segmentation:**
   - Segment the continuous EEG data into epochs based on event markers. Each epoch corresponds to a trial of motor imagery, such as imagining left-hand or right-hand movements.

3. **Feature Extraction:**
   - Apply a Butterworth bandpass filter to isolate the frequency bands of interest.
   - Use the Common Spatial Patterns (CSP) technique to enhance the signal-to-noise ratio of the EEG data for each class.
   - Compute the energy distribution features from the TFR of each epoch using the Wigner-Ville distribution, which provides a detailed view of how signal energy varies over time and frequency.

4. **Classification:**
   - Implement k-fold cross-validation to train and evaluate the performance of a neural network classifier on the extracted features.
   - Use feature selection techniques to identify and retain the most discriminative features for classification.

5. **Results Visualization:**
   - Calculate and display the classification accuracy for each fold, providing an assessment of the model's performance.
   - Visualize the event-related potentials (ERP) and time-frequency representations (TFRs) to show the differences in brain activity patterns associated with different motor imagery tasks.

### Significance

The code provides a framework for developing a BCI system that can accurately classify motor imagery tasks based on EEG signals. By extracting energy distribution features and employing advanced classification techniques, this system demonstrates the potential to enhance the control capabilities of BCIs, making them more responsive and reliable for end-users. This work is a step toward practical applications in neurorehabilitation and assistive technologies, where BCIs can empower individuals with physical limitations to regain control and independence.

## Features

- **EEG Data Loading:** Load and preprocess EEG data in GDF format using the Biosig toolbox.
- **Bandpass Filtering:** Apply a Butterworth bandpass filter to extract specific frequency bands.
- **Data Segmentation:** Segment EEG signals into epochs based on event markers.
- **Feature Extraction:** Compute entropy features from time-frequency representations (TFRs).
- **Classification:** Perform classification using CSP and neural networks with k-fold cross-validation.
- **ERP and TFR Analysis:** Compute and visualize event-related potentials (ERP) and TFRs.

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/eeg-signal-processing.git
   cd eeg-signal-processing
   ```

2. Ensure MATLAB is installed on your system.

3. Add required toolboxes to your MATLAB path:

   ```matlab
   addpath('/path/to/your/tftb-toolbox');
   addpath('/path/to/your/eeglab-toolbox');
   ```

## Usage

1. **Prepare Data:** Ensure your EEG data in GDF format is accessible and modify the `filename` variable in the script to point to your data file.

2. **Run the Script:** Execute the MATLAB script `eeg_processing.m` to start processing and classification:

   ```matlab
   run('eeg_processing.m');
   ```

3. **View Results:** Classification accuracy and plots for ERP and TFR will be displayed upon completion.

## File Structure

- `eeg_processing.m`: Main MATLAB script for EEG signal processing and classification.
- `emptyEEG.mat`: Placeholder EEG structure for further use.
- `tftb-0.2/mfiles`: Folder containing the Time-Frequency Toolbox scripts.
- `eeglab2024.0`: Folder containing the EEGLAB toolbox.

## Data

- **Input:** EEG data in GDF format. The script processes channels related to motor imagery tasks.
- **Output:** Classification accuracy, entropy or energy distribution features, and plots of ERP and TFR.

## Dependencies

- MATLAB (R2024 or later recommended)
- [Biosig Toolbox](https://sourceforge.net/projects/biosig/)
- [EEGLAB Toolbox](https://sccn.ucsd.edu/eeglab/)
- [Time-Frequency Toolbox (TFTB)](https://tftb.nongnu.org/)

Ensure these dependencies are installed and properly configured in your MATLAB environment.

## Results

The script calculates and displays the classification accuracy for each fold in the k-fold cross-validation process. It also visualizes the time-frequency representations and both entropy and energy distribution plots for both classes.

