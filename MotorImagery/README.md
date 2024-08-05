# Time-Frequency-MotorImagery-BCI: Technical Overview

This document provides a comprehensive technical overview of the **Time-Frequency-MotorImagery-BCI** project. The focus is on the mathematical and algorithmic aspects of the code, including detailed explanations of signal processing, feature extraction, and classification methods.

## Table of Contents

- [Introduction](#introduction)
- [Methodology](#methodology)
  - [EEG Signal Preprocessing](#eeg-signal-preprocessing)
  - [Time-Frequency Analysis](#time-frequency-analysis)
  - [Feature Extraction](#feature-extraction)
  - [Classification](#classification)
- [Algorithms](#algorithms)
  - [Common Spatial Patterns (CSP)](#common-spatial-patterns-csp)
  - [Pseudo Wigner-Ville Distribution (PWVD)](#pseudo-wigner-ville-distribution-pwvd)
  - [Entropy Calculation](#entropy-calculation)
- [Implementation Details](#implementation-details)
- [Conclusion](#conclusion)

## Introduction

Brain-Computer Interfaces (BCIs) allow direct communication between the human brain and external devices. This project utilizes EEG signals for classifying motor imagery tasks, which involves imagining movements such as left-hand or right-hand motions. The core of this project is the application of time-frequency analysis to extract features that represent energy distribution in the EEG data, followed by classification using machine learning techniques.

## Methodology

### EEG Signal Preprocessing

EEG signal preprocessing is crucial for enhancing signal quality and isolating relevant information. The preprocessing steps include:

1. **Loading EEG Data:**
   - EEG data is loaded from a General Data Format (GDF) file using the Biosig toolbox.
   - The data typically contains multiple channels recorded at a specific sampling rate.

2. **Normalization and Filtering:**
   - Each EEG channel is normalized by subtracting the mean and dividing by the standard deviation to ensure that the signals have zero mean and unit variance.
   - A Butterworth bandpass filter is applied to isolate the mu (8-12 Hz) and beta (13-30 Hz) frequency bands, which are significant for motor imagery tasks. The filter is designed with specified low (`fl`) and high (`fh`) cut-off frequencies and an order, which defines the steepness of the filter's roll-off.

3. **Epoch Segmentation:**
   - The continuous EEG data is segmented into epochs based on event markers. Each epoch corresponds to a specific trial of motor imagery, which is defined by the latency and duration parameters (`ti` and `latency`).

### Time-Frequency Analysis

Time-frequency analysis is used to examine how signal energy is distributed over time and frequency. The pseudo Wigner-Ville distribution (PWVD) is employed for this purpose:

- **PWVD:** A quadratic time-frequency distribution that provides a high-resolution representation of signal energy. It is particularly useful for analyzing non-stationary signals like EEG. The PWVD is used to capture the energy changes in EEG signals associated with motor imagery tasks.

### Feature Extraction

Features are extracted from the time-frequency representations using entropy-based measures:

- **Energy Distribution:** The time-frequency representation is used to calculate energy distribution across time and frequency, which reflects how signal energy changes during motor imagery tasks.

- **Entropy Features:** Entropy is computed to quantify the complexity or randomness of the energy distribution in the TFR. Both Shannon and Renyi entropies are common measures used in this context, providing insights into the information content of the signal.

### Classification

Classification is performed using machine learning models, with k-fold cross-validation to evaluate performance:

- **Feature Selection:** Techniques like Minimum Redundancy Maximum Relevance (mRMR) are used to select the most discriminative features that contribute to classifying different motor imagery tasks effectively.

- **Classifier Training:** Models such as k-Nearest Neighbors (k-NN) and Linear Discriminant Analysis (LDA) are trained on the selected features to distinguish between different classes of motor imagery.

- **Cross-Validation:** K-fold cross-validation is employed to ensure robust evaluation and prevent overfitting by repeatedly partitioning the data into training and testing sets.

## Algorithms

### Common Spatial Patterns (CSP)

CSP is a powerful method for feature extraction in BCI applications. It is used to enhance the discriminability of EEG signals by maximizing variance between two classes:

- **Objective:** Find spatial filters that maximize the variance of one class while minimizing it for another, making the signal components most relevant for discrimination more prominent.

- **Mathematical Formulation:**

  1. **Compute Covariance Matrices:** For each class, compute the average covariance matrix:
     \[
     R_i = \frac{1}{N_i} \sum_{n=1}^{N_i} \frac{x_n x_n^T}{\text{trace}(x_n x_n^T)}
     \]
     where \( x_n \) is the EEG data matrix for the \( n \)-th trial, \( N_i \) is the number of trials for class \( i \), and \( i \) indicates each class.

  2. **Composite Covariance Matrix:** Calculate the composite covariance matrix as the sum of covariance matrices from both classes:
     \[
     R_c = R_1 + R_2
     \]

  3. **Whitening Transformation:** Use the eigenvalue decomposition of \( R_c \) to whiten the data:
     \[
     R_c = U \Lambda U^T
     \]
     The whitening matrix is \( P = \Lambda^{-\frac{1}{2}} U^T \).

  4. **Whitened Covariance Matrices:** Transform the covariance matrices using the whitening transformation:
     \[
     \tilde{R}_i = P R_i P^T
     \]

  5. **Generalized Eigenvalue Decomposition:** Perform eigenvalue decomposition on the whitened matrices:
     \[
     \tilde{R}_1 v_j = \lambda_j \tilde{R}_2 v_j
     \]
     where \( \lambda_j \) are the eigenvalues and \( v_j \) are the eigenvectors.

  6. **Spatial Filters:** Select the eigenvectors corresponding to the largest and smallest eigenvalues to form the spatial filters:
     \[
     W = \left[ v_1, v_2, \ldots, v_m, v_{N-m+1}, \ldots, v_N \right]
     \]
     where \( m \) is the number of filters to select.

### Pseudo Wigner-Ville Distribution (PWVD)

PWVD provides a detailed time-frequency representation of a signal:

- **Formula:** The PWVD is defined as:
  \[
  PWVD(t, f) = \int_{-\infty}^{\infty} x(t+\tau/2) x^*(t-\tau/2) e^{-j2\pi f \tau} d\tau
  \]
  where \(x(t)\) is the signal, \(\tau\) is the time lag, and \(f\) is the frequency.

- **Advantages:** Offers high time and frequency resolution, making it suitable for capturing the dynamics of EEG signals during motor imagery tasks. It provides insights into the temporal evolution of frequency components, crucial for understanding brain activities associated with motor imagery.

### Entropy Calculation

Entropy is a measure of uncertainty or randomness in a system. In this project, entropy is used to capture the complexity of energy distribution:

- **Shannon Entropy:** 
  \[
  H(X) = -\sum_{i} p(x_i) \log_2 p(x_i)
  \]
  where \(p(x_i)\) is the probability of occurrence of each state. Shannon entropy quantifies the average information content of a signal.

- **Renyi Entropy:**
  \[
  H_\alpha(X) = \frac{1}{1-\alpha} \log_2 \left( \sum_i p(x_i)^\alpha \right)
  \]
  where \(\alpha\) is a parameter controlling the order of entropy. Renyi entropy is a generalized form of entropy that emphasizes certain aspects of the probability distribution.

## Implementation Details

- **Data Handling:** Data is loaded and managed using MATLAB scripts, with efficient handling of epochs and signal channels. The scripts ensure that data is properly segmented and organized for further processing.

- **Parallel Processing:** Computational efficiency is enhanced using parallel processing capabilities in MATLAB (e.g., `parfor` loops), allowing for faster execution of cross-validation and feature extraction processes.

- **Visualization:** Plots of event-related potentials (ERPs) and time-frequency representations are generated for analysis and interpretation. Visualization aids in understanding the differences in brain activity patterns associated with different motor imagery tasks.

## Conclusion

The **Time-Frequency-MotorImagery-BCI** project demonstrates a comprehensive approach to EEG signal classification using advanced time-frequency analysis and feature extraction techniques. By leveraging methods such as CSP and PWVD, the project highlights the potential of BCIs in decoding motor imagery tasks, paving the way for applications in assistive technologies and neurorehabilitation.
