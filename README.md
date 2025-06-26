# Network Intrusion Detection with Reinforcement Learning

## Zero-Day Attack Detection Research using Q-Learning and Deep Q-Network

This repository contains the implementation and comparison of two reinforcement learning algorithms for network intrusion detection, with a specific focus on the ability to detect zero-day attacks.

## 📋 Research Description

This research develops and compares two reinforcement learning approaches for network intrusion detection:
- **Q-Learning**: A simple yet efficient tabular algorithm
- **Deep Q-Network (DQN)**: A more complex neural network-based algorithm

Both algorithms are trained to detect normal attacks and evaluated on their ability to detect zero-day attacks that were not seen during training. We use the UNSW-NB15 dataset with Fuzzers and Reconnaissance attack categories as a simulation of zero-day attacks.

## 🎯 Research Objectives

1. Develop a reinforcement learning-based network intrusion detection model
2. Evaluate and compare the performance of Q-Learning and DQN in detecting:
   - Recognized standard attacks
   - Previously unseen zero-day attacks
3. Optimize parameters to achieve the best detection performance
4. Analyze the generalization capabilities and computational efficiency of both algorithms

## 📊 Dataset

This research uses the **UNSW-NB15** dataset containing normal network traffic data and various types of attacks. The dataset is divided into:
- **Training data**: Common attacks (without Fuzzers and Reconnaissance)
- **Testing data**: All types of attacks
- **Zero-day evaluation data**: Only Fuzzers and Reconnaissance attacks + normal traffic

> **Note:** Due to GitHub file size limitations, the dataset files (UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv) are not included in this repository. Please download them from the [UNSW-NB15 official site](https://research.unsw.edu.au/projects/unsw-nb15-dataset).

##  Research Results

### Q-Learning
- Best model: **Scheme 1** (alpha=0.01, gamma=0.8, epsilon=1.0, epsilon_decay=0.8, epsilon_min=0.05)
- Standard test data accuracy: 94.18%, F1 score: 0.9447
- Zero-day data accuracy: 91.87%, F1 score: 0.7653
- Zero-day data detection rate (recall): 64.56%
- Training time: 36.03 seconds
- Model size: 0.000458 MB
- Inference speed: 997,931 samples/second

### Deep Q-Network (DQN)
- Best model: **Scheme 1** (learning_rate=0.0001, epsilon_decay=0.93, batch_size=32, buffer_size=50000)
- Standard test data accuracy: 99.09%, F1 score: 0.9918
- Zero-day data accuracy: 98.40%, F1 score: 0.9624
- Zero-day data detection rate (recall): 99.91%
- Training time: 4,852.5 seconds
- Model size: 0.03 MB
- Inference speed: 3,623.4 samples/second

### Comparison with Traditional Methods
- **Random Forest**: Standard test accuracy: 89.78%, Zero-day detection rate: 43.37%
- **SVM**: Standard test accuracy: 87.40%, Zero-day detection rate: 47.68%

DQN shows significantly better performance than Q-Learning and traditional methods, especially in detecting zero-day attacks. However, Q-Learning offers considerable computational efficiency advantages in terms of training time, model size, and inference speed.

## 🔍 Key Findings

1. DQN outperformed Q-Learning across all metrics, with a detection rate difference of 35.35% on zero-day attacks
2. DQN showed exceptional generalization capability with minimal performance degradation (<1%) when facing novel threats
3. Q-Learning exhibited higher computational efficiency (275x faster inference speed and much shorter training time)
4. Feature analysis revealed that service, dbytes, sload, ct_srv_src, and swin had the strongest impact on detection outcomes
5. t-SNE visualization demonstrated how DQN effectively transforms feature space to create clear boundaries between normal and attack traffic

## 🔬 Methodology

Both algorithms use the same techniques for data preprocessing:
1. Feature cleaning and normalization
2. Handling missing values
3. Feature selection using SelectKBest
4. Class balancing using SMOTE-Tomek

The main differences are in the model architecture:
- Q-Learning: Tabular approach with state discretization using K-Means clustering
- DQN: Neural network with 2 hidden layers (64 and 32 neurons)

##  Usage Scenarios

Based on our findings, we recommend:
- **DQN**: For enterprise-scale systems, cloud infrastructures, and data center networks where detection accuracy is paramount
- **Q-Learning**: For resource-constrained environments like branch offices, embedded IoT networks, and industrial control systems with hardware or latency constraints

## 🙏 Acknowledgements

- UNSW-NB15 Dataset: [https://research.unsw.edu.au/projects/unsw-nb15-dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

## 📧 Contact
adhit1312@gmail.com
adhitamaw@student.telkomuniversity.ac.id

---

© 2025. All rights reserved.
