# Smart-Traffic-LiDAR: A Full-Stack Roadside LiDAR Processing Framework

![Status](https://img.shields.io/badge/Status-PHD_Heritage-blue)
![Field](https://img.shields.io/badge/Domain-ITS_/_Traffic_Engineering-green)

## 🚀 Overview
This repository contains a comprehensive suite of tools and deep learning models for **Roadside LiDAR-based Object Detection, Tracking, and Trajectory Reconstruction**. 

Originally developed during my PhD research and validated in collaboration with real-world infrastructure data (including Caltrans-related scenarios), this framework focuses on solving critical challenges in Smart City infrastructure, specifically **Occlusion Handling** in complex urban intersections.

---

## 🛠 Core Components

### 1. The Processing Engine (`LiDAR_Tracker_Project_v3.1`)
Our most stable, engineering-ready pipeline featuring a multi-tab graphical interface for:
- **Data Ingestion**: Pcap processing and synchronization.
- **Object Detection**: Fast clustering and point cloud segmentation.
- **Multi-Object Tracking (MOT)**: Robust association algorithms.
- **Real-time Reporting**: Socket-based data streaming for ITS integration.

### 2. Context-Aware Deep Learning (`ContextAware`)
Advanced trajectory reconstruction models designed to mitigate occlusion issues:
- **SocialLSTM / SimpleLSTM**: Predicting vehicle movements in high-density environments.
- **Transformer Baselines**: State-of-the-art attention mechanisms for spatial-temporal trajectory smoothing.

### 3. Historical Archive (`legacy/`)
A preserved collection of research milestones, including:
- Early MOT experiments.
- Raspberry Pi edge computing implementation tests.
- Classifier training scripts and raw model files (.sav).

---

## 📈 Key Features
- **Occlusion Resilience**: Specialized algorithms to "see" through blind spots using spatial-temporal context.
- **High Efficiency**: Optimized for roadside deployment with low-latency requirements.
- **Modular Design**: Easy to extract individual components for custom traffic analysis tasks.

---

## 📝 Citation & Contact
If you use this work for your research, please refer to the papers in the `ContextAware` directory. 

Developed by **Joshua (Zhihui) Chen**, Traffic Engineer at Caltrans.
