# MetroFlow

**Passengers’ Origin-Destination Distribution Prediction Over Metro Networks**

## 📌 Overview

**MetroFlow** is an AI-powered model that predicts passenger flow and origin-destination (OD) distributions in metro networks, especially for newly introduced lines that lack historical ridership data. It integrates aggregate and disaggregate modeling techniques to forecast travel behavior, enabling smarter transit planning, efficient resource allocation, and better infrastructure decisions.

## 👥 Team

* K S Sreekumar
* Nandana
* Anchana Shaji

## 🧠 Problem Statement

Urban metro systems face a critical challenge: predicting how new metro lines will affect passenger flow without existing historical data. Inaccurate OD forecasting can lead to poor network optimization and misallocation of resources.

## 📚 Background

* AFC (Automated Fare Collection) systems provide valuable insights into current travel behavior.
* However, these systems fall short when forecasting future passenger patterns for yet-to-be-launched lines.
* There is a need to infer how passengers will reroute, shift preferences, or adopt new lines post-expansion.

## 🔍 Literature Insights

Key findings from research include:

* New metro lines affect land use and urban development.
* AFC and GPS data are valuable but require complex processing.
* Mobile phone data, spatial analysis, and urban modeling contribute to travel demand estimation.
* Prior models often suffer from high computational costs, calibration complexity, and dependency on quality data.

## 💡 Proposed Solution

MetroFlow bridges the data gap by combining AFC patterns and station metadata to simulate realistic OD matrices for new routes. Using machine learning, it refines predictions based on behavior observed in similar station pairs.

### ✅ Core Functionalities

* **OD Prediction:** Estimates new origin-destination pairs based on passenger behavior.
* **Event-Based Simulation:** Predicts spikes in travel patterns due to public events.
* **Synthetic OD Generation:** Creates realistic OD matrices for future expansions.
* **Validation Tools:** Compares predicted vs. actual OD flows using error metrics and heatmaps.

## 🔄 Workflow

1. **Data Collection:** Gather AFC data, station metadata, and metro network layouts.
2. **Feature Engineering:** Extract travel behavior indicators and station similarities.
3. **Model Training:** Apply machine learning models (e.g., XGBoost, MNL) to infer passenger choices.
4. **Prediction & Validation:** Generate OD matrices and validate using historical trends and accuracy metrics.

## 🏗️ System Architecture

* **Input Sources:** AFC data, Metro Station info, Expansion plans
* **Core Modules:**

  * Feature Engineering
  * OD Matrix Generator
  * Event Simulation
  * Visualizer (Heatmaps, Comparative Graphs)
* **Output:** Predicted OD Distributions, Profit Estimations, Performance Visualizations

## 📈 Impact

MetroFlow aims to support:

* **Metro authorities** in resource and capacity planning
* **Urban planners** in infrastructure development
* **Policy makers** in transport policy decisions
* **Developers** in identifying high-footfall investment zones
