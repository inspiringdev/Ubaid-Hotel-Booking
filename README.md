# Ubaid-Hotel-Booking

## Big Data Analytics (CS404) — Project

---

## Overview
Ubaid Mahal is an end-to-end machine learning pipeline designed to maximize hotel revenue through dynamic pricing. Leveraging a large-scale dataset of real hotel bookings, the system utilizes supervised learning, reinforcement learning, and advanced business intelligence to recommend the optimal room rate (ADR) in real-time.

The solution balances competitive pricing with demand elasticity, ensuring rates are high enough to maximize profit but attractive enough to drive occupancy.

---

## Dataset profile
- **Source:** Hotel Booking Demand Datasets (Antonio, Almeida, & Nunes, Data in Brief 2019)
- **Volume:** 119,390 booking records (City & Resort hotels)
- **Timeline:** July 2015 – August 2017
- **Granularity:** Daily transactional data including cancellations
- **Features:** 30+ features (Lead Time, Seasonality, Competitor Pricing, Events, etc.)
- **Target:** `actual_price` (Optimized Average Daily Rate)

---

## Project architecture
The project is structured into a modular pipeline with dedicated notebooks for the rubric requirements:

```
hotel_pricing_project/
├── data/
│ ├── hotel_booking_data.csv 
│ └── hotel_booking_clean.csv #datacleaned
├── notebooks/ # Jupyter Notebooks
│ ├── 01_Data_Preprocessing.ipynb #cleaning, encoding, splitting
│ ├── 02_Exploratory_EDA.ipynb # 12 EDA visualizations
│ ├── 03_Models_and_RL.ipynb # 6 Regression models + Q-Learning Agent
│ ├── 04_Advanced_Analysis.ipynb # Segmentation, elasticity, anomaly detection
│ └── 05_Price_Recommender.ipynb # Business logic & final insights
├── src/
│ ├── preprocessing.py #data cleaning 
│ ├── feature_engineering.py 
│ ├── eda.py #plotting functions
 │ ├── models.py #model training wrappers
 │ └── advanced_analysis.py # Complex analytics modules
 │ 
└── outputs/
 │ ├── figures/ #it contains eda and plots
 │ ├── adv_figures/ # Advanced analysis visualizations
 │ ├── models/ # Saved .pkl model artifacts
 │
 │ 
├── main.py # Full pipeline orchestrator
 │ 
├── requirements.txt
 │ 
└── README.md"""
