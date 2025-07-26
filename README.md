# Daily Commodity prediction


## 🌾 Real-Time Agricultural Commodity Price Prediction & Analysis System

**Project Title:** Agricultural Commodities Price Prediction
**Domain:** AgriTech | Machine Learning | Real-Time Data Analytics
**Tech Stack:** Python, Pandas, Scikit-learn, Plotly, Streamlit

---

### 📌 **Project Summary:**

This project presents a real-time, interactive platform for analyzing and predicting agricultural commodity prices using machine learning and exploratory data analysis (EDA). Developed to support the agricultural ecosystem in states like **Andhra Pradesh** and **Tamil Nadu**, the system empowers **farmers, traders, and decision-makers** by delivering actionable insights on **daily market price fluctuations**.

Users can upload commodity datasets, map relevant fields, explore price trends, compare markets, and generate predictions—all through a user-friendly **Streamlit dashboard**.

---

### 🧭 **Core Objectives:**

* Predict future modal prices of commodities using regression models.
* Perform exploratory analysis on commodity pricing trends.
* Enable dynamic filtering by **state**, **district**, **market**, and **commodity**.
* Visualize and compare prices across **date ranges** and **locations**.
* Deliver a low-barrier tool for stakeholders to anticipate market behavior.

---

### 📂 **Application Features (as per UI Screenshots):**

#### 🔄 **Interactive Dashboard Modules:**

* **📅 Date Range Selector:** Filter price data for specific time windows.
* **📍 Location Filter:** Choose state (e.g., Tamil Nadu), district (e.g., Madurai), and market (e.g., Usilampatty).
* **🌽 Commodity Filter:** Select crops like Paddy(Dhan) for focused analysis.
* **📤 Upload CSV:** Supports structured agricultural datasets up to 200MB.
* **🧩 Column Mapping:** Users define key data points (e.g., min, max, modal price; market; date).

#### 📊 **Visualization & Analysis Tabs:**

* **Price Analysis:** Displays average prices (Min: ₹1833.33, Max: ₹1933.33, Modal: ₹1883.33).
* **Market Compare:** Side-by-side comparison of prices across multiple markets.
* **Trend Analysis:** Time-series exploration to identify seasonal or weekly patterns.
* **Data View:** Raw data exploration post-filtering.
* **Advanced Analysis:** Additional visual insights for deeper price behavior.

---

### 🤖 **Modeling & Methodology:**

* **EDA Techniques:**

  * Boxplots, heatmaps, and bar charts to interpret price spread and volatility.
* **Machine Learning Models:**

  * Linear Regression, Random Forest, XGBoost for modal price prediction.
  * Random Forest showed the highest accuracy with **R² = 0.93**.
* **Evaluation Metrics:** R² Score, RMSE, MAE.

---

### 💡 **Key Insights:**

* Prices are **highly location-specific**, with variety and market as dominant influencers.
* Commodities like **groundnut and turmeric** show greater seasonal fluctuation.
* Dashboards enable **granular visibility** into average pricing and comparative trends.

---

### 📈 **Business Impact & Use Case:**

* Helps **farmers** decide the best time and place to sell produce.
* Equips **traders** with market intelligence for inventory planning.
* Assists **policymakers** in analyzing regional demand and price stability.

---

### 🔮 **Limitations & Future Scope:**

* Currently lacks time-series models (e.g., ARIMA, LSTM) due to limited timestamp granularity.
* Focus is localized (Andhra Pradesh, Tamil Nadu); expansion to pan-India is feasible.
* Future Enhancements:

  * Weather-based modeling (rainfall, temperature)
  * Mobile app integration for direct farmer access
  * SMS alert system for price recommendations

---

This project exemplifies the power of **data-driven agriculture**, seamlessly blending **ML models, real-time dashboards, and EDA** to foster better price transparency and decision-making in India’s agri-economy.


<img width="1814" height="893" alt="Screenshot 2025-07-27 004514" src="https://github.com/user-attachments/assets/5d39705f-bd54-4152-b720-0cb8646a99a5" />
<img width="1280" height="824" alt="Screenshot 2025-07-27 004543" src="https://github.com/user-attachments/assets/2a005b65-1221-4da4-b052-eb35b6404c02" />
<img width="852" height="116" alt="Screenshot 2025-07-27 004604" src="https://github.com/user-attachments/assets/6f62fb65-3de9-48ca-b3d4-c87383e27f9a" />



