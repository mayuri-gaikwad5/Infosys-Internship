# Infosys-Internship

# Hotel Revenue Analysis: A Data-Driven Solution

This project delivers a robust analytical solution built in **Power BI** and complemented by an **AI-powered Streamlit application**. The primary goal is to help hotel management accurately forecast demand, analyze guest behavior, and optimize pricing strategies to ultimately **improve revenue**.

---
## 🌟 Key Project Achievements

The project's success is highlighted by the following key technical and analytical milestones:

### Data Modeling and Quality
* **Structured Data Model:** Designed and validated a scalable **Star Schema** with a central `FactBooking` table linked to dimensions for Customer, Room, Date, and Hotel Branch, ensuring efficient query performance.
* **Data Transformation (Power Query):** Successfully performed extensive data cleaning, validating for duplicates and inconsistencies, and creating essential **derived columns** (e.g., `BookingDuration`, `Total Guests`) to form a reliable foundation for accurate business reporting.

### Core Revenue & Performance Tracking
* **KPI Development:** Accurately developed and visualized key performance indicator (KPI) measures, including **Occupancy %** (e.g., 12.45%), **Total Revenue** (e.g., $13.75M), and **RevPAR** (e.g., $2.17K).
* **Dynamic Trend Analysis:** Visualized and enabled dynamic time-based analysis of occupancy, ADR, and RevPAR using interactive line charts, allowing stakeholders to easily monitor performance trends across custom periods like year, month, or season.
* **Channel Performance:** Successfully compared and tracked booking performance across various channels, including **Direct** bookings versus those from **OTA/TO** (Online Travel Agencies/Tour Operators).

### Advanced Analytics & Strategy Modules
* **Guest Segmentation:** Segmented the customer base into categories like **first-time, loyal, and high-spending customers** (Customer Cluster) to enable targeted marketing and loyalty program development. The analysis also determined a significant portion of bookings (74.11%) were made by **Solo guests**.
* **Forecasting & Risk Management:** Implemented forecasting visuals to predict future occupancy and analyzed a high overall **Cancellation Rate of 36.68%** across lead times and booking channels to identify high-risk reservations and quantify revenue loss.
* **Revenue Strategy Dashboard:** Created a strategic dashboard to monitor financial performance, integrate **pricing tiers by season and room type** (e.g., highest ADR for room type G in Summer at $138.77), and pinpoint high **upsell potential** across services (spa, dining).
* **AI Module (Streamlit App):** Developed an integrated AI-powered interface for real-time insights, allowing staff to input booking details and instantly receive a predicted **cancellation probability** (e.g., 29.35%) and automated recommendations (e.g., *Low risk. Booking is likely secure.*).

---
## 💻 Tech Stack
* **Business Intelligence Tool:** Power BI
* **Data Transformation:** Power Query
* **Data Application & ML Backend:** Streamlit, Python (NumPy, Pandas)

---
## 📜 License
This project is licensed under the **MIT License**.
