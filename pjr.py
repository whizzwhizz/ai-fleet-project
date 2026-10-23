import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
import time

# ==========================================
# 1. UI Setup
# ==========================================
st.set_page_config(page_title="AI Fleet Orchestration", layout="wide")
st.title("NextGen Mobility - AI Fleet Orchestration")
st.write("**Student:** KAVYANSH SHARMA (24BYB0169)")
st.write("This dashboard demonstrates real-time Neural Network forecasting and mathematical fleet optimization.")

# Sidebar controls
st.sidebar.header("Simulation Parameters")
fleet_size = st.sidebar.slider("Fleet Size", 20, 100, 40)
sim_days = st.sidebar.slider("Training Days (Data Volume)", 10, 100, 30)

if st.sidebar.button("Run AI Simulation"):

    # Create placeholders for loading states
    status_text = st.empty()
    progress_bar = st.progress(0)

    # ==========================================
    # 2. City Grid & Data Generation
    # ==========================================
    status_text.info("Step 1: Generating Smart City Environment & Historical Data...")
    progress_bar.progress(25)
    time.sleep(1)  # Slight pause for visual effect during presentation

    grid_size = 10
    total_zones = grid_size * grid_size

    # Generate synthetic historical demand (Features: day, hour, zone)
    np.random.seed(42)
    X_train = []
    y_train = []

    for day in range(sim_days):
        for hour in range(24):
            for zone in range(total_zones):
                # Add logic where certain zones (downtown) have higher demand at certain hours
                base_demand = np.random.poisson(2)
                if 8 <= hour <= 10 and zone % 3 == 0:  # Morning rush
                    base_demand += np.random.poisson(5)

                X_train.append([day, hour, zone])
                y_train.append(base_demand)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # ==========================================
    # 3. AI Model Training (Neural Network)
    # ==========================================
    status_text.info("Step 2: Training Neural Network (MLPRegressor) on historical patterns...")
    progress_bar.progress(50)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # The actual AI model
    ai_model = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=200, random_state=42)
    ai_model.fit(X_train_scaled, y_train)

    # Predict a sample day (Day 0) to show the forecast
    sample_day = np.array([[0, h, 15] for h in range(24)])  # Zone 15
    sample_day_scaled = scaler.transform(sample_day)
    predictions = ai_model.predict(sample_day_scaled)
    actuals = y_train[(X_train[:, 0] == 0) & (X_train[:, 2] == 15)]

    # ==========================================
    # 4. Fleet Optimization (Hungarian Algorithm)
    # ==========================================
    status_text.info("Step 3: Running AI Dispatch Optimizer vs Traditional Baseline...")
    progress_bar.progress(75)
    time.sleep(1)

    # Simulate a scenario: 10 riders requesting cars, and our fleet size
    riders = np.random.rand(10, 2) * grid_size
    vehicles = np.random.rand(fleet_size, 2) * grid_size

    # Calculate distance matrix
    cost_matrix = np.zeros((10, fleet_size))
    for i in range(10):
        for j in range(fleet_size):
            cost_matrix[i, j] = np.linalg.norm(riders[i] - vehicles[j])

    # AI Dispatch (Optimal matching minimizing total distance)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    ai_total_distance = cost_matrix[row_ind, col_ind].sum()

    # Traditional Dispatch (Naive nearest neighbor, non-optimized)
    traditional_distance = 0
    used_vehicles = set()
    for i in range(10):
        best_v, min_dist = -1, float('inf')
        for j in range(fleet_size):
            if j not in used_vehicles and cost_matrix[i, j] < min_dist:
                min_dist = cost_matrix[i, j]
                best_v = j
        traditional_distance += min_dist
        used_vehicles.add(best_v)

    status_text.success("Simulation Complete!")
    progress_bar.progress(100)

    # ==========================================
    # 5. Displaying the Real Results
    # ==========================================

    # Top Metrics
    st.write("### Simulation Results")
    col1, col2, col3 = st.columns(3)

    ai_savings = ((traditional_distance - ai_total_distance) / traditional_distance) * 100

    col1.metric("Total Empty Distance (AI)", f"{ai_total_distance:.1f} km", f"-{ai_savings:.1f}% vs Traditional",
                delta_color="inverse")
    col2.metric("Total Empty Distance (Traditional)", f"{traditional_distance:.1f} km")
    col3.metric("Neural Network Score (R²)", f"{ai_model.score(X_train_scaled, y_train):.2f}")

    # Charts
    st.write("---")
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.write("#### AI Demand Forecast vs Actual (Zone 15)")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(24), actuals, label='Actual Demand', color='gray', linestyle='--')
        ax.plot(range(24), predictions, label='AI Forecast', color='blue', linewidth=2)
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Ride Requests")
        ax.legend()
        st.pyplot(fig)

    with col_chart2:
        st.write("#### Fleet Dispatch Efficiency Comparison")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        bars = ax2.bar(['Traditional Dispatch', 'AI Optimized (Hungarian)'],
                       [traditional_distance, ai_total_distance],
                       color=['#ff9999', '#66b3ff'])
        ax2.set_ylabel("Total Empty Kilometers (Lower is better)")
        st.pyplot(fig2)

else:
    st.info("👈 Adjust the parameters in the sidebar and click **Run AI Simulation** to start.")
    