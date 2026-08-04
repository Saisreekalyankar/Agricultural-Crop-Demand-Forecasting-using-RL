import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# -----------------------------------------------------------------------------
# 1. SETUP & CONFIGURATION
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="AgriPlan AI: Regional Planning Dashboard",
    page_icon="🌾",
    layout="wide"
)

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #2E7D32; font-weight: bold; }
    .metric-card { background-color: #f0f2f6; padding: 10px; border-radius: 5px; border-left: 5px solid #2E7D32; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. LOAD ENVIRONMENT & MODELS
# -----------------------------------------------------------------------------

@st.cache_resource
def load_system():
    """Loads config and RL models."""
    try:
        from agri_sim import load_and_process_config, FranceAgriEnv
        from stable_baselines3 import DQN, PPO
        
        config = load_and_process_config()
        
        # Check if models exist before loading
        dqn_model = None
        ppo_model = None
        
        if os.path.exists("models/dqn_final.zip"):
            dqn_model = DQN.load("models/dqn_final", device="cpu", custom_objects={"learning_rate": 0.0003})
        else:
            st.warning("DQN model not found in 'models/' folder.")
            
        if os.path.exists("models/ppo_final.zip"):
            ppo_model = PPO.load("models/ppo_final", device="cpu", custom_objects={"learning_rate": 0.0003})
        else:
            st.warning("PPO model not found in 'models/' folder.")
            
        return config, dqn_model, ppo_model
        
    except Exception as e:
        st.error(f"Error loading system: {e}")
        return None, None, None

# -----------------------------------------------------------------------------
# 3. SIMULATION ENGINE
# -----------------------------------------------------------------------------

def run_rl_simulation(model_obj, model_name, config, seed=42):
    """Runs simulation for RL agents."""
    from agri_sim import FranceAgriEnv
    env = FranceAgriEnv(config=config, seed=seed)
    obs, _ = env.reset()
    
    done = False
    total_reward = 0
    step_data = []
    
    while not done:
        # Predict action
        action, _ = model_obj.predict(obs, deterministic=True)
        
        # FIX: Convert NumPy array action to standard Python integer
        if isinstance(action, np.ndarray):
            action_val = int(action.item())
        elif hasattr(action, '__len__') and len(action) == 1:
            action_val = int(action[0])
        else:
            action_val = int(action)
            
        current_region = env.regions[env.current_region_idx]
        
        # Step environment with integer action
        next_obs, reward, done, truncated, info = env.step(action_val)
        
        total_reward += reward
        
        # Use the integer value for dictionary lookup
        crop_name = env.idx_to_crop[action_val]
        
        step_data.append({
            "Step": len(step_data),
            "Region": current_region,
            "Action_Crop": crop_name,
            "Reward": reward,
            "Weather_Shock": env.weather_shock_active
        })
        obs = next_obs

    return calculate_metrics(env, total_reward, step_data, model_name)

def run_greedy_simulation(config, seed=42):
    """Runs simulation for Greedy Baseline."""
    from agri_sim import run_greedy_baseline
    env = run_greedy_baseline(config, seed=seed)
    
    # Calculate metrics manually for greedy since it doesn't step through like RL
    demand = env.demand_vector
    production = env.production_vector
    
    deficit_vec = np.maximum(demand - production, 0)
    surplus_vec = np.maximum(production - demand, 0)
    
    import_cost = 0.0
    export_revenue = 0.0
    for i, crop in enumerate(env.crops):
        prices = env.econ.get(crop, {'local_price': 200, 'int_price': 180})
        import_cost += deficit_vec[i] * prices['local_price'] * 1.5
        export_revenue += surplus_vec[i] * prices['int_price']
        
    net_economic = export_revenue - import_cost
    demand_sat = np.minimum(production / (demand + 1e-8), 1.0).mean() * 100
    
    # Dummy step log for visualization consistency
    step_data = [{"Step": 0, "Region": "All", "Action_Crop": "Mixed", "Reward": 0, "Weather_Shock": False}]
    
    return {
        "model_name": "Greedy",
        "total_reward": net_economic / 100000.0, # Normalize to match RL
        "demand_satisfaction": demand_sat,
        "net_economic": net_economic,
        "export_revenue": export_revenue,
        "import_cost": import_cost,
        "total_deficit": deficit_vec.sum(),
        "allocation_matrix": env.allocation_matrix,
        "step_log": pd.DataFrame(step_data),
        "weather_history": ["Normal"] * 10, # Placeholder
        "crops": env.crops,
        "regions": env.regions
    }

def calculate_metrics(env, total_reward, step_data, model_name):
    """Helper to extract metrics from an finished env episode."""
    demand = env.demand_vector
    production = env.production_vector
    
    deficit_vec = np.maximum(demand - production, 0)
    surplus_vec = np.maximum(production - demand, 0)
    
    import_cost = 0.0
    export_revenue = 0.0
    for i, crop in enumerate(env.crops):
        prices = env.econ.get(crop, {'local_price': 200, 'int_price': 180})
        import_cost += deficit_vec[i] * prices['local_price'] * 1.5
        export_revenue += surplus_vec[i] * prices['int_price']
        
    net_economic = export_revenue - import_cost
    demand_sat = np.minimum(production / (demand + 1e-8), 1.0).mean() * 100
    
    return {
        "model_name": model_name,
        "total_reward": total_reward,
        "demand_satisfaction": demand_sat,
        "net_economic": net_economic,
        "export_revenue": export_revenue,
        "import_cost": import_cost,
        "total_deficit": deficit_vec.sum(),
        "allocation_matrix": env.allocation_matrix,
        "step_log": pd.DataFrame(step_data),
        "weather_history": env.weather_history,
        "crops": env.crops,
        "regions": env.regions
    }

# -----------------------------------------------------------------------------
# 4. STREAMLIT UI
# -----------------------------------------------------------------------------

def main():
    st.markdown('<p class="main-header">🌾 AgriPlan AI: Regional Agricultural Planning</p>', unsafe_allow_html=True)
    st.markdown("### Decision-Support System for Reinforcement Learning Agents")

    st.sidebar.header("Simulation Controls")
    config, dqn_model, ppo_model = load_system()
    
    if config is None:
        st.stop()

    model_choice = st.sidebar.selectbox(
        "Select AI Agent",
        ["PPO (Proximal Policy Optimization)", "DQN (Deep Q-Network)", "Greedy Baseline"]
    )
    
    seed_val = st.sidebar.number_input("Random Seed", min_value=0, max_value=100000, value=42)
    run_button = st.sidebar.button("🚀 Run Simulation")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🗺️ Regional Allocation", "🌦️ Weather & Robustness", "🧠 Explainability"])
    
    if run_button:
        with st.spinner(f"Running {model_choice} simulation..."):
            
            if "PPO" in model_choice:
                if ppo_model: results = run_rl_simulation(ppo_model, "PPO", config, seed=int(seed_val))
                else: st.error("PPO Model missing"); st.stop()
            elif "DQN" in model_choice:
                if dqn_model: results = run_rl_simulation(dqn_model, "DQN", config, seed=int(seed_val))
                else: st.error("DQN Model missing"); st.stop()
            else:
                results = run_greedy_simulation(config, seed=int(seed_val))

            # --- TAB 1: DASHBOARD ---
            with tab1:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Demand Satisfaction", f"{results['demand_satisfaction']:.1f}%")
                col2.metric("Net Economic Outcome", f"€{results['net_economic']:,.0f}")
                col3.metric("Export Revenue", f"€{results['export_revenue']:,.0f}")
                col4.metric("Import Cost", f"€{results['import_cost']:,.0f}")
                
                st.subheader("Crop Allocation Distribution")
                alloc_df = pd.DataFrame(
                    results['allocation_matrix'],
                    index=results['regions'],
                    columns=results['crops']
                )
                alloc_melted = alloc_df.reset_index().melt(id_vars='index', var_name='Crop', value_name='Units Allocated')
                alloc_melted.rename(columns={'index': 'Region'}, inplace=True)
                
                fig_alloc = px.bar(alloc_melted, x='Region', y='Units Allocated', color='Crop', barmode='stack')
                st.plotly_chart(fig_alloc, use_container_width=True)

            # --- TAB 2: REGIONAL ALLOCATION ---
            with tab2:
                st.subheader("Regional Land Use Efficiency")
                fig_heat = px.imshow(
                    results['allocation_matrix'],
                    labels=dict(x="Crop", y="Region", color="Units"),
                    x=results['crops'],
                    y=results['regions'],
                    aspect="auto"
                )
                st.plotly_chart(fig_heat, use_container_width=True)

            # --- TAB 3: WEATHER & ROBUSTNESS ---
            with tab3:
                st.subheader("Weather Shock Analysis")
                weather_counts = pd.Series(results['weather_history']).value_counts()
                fig_weather = px.pie(values=weather_counts.values, names=weather_counts.index, title="Weather Events")
                st.plotly_chart(fig_weather, use_container_width=True)

            # --- TAB 4: EXPLAINABILITY ---
            with tab4:
                st.subheader("Decision Factors")
                features = ["Region Progress", "Unit Allocation", "Deficit: Cereals", "Deficit: Pulses", "Deficit: Grass", "Weather Shock"]
                importance = [0.1, 0.15, 0.3, 0.25, 0.1, 0.1] 
                
                fig_shap = go.Figure(go.Bar(x=importance, y=features, orientation='h', marker_color='#2E7D32'))
                fig_shap.update_layout(title="Simulated Feature Impact on Action Selection")
                st.plotly_chart(fig_shap, use_container_width=True)

    else:
        st.info("👈 Select a model and click 'Run Simulation' to start.")

if __name__ == "__main__":
    main()
