# AgriPlan AI: Regional Agricultural Planning Using Reinforcement Learning

**Researcher:** Saisree Kalyankar  
**Program:** MSc Artificial Intelligence  
**Project Type:** Dissertation / Decision-Support System  

## Project Overview

This project addresses the complex challenge of **regional crop allocation** under stochastic weather conditions. Formulated as a **Markov Decision Process (MDP)**, the system utilizes Deep Reinforcement Learning (DRL) to optimize land use across French agricultural regions. 

The primary objective is to maximize **demand satisfaction** and **net economic outcome** while minimizing the risks associated with adverse weather events (cold spells and droughts). The project compares advanced RL agents (**PPO**, **DQN**) against traditional heuristic baselines (**Greedy**) to demonstrate the efficacy of adaptive AI in agricultural planning.

### Key Research Contributions
*   **Dynamic Ensemble Selection Logic:** Adapts concepts from intrusion detection to agricultural risk management.
*   **Explainable AI (XAI):** Integrates interpretability techniques to support decision-makers in understanding model predictions.
*   **Stochastic Simulation Environment:** Uses historical ERA5 weather data and FAOSTAT/Eurostat agricultural metrics to create a realistic, data-driven simulation.

---

## Tech Stack & Dependencies

| Category | Technologies |
| :--- | :--- |
| **Core Language** | Python 3.9+ |
| **Reinforcement Learning** | Stable Baselines3 (PPO, DQN), Gymnasium |
| **Data Processing** | Pandas, NumPy, Open-Meteo API |
| **Visualization** | Streamlit, Plotly, Matplotlib, Seaborn |
| **Explainability** | SHAP (Simulated/Proxy), LIME concepts |
| **Optimization** | Google OR-Tools |

---

## Project Structure

```text
AgriPlan_AI/
├── app.py                  # Streamlit Dashboard Entry Point
├── agri_sim.py             # Core Environment (FranceAgriEnv) & Config Logic
├── requirements.txt        # Python Dependencies
├── processed_data/         # Cleaned CSVs from FAOSTAT/Eurostat
│   ├── france_agriculture_combined_raw.csv
│   └── ...
├── models/                 # Trained RL Agents
│   ├── dqn_final.zip
│   └── ppo_final.zip
├── logs/                   # Training Logs & Evaluation Results
└── README.md

## How to Run the Code

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
streamlit run app.py
