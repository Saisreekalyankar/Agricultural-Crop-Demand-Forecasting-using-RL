import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# RL & Optimization
import gymnasium as gym
from gymnasium import spaces

def load_and_process_config(data_path="processed_data/france_agriculture_combined_raw.csv"):
    """
    Processes raw combined data into simulation configuration.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}.")

    df = pd.read_csv(data_path)

    # Filter relevant crops
    relevant_categories = [
        'Cereals for the production of grain (including seed)',
        'Dry pulses and protein crops for the production of grain (including seed and mixtures of cereals and pulses)',
        'Permanent grassland'
    ]
    sim_df = df[df['eurostat_crop_category'].isin(relevant_categories)].copy()

    # --- Phase 2: Regional Structure & Stats ---
    region_crop_stats = sim_df.groupby(['region_normalised', 'eurostat_crop_category']).agg(
        avg_area_ha=('eurostat_area_ha', 'mean'),
        count_years=('year', 'count')
    ).reset_index()

    # --- Yield Mapping ---
    national_yearly = sim_df.dropna(subset=['fao_total_production_tonnes', 'fao_total_harvested_area_ha']).copy()
    national_yearly['yield_kg_ha'] = (national_yearly['fao_total_production_tonnes'] * 1000) / national_yearly['fao_total_harvested_area_ha']
    
    national_crop_yield = national_yearly.groupby('eurostat_crop_category')['yield_kg_ha'].mean().reset_index()
    national_crop_yield.columns = ['eurostat_crop_category', 'national_avg_yield']

    region_crop_stats = region_crop_stats.merge(national_crop_yield, on='eurostat_crop_category', how='left')
    region_crop_stats['base_yield_kg_ha'] = region_crop_stats['national_avg_yield']

    global_avg_yield = region_crop_stats['base_yield_kg_ha'].mean()
    region_crop_stats['base_yield_kg_ha'] = region_crop_stats['base_yield_kg_ha'].fillna(global_avg_yield)

    # --- Weather Event Modelling ---
    weather_df = sim_df[['region_normalised', 'year', 'annual_mean_temperature_c', 'annual_precipitation_mm']].dropna()

    weather_stats = weather_df.groupby('region_normalised').agg(
        mean_temp=('annual_mean_temperature_c', 'mean'),
        std_temp=('annual_mean_temperature_c', 'std'),
        mean_precip=('annual_precipitation_mm', 'mean'),
        std_precip=('annual_precipitation_mm', 'std'),
        count_years=('year', 'count')
    ).reset_index()

    weather_stats['cold_threshold'] = weather_stats['mean_temp'] - weather_stats['std_temp']
    weather_stats['low_rain_threshold'] = weather_stats['mean_precip'] - weather_stats['std_precip']

    def calc_empirical_probs(row):
        region_data = weather_df[weather_df['region_normalised'] == row['region_normalised']]
        if region_data.empty or row['count_years'] == 0:
            return 0.16, 0.16
        
        n_cold = (region_data['annual_mean_temperature_c'] < row['cold_threshold']).sum()
        n_dry = (region_data['annual_precipitation_mm'] < row['low_rain_threshold']).sum()
        
        p_cold = n_cold / row['count_years']
        p_rain = n_dry / row['count_years']
        
        return max(0.05, min(0.5, p_cold)), max(0.05, min(0.5, p_rain))

    probs = weather_stats.apply(calc_empirical_probs, axis=1, result_type='expand')
    weather_stats['p_cold'], weather_stats['p_low_rain'] = probs[0], probs[1]

    # --- Crop Sensitivities ---
    crop_sensitivity = {
        'Cereals for the production of grain (including seed)': {'cold': 0.10, 'rain': 0.15},
        'Dry pulses and protein crops for the production of grain (including seed and mixtures of cereals and pulses)': {'cold': 0.05, 'rain': 0.20},
        'Permanent grassland': {'cold': 0.02, 'rain': 0.25}
    }

    region_crop_stats['cold_sens'] = region_crop_stats['eurostat_crop_category'].map(lambda x: crop_sensitivity.get(x, {'cold':0.1, 'rain':0.1})['cold'])
    region_crop_stats['rain_sens'] = region_crop_stats['eurostat_crop_category'].map(lambda x: crop_sensitivity.get(x, {'cold':0.1, 'rain':0.1})['rain'])

    # --- Economic Parameters ---
    economic_params = {
        'Cereals for the production of grain (including seed)': {'local_price': 220, 'int_price': 200},
        'Dry pulses and protein crops for the production of grain (including seed and mixtures of cereals and pulses)': {'local_price': 350, 'int_price': 320},
        'Permanent grassland': {'local_price': 150, 'int_price': 140}
    }

    # --- Discretization ---
    regions = region_crop_stats['region_normalised'].unique().tolist()
    crops = region_crop_stats['eurostat_crop_category'].unique().tolist()

    region_total_area = region_crop_stats.groupby('region_normalised')['avg_area_ha'].sum()
    total_france_area = region_total_area.sum()

    n_units_total = 100
    region_units = (region_total_area / total_france_area * n_units_total).round().astype(int)

    diff = n_units_total - region_units.sum()
    if diff != 0:
        region_units.iloc[0] += diff

    config = {
        'regions': regions,
        'crops': crops,
        'region_crop_stats': region_crop_stats,
        'weather_stats': weather_stats,
        'region_units': region_units.to_dict(),
        'n_units_total': n_units_total,
        'crop_sensitivity': crop_sensitivity,
        'economic_params': economic_params
    }

    return config


class FranceAgriEnv(gym.Env):
    def __init__(self, config=None, seed=None):
        super().__init__()

        self.config = config if config else load_and_process_config()
        self.regions = self.config['regions']
        self.crops = self.config['crops']
        self.region_units = self.config['region_units']
        self.stats = self.config['region_crop_stats']
        self.weather_stats = self.config['weather_stats']
        
        self.econ = self.config.get('economic_params', {
            'Cereals for the production of grain (including seed)': {'local_price': 220, 'int_price': 200},
            'Dry pulses and protein crops for the production of grain (including seed and mixtures of cereals and pulses)': {'local_price': 350, 'int_price': 320},
            'Permanent grassland': {'local_price': 150, 'int_price': 140}
        })

        self.n_regions = len(self.regions)
        self.n_crops = len(self.crops)

        self.region_to_idx = {r: i for i, r in enumerate(self.regions)}
        self.crop_to_idx = {c: i for i, c in enumerate(self.crops)}
        self.idx_to_crop = {i: c for c, i in self.crop_to_idx.items()}

        self.action_space = spaces.Discrete(self.n_crops)

        obs_size = 1 + 1 + self.n_crops + 1
        self.observation_space = spaces.Box(low=-10, high=10, shape=(obs_size,), dtype=np.float32)

        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_region_idx = 0
        self.units_allocated_in_current_region = 0
        self.max_units_in_current_region = self.region_units[self.regions[0]]

        self.demand_vector = np.zeros(self.n_crops)
        for i, crop in enumerate(self.crops):
            national_prod = self.config.get('national_avg_prod', {}).get(crop, 10000)
            noise = max(0.1, self.rng.normal(1.0, 0.05))
            self.demand_vector[i] = national_prod * 1.05 * noise 

        self.demand_vector[self.demand_vector == 0] = 1.0
        self.production_vector = np.zeros(self.n_crops)
        
        self.allocation_matrix = np.zeros((self.n_regions, self.n_crops))
        self.weather_history = []
        self.weather_shock_active = False

        return self._get_obs(), {}

    def _get_obs(self):
        region_norm = self.current_region_idx / max(1, self.n_regions)
        units_norm = self.units_allocated_in_current_region / max(1, self.max_units_in_current_region)

        deficit = np.maximum(self.demand_vector - self.production_vector, 0)
        deficit_norm = deficit / (self.demand_vector + 1e-8)
        deficit_norm = np.clip(deficit_norm, 0, 5.0)

        weather_norm = 1.0 if self.weather_shock_active else 0.0

        obs = np.array([region_norm, units_norm, *deficit_norm, weather_norm], dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

        return obs

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = int(action.item())
        elif hasattr(action, '__len__') and len(action) == 1:
            action = int(action[0])
        else:
            action = int(action)

        if action < 0 or action >= self.n_crops:
            action = 0

        crop_name = self.idx_to_crop[action]
        region_name = self.regions[self.current_region_idx]

        self.allocation_matrix[self.current_region_idx, action] += 1

        mask = (self.stats['region_normalised'] == region_name) & \
               (self.stats['eurostat_crop_category'] == crop_name)
        subset = self.stats[mask]

        if subset.empty:
            base_yield = 5000
            unit_area = 100
            cold_sens = 0.1
            rain_sens = 0.1
        else:
            base_yield = subset['base_yield_kg_ha'].values[0]
            region_total_area = self.stats[self.stats['region_normalised'] == region_name]['avg_area_ha'].sum()
            if np.isnan(region_total_area) or region_total_area == 0:
                region_total_area = 10000
            unit_area = region_total_area / max(1, self.max_units_in_current_region)
            cold_sens = subset['cold_sens'].values[0]
            rain_sens = subset['rain_sens'].values[0]

        w_stats = self.weather_stats[self.weather_stats['region_normalised'] == region_name]
        p_cold = w_stats['p_cold'].values[0] if not w_stats.empty else 0.16
        p_rain = w_stats['p_low_rain'].values[0] if not w_stats.empty else 0.16

        is_cold = self.rng.random() < p_cold
        is_dry = self.rng.random() < p_rain
        
        current_weather_state = "Normal"
        if is_cold and is_dry: current_weather_state = "Adverse_Double"
        elif is_cold: current_weather_state = "Adverse_Cold"
        elif is_dry: current_weather_state = "Adverse_Dry"
        self.weather_history.append(current_weather_state)

        loss_cold = cold_sens if is_cold else 0.0
        loss_rain = rain_sens if is_dry else 0.0
        weather_factor = 1.0 - loss_cold - loss_rain
        weather_factor = max(0.1, min(1.0, weather_factor))

        produced_kg = unit_area * base_yield * weather_factor
        produced_tonnes = produced_kg / 1000.0

        if np.isnan(produced_tonnes):
            produced_tonnes = 0.0

        self.production_vector[action] += produced_tonnes

        ratio = self.production_vector[action] / (self.demand_vector[action] + 1e-8)
        demand_met_ratio = min(1.0, max(0.0, ratio))

        reward = float(demand_met_ratio * 5.0)

        if weather_factor < 1.0:
            reward -= 2.0

        self.units_allocated_in_current_region += 1
        done = False
        truncated = False

        if self.units_allocated_in_current_region >= self.max_units_in_current_region:
            self.current_region_idx += 1
            self.units_allocated_in_current_region = 0

            if self.current_region_idx >= self.n_regions:
                done = True
                total_deficit = np.maximum(self.demand_vector - self.production_vector, 0)
                total_surplus = np.maximum(self.production_vector - self.demand_vector, 0)

                import_cost = 0.0
                export_revenue = 0.0
                for i, crop in enumerate(self.crops):
                    prices = self.econ.get(crop, {'local_price': 200, 'int_price': 180})
                    import_cost += total_deficit[i] * prices['local_price'] * 1.5
                    export_revenue += total_surplus[i] * prices['int_price']
                
                net_economic = export_revenue - import_cost
                reward += float(net_economic / 100000.0) 
                
            else:
                self.max_units_in_current_region = self.region_units[self.regions[self.current_region_idx]]

        self.weather_shock_active = (is_cold or is_dry)

        if np.isinf(reward) or np.isnan(reward):
            reward = 0.0

        return self._get_obs(), reward, done, truncated, {}

def run_greedy_baseline(config, seed=42):
    """
    Runs a greedy baseline simulation for comparison.
    """
    env = FranceAgriEnv(config=config, seed=seed)
    env.reset()
    
    # Calculate Expected Profit per Unit for each Crop in each Region
    expected_profits = np.zeros((env.n_regions, env.n_crops))

    for r_idx, region in enumerate(env.regions):
        region_area = env.stats[env.stats['region_normalised'] == region]['avg_area_ha'].sum()
        unit_area = region_area / env.region_units[region]

        for c_idx, crop in enumerate(env.crops):
            mask = (env.stats['region_normalised'] == region) & (env.stats['eurostat_crop_category'] == crop)
            subset = env.stats[mask]
            if not subset.empty:
                base_yield = subset['base_yield_kg_ha'].values[0]
                expected_profits[r_idx, c_idx] = base_yield * unit_area
            else:
                expected_profits[r_idx, c_idx] = 0

    # Greedy Allocation
    for r_idx in range(env.n_regions):
        units_available = env.region_units[env.regions[r_idx]]
        sorted_crops = np.argsort(expected_profits[r_idx])[::-1]

        for c_idx in sorted_crops:
            if units_available <= 0:
                break
            env.allocation_matrix[r_idx, c_idx] = units_available
            units_available = 0
            
    # Simulate production based on this allocation
    total_prod = np.zeros(env.n_crops)
    for r_idx, region in enumerate(env.regions):
        p_cold = env.weather_stats[env.weather_stats['region_normalised'] == region]['p_cold'].values[0] if not env.weather_stats[env.weather_stats['region_normalised'] == region].empty else 0.16
        p_rain = env.weather_stats[env.weather_stats['region_normalised'] == region]['p_low_rain'].values[0] if not env.weather_stats[env.weather_stats['region_normalised'] == region].empty else 0.16
        
        region_area = env.stats[env.stats['region_normalised'] == region]['avg_area_ha'].sum()
        unit_area = region_area / max(1, env.region_units[region])

        for c_idx, crop in enumerate(env.crops):
            n_units = int(env.allocation_matrix[r_idx, c_idx])
            if n_units <= 0: continue

            mask = (env.stats['region_normalised'] == region) & (env.stats['eurostat_crop_category'] == crop)
            subset = env.stats[mask]
            if subset.empty: continue

            base_yield = subset['base_yield_kg_ha'].values[0]
            cold_sens = subset['cold_sens'].values[0]
            rain_sens = subset['rain_sens'].values[0]

            # Simulate weather for these units
            rand_cold = env.rng.random(n_units) < p_cold
            rand_rain = env.rng.random(n_units) < p_rain

            wf = 1.0 - (cold_sens * rand_cold) - (rain_sens * rand_rain)
            wf = np.maximum(0.1, wf)

            total_prod[c_idx] += np.sum(unit_area * base_yield * wf) / 1000.0
            
    env.production_vector = total_prod
    return env