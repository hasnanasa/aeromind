import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# ==========================================
# 1. L'ENVIRONNEMENT (Identique à ta version)
# ==========================================
class OrbitMindEnv(gym.Env):
    def __init__(self, dataset_path='orbitmind_full_dataset.csv'):
        super(OrbitMindEnv, self).__init__()
        self.df = pd.read_csv(dataset_path)
        self.satellites = self.df.to_dict('records')
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.MultiDiscrete([3, 3, 2, 2, 2])
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sat = random.choice(self.satellites)
        self._load_satellite_physics()
        return self._get_observation(), {}

    def _load_satellite_physics(self):
        self.time_step = 0
        self.true_battery = self.sat['Capacite_Batterie_Wh']
        self.estimated_battery = self.true_battery 
        self.data_buffer = 0.0
        self.max_solar = self.sat['Puissance_Solaire_Max_W']
        self.survival_power = self.sat['Puissance_Survie_W']
        self.science_power = self.sat['Puissance_Science_W']
        self.science_rate = self.sat['Debit_Science_Mo_s']
        self.antenna_power = self.sat['Puissance_Antenne_W']
        self.antenna_rate = self.sat['Debit_Antenne_Mo_s']
        self.buffer_max = self.sat['Capacite_Buffer_Mo']
        self.orbit_period = self.sat['Periode_Orbite_s']
        self.eclipse_duration = self.sat['Duree_Eclipse_s']
        self.E_min = self.sat['E_min_Wh']
        self.current_attitude = 0 
        self.fault_time = int(self.orbit_period * 0.5)
        self.solar_fault = False
        self.sensor_fault = False

    def is_in_eclipse(self):
        time_in_orbit = self.time_step % self.orbit_period
        return time_in_orbit >= (self.orbit_period - self.eclipse_duration)

    def is_station_visible(self):
        time_in_orbit = self.time_step % self.orbit_period
        return time_in_orbit < 300 

    def _get_observation(self):
        measured_battery = self.true_battery
        if self.sensor_fault:
            measured_battery += self.sat['Biais_Capteur_Wh']
            
        residual = abs(measured_battery - self.estimated_battery)
        anomaly_detected = 1 if residual > self.sat['Seuil_Anomalie_Wh'] else 0
        
        if anomaly_detected:
            ai_view_battery = self.estimated_battery
        else:
            ai_view_battery = measured_battery
            self.estimated_battery = measured_battery 
            
        b_pct = max(0, ai_view_battery) / self.sat['Capacite_Batterie_Wh']
        b_bin = 0 if b_pct < 0.4 else (1 if b_pct < 0.7 else 2)
            
        d_pct = self.data_buffer / self.buffer_max
        d_bin = 0 if d_pct < 0.3 else (1 if d_pct < 0.7 else 2)
        
        sun_visible = 0 if self.is_in_eclipse() else 1
        station_visible = 1 if self.is_station_visible() else 0
        
        return (b_bin, d_bin, sun_visible, station_visible, anomaly_detected)

    def step(self, action):
        reward = 0
        action_was_masked = False

        if action == 2 and not self.is_station_visible():
            reward -= 15.0  
            action = 0
            action_was_masked = True
            
        target_attitude = 0 if action == 0 else 1 
        t_rotation = 0
        E_rotation = 0
        if self.current_attitude != target_attitude:
            t_rotation = 45.0 
            P_rotation = 30.0 + (self.sat['Masse_kg'] * 0.1) 
            E_rotation = P_rotation * (t_rotation / 3600.0)
            
        cost_action = 0.0
        if action == 1: 
            cost_action = self.science_power * ((t_rotation + 10.0) / 3600.0) + E_rotation
        elif action == 2: 
            data_to_send = min(10.0, self.data_buffer)
            t_transmission = data_to_send / self.antenna_rate
            cost_action = self.antenna_power * ((t_rotation + t_transmission) / 3600.0) + E_rotation

        if (self.estimated_battery - cost_action) < self.E_min:
            if not action_was_masked:
                reward -= 5.0 
            action = 0
            target_attitude = 0
            t_rotation = 0 if self.current_attitude == 0 else 45.0
            P_rotation = 30.0 + (self.sat['Masse_kg'] * 0.1)
            E_rotation = P_rotation * (t_rotation / 3600.0)

        self.current_attitude = target_attitude
            
        dt = 15.0 
        power_draw = self.survival_power
        data_gen = 0.0
        data_sent = 0.0
        
        b_pct = self.estimated_battery / self.sat['Capacite_Batterie_Wh']
        
        if action == 1: 
            dt = max(dt, t_rotation + 10.0)
            power_draw += self.science_power
            data_gen = self.science_rate * 10.0 
            if b_pct < 0.5: reward -= 20.0 
            else: reward += 1.0
                
        elif action == 2: 
            data_to_send = min(10.0, self.data_buffer)
            t_transmission = data_to_send / self.antenna_rate
            dt = max(dt, t_rotation + t_transmission)
            power_draw += self.antenna_power
            data_sent = data_to_send
            reward += data_sent * 100.0 
            
        elif action == 0: 
            if b_pct < 0.5: reward += 5.0 
                
        solar_efficiency = 1.0 if self.current_attitude == 0 else 0.4 
        if self.is_in_eclipse() or self.solar_fault:
            solar_gen = 0.0 if self.is_in_eclipse() else (self.max_solar * 0.3)
        else:
            solar_gen = self.max_solar * solar_efficiency
            
        net_power = solar_gen - power_draw
        energy_change = (net_power * (dt / 3600.0)) - E_rotation
        
        self.true_battery += energy_change
        self.estimated_battery += energy_change 
        
        self.true_battery = max(0.0, min(self.true_battery, self.sat['Capacite_Batterie_Wh']))
        self.estimated_battery = max(0.0, min(self.estimated_battery, self.sat['Capacite_Batterie_Wh'])) 
        self.data_buffer = max(0.0, min(self.data_buffer + data_gen - data_sent, self.buffer_max))
        
        if self.data_buffer >= self.buffer_max - 1.0:
            reward -= 50.0 
            
        self.time_step += int(dt)
        
        if self.time_step >= self.fault_time and not self.solar_fault:
            self.solar_fault = True
            self.sensor_fault = True
            
        terminated = False
        if self.true_battery <= self.E_min:
            reward = -1000.0 
            terminated = True
            
        if self.time_step >= (self.orbit_period * 2):
            terminated = True
            
        return self._get_observation(), reward, terminated, False, {"actual_action": action}

# ==========================================
# 2. L'AGENT (Identique à ta version)
# ==========================================
class QLearningAgent:
    def __init__(self):
        self.q_table = {}
        self.alpha = 0.1
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.9995 
        self.epsilon_min = 0.01    

    def get_q(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0, 0.0]
        return self.q_table[state]

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 2)
        return np.argmax(self.get_q(state))

    def learn(self, state, action, reward, next_state):
        q_values = self.get_q(state)
        next_q_values = self.get_q(next_state)
        current_q = q_values[action]
        max_next_q = max(next_q_values)
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# ==========================================
# 3. BOUCLE D'ENTRAÎNEMENT ET ÉVALUATION
# ==========================================
if __name__ == "__main__":
    print("🚀 Entraînement en cours...")
    env = OrbitMindEnv()
    agent = QLearningAgent()
    
    for ep in range(3000): # On réduit un peu pour gagner du temps
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.learn(state, action, reward, next_state)
            state = next_state
            
    print("✅ Entraînement terminé ! Lancement du Test d'Évaluation...")
    
    # ==========================================
    # 4. TEST D'ÉVALUATION SUR UN SATELLITE SPÉCIFIQUE
    # ==========================================
    # On choisit Aalto-1 (ou un autre) pour le diagnostic
    test_env = OrbitMindEnv()
    test_sat = next(s for s in test_env.satellites if "Aalto" in s['Nom_Satellite'])
    test_env.sat = test_sat
    test_env._load_satellite_physics()
    
    agent.epsilon = 0.0 # Mode "Expert" (0% de hasard)
    state = test_env._get_observation()
    done = False
    
    # Variables pour stocker l'historique du test
    t_axis = []
    batt_true, batt_sensor, batt_est = [], [], []
    actions = []
    buffer_levels = []
    anomalies = []
    total_data_sent = 0.0

    while not done:
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
        
        # Enregistrement des données
        t_axis.append(test_env.time_step)
        batt_true.append(test_env.true_battery)
        
        # On recalcule ce que le capteur affiche pour le graphique
        sensor_val = test_env.true_battery + (test_env.sat['Biais_Capteur_Wh'] if test_env.sensor_fault else 0)
        batt_sensor.append(sensor_val)
        
        batt_est.append(test_env.estimated_battery)
        actions.append(info["actual_action"])
        buffer_levels.append(test_env.data_buffer)
        
        # On vérifie si l'anomalie est détectée
        residual = abs(sensor_val - test_env.estimated_battery)
        anomalies.append(1 if residual > test_env.sat['Seuil_Anomalie_Wh'] else 0)
        
        # Compter la data envoyée
        if info["actual_action"] == 2:
            total_data_sent += min(10.0, buffer_levels[-2] if len(buffer_levels)>1 else 0) # approx

        state = next_state

    # ==========================================
    # 5. GÉNÉRATION DES GRAPHIQUES DE PREUVE
    # ==========================================
    fig, axs = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    fig.suptitle(f"Rapport de Mission : {test_sat['Nom_Satellite']}", fontsize=16)

    # Graph 1 : La Bataille des Trois Batteries
    axs[0].plot(t_axis, batt_true, 'b-', linewidth=2, label='Vraie Batterie (Physique)')
    axs[0].plot(t_axis, batt_sensor, 'r--', linewidth=1.5, label='Capteur (Menteur)')
    axs[0].plot(t_axis, batt_est, 'g-.', linewidth=2, label='Estimation IA (Physique)')
    axs[0].axhline(y=test_env.E_min, color='black', linestyle=':', label='E_min (Survie)')
    axs[0].set_ylabel("Énergie (Wh)")
    axs[0].legend(loc='lower left')
    axs[0].set_title("1. Gestion de l'Énergie et Détection du Mensonge")

    # Graph 2 : Le Buffer de Data
    axs[1].plot(t_axis, buffer_levels, color='purple', linewidth=2)
    axs[1].axhline(y=test_env.buffer_max, color='red', linestyle='--', label='Buffer Max')
    axs[1].set_ylabel("Data (Mo)")
    axs[1].legend(loc='upper left')
    axs[1].set_title("2. Accumulation de Data Scientifique")

    # Graph 3 : Les Décisions de l'IA
    axs[2].set_yticks([0, 1, 2])
    axs[2].set_yticklabels(['IDLE (Survie)', 'SCIENCE', 'DOWNLINK'])
    axs[2].plot(t_axis, actions, color='black', drawstyle='steps-pre')
    axs[2].set_ylabel("Action")
    axs[2].set_title("3. Stratégie de l'IA")

    # Graph 4 : Détection d'Anomalie
    axs[3].fill_between(t_axis, anomalies, color='orange', alpha=0.5, step='pre')
    axs[3].set_yticks([0, 1])
    axs[3].set_yticklabels(['Normal', 'Anomalie'])
    axs[3].set_xlabel("Temps (secondes)")
    axs[3].set_title("4. Alerte de Diagnostic (Capteur Défaillant)")

    # Ajout des zones d'éclipse
    for ax in axs:
        # On grise l'éclipse
        for t in t_axis:
            if test_env.is_in_eclipse() if False else (t % test_env.orbit_period) >= (test_env.orbit_period - test_env.eclipse_duration):
                ax.axvspan(t, t+15, color='gray', alpha=0.1)

    plt.tight_layout()
    plt.show()
    
    print(f"🏆 Mission accomplie ! Data totale envoyée : {total_data_sent:.1f} Mo")