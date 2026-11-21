import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# ==========================================
# 1. L'ENVIRONNEMENT PHYSIQUE (ORBITMIND)
# ==========================================
class OrbitMindEnv(gym.Env):
    def __init__(self, dataset_path='orbitmind_full_dataset.csv'):
        super(OrbitMindEnv, self).__init__()
        
        self.df = pd.read_csv(dataset_path)
        self.satellites = self.df.to_dict('records')
        
        self.action_space = spaces.Discrete(3)
        # Observation : [Batterie(0-2), Buffer(0-2), Soleil(0-1), Station(0-1), Anomalie(0-1)]
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
            
        # Nouvelle perception de la batterie (Marge par rapport à E_min)
        margin = (ai_view_battery - self.E_min) / self.sat['Capacite_Batterie_Wh']
        if margin < 0.15:
            b_bin = 0  # Critique (Moins de 15% de marge)
        elif margin < 0.5:
            b_bin = 1  # Moyen (15% à 50% de marge)
        else:
            b_bin = 2  # Excellent (Plus de 50% de marge)
            
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

        # Filtre de Sécurité (Anti-Suicide Physique)
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
        
        # --- NOUVELLES RÉCOMPENSES STRICTES ---
        if action == 1: # SCIENCE
            dt = max(dt, t_rotation + 10.0)
            power_draw += self.science_power
            data_gen = self.science_rate * 10.0 
            
            if b_pct < 0.5:
                reward -= 20.0  # Punition sévère : Ne fais pas de science si batterie faible !
            else:
                reward += 1.0   # Petit bonus pour la génération de data
                
        elif action == 2: # DOWNLINK
            data_to_send = min(10.0, self.data_buffer)
            t_transmission = data_to_send / self.antenna_rate
            dt = max(dt, t_rotation + t_transmission)
            power_draw += self.antenna_power
            data_sent = data_to_send
            
            reward += data_sent * 100.0 # Récompense massive pour l'envoi de data
            
        elif action == 0: # IDLE
            if b_pct < 0.5:
                reward += 5.0 # Récompense le repos quand on a besoin de recharger
                
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
        
        # Punition Buffer Plein
        if self.data_buffer >= self.buffer_max - 1.0:
            reward -= 50.0 
            
        self.time_step += int(dt)
        
        if self.time_step >= self.fault_time and not self.solar_fault:
            self.solar_fault = True
            self.sensor_fault = True
            
        terminated = False
        if self.true_battery <= self.E_min:
            reward = -1000.0 # Mort instantanée
            terminated = True
            
        if self.time_step >= (self.orbit_period * 2):
            terminated = True
            
        return self._get_observation(), reward, terminated, False, {"actual_action": action}

# ==========================================
# 2. L'AGENT INTELLIGENT (Q-LEARNING)
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
# 3. BOUCLE D'ENTRAÎNEMENT ET TEST
# ==========================================
if __name__ == "__main__":
    print("🚀 Démarrage de l'entraînement (Stratégie Stricte)...")
    env = OrbitMindEnv()
    agent = QLearningAgent()
    
    episodes = 5000
    rewards_history = []
    
    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            
        rewards_history.append(total_reward)
        
        if (ep + 1) % 1000 == 0:
            print(f"Épisode {ep+1}/{episodes} | Récompense: {total_reward:.1f} | Epsilon: {agent.epsilon:.3f}")
            
    print("\n✅ Entraînement terminé !")
    
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history, alpha=0.3, color='blue', label='Récompense')
    moving_avg = np.convolve(rewards_history, np.ones(200)/200, mode='valid')
    plt.plot(moving_avg, color='red', linewidth=2, label='Tendance')
    plt.title("Apprentissage de l'IA (Punition Suicide + Repos Récompensé)")
    plt.legend()
    plt.show()

    # Test sur un satellite
    print("\n🎬 --- TEST DE L'IA SUR UN SATELLITE ---")
    test_env = OrbitMindEnv()
    # On prend Aalto-1 pour tester la petite batterie
    test_sat = next(s for s in test_env.satellites if "Aalto" in s['Nom_Satellite'])
    test_env.sat = test_sat
    test_env._load_satellite_physics()
    
    state = test_env._get_observation()
    done = False
    history_battery, history_actions, time_axis = [], [], []
    agent.epsilon = 0.0 
    
    while not done:
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
        
        history_battery.append(test_env.true_battery)
        history_actions.append(info["actual_action"]) 
        time_axis.append(test_env.time_step)
        state = next_state

    # Graphique de Test avec 2 sous-graphiques (Batterie + Actions)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Sous-graphique 1 : Batterie
    ax1.plot(time_axis, history_battery, color='blue', linewidth=2, label='Batterie')
    ax1.axhline(y=test_env.E_min, color='red', linestyle='--', label='E_min')
    ax1.set_ylabel("Batterie (Wh)")
    ax1.legend()
    
    # Sous-graphique 2 : Actions
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['IDLE (Repos)', 'SCIENCE', 'DOWNLINK'])
    ax2.plot(time_axis, history_actions, color='black', drawstyle='steps-pre')
    
    plt.title(f"Décisions de l'IA pour : {test_sat['Nom_Satellite']}")
    plt.xlabel("Temps (secondes)")
    plt.tight_layout()
    plt.show()