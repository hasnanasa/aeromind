import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import pickle
# ==========================================
# 1. L'ENVIRONNEMENT PHYSIQUE (ORBITMIND)
# ==========================================
class OrbitMindEnv(gym.Env):
    def __init__(self, dataset_path='orbitmind_full_dataset.csv'):
        super(OrbitMindEnv, self).__init__()
        self.df = pd.read_csv(dataset_path)
        self.satellites = self.df.to_dict('records')
        
        self.action_space = spaces.Discrete(3)
        
        # NOUVEAU : Ajout de la 6ème dimension (Temps avant l'éclipse)
        # [Batterie, Buffer, Soleil, Station, Anomalie, Eclipse]
        self.observation_space = spaces.MultiDiscrete([3, 3, 2, 2, 2, 3])
        
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
            
        # Marge d'énergie UTILE
        max_usable_energy = self.sat['Capacite_Batterie_Wh'] - self.E_min
        current_usable_energy = ai_view_battery - self.E_min
        margin_pct = max(0.0, min(1.0, current_usable_energy / max_usable_energy)) if max_usable_energy > 0 else 0
        
        if margin_pct < 0.25: b_bin = 0  
        elif margin_pct < 0.60: b_bin = 1  
        else: b_bin = 2  
            
        d_pct = self.data_buffer / self.buffer_max
        d_bin = 0 if d_pct < 0.3 else (1 if d_pct < 0.7 else 2)
        
        sun_visible = 0 if self.is_in_eclipse() else 1
        station_visible = 1 if self.is_station_visible() else 0
        
        # NOUVEAU : Perception du temps avant l'éclipse
        time_in_orbit = self.time_step % self.orbit_period
        if time_in_orbit < (self.orbit_period - self.eclipse_duration):
            time_to_eclipse = (self.orbit_period - self.eclipse_duration) - time_in_orbit
            if time_to_eclipse > 1800: eclipse_bin = 2 # > 30 mins (Sûr)
            else: eclipse_bin = 1 # < 30 mins (Imminent)
        else:
            eclipse_bin = 0 # Dans l'éclipse
            
        return (b_bin, d_bin, sun_visible, station_visible, anomaly_detected, eclipse_bin)

    def step(self, action):
        reward = 0
        action_was_masked = False

        if action == 2 and not self.is_station_visible():
            reward -= 20.0  
            action = 0
            action_was_masked = True
            
        target_attitude = 0 if action == 0 else 1 
        t_rotation = 0
        E_rotation = 0

        if action == 2 and not self.is_station_visible():
            reward -= 20.0  
            action = 0
            action_was_masked = True

            
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

        # État de l'éclipse
        time_in_orbit = self.time_step % self.orbit_period
        if time_in_orbit < (self.orbit_period - self.eclipse_duration):
            time_to_eclipse = (self.orbit_period - self.eclipse_duration) - time_in_orbit
            is_eclipse_imminent = time_to_eclipse < 1800
        else:
            is_eclipse_imminent = True
            
        # FILTRE PRÉDICTIF (Anti-Suicide)
        energy_needed_eclipse = self.survival_power * (self.eclipse_duration / 3600.0)
        if (self.estimated_battery - cost_action) < (self.E_min + energy_needed_eclipse):
            if not action_was_masked: reward -= 50.0 
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
        in_eclipse = self.is_in_eclipse()
        
        # --- HIÉRARCHIE DES RÉCOMPENSES ---
        if action == 1: # SCIENCE
            dt = max(dt, t_rotation + 10.0)
            power_draw += self.science_power
            data_gen = self.science_rate * 10.0 
            if self.data_buffer > (self.buffer_max * 0.8):
                reward -= 40.0
            elif is_eclipse_imminent:
                reward -= 50.0 # Interdit de faire de la science trop près de l'éclipse
            else:
                reward += 2.0 # Encourage à générer de la data
                
        elif action == 2: # DOWNLINK
            data_to_send = min(10.0, self.data_buffer)
            t_transmission = data_to_send / self.antenna_rate
            dt = max(dt, t_rotation + t_transmission)
            power_draw += self.antenna_power
            data_sent = data_to_send
            reward += data_sent * 500.0 # Objectif principal
            
        elif action == 0: # IDLE
            if self.is_station_visible() and self.data_buffer > 0:
                reward -= 100.0 # Punition si rate la station
            elif is_eclipse_imminent:
                reward += 10.0 # Encourage à recharger avant l'éclipse
                
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
            reward = -10000.0 
            terminated = True
            
        if self.time_step >= (self.orbit_period * 2):
            terminated = True
            
        return self._get_observation(), reward, terminated, False, {"actual_action": action}

# ==========================================
# 2. L'AGENT INTELLIGENT
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
# 3. GRAND TEST DE ROBUSTESSE
# ==========================================
if __name__ == "__main__":
    print("🚀 Entraînement intensif (15 000 épisodes)...")
    env = OrbitMindEnv()
    agent = QLearningAgent()
    # On augmente à 15000 épisodes pour qu'elle apprenne la stratégie complexe
    for ep in range(15000): 
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.learn(state, action, reward, next_state)
            state = next_state
            
    print("✅ IA prête. Lancement du test sur 4 satellites extrêmes...\n")
    
    test_env = OrbitMindEnv()
    
    sat_cubesat = min(test_env.satellites, key=lambda x: x['Capacite_Batterie_Wh']) 
    sat_geant = max(test_env.satellites, key=lambda x: x['Capacite_Batterie_Wh']) 
    sat_rapide = max(test_env.satellites, key=lambda x: x['Debit_Antenne_Mo_s'])   
    sat_lent = min(test_env.satellites, key=lambda x: x['Debit_Antenne_Mo_s'])     
    
    satellites_a_tester = [sat_cubesat, sat_geant, sat_rapide, sat_lent]
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Test de Robustesse : L'IA s'adapte-t-elle à tous les satellites ?", fontsize=16)
    axs = axs.flatten()
    
    agent.epsilon = 0.0 
    
    for i, sat in enumerate(satellites_a_tester):
        test_env.sat = sat
        test_env._load_satellite_physics()
        
        state = test_env._get_observation()
        done = False
        
        t_axis, batt_true, actions = [], [], []
        total_data = 0.0
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            
            t_axis.append(test_env.time_step)
            batt_true.append(test_env.true_battery)
            actions.append(info["actual_action"])
            
            if info["actual_action"] == 2:
                total_data += min(10.0, test_env.data_buffer)
                
            state = next_state
            
        ax = axs[i]
        ax.plot(t_axis, batt_true, color='blue', linewidth=2, label='Batterie')
        ax.axhline(y=test_env.E_min, color='red', linestyle='--', label='E_min')
        
        for j in range(len(actions)-1):
            x_vals = [t_axis[j], t_axis[j+1]]
            y_vals = [batt_true[j], batt_true[j+1]]
            if actions[j] == 1: 
                ax.plot(x_vals, y_vals, color='green', linewidth=3) 
            elif actions[j] == 2: 
                ax.plot(x_vals, y_vals, color='orange', linewidth=3) 
            else: 
                ax.plot(x_vals, y_vals, color='gray', linewidth=1) 
                
        statut = "✅ SURVIT" if test_env.true_battery > test_env.E_min else "💀 MORT"
        ax.set_title(f"{sat['Nom_Satellite'][:20]}\nCap: {sat['Capacite_Batterie_Wh']:.0f}Wh | {statut} | Data: {total_data:.0f} Mo", fontsize=10)
        ax.set_ylabel("Énergie (Wh)")
        ax.set_xlabel("Temps (s)")
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            legend_elements = [
                mpatches.Patch(color='green', label='SCIENCE'),
                mpatches.Patch(color='orange', label='DOWNLINK'),
                mpatches.Patch(color='gray', label='IDLE (Survie)'),
                plt.Line2D([0], [0], color='blue', label='Batterie'),
                plt.Line2D([0], [0], color='red', linestyle='--', label='E_min')
            ]
            ax.legend(handles=legend_elements, loc='lower left', fontsize=8)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.show()

    # --- SAUVEGARDE DU CERVEAU DE L'IA ---
    with open('orbitmind_brain.pkl', 'wb') as f:
        pickle.dump(agent.q_table, f)
    print("🧠 Cerveau de l'IA sauvegardé dans 'orbitmind_brain.pkl' !")

    # ==========================================
# 4. STRESS TEST : SCÉNARIO DE L'IMPRÉVISIBLE
# ==========================================
    print("\n💀 --- LANCEMENT DU STRESS TEST (Pannes Imprévisibles) ---")
    
    # On crée un environnement modifié pour le test
    class StressTestEnv(OrbitMindEnv):
        def _load_satellite_physics(self):
            super()._load_satellite_physics()
            # 1. L'éclipse est 20% plus longue que prévu
            self.eclipse_duration = int(self.eclipse_duration * 1.2)
            # 2. Le moment de la panne est aléatoire (entre 25% et 75%)
            self.fault_time = int(self.orbit_period * random.uniform(0.25, 0.75))
            # 3. Le biais du capteur est aléatoire (+100 à +400 Wh)
            self.random_sensor_bias = random.uniform(100.0, 400.0)
            # 4. Panne de comms : L'antenne consomme mais n'envoie rien
            self.comms_broken = random.choice([True, False])

        def _get_observation(self):
            # On utilise la logique parente, mais avec notre biais aléatoire
            measured_battery = self.true_battery
            if self.sensor_fault:
                measured_battery += self.random_sensor_bias
                
            residual = abs(measured_battery - self.estimated_battery)
            anomaly_detected = 1 if residual > self.sat['Seuil_Anomalie_Wh'] else 0
            
            if anomaly_detected:
                ai_view_battery = self.estimated_battery
            else:
                ai_view_battery = measured_battery
                self.estimated_battery = measured_battery 
                
            max_usable_energy = self.sat['Capacite_Batterie_Wh'] - self.E_min
            current_usable_energy = ai_view_battery - self.E_min
            margin_pct = max(0.0, min(1.0, current_usable_energy / max_usable_energy)) if max_usable_energy > 0 else 0
            
            if margin_pct < 0.25: b_bin = 0  
            elif margin_pct < 0.60: b_bin = 1  
            else: b_bin = 2  
                
            d_pct = self.data_buffer / self.buffer_max
            d_bin = 0 if d_pct < 0.3 else (1 if d_pct < 0.7 else 2)
            
            sun_visible = 0 if self.is_in_eclipse() else 1
            station_visible = 1 if self.is_station_visible() else 0
            
            time_in_orbit = self.time_step % self.orbit_period
            if time_in_orbit < (self.orbit_period - self.eclipse_duration):
                time_to_eclipse = (self.orbit_period - self.eclipse_duration) - time_in_orbit
                if time_to_eclipse > 1800: eclipse_bin = 2 
                else: eclipse_bin = 1 
            else:
                eclipse_bin = 0 
                
            return (b_bin, d_bin, sun_visible, station_visible, anomaly_detected, eclipse_bin)

        def step(self, action):
            # On intercepte l'action pour simuler la panne de comms
            info = super().step(action)
            if self.comms_broken and self.sensor_fault: # La panne de comms arrive avec la panne générale
                self.data_buffer += min(10.0, self.data_buffer) # On remet la data qu'on croyait avoir envoyée
            return info

    stress_env = StressTestEnv()
    
    # On choisit un satellite moyen pour le test
    test_sat = next(s for s in stress_env.satellites if "Aalto" in s['Nom_Satellite'])
    stress_env.sat = test_sat
    stress_env._load_satellite_physics()
    
    print(f"Satellite : {test_sat['Nom_Satellite']}")
    print(f"Panne prévue à t = {stress_env.fault_time}s")
    print(f"Biais du capteur : +{stress_env.random_sensor_bias:.1f} Wh")
    print(f"Antenne cassée : {stress_env.comms_broken}")
    
    state = stress_env._get_observation()
    done = False
    t_axis, batt_true, actions, anomalies = [], [], [], []
    
    agent.epsilon = 0.0 
    
    while not done:
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, info = stress_env.step(action)
        done = terminated or truncated
        
        t_axis.append(stress_env.time_step)
        batt_true.append(stress_env.true_battery)
        actions.append(info["actual_action"])
        
        # On vérifie si l'IA a détecté l'anomalie
        residual = abs((stress_env.true_battery + (stress_env.random_sensor_bias if stress_env.sensor_fault else 0)) - stress_env.estimated_battery)
        anomalies.append(1 if residual > stress_env.sat['Seuil_Anomalie_Wh'] else 0)
        
        state = next_state

    # --- AFFICHAGE DU STRESS TEST ---
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Stress Test (Pannes Imprévisibles) : {test_sat['Nom_Satellite']}", fontsize=16)

    # Graph 1 : Batterie
    axs[0].plot(t_axis, batt_true, color='blue', linewidth=2, label='Vraie Batterie')
    axs[0].axhline(y=stress_env.E_min, color='red', linestyle='--', label='E_min')
    axs[0].axvline(x=stress_env.fault_time, color='black', linestyle=':', linewidth=2, label='Panne Aléatoire')
    axs[0].set_ylabel("Énergie (Wh)")
    axs[0].legend()
    axs[0].set_title("1. État de la Batterie")

    # Graph 2 : Actions
    axs[1].set_yticks([0, 1, 2])
    axs[1].set_yticklabels(['IDLE (Survie)', 'SCIENCE', 'DOWNLINK'])
    axs[1].plot(t_axis, actions, color='black', drawstyle='steps-pre')
    axs[1].set_ylabel("Action")
    axs[1].set_title("2. Décisions de l'IA")

    # Graph 3 : Anomalie
    axs[2].fill_between(t_axis, anomalies, color='orange', alpha=0.5, step='pre')
    axs[2].set_yticks([0, 1])
    axs[2].set_yticklabels(['Normal', 'Anomalie'])
    axs[2].set_xlabel("Temps (secondes)")
    axs[2].set_title("3. Détection du Mensonge (Diagnostic)")

    plt.tight_layout()
    plt.show()



    # ==========================================
# 6. TEST FINAL : SATELLITES FANTÔMES (DONNÉES INÉDITES)
# ==========================================
    print("\n👻 --- TEST FINAL SUR SATELLITES FANTÔMES (Zéro-Shot Learning) ---")
    
    def generate_random_satellite(idx):
        """Génère un satellite physiquement valide mais totalement inédit"""
        mass = random.uniform(10.0, 300.0)
        survival_power = 10.0 + (mass * 0.05)
        
        # On s'assure que les panneaux solaires peuvent recharger
        max_solar = random.uniform(survival_power * 1.5, 400.0)
        
        # Calcul de E_min (selon notre formule)
        eclipse_duration = 2100.0
        e_min = (survival_power * (eclipse_duration / 3600.0)) + 100.0
        
        return {
            'Nom_Satellite': f'Fantome-{idx}',
            'Masse_kg': mass,
            'Capacite_Batterie_Wh': random.uniform(e_min * 2, 3000.0),
            'Puissance_Solaire_Max_W': max_solar,
            'Puissance_Survie_W': survival_power,
            'Puissance_Science_W': random.uniform(20.0, 80.0),
            'Debit_Science_Mo_s': random.uniform(0.5, 3.0),
            'Puissance_Antenne_W': random.uniform(15.0, 120.0),
            'Debit_Antenne_Mo_s': random.uniform(0.5, 10.0),
            'Capacite_Buffer_Mo': random.uniform(100.0, 600.0),
            'Periode_Orbite_s': 5677.0,
            'Duree_Eclipse_s': eclipse_duration,
            'E_min_Wh': e_min,
            'Seuil_Anomalie_Wh': 100.0,
            'Biais_Capteur_Wh': random.uniform(100.0, 350.0)
        }

    # On génère 4 nouveaux satellites inédits
    fantomes = [generate_random_satellite(i) for i in range(4)]
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Test Zéro-Shot : L'IA sur des Satellites Totalement Inédits", fontsize=16)
    axs = axs.flatten()
    
    agent.epsilon = 0.0 # L'IA utilise son cerveau (Q-Table) sans aucun hasard
    
    for i, sat in enumerate(fantomes):
        test_env.sat = sat
        test_env._load_satellite_physics()
        
        state = test_env._get_observation()
        done = False
        t_axis, batt_true, actions = [], [], []
        total_data = 0.0
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            
            t_axis.append(test_env.time_step)
            batt_true.append(test_env.true_battery)
            actions.append(info["actual_action"])
            
            if info["actual_action"] == 2:
                total_data += min(10.0, test_env.data_buffer)
                
            state = next_state
            
        ax = axs[i]
        ax.plot(t_axis, batt_true, color='blue', linewidth=2, label='Batterie')
        ax.axhline(y=test_env.E_min, color='red', linestyle='--', label='E_min')
        
        for j in range(len(actions)-1):
            x_vals = [t_axis[j], t_axis[j+1]]
            y_vals = [batt_true[j], batt_true[j+1]]
            if actions[j] == 1: 
                ax.plot(x_vals, y_vals, color='green', linewidth=3) 
            elif actions[j] == 2: 
                ax.plot(x_vals, y_vals, color='orange', linewidth=3) 
            else: 
                ax.plot(x_vals, y_vals, color='gray', linewidth=1.5) 
                
        statut = "✅ SURVIT" if test_env.true_battery > test_env.E_min else "💀 MORT"
        ax.set_title(f"{sat['Nom_Satellite']} | Masse: {sat['Masse_kg']:.0f}kg | Cap: {sat['Capacite_Batterie_Wh']:.0f}Wh\nStatut: {statut} | Data: {total_data:.0f} Mo", fontsize=10)
        ax.set_ylabel("Énergie (Wh)")
        ax.set_xlabel("Temps (s)")
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            legend_elements = [
                mpatches.Patch(color='green', label='SCIENCE'),
                mpatches.Patch(color='orange', label='DOWNLINK'),
                mpatches.Patch(color='gray', label='IDLE (Survie)'),
                plt.Line2D([0], [0], color='blue', label='Batterie'),
                plt.Line2D([0], [0], color='red', linestyle='--', label='E_min')
            ]
            ax.legend(handles=legend_elements, loc='lower left', fontsize=8)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()