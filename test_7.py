import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import pickle

from Q_learning_avancee import OrbitMindEnv, QLearningAgent

# ==========================================
# 1. L'ENVIRONNEMENT ULTIME (Hardcore + Pannes Totales)
# ==========================================
class UltimateEnv(OrbitMindEnv):
    def _get_observation(self):
        # L'IA voit le mensonge du capteur SANS correction
        measured_battery = self.true_battery
        if self.sensor_fault:
            measured_battery += self.sat['Biais_Capteur_Wh']
            
        residual = abs(measured_battery - self.estimated_battery)
        anomaly_detected = 1 if residual > self.sat['Seuil_Anomalie_Wh'] else 0
        
        # L'observation utilise la valeur MENTEUSE (l'IA doit utiliser anomaly_detected)
        ai_view_battery = measured_battery 
            
        b_pct = max(0, ai_view_battery) / self.sat['Capacite_Batterie_Wh']
        b_bin = 0 if b_pct < 0.3 else (1 if b_pct < 0.7 else 2)
            
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
        reward = 0
        
        # 1. Panne d'antenne aléatoire (30% de chance d'être cassée au moment du downlink)
        if action == 2 and random.random() < 0.3 and not self.sensor_fault: 
            # Si l'antenne casse, elle consomme mais n'envoie rien
            target_attitude = 1
            self.current_attitude = target_attitude
            self.true_battery -= self.antenna_power * (15.0 / 3600.0) # Consommation
            self.time_step += 15
            return self._get_observation(), -50.0, False, False, {"actual_action": action, "data_sent": 0}

        # 2. Physique normale (SANS FILTRE DE SÉCURITÉ)
        target_attitude = 0 if action == 0 else 1 
        t_rotation = 45.0 if self.current_attitude != target_attitude else 0
        P_rotation = 30.0 + (self.sat['Masse_kg'] * 0.1) 
        E_rotation = P_rotation * (t_rotation / 3600.0) if t_rotation > 0 else 0
        
        self.current_attitude = target_attitude
        dt = 15.0 
        power_draw = self.survival_power
        data_gen = 0.0
        data_sent = 0.0
        
        if action == 1: # SCIENCE
            dt = max(dt, t_rotation + 10.0)
            power_draw += self.science_power
            data_gen = self.science_rate * 10.0 
            reward += 2.0 
                
        elif action == 2: # DOWNLINK
            if self.is_station_visible() and self.data_buffer > 0:
                data_to_send = min(10.0, self.data_buffer)
                t_transmission = data_to_send / self.antenna_rate
                dt = max(dt, t_rotation + t_transmission)
                power_draw += self.antenna_power
                data_sent = data_to_send
                reward += data_sent * 500.0 
            else:
                reward -= 20.0 
                
        elif action == 0: # IDLE
            if self.is_station_visible() and self.data_buffer > 0:
                reward -= 100.0 
            else:
                reward += 2.0
                
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
        
        self.time_step += int(dt)
        
        # Injection de la panne générale (Capteur + Solaire) à un moment aléatoire
        if self.time_step >= self.fault_time and not self.solar_fault:
            self.solar_fault = True
            self.sensor_fault = True
            
        terminated = False
        if self.true_battery <= self.E_min:
            reward = -10000.0 
            terminated = True
            
        if self.time_step >= (self.orbit_period * 2):
            terminated = True
            
        return self._get_observation(), reward, terminated, False, {"actual_action": action, "data_sent": data_sent}

# ==========================================
# 2. LE TEST ULTIME
# ==========================================
if __name__ == "__main__":
    print("🧠 Chargement du cerveau de l'IA...")
    env = UltimateEnv()
    agent = QLearningAgent()
    with open('orbitmind_brain.pkl', 'rb') as f:
        agent.q_table = pickle.load(f)
    agent.epsilon = 0.0 # Mode Expert

    def generate_random_sat(idx):
        mass = random.uniform(10.0, 200.0)
        surv_pwr = 10.0 + (mass * 0.05)
        max_solar = random.uniform(surv_pwr * 1.2, 100.0) # Panneaux plus faibles
        eclipse_dur = 2100.0
        e_min = (surv_pwr * (eclipse_dur / 3600.0)) + 50.0
        
        # LA MODIFICATION EST ICI :
        # La batterie ne fait que 20% à 50% de plus que le strict minimum !
        capacite_batterie = e_min * random.uniform(1.2, 1.5)
        
        return {
            'Nom_Satellite': f'PAUVRE-{idx}',
            'Masse_kg': mass,
            'Capacite_Batterie_Wh': capacite_batterie, # <--- Batterie minuscule
            'Puissance_Solaire_Max_W': max_solar,
            'Puissance_Survie_W': surv_pwr,
            'Puissance_Science_W': random.uniform(20.0, 70.0),
            'Debit_Science_Mo_s': random.uniform(0.5, 3.0),
            'Puissance_Antenne_W': random.uniform(20.0, 100.0),
            'Debit_Antenne_Mo_s': random.uniform(0.5, 8.0),
            'Capacite_Buffer_Mo': random.uniform(100.0, 500.0),
            'Periode_Orbite_s': 5677.0,
            'Duree_Eclipse_s': eclipse_dur,
            'E_min_Wh': e_min,
            'Seuil_Anomalie_Wh': 100.0,
            'Biais_Capteur_Wh': random.uniform(100.0, 300.0)
        }

    print("🔥 Lancement du TEST ULTIME (4 Satellites Inconnus, Pannes Totales, Sans Filet)...\n")
    
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("TEST ULTIME : Généralisation, Survie, et Transmission face à l'Inconnu", fontsize=16)
    axs = axs.flatten()

    for i in range(4):
        sat = generate_random_sat(i)
        env.sat = sat
        env._load_satellite_physics()
        # Panne aléatoire entre 30% et 70% de l'orbite
        env.fault_time = int(env.orbit_period * random.uniform(0.3, 0.7))
        
        state = env._get_observation()
        done = False
        t_axis, batt_true, actions, buffer_levels = [], [], [], []
        total_data = 0.0
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            t_axis.append(env.time_step)
            batt_true.append(env.true_battery)
            actions.append(info["actual_action"])
            buffer_levels.append(env.data_buffer)
            
            if info["actual_action"] == 2:
                total_data += info["data_sent"]
                
            state = next_state

        # --- AFFICHAGE ---
        ax = axs[i]
        # Batterie
        ax.plot(t_axis, batt_true, color='blue', linewidth=2, label='Batterie')
        ax.axhline(y=env.E_min, color='red', linestyle='--', label='E_min')
        ax.axvline(x=env.fault_time, color='black', linestyle=':', label='Panne Aléatoire')
        
        # Couleurs des actions
        for j in range(len(actions)-1):
            x_vals = [t_axis[j], t_axis[j+1]]
            y_vals = [batt_true[j], batt_true[j+1]]
            if actions[j] == 1: ax.plot(x_vals, y_vals, color='green', linewidth=3) 
            elif actions[j] == 2: ax.plot(x_vals, y_vals, color='orange', linewidth=3) 
            else: ax.plot(x_vals, y_vals, color='gray', linewidth=1.5)

        statut = "✅ SURVIT" if env.true_battery > env.E_min else "💀 MORT"
        ax.set_title(f"{sat['Nom_Satellite']} (Masse: {sat['Masse_kg']:.0f}kg, Cap: {sat['Capacite_Batterie_Wh']:.0f}Wh)\nStatut: {statut} | Data: {total_data:.1f} Mo", fontsize=11)
        ax.set_ylabel("Énergie (Wh)")
        ax.set_xlabel("Temps (s)")
        if i == 0:
            ax.legend(handles=[
                mpatches.Patch(color='green', label='SCIENCE'),
                mpatches.Patch(color='orange', label='DOWNLINK'),
                mpatches.Patch(color='gray', label='IDLE'),
                plt.Line2D([0], [0], color='blue', label='Batterie'),
                plt.Line2D([0], [0], color='red', linestyle='--', label='E_min')
            ], loc='upper right')

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.show()