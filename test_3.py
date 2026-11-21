import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import pickle

# On importe les classes de base
from Q_learning_avancee import OrbitMindEnv, QLearningAgent

# --- CRÉATION D'UN ENVIRONNEMENT SANS FILTRE (Le Dilemme) ---
class DilemmaEnv(OrbitMindEnv):
    def step(self, action):
        reward = 0
        
        # On supprime le filtre anti-suicide et le masquage !
        # L'IA est seule face à ses responsabilités.
        
        target_attitude = 0 if action == 0 else 1 
        t_rotation = 45.0 if self.current_attitude != target_attitude else 0
        P_rotation = 30.0 + (self.sat['Masse_kg'] * 0.1) 
        E_rotation = P_rotation * (t_rotation / 3600.0) if t_rotation > 0 else 0
        
        self.current_attitude = target_attitude
            
        dt = 15.0 
        power_draw = self.survival_power
        data_gen = 0.0
        data_sent = 0.0
        
        in_eclipse = self.is_in_eclipse()
        
        # Calcul des récompenses sans filtre
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
                reward -= 20.0 # Punition si elle tente sans station
                
        elif action == 0: # IDLE
            if self.is_station_visible() and self.data_buffer > 0:
                reward -= 100.0 
            else:
                reward += 2.0
                
        solar_efficiency = 1.0 if self.current_attitude == 0 else 0.4 
        if in_eclipse or self.solar_fault:
            solar_gen = 0.0 if in_eclipse else (self.max_solar * 0.3)
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
        
        terminated = False
        if self.true_battery <= self.E_min:
            reward = -10000.0 
            terminated = True
            
        if self.time_step >= (self.orbit_period * 2): # On raccourcit l'orbite pour le test
            terminated = True
            
        return self._get_observation(), reward, terminated, False, {"actual_action": action}

if __name__ == "__main__":
    print("🧠 Chargement du cerveau de l'IA...")
    env = DilemmaEnv()
    agent = QLearningAgent()
    
    with open('orbitmind_brain.pkl', 'rb') as f:
        agent.q_table = pickle.load(f)
        
    agent.epsilon = 0.0 # Mode Expert pur
    
    # On prend Aalto-1 (petite batterie)
    test_sat = next(s for s in env.satellites if "Aalto" in s['Nom_Satellite'])
    env.sat = test_sat
    env._load_satellite_physics()
    
    # --- SCÉNARIO CATASTROPHE ---
    # 1. On force la batterie à être très basse (juste au-dessus de E_min)
    env.true_battery = env.E_min + (test_sat['Capacite_Batterie_Wh'] * 0.05) 
    env.estimated_battery = env.true_battery
    
    # 2. On force le buffer à être plein (beaucoup de data à sauver)
    env.data_buffer = test_sat['Capacite_Buffer_Mo'] * 0.9
    
    # 3. On place le temps juste avant la fin de l'éclipse (quand la station est visible)
    env.time_step = 0  # Au début de l'orbite (la station est visible < 300s)
    
    state = env._get_observation()
    
    print(f"🚨 SITUATION CRITIQUE : {test_sat['Nom_Satellite']}")
    print(f"   Batterie : {env.true_battery:.1f} Wh (E_min : {env.E_min:.1f} Wh)")
    print(f"   Data : {env.data_buffer:.1f} Mo (Buffer Max : {test_sat['Capacite_Buffer_Mo']:.1f} Mo)")
    print(f"   Station Visible : {env.is_station_visible()}")
    
    # --- TEST DE DÉCISION ---
    # On demande à l'IA ce qu'elle veut faire dans cet état précis
    action_choisie = agent.choose_action(state)
    
    noms_actions = ["IDLE (Survie)", "SCIENCE", "DOWNLINK (Envoi)"]
    print(f"\n🧠 DÉCISION DE L'IA : {noms_actions[action_choisie]}")
    
    if action_choisie == 2:
        print("   ➡️ L'IA a choisi de prendre le risque de MOURIR pour transmettre la data !")
    elif action_choisie == 0:
        print("   ➡️ L'IA a choisi de SURVIVRE et de sacrifier la data !")
        
    # --- SIMULATION DE LA CONSÉQUENCE ---
    print("\n⏳ Simulation de la conséquence de son choix...")
    next_state, reward, terminated, _, info = env.step(action_choisie)
    
    print(f"   Récompense reçue : {reward:.1f}")
    print(f"   Batterie après action : {env.true_battery:.1f} Wh")
    
    if terminated and env.true_battery <= env.E_min:
        print("💀 RÉSULTAT : Le satellite est MORT.")
    else:
        print("✅ RÉSULTAT : Le satellite a SURVÉCU.")