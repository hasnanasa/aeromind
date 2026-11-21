import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import pickle

from Q_learning_avancee import OrbitMindEnv, QLearningAgent

# ==========================================
# 1. L'ENVIRONNEMENT HARDCORE (Sans Filet)
# ==========================================
class HardcoreEnv(OrbitMindEnv):
    def _get_observation(self):
        # L'agent voit le mensonge SANS correction
        measured_battery = self.true_battery
        if self.sensor_fault:
            measured_battery += self.sat['Biais_Capteur_Wh']
            
        # On calcule l'anomalie (mais on ne corrige PAS la batterie pour l'agent)
        residual = abs(measured_battery - self.estimated_battery)
        anomaly_detected = 1 if residual > self.sat['Seuil_Anomalie_Wh'] else 0
        
        # On utilise la valeur MENTEUSE pour l'observation
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
        
        # ON SUPPRIME LE FILTRE ANTI-SUICIDE
        # Si l'agent fait une action, elle est exécutée, même si elle le tue.
        
        target_attitude = 0 if action == 0 else 1 
        t_rotation = 45.0 if self.current_attitude != target_attitude else 0
        P_rotation = 30.0 + (self.sat['Masse_kg'] * 0.1) 
        E_rotation = P_rotation * (t_rotation / 3600.0) if t_rotation > 0 else 0
        
        self.current_attitude = target_attitude
            
        dt = 15.0 
        power_draw = self.survival_power
        data_gen = 0.0
        data_sent = 0.0
        
        # Calcul des récompenses
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
        self.estimated_battery += energy_change # L'estimation physique continue en interne
        
        self.true_battery = max(0.0, min(self.true_battery, self.sat['Capacite_Batterie_Wh']))
        self.estimated_battery = max(0.0, min(self.estimated_battery, self.sat['Capacite_Batterie_Wh'])) 
        self.data_buffer = max(0.0, min(self.data_buffer + data_gen - data_sent, self.buffer_max))
        
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
# 2. L'AGENT CLASSIQUE (Rule-Based)
# ==========================================
class RuleBasedAgent:
    def choose_action(self, state):
        b_bin, d_bin, sun_visible, station_visible, anomaly, eclipse_bin = state
        
        # Le système classique ne sait pas quoi faire de "anomaly", il l'ignore.
        # Il croit bêtement le capteur.
        
        if b_bin == 0:
            return 0 # IDLE
        if b_bin == 1 and eclipse_bin == 0:
            return 0 # IDLE
        if station_visible == 1 and d_bin > 0:
            return 2 # DOWNLINK
        return 1 # SCIENCE

# ==========================================
# 3. LE TEST (Le Vrai Verdict)
# ==========================================
if __name__ == "__main__":
    print("🧠 Chargement de l'IA...")
    env = HardcoreEnv()
    ai_agent = QLearningAgent()
    with open('orbitmind_brain.pkl', 'rb') as f:
        ai_agent.q_table = pickle.load(f)
    ai_agent.epsilon = 0.0 
    
    baseline_agent = RuleBasedAgent()
    
    num_missions = 100
    print(f"🚀 Lancement de {num_missions} missions HARDCORE (Sans filet de sécurité)...\n")
    
    results = {
        'AI': {'survivals': 0, 'total_data': 0},
        'Baseline': {'survivals': 0, 'total_data': 0}
    }
    
    test_sat = next(s for s in env.satellites if "Aalto" in s['Nom_Satellite'])
    
    for mission in range(num_missions):
        # --- TEST IA ---
        env.sat = test_sat
        env._load_satellite_physics()
        state = env._get_observation()
        done = False
        data_sent_ai = 0
        
        while not done:
            action = ai_agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if info["actual_action"] == 2:
                data_sent_ai += min(10.0, env.data_buffer)
            state = next_state
            
        if env.true_battery > env.E_min:
            results['AI']['survivals'] += 1
            results['AI']['total_data'] += data_sent_ai
            
        # --- TEST SYSTEME CLASSIQUE ---
        env.sat = test_sat
        env._load_satellite_physics()
        state = env._get_observation()
        done = False
        data_sent_baseline = 0
        
        while not done:
            action = baseline_agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if info["actual_action"] == 2:
                data_sent_baseline += min(10.0, env.data_buffer)
            state = next_state
            
        if env.true_battery > env.E_min:
            results['Baseline']['survivals'] += 1
            results['Baseline']['total_data'] += data_sent_baseline

    # Résultats
    ai_surv_rate = (results['AI']['survivals'] / num_missions) * 100
    base_surv_rate = (results['Baseline']['survivals'] / num_missions) * 100
    ai_avg_data = results['AI']['total_data'] / max(1, results['AI']['survivals'])
    base_avg_data = results['Baseline']['total_data'] / max(1, results['Baseline']['survivals'])

    print("==================================================")
    print("🏆 RÉSULTATS DU VRAI BENCHMARK HARDCORE")
    print("==================================================")
    print(f"🤖 IA ORBITMIND : Taux de survie = {ai_surv_rate:.1f}% | Data = {ai_avg_data:.1f} Mo")
    print(f"📏 SYSTÈME CLASSIQUE : Taux de survie = {base_surv_rate:.1f}% | Data = {base_avg_data:.1f} Mo")
    
    print("\n📊 VERDICT :")
    if ai_surv_rate > base_surv_rate:
        print(f"✅ L'IA survit plus que le système classique.")
    elif ai_surv_rate == base_surv_rate:
        if ai_avg_data > base_avg_data:
            print(f"✅ Égalité de survie, MAIS l'IA transmet {ai_avg_data - base_avg_data:.1f} Mo de plus par mission !")
            print(f"   L'IA est {ai_avg_data / max(1, base_avg_data):.1f} fois plus efficace !")
        else:
            print("⚖️ Égalité de survie et de data.")
    else:
        print("❌ Le système classique survit plus.")