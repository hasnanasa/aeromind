import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import pickle

# On importe ton environnement
from Q_learning_avancee import OrbitMindEnv, QLearningAgent

# ==========================================
# 1. L'AGENT CLASSIQUE (Rule-Based / Idiot)
# ==========================================
class RuleBasedAgent:
    def choose_action(self, state):
        # state = (b_bin, d_bin, sun_visible, station_visible, anomaly, eclipse)
        b_bin, d_bin, sun_visible, station_visible, anomaly, eclipse_bin = state
        
        # Règle 1 : Si la batterie est basse (0), on se repose.
        if b_bin == 0:
            return 0 # IDLE
            
        # Règle 2 : Si la batterie est moyenne et qu'on est dans l'éclipse, on se repose
        if b_bin == 1 and eclipse_bin == 0:
            return 0 # IDLE
            
        # Règle 3 : Si la station est visible et qu'on a de la data, on envoie
        if station_visible == 1 and d_bin > 0:
            return 2 # DOWNLINK
            
        # Règle 4 : Sinon, on fait de la science
        return 1 # SCIENCE

# ==========================================
# 2. LE TEST DE MONTE CARLO (100 Missions)
# ==========================================
if __name__ == "__main__":
    print("🧠 Chargement de l'IA...")
    env = OrbitMindEnv()
    ai_agent = QLearningAgent()
    with open('orbitmind_brain.pkl', 'rb') as f:
        ai_agent.q_table = pickle.load(f)
    ai_agent.epsilon = 0.0 # Mode Expert
    
    baseline_agent = RuleBasedAgent()
    
    num_missions = 100
    print(f"🚀 Lancement de {num_missions} missions pour comparer l'IA vs Système Classique...\n")
    
    results = {
        'AI': {'survivals': 0, 'total_data': 0},
        'Baseline': {'survivals': 0, 'total_data': 0}
    }
    
    # On utilise le même satellite pour les deux pour être équitable
    test_sat = next(s for s in env.satellites if "Aalto" in s['Nom_Satellite'])
    
    for mission in range(num_missions):
        # --- TEST DE L'IA ---
        env.sat = test_sat
        env._load_satellite_physics()
        state = env._get_observation()
        done = False
        data_sent_ai = 0
        
        while not done:
            action = ai_agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # On compte la data envoyée
            if info["actual_action"] == 2:
                data_sent_ai += min(10.0, env.data_buffer)
                
            state = next_state
            
        if env.true_battery > env.E_min:
            results['AI']['survivals'] += 1
            results['AI']['total_data'] += data_sent_ai
            
        # --- TEST DU SYSTEME CLASSIQUE ---
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

    # ==========================================
    # 3. AFFICHAGE DES RÉSULTATS (Le Verdict)
    # ==========================================
    ai_surv_rate = (results['AI']['survivals'] / num_missions) * 100
    base_surv_rate = (results['Baseline']['survivals'] / num_missions) * 100
    
    ai_avg_data = results['AI']['total_data'] / max(1, results['AI']['survivals'])
    base_avg_data = results['Baseline']['total_data'] / max(1, results['Baseline']['survivals'])

    print("==================================================")
    print("🏆 RÉSULTATS DU BENCHMARK (100 MISSIONS)")
    print("==================================================")
    print(f"🤖 IA ORBITMIND :")
    print(f"   - Taux de survie : {ai_surv_rate:.1f} %")
    print(f"   - Data moyenne envoyée : {ai_avg_data:.1f} Mo par mission réussie")
    
    print(f"\n📏 SYSTÈME CLASSIQUE (Rule-Based) :")
    print(f"   - Taux de survie : {base_surv_rate:.1f} %")
    print(f"   - Data moyenne envoyée : {base_avg_data:.1f} Mo par mission réussie")
    
    print("\n📊 VERDICT :")
    if ai_surv_rate > base_surv_rate:
        print(f"✅ L'IA est plus robuste de {ai_surv_rate - base_surv_rate:.1f}% face aux pannes.")
    if ai_avg_data > base_avg_data:
        print(f"✅ L'IA transmet en moyenne {ai_avg_data - base_avg_data:.1f} Mo de plus par mission.")