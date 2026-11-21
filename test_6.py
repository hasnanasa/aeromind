import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
from Q_learning_avancee import OrbitMindEnv, QLearningAgent

# ==========================================
# 1. L'ENVIRONNEMENT AVEC PANNES D'ANTENNE
# ==========================================
class BrokenCommsEnv(OrbitMindEnv):
    def _load_satellite_physics(self):
        super()._load_satellite_physics()
        self.comms_broken = False
        self.comms_fault_time = 2500 # La panne d'antenne frappe à 2500s

    def step(self, action):
        # On laisse l'environnement parent gérer la physique normale
        next_state, reward, terminated, truncated, info = super().step(action)
        
        # --- LA PANNE SILENCIEUSE ---
        # Si la panne a eu lieu et que l'IA a tenté un DOWNLINK (action 2)
        if self.time_step >= self.comms_fault_time and not self.comms_broken:
            self.comms_broken = True
            print(f"🚨 PANNE ANTENNE À t={self.time_step}s ! L'antenne consomme mais n'émet plus.")

        if self.comms_broken and info["actual_action"] == 2:
            # L'antenne a consommé l'énergie, MAIS on remet la data qui n'a pas pu être envoyée
            data_lost = min(10.0, self.data_buffer + (self.sat['Debit_Antenne_Mo_s'] * 15.0)) # Approximation
            # En fait, on annule l'envoi en remettant le buffer à son niveau d'avant
            # (Pour simuler que rien n'est parti)
            self.data_buffer = min(self.buffer_max, self.data_buffer + data_lost)
            
            # L'IA reçoit une punition pour l'échec de transmission (pour qu'elle apprenne)
            reward -= 200.0 
            info["data_sent"] = 0 # Aucune data n'est partie
            
        return next_state, reward, terminated, truncated, info

if __name__ == "__main__":
    print("🧠 Chargement de l'IA...")
    env = BrokenCommsEnv()
    agent = QLearningAgent()
    
    with open('orbitmind_brain.pkl', 'rb') as f:
        agent.q_table = pickle.load(f)
    agent.epsilon = 0.0 # Mode Expert pur
    
    # On prend Aalto-1
    test_sat = next(s for s in env.satellites if "Aalto" in s['Nom_Satellite'])
    env.sat = test_sat
    env._load_satellite_physics()
    
    state = env._get_observation()
    done = False
    
    t_axis, batt_true, actions, buffer_levels = [], [], [], []
    
    while not done:
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        t_axis.append(env.time_step)
        batt_true.append(env.true_battery)
        actions.append(info["actual_action"])
        buffer_levels.append(env.data_buffer)
        
        state = next_state

    # --- AFFICHAGE ---
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Test de Panne d'Antenne (Comms Failure) : {test_sat['Nom_Satellite']}", fontsize=16)

    # Batterie
    axs[0].plot(t_axis, batt_true, color='blue', linewidth=2, label='Batterie')
    axs[0].axhline(y=env.E_min, color='red', linestyle='--', label='E_min')
    axs[0].axvline(x=env.comms_fault_time, color='black', linestyle=':', linewidth=2, label='Panne Antenne')
    
    for j in range(len(actions)-1):
        x_vals = [t_axis[j], t_axis[j+1]]
        y_vals = [batt_true[j], batt_true[j+1]]
        if actions[j] == 1: axs[0].plot(x_vals, y_vals, color='green', linewidth=3) 
        elif actions[j] == 2: axs[0].plot(x_vals, y_vals, color='orange', linewidth=3) 
        else: axs[0].plot(x_vals, y_vals, color='gray', linewidth=1.5)
            
    axs[0].set_ylabel("Énergie (Wh)")
    axs[0].legend(loc='upper right')

    # Actions
    axs[1].set_yticks([0, 1, 2])
    axs[1].set_yticklabels(['IDLE', 'SCIENCE', 'DOWNLINK'])
    axs[1].plot(t_axis, actions, color='black', drawstyle='steps-pre')
    axs[1].axvline(x=env.comms_fault_time, color='black', linestyle=':', linewidth=2)
    axs[1].set_ylabel("Action")

    # Buffer
    axs[2].plot(t_axis, buffer_levels, color='purple', linewidth=2)
    axs[2].axhline(y=test_sat['Capacite_Buffer_Mo'], color='red', linestyle='--', label='Buffer Max')
    axs[2].axvline(x=env.comms_fault_time, color='black', linestyle=':', linewidth=2)
    axs[2].set_ylabel("Data (Mo)")
    axs[2].set_xlabel("Temps (s)")
    axs[2].legend()

    plt.tight_layout()
    plt.show()