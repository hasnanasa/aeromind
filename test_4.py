import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import pickle

from Q_learning_avancee import OrbitMindEnv, QLearningAgent

if __name__ == "__main__":
    print("🧠 Chargement de l'IA...")
    env = OrbitMindEnv()
    agent = QLearningAgent()
    
    with open('orbitmind_brain.pkl', 'rb') as f:
        agent.q_table = pickle.load(f)
        
    agent.epsilon = 0.0 # Mode Expert
    
    # On prend un satellite moyen pour voir une belle courbe
    test_sat = next(s for s in env.satellites if "CARTOSAT" in s['Nom_Satellite'].upper())
    env.sat = test_sat
    env._load_satellite_physics()
    
    print(f"🛰️ Satellite : {test_sat['Nom_Satellite']}")
    print(f"   Capacité Buffer : {test_sat['Capacite_Buffer_Mo']:.1f} Mo")
    print(f"   Débit Antenne : {test_sat['Debit_Antenne_Mo_s']:.1f} Mo/s\n")
    
    state = env._get_observation()
    done = False
    
    t_axis, batt_true, actions = [], [], []
    buffer_history = []
    cumulative_sent = []
    
    total_sent = 0.0
    
    while not done:
        # On sauvegarde l'état du buffer AVANT l'action
        buffer_before = env.data_buffer
        
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        t_axis.append(env.time_step)
        batt_true.append(env.true_battery)
        actions.append(info["actual_action"])
        buffer_history.append(env.data_buffer)
        
        # Calcul de la data envoyée pendant cette action
        if info["actual_action"] == 2:
            # Si on a fait un downlink, la data envoyée est la différence de buffer
            # (ou 10 Mo max si le buffer était plein)
            data_evacuated = max(0, buffer_before - env.data_buffer)
            # Sécurité si le buffer a aussi généré de la data en même temps (impossible ici car action 2)
            total_sent += data_evacuated
            
        cumulative_sent.append(total_sent)
        state = next_state

    # --- GRAPHIQUE D'EFFICACITÉ DE LA MISSION ---
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(f"Rapport de Mission : {test_sat['Nom_Satellite']} (Efficacité de Transmission)", fontsize=16)

    # Graphique 1 : Batterie et Actions
    axs[0].plot(t_axis, batt_true, color='blue', linewidth=2, label='Batterie')
    axs[0].axhline(y=env.E_min, color='red', linestyle='--', label='E_min')
    
    for j in range(len(actions)-1):
        x_vals = [t_axis[j], t_axis[j+1]]
        y_vals = [batt_true[j], batt_true[j+1]]
        if actions[j] == 1: 
            axs[0].plot(x_vals, y_vals, color='green', linewidth=3) 
        elif actions[j] == 2: 
            axs[0].plot(x_vals, y_vals, color='orange', linewidth=3) 
        else: 
            axs[0].plot(x_vals, y_vals, color='gray', linewidth=1.5)
            
    axs[0].set_ylabel("Énergie (Wh)")
    axs[0].set_title("1. Batterie & Actions")
    axs[0].legend(loc='upper right')

    # Graphique 2 : Niveau du Buffer
    axs[1].plot(t_axis, buffer_history, color='purple', linewidth=2)
    axs[1].axhline(y=test_sat['Capacite_Buffer_Mo'], color='red', linestyle='--', label='Buffer Max (Plein)')
    axs[1].set_ylabel("Data en Mémoire (Mo)")
    axs[1].set_title("2. Remplissage du Buffer (Science) et Vidange (Downlink)")
    axs[1].legend()

    # Graphique 3 : Data Cumulée
    axs[2].plot(t_axis, cumulative_sent, color='darkorange', linewidth=3)
    axs[2].set_ylabel("Data Totale Envoyée (Mo)")
    axs[2].set_xlabel("Temps (s)")
    axs[2].set_title("3. Mission Accomplie : Volume Total Transmis à la Terre")

    # Légende pour les couleurs de la batterie
    legend_elements = [
        mpatches.Patch(color='green', label='SCIENCE'),
        mpatches.Patch(color='orange', label='DOWNLINK'),
        mpatches.Patch(color='gray', label='IDLE (Survie)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    plt.show()
    
    print(f"\n📊 BILAN DE LA MISSION :")
    print(f"   Data totale transmise : {total_sent:.1f} Mo")
    if test_sat['Capacite_Buffer_Mo'] > 0:
         print(f"   Taux d'occupation final du buffer : {(env.data_buffer/test_sat['Capacite_Buffer_Mo'])*100:.1f}%")