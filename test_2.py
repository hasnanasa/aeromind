import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import pickle

# Importation depuis ton fichier d'entraînement
from Q_learning_avancee import OrbitMindEnv, QLearningAgent

if __name__ == "__main__":
    print("🧠 Chargement du cerveau de l'IA (avec Marge de Sécurité)...")
    env = OrbitMindEnv()
    agent = QLearningAgent()
    
    try:
        with open('orbitmind_brain.pkl', 'rb') as f:
            agent.q_table = pickle.load(f)
        print("✅ Cerveau chargé avec succès !")
    except FileNotFoundError:
        print("❌ ERREUR : Lance d'abord 'Q_learning_avancee.py' pour créer le cerveau.")
        exit()

    print("🎬 Lancement du Test Scientifique A/B (Même taille)...\n")
    agent.epsilon = 0.0 
    
    test_env = OrbitMindEnv()
    
    # --- SATELLITE 1 : Connu (Aalto-1 du Dataset) ---
    sat_connu = next(s for s in test_env.satellites if "Aalto" in s['Nom_Satellite'])
    
    # --- SATELLITE 2 : Inconnu (Le Jumeau Fantôme, même capacité ~65Wh) ---
    sat_inconnu = {
        'Nom_Satellite': 'Ghost-Twin-65',
        'Masse_kg': 2.5, # Masse similaire
        'Capacite_Batterie_Wh': 65.0, # Même batterie !
        'Puissance_Solaire_Max_W': 18.0, # Panneau similaire
        'Puissance_Survie_W': 11.0, # Survie similaire
        'Puissance_Science_W': 22.0,
        'Debit_Science_Mo_s': 0.6,
        'Puissance_Antenne_W': 18.0,
        'Debit_Antenne_Mo_s': 1.2,
        'Capacite_Buffer_Mo': 120.0,
        'Periode_Orbite_s': 5677.0,
        'Duree_Eclipse_s': 2100.0,
        'E_min_Wh': 25.0, # Même E_min (calculé pour cette taille)
        'Seuil_Anomalie_Wh': 50.0,
        'Biais_Capteur_Wh': 100.0 # Biais plus petit car petite batterie
    }
    
    def run_simulation(satellite_profile):
        test_env.sat = satellite_profile
        test_env._load_satellite_physics()
        state = test_env._get_observation()
        done = False
        
        t_axis, batt_true, actions, data_sent_log = [], [], [], []
        total_data = 0.0
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            
            t_axis.append(test_env.time_step)
            batt_true.append(test_env.true_battery)
            actions.append(info["actual_action"])
            
            if info["actual_action"] == 2 and test_env.data_buffer > 0:
                data_packet = min(10.0, test_env.data_buffer)
                total_data += data_packet
                data_sent_log.append((test_env.time_step, data_packet))
                
            state = next_state
            
        return t_axis, batt_true, actions, total_data, data_sent_log

    # Lancer les simulations
    t1, b1, a1, total_d1, log_d1 = run_simulation(sat_connu)
    t2, b2, a2, total_d2, log_d2 = run_simulation(sat_inconnu)

    # --- AFFICHAGE COMPARATIF ---
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Test Scientifique : Comparaison de Satellites de Même Taille (~65Wh)", fontsize=16)

    def plot_sat(ax_batt, ax_action, t, b, a, sat_name, total_d, log_d):
        # Graphique Batterie
        ax_batt.plot(t, b, color='blue', linewidth=2, label='Batterie')
        ax_batt.axhline(y=test_env.E_min, color='red', linestyle='--', label='E_min')
        ax_batt.set_title(f"{sat_name}\nData Envoyée : {total_d:.1f} Mo", fontsize=12)
        ax_batt.set_ylabel("Énergie (Wh)")
        ax_batt.legend(loc='upper right')
        
        for j in range(len(a)-1):
            x_vals = [t[j], t[j+1]]
            y_vals = [b[j], b[j+1]]
            if a[j] == 1: 
                ax_batt.plot(x_vals, y_vals, color='green', linewidth=3) 
            elif a[j] == 2: 
                ax_batt.plot(x_vals, y_vals, color='orange', linewidth=3) 
            else: 
                ax_batt.plot(x_vals, y_vals, color='gray', linewidth=2)
                
        # Graphique Actions
        ax_action.set_yticks([0, 1, 2])
        ax_action.set_yticklabels(['IDLE (Survie)', 'SCIENCE', 'DOWNLINK'])
        ax_action.plot(t, a, color='black', drawstyle='steps-pre')
        
        for time_sent, size in log_d:
            ax_action.text(time_sent, 1.5, f"📦 {size:.0f}Mo", color='orange', fontsize=8, rotation=45)

        ax_action.set_title("Actions et Paquets Transmis")
        ax_action.set_xlabel("Temps (s)")
        ax_action.set_ylabel("Action")

    plot_sat(axs[0, 0], axs[1, 0], t1, b1, a1, f"CONNU : {sat_connu['Nom_Satellite']}", total_d1, log_d1)
    plot_sat(axs[0, 1], axs[1, 1], t2, b2, a2, f"INCONNU : {sat_inconnu['Nom_Satellite']}", total_d2, log_d2)

    legend_elements = [
        mpatches.Patch(color='green', label='SCIENCE'),
        mpatches.Patch(color='orange', label='DOWNLINK'),
        mpatches.Patch(color='gray', label='IDLE (Survie)'),
        plt.Line2D([0], [0], color='blue', label='Batterie'),
        plt.Line2D([0], [0], color='red', linestyle='--', label='E_min')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.show()