import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Charger le dataset
df = pd.read_csv('orbitmind_full_dataset.csv')
sats_to_simulate = df.sample(n=10, random_state=42).reset_index(drop=True)

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

print("🚀 Simulation avec injection de PANNES au milieu de l'orbite...\n")

for i, row in sats_to_simulate.iterrows():
    name = row['Nom_Satellite']
    
    # Paramètres
    battery_cap = row['Capacite_Batterie_Wh']
    max_solar = row['Puissance_Solaire_Max_W']
    survival_power = row['Puissance_Survie_W']
    science_power = row['Puissance_Science_W']
    E_min = row['E_min_Wh']
    period = int(row['Periode_Orbite_s'])
    eclipse_dur = int(row['Duree_Eclipse_s'])
    
    # Variables
    battery = battery_cap
    history_soc = [] # On trace le % (State of Charge)
    history_solar = []
    
    # Panne injectée à la moitié de l'orbite
    fault_time = period // 2
    
    for t in range(period * 2): # On simule 2 orbites pour bien voir
        time_in_orbit = t % period
        in_eclipse = time_in_orbit >= (period - eclipse_dur)
        
        # --- INJECTION DE LA PANNE ---
        solar_fault = t >= fault_time
        sensor_fault = t >= fault_time
        
        # Production solaire (réduite de 70% si panne)
        if in_eclipse:
            solar_gen = 0.0
        else:
            solar_gen = max_solar * (0.3 if solar_fault else 1.0)
            
        # Consommation
        power_draw = survival_power + science_power
        
        # Physique
        net_power = solar_gen - power_draw
        battery += net_power * (1.0 / 3600.0)
        battery = max(0.0, min(battery, battery_cap))
        
        # Calcul du % et du capteur (qui ment)
        true_soc = (battery / battery_cap) * 100
        obs_soc = true_soc + (25.0 if sensor_fault else 0.0) # +25% de mensonge
        
        history_soc.append(obs_soc) # On trace ce que le capteur voit
        
    # --- Tracé ---
    ax = axes[i]
    # Ligne bleue : ce que le capteur montre à l'IA
    ax.plot(history_soc, label='Capteur (%)', color='blue')
    ax.axhline(y=(E_min / battery_cap)*100, color='red', linestyle='--', label='E_min')
    
    # Zones (Soleil = blanc, Éclipse = gris, Panne = orange)
    ax.axvspan(period - eclipse_dur, period, color='gray', alpha=0.3)
    ax.axvspan(period*1.5, period*2, color='gray', alpha=0.3) # 2ème éclipse
    ax.axvline(x=fault_time, color='orange', linestyle=':', linewidth=2, label='Panne !')
    
    ax.set_title(f"{name[:15]}...", fontsize=10)
    ax.set_ylim(0, 110)
    if i == 0 or i == 5: ax.set_ylabel('Batterie (%)')
    if i >= 5: ax.set_xlabel('Temps (s)')
        
    # Statut
    min_soc = min(history_soc)
    if min_soc <= 1.0:
        ax.set_facecolor('#ffcccc') # Rouge
        print(f"💀 {name[:20]} : MORT (Capteur à {min_soc:.1f}%)")
    else:
        ax.set_facecolor('#ccffcc') # Vert
        print(f"✅ {name[:20]} : SURVIT (Capteur à {min_soc:.1f}%)")

plt.tight_layout()
plt.show()