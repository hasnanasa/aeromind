import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Charger le dataset
df = pd.read_csv('orbitmind_full_dataset.csv')

# 2. Prendre 10 satellites au hasard (ou les 10 premiers)
# On utilise random_state pour que tu aies toujours les mêmes 10 pour comparer
sats_to_simulate = df.sample(n=10, random_state=42).reset_index(drop=True)

# Création de la figure (2 lignes, 5 colonnes)
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

print("🚀 Lancement de la simulation physique pour 10 satellites...\n")

for i, row in sats_to_simulate.iterrows():
    name = row['Nom_Satellite']
    
    # --- Extraction des paramètres depuis le CSV ---
    battery_cap = row['Capacite_Batterie_Wh']
    max_solar = row['Puissance_Solaire_Max_W']
    survival_power = row['Puissance_Survie_W']
    science_power = row['Puissance_Science_W']
    E_min = row['E_min_Wh']
    period = int(row['Periode_Orbite_s'])
    eclipse_dur = int(row['Duree_Eclipse_s'])
    
    # --- Variables de simulation ---
    battery = battery_cap # Batterie pleine au départ
    history_battery = []
    history_solar = []
    
    # --- Boucle de simulation (1 orbite complète) ---
    for t in range(period):
        # Phase orbitale : L'éclipse se produit sur la fin de l'orbite
        time_in_orbit = t % period
        in_eclipse = time_in_orbit >= (period - eclipse_dur)
        
        # Production solaire
        solar_gen = 0.0 if in_eclipse else max_solar
        
        # Consommation (On suppose que le satellite fait toujours de la science pour tester)
        power_draw = survival_power + science_power
        
        # Physique : E = P * t (en Wh)
        net_power = solar_gen - power_draw
        battery += net_power * (1.0 / 3600.0)
        
        # Contraintes
        battery = max(0.0, min(battery, battery_cap))
        
        history_battery.append(battery)
        history_solar.append(solar_gen)
        
    # --- Tracé du graphique pour ce satellite ---
    ax = axes[i]
    ax.plot(history_battery, label='Batterie (Wh)', color='blue')
    ax.axhline(y=E_min, color='red', linestyle='--', label='E_min (Survie)')
    
    # Remplir la zone d'éclipse en gris
    ax.axvspan(period - eclipse_dur, period, color='gray', alpha=0.3, label='Éclipse')
    
    # Mise en forme
    ax.set_title(f"{name[:15]}...\nCap: {battery_cap:.0f}Wh", fontsize=10)
    ax.set_ylim(0, battery_cap * 1.1)
    if i == 0 or i == 5:
        ax.set_ylabel('Énergie (Wh)')
    if i >= 5:
        ax.set_xlabel('Temps (s)')
        
    # Statut de survie
    min_batt = min(history_battery)
    if min_batt <= 10.0:
        ax.set_facecolor('#ffcccc') # Fond rouge si mort
        print(f"💀 {name[:20]} : MORT (Batterie vide)")
    elif min_batt < E_min:
        ax.set_facecolor('#ffffcc') # Fond jaune si zone de danger
        print(f"⚠️ {name[:20]} : DANGER (Passe sous E_min)")
    else:
        ax.set_facecolor('#ccffcc') # Fond vert si sain
        print(f"✅ {name[:20]} : SAIN (Survit à l'éclipse)")

# Ajouter une légende globale
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05))

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.suptitle("Simulation Physique d'une Orbite pour 10 Satellites (Scénario de Survie)", fontsize=16)
plt.show()