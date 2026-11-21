import pandas as pd
import numpy as np
import math

# --- CONSTANTES PHYSIQUES ---
MU_EARTH = 398600.4418 # km^3/s^2
R_EARTH = 6371.0 # km
SOLAR_CONSTANT = 1361.0 # W/m^2

def generate_full_dataset(input_file, output_file):
    print("🔄 Chargement des données de base (UCS)...")
    df = pd.read_csv(input_file)
    
    # On va stocker les nouvelles lignes complètes ici
    full_data = []
    
    for index, row in df.iterrows():
        name = row['Nom_Satellite']
        mass = row['Masse_kg']
        max_power = row['Puissance_Solaire_W']
        
        # --- 1. PARAMÈTRES ORBITAUX (Saisie/Déduction) ---
        # On simule une altitude aléatoire réaliste pour LEO (entre 400 et 800 km)
        altitude = np.random.uniform(400, 800)
        r_orbit = R_EARTH + altitude
        
        # Calculs dérivés (Lois de Kepler)
        velocity = math.sqrt(MU_EARTH / r_orbit)
        period_s = 2 * math.pi * math.sqrt((r_orbit**3) / MU_EARTH)
        
        # Calcul de l'éclipse (Approximation géométrique pour LEO)
        # L'ombre de la Terre couvre environ 35 à 38% de l'orbite
        eclipse_duration_s = period_s * np.random.uniform(0.35, 0.38)
        
        # --- 2. PANNEAUX SOLAIRES (Déduction) ---
        # Puissance donnée par l'UCS. On déduit la surface (en supposant un rendement de 30%)
        panel_area = max_power / (SOLAR_CONSTANT * 0.30)
        panel_efficiency = 0.30
        degradation_rate = 0.02 # 2% par an (standard spatial)
        
        # --- 3. BATTERIE (Déduction) ---
        # Règle de l'art : La batterie doit pouvoir fournir l'énergie de 12 à 16h de production solaire
        battery_capacity = max_power * np.random.uniform(12.0, 16.0)
        dod_max = 0.20 # Profondeur de décharge max 20%
        charge_efficiency = 0.95
        
        # --- 4. CONSOMMATION (Déduction) ---
        # Puissance survie : 10W de base + 5% de la masse (pour les roues à réaction/ADCS)
        survival_power = 10.0 + (mass * 0.05)
        
        # Puissance science : Une fraction de la puissance solaire
        science_power = max_power * np.random.uniform(0.2, 0.4)
        
        # Débit science : dépend de la puissance (ex: 0.1 Mo/s par Watt alloué)
        science_data_rate = science_power * 0.05
        
        # --- 5. ANTENNE (Déduction) ---
        # Puissance d'émission : Environ 20% de la puissance solaire max (min 5W, max 150W)
        antenna_tx_power = np.clip(max_power * 0.2, 5.0, 150.0)
        
        # Débit antenne : plus l'antenne est puissante, plus elle transmet vite
        antenna_datarate = np.clip(antenna_tx_power * 0.05, 0.1, 20.0)
        
        # Taille du Buffer (Mémoire)
        buffer_capacity = np.random.uniform(100.0, 1000.0)
        
        # --- 6. PARAMÈTRES DE PANNES (IA) ---
        sensor_bias = 250.0 # Le mensonge du capteur (+250 Wh)
        anomaly_threshold = 100.0 # Seuil de détection du gradient
        
        # --- 7. CALCUL DE L'ÉNERGIE MINIMALE DE SURVIE (E_min) ---
        # E_min = (P_survie * t_eclipse_en_heures) * Marge(1.5) + (Capacite * DoD)
        time_eclipse_hours = eclipse_duration_s / 3600.0
        energy_for_eclipse = survival_power * time_eclipse_hours
        E_min = (energy_for_eclipse * 1.5) + (battery_capacity * dod_max)
        
        # --- SAUVEGARDE DE LA LIGNE COMPLÈTE ---
        full_data.append({
            "Nom_Satellite": name,
            "Masse_kg": round(mass, 2),
            "Altitude_km": round(altitude, 2),
            "Vitesse_km_s": round(velocity, 3),
            "Periode_Orbite_s": round(period_s, 1),
            "Duree_Eclipse_s": round(eclipse_duration_s, 1),
            "Puissance_Solaire_Max_W": round(max_power, 2),
            "Surface_Panneau_m2": round(panel_area, 3),
            "Rendement_Panneau": panel_efficiency,
            "Capacite_Batterie_Wh": round(battery_capacity, 2),
            "DoD_Max": dod_max,
            "Puissance_Survie_W": round(survival_power, 2),
            "Puissance_Science_W": round(science_power, 2),
            "Debit_Science_Mo_s": round(science_data_rate, 2),
            "Puissance_Antenne_W": round(antenna_tx_power, 2),
            "Debit_Antenne_Mo_s": round(antenna_datarate, 2),
            "Capacite_Buffer_Mo": round(buffer_capacity, 2),
            "E_min_Wh": round(E_min, 2),
            "Seuil_Anomalie_Wh": anomaly_threshold,
            "Biais_Capteur_Wh": sensor_bias
        })

    # Création du DataFrame final
    df_final = pd.DataFrame(full_data)
    
    # Sauvegarde en CSV
    df_final.to_csv(output_file, index=False)
    print(f"✅ Dataset complet généré avec succès : '{output_file}'")
    print(f"   Nombre de satellites configurés : {len(df_final)}")
    
    # Afficher un exemple pour vérifier
    print("\n--- APERÇU D'UN SATELLITE GÉNÉRÉ ALÉATOIREMENT ---")
    sample = df_final.sample(1).iloc[0]
    for col, val in sample.items():
        print(f"  {col:<25}: {val}")

# --- EXÉCUTION ---
if __name__ == "__main__":
    generate_full_dataset('orbitmind_dataset.csv', 'orbitmind_full_dataset.csv')