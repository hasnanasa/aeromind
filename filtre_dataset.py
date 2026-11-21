import pandas as pd
import math
import random

# --- 1. CHARGEMENT DU FICHIER ---
file_path = 'satellites.csv'

try:
    # On charge le fichier (latin-1 gère les accents des noms de satellites)
    df = pd.read_csv(file_path, encoding='latin-1')
    
    print(f"Total de satellites dans le fichier : {len(df)}")

    # --- 2. FILTRAGE (Uniquement LEO) ---
    # On ne garde que l'orbite basse (Low Earth Orbit)
    df_leo = df[df['Class of Orbit'] == 'LEO'].copy()
    print(f"Satellites en orbite LEO : {len(df_leo)}")

    # --- 3. SÉLECTION DES COLONNES ---
    # Attention : la première colonne a des guillemets et une virgule dans son nom !
    col_name = '"Name of Satellite, Alternate Names"'
    
    # Si les guillemets ne sont pas lus comme ça par Pandas, on essaie sans :
    if col_name not in df_leo.columns:
        col_name = 'Name of Satellite, Alternate Names'

    # On extrait uniquement ce dont on a besoin
    df_clean = df_leo[[col_name, 'Launch Mass (kg.)', 'Power (watts)']].copy()
    
    # On renomme les colonnes pour notre projet
    df_clean.columns = ['Nom_Satellite', 'Masse_kg', 'Puissance_Solaire_W']

    # --- 4. NETTOYAGE DES DONNÉES ---
    # Conversion en nombres (si Pandas a lu du texte par erreur)
    df_clean['Masse_kg'] = pd.to_numeric(df_clean['Masse_kg'], errors='coerce')
    df_clean['Puissance_Solaire_W'] = pd.to_numeric(df_clean['Puissance_Solaire_W'], errors='coerce')
    
    # On supprime les lignes où la masse ou la puissance est vide (NaN)
    df_clean = df_clean.dropna()
    
    # On supprime les satellites qui ont une masse nulle ou une puissance nulle (erreurs de données)
    df_clean = df_clean[(df_clean['Masse_kg'] > 0) & (df_clean['Puissance_Solaire_W'] > 0)]

    # --- 5. SAUVEGARDE DU DATASET FINAL ---
    df_clean.to_csv('orbitmind_dataset.csv', index=False)
    
    print(f"\n✅ Succès ! Dataset nettoyé créé avec {len(df_clean)} satellites LEO.")
    print("   Fichier sauvegardé : 'orbitmind_dataset.csv'")
    print("\nAperçu des 5 premiers satellites :")
    print(df_clean.head())

except Exception as e:
    print(f"❌ Erreur : {e}")