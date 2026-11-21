import pandas as pd

# 1. Lire le fichier Excel
df = pd.read_excel('satellites.xlsx')

# 2. Le sauvegarder en CSV
# index=False permet d'éviter d'ajouter une colonne de numéros au début
# encoding='utf-8' permet de gérer les caractères spéciaux (accents, etc.)
df.to_csv('satellites.csv', index=False, encoding='utf-8')

print("✅ Conversion réussie ! Le fichier satellites.csv est prêt.")