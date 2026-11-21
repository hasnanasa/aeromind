import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- PARAMÈTRES PHYSIQUES ---
R_earth = 6371  # Rayon de la Terre (km)
R_orbit = 6871  # Rayon de l'orbite (500 km d'altitude)
orbit_period = 5677  # Temps d'un tour complet (secondes)

# Position de la station sol (un point rouge sur Terre)
# On la place à un angle de 45 degrés sur le côté droit
gs_angle = np.radians(45)
gs_x = R_earth * np.cos(gs_angle)
gs_y = R_earth * np.sin(gs_angle)

# Configuration de la figure
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-R_orbit - 1000, R_orbit + 1000)
ax.set_ylim(-R_orbit - 1000, R_orbit + 1000)
ax.set_aspect('equal')
ax.axis('off') # On cache les axes pour faire plus propre

# 1. Dessiner la Terre
earth = plt.Circle((0, 0), R_earth, color='#1E90FF', alpha=0.8, label='Terre')
ax.add_patch(earth)

# 2. Dessiner la zone d'ombre (Éclipse) à gauche
# On dessine un grand rectangle gris pour l'ombre de la Terre
shadow = plt.Rectangle((-R_orbit - 1000, -R_earth), R_orbit + 1000, 2*R_earth, 
                       color='gray', alpha=0.3, label='Zone d\'éclipse')
ax.add_patch(shadow)

# 3. Dessiner l'orbite (cercle en pointillés)
orbit = plt.Circle((0, 0), R_orbit, fill=False, color='white', linestyle='--', linewidth=1)
ax.add_patch(orbit)

# 4. Dessiner la Station Sol (Point Rouge)
ax.plot(gs_x, gs_y, 'ro', markersize=10, label='Station Sol')
ax.text(gs_x + 300, gs_y + 300, 'Station Sol', color='red', fontweight='bold')

# 5. Créer le Satellite (Point Blanc)
satellite, = ax.plot([], [], 'wo', markersize=8, label='Satellite')

# --- INITIALISATION DE L'ANIMATION ---
def init():
    satellite.set_data([], [])
    return satellite,

# --- MISE À JOUR À CHAQUE IMAGE ---
def update(frame):
    # Calcul de l'angle du satellite en fonction du temps
    theta = (frame / 100) * 2 * np.pi # L'angle tourne
    
    # Position X et Y du satellite
    x = R_orbit * np.cos(theta)
    y = R_orbit * np.sin(theta)
    
    satellite.set_data([x], [y])
    
    # Changer la couleur du satellite s'il est dans l'ombre (à gauche)
    if x < 0: 
        satellite.set_color('gray') # Dans l'éclipse
    else:
        satellite.set_color('yellow') # Au soleil
        
    return satellite,

# Lancer l'animation
ani = animation.FuncAnimation(fig, update, frames=300, init_func=init, blit=True, interval=50)

plt.title("Simulation de l'orbite ORBITMIND")
plt.legend(loc='upper right')
plt.show()