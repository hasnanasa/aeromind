import socket
import time
import pickle
from Q_learning_avancee import OrbitMindEnv, QLearningAgent

# 1. Connexion à Vizard
HOST = 'localhost'
PORT = 5556
print("Connexion à Vizard...")
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((HOST, PORT))
print("✅ Connecté ! Démarrage de la mission 3D...\n")

# 2. Chargement de l'IA
env = OrbitMindEnv()
agent = QLearningAgent()
with open('orbitmind_brain.pkl', 'rb') as f:
    agent.q_table = pickle.load(f)
agent.epsilon = 0.0 # Mode Expert

# On choisit un satellite moyen
test_sat = next(s for s in env.satellites if "Aalto" in s['Nom_Satellite'])
env.sat = test_sat
env._load_satellite_physics()

state = env._get_observation()
done = False

# 3. Boucle de simulation
while not done:
    action = agent.choose_action(state)
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    actual_action = info["actual_action"]
    battery = env.true_battery
    buffer = env.data_buffer
    is_eclipse = 1 if env.is_in_eclipse() else 0
    
    # Envoi à Vizard (Format: "Action, Batterie, Data, Eclipse")
    message = f"{actual_action},{battery:.1f},{buffer:.1f},{is_eclipse}"
    client.send(message.encode('utf-8'))
    
    # Affichage terminal
    actions_names = ["IDLE (Soleil)", "SCIENCE (Terre)", "DOWNLINK (Antenne)"]
    eclipse_str = "🌑 ECLIPSE" if is_eclipse else "☀️ SOLEIL"
    print(f"{eclipse_str} | Action: {actions_names[actual_action]:<20} | Batterie: {battery:.1f} Wh | Data: {buffer:.1f} Mo")
    
    state = next_state
    
    # Ralentir pour voir l'animation 3D (1 seconde par action)
    time.sleep(1.0) 

print("\nMission terminée. Le satellite a survécu.")
client.close()