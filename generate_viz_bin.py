import sys
import os
import pickle
import numpy as np

from Basilisk.utilities import SimulationBaseClass, macros
from Basilisk.simulation import spacecraft
from Basilisk.utilities import vizSupport
from Basilisk.architecture import sysModel

# Importation de ton IA
from Q_learning_avancee import OrbitMindEnv, QLearningAgent

class AIController(sysModel.SysModel):
    def __init__(self, env, agent, scObject):
        super().__init__()
        self.env = env
        self.agent = agent
        self.scObject = scObject
        self.state = self.env._get_observation()
        self.last_action = 0

    def UpdateState(self, CurrentSimNanos):
        # On ne prend une décision que toutes les 15 secondes
        if CurrentSimNanos % macros.sec2nano(15.0) == 0:
            action = self.agent.choose_action(self.state)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            self.state = next_state
            self.last_action = info["actual_action"]
            
            # Logique d'orientation 3D
            if self.last_action == 0: # IDLE
                self.scObject.hub.sigma_BNInit = [[0], [0], [0]]
            elif self.last_action in [1, 2]: # SCIENCE / DOWNLINK
                self.scObject.hub.sigma_BNInit = [[np.pi/2], [0], [0]]

            # OPTIMISATION : On n'affiche le texte que toutes les 1000 secondes
            current_sec = int(CurrentSimNanos * 1e-9)
            if current_sec % 1000 == 0:
                actions_names = ["IDLE", "SCIENCE", "DOWNLINK"]
                print(f"t={current_sec}s | Action: {actions_names[self.last_action]:<8} | Batt: {self.env.true_battery:.1f} Wh")

def run_simulation_with_ai():
    if not os.path.exists('_VizFiles'):
        os.makedirs('_VizFiles')

    print("🧠 Chargement de l'IA...")
    env = OrbitMindEnv()
    agent = QLearningAgent()
    
    with open('orbitmind_brain.pkl', 'rb') as f:
        agent.q_table = pickle.load(f)
    agent.epsilon = 0.0

    test_sat = next(s for s in env.satellites if "Aalto" in s['Nom_Satellite'])
    env.sat = test_sat
    env._load_satellite_physics()

    sim = SimulationBaseClass.SimBaseClass()
    process = sim.CreateNewProcess("proc")
    # Pas de temps de 1 seconde
    task = sim.CreateNewTask("task", macros.sec2nano(1.0)) 

    scObject = spacecraft.Spacecraft()
    scObject.ModelTag = "OrbitMindSat"
    
    r = 6371 * 1000 + 500 * 1000
    mu = 398600.4418 * 1e9
    v = np.sqrt(mu / r)
    scObject.hub.r_CN_NInit = [[r], [0], [0]]
    scObject.hub.v_CN_NInit = [[0], [v], [0]]

    sim.AddModelToTask("task", scObject)

    vizSupport.enableUnityVisualization(
        sim, "task", scObject,
        saveFile="_VizFiles/orbitmind_ai.bin",
        liveStream=False
    )

    ai_model = AIController(env, agent, scObject)
    sim.AddModelToTask("task", ai_model, 100)

    print("🚀 Génération du fichier .bin en cours... (Veuillez patienter 1 à 2 minutes)")
    
    sim.InitializeSimulation()
    # OPTIMISATION : On simule seulement 1 orbite (5677 secondes) au lieu de 2
    sim.ConfigureStopTime(macros.sec2nano(5677)) 
    sim.ExecuteSimulation()

    print(f"\n✅ Fichier généré avec succès : _VizFiles/orbitmind_ai.bin")

if __name__ == "__main__":
    run_simulation_with_ai()