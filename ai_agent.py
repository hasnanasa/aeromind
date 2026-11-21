import numpy as np
import random
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces

# ==========================================
# 1. THE ENVIRONMENT (Fixed Physics)
# ==========================================
class OrbitMindEnv(gym.Env):
    def __init__(self):
        super(OrbitMindEnv, self).__init__()
        
        # Physics parameters
        self.battery_capacity = 1000.0  # Wh (Realistic for small sat)
        self.min_energy = 100.0         
        self.solar_max_power = 150.0    # W
        self.dt = 1.0                   # 1 second per step
        self.max_buffer = 500.0         # MB
        
        # Actions: 0=IDLE, 1=SCIENCE, 2=DOWNLINK
        self.action_space = spaces.Discrete(3)
        
        # Observations: [Energy Bucket (0-2), Buffer Bucket (0-2), Solar Bucket (0-1)]
        self.observation_space = spaces.MultiDiscrete([3, 3, 2])
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.time_step = 0
        self.battery_energy = 800.0
        self.data_buffer = 0.0
        self.solar_fault = False
        self.sensor_fault = False
        self.comms_fault = False
        return self._get_obs(), {}

    def _get_solar_power(self, t):
        if (t % 1000) < 400: base_solar = 0.0 
        else: base_solar = self.solar_max_power * np.sin(np.pi * ((t % 1000) - 400) / 600)
        return base_solar * 0.4 if self.solar_fault else base_solar

    def _get_obs(self):
        # Discretize state for the Q-Table
        e = self.battery_energy + (200.0 if self.sensor_fault else 0.0) # Sensor lies!
        e_bin = 0 if e < 250 else (1 if e < 600 else 2) 
        
        b = self.data_buffer
        b_bin = 0 if b < 150 else (1 if b < 350 else 2) 
        
        s = self._get_solar_power(self.time_step)
        s_bin = 0 if s < 10.0 else 1 
        
        return (e_bin, b_bin, s_bin)

    def step(self, action):
        self.time_step += 1
        
        # Inject faults at step 1500
        if self.time_step == 1500:
            self.solar_fault = True
            self.sensor_fault = True
            self.comms_fault = True

        solar_gen = self._get_solar_power(self.time_step)
        power_draw = 20.0 
        data_generated = 0.0
        data_sent = 0.0
        
        if action == 1: # SCIENCE
            power_draw += 60.0
            data_generated = 2.0
        elif action == 2: # DOWNLINK
            power_draw += 100.0
            if not self.comms_fault:
                data_sent = min(5.0, self.data_buffer)
                
        # --- FIXED PHYSICS MATH HERE ---
        # Power is in Watts (Joules/sec). Battery is in Watt-hours.
        # Convert seconds to hours: dt / 3600
        net_power = solar_gen - power_draw
        energy_change_wh = net_power * (self.dt / 3600.0) 
        
        self.battery_energy = max(0.0, min(self.battery_energy + energy_change_wh, self.battery_capacity))
        self.data_buffer = max(0.0, min(self.data_buffer + data_generated - data_sent, self.max_buffer))
        
        # Calculate Reward
        reward = 0
        terminated = False
        
        if action == 2 and not self.comms_fault and data_sent > 0:
            reward += 2.0 # Good! You sent data!
        if self.data_buffer >= self.max_buffer - 1.0:
            reward -= 2.0 # Bad! Buffer is full, losing science!
        if self.battery_energy <= 10.0:
            reward -= 100.0 
            terminated = True
            
        if self.time_step >= 3000:
            terminated = True
            
        return self._get_obs(), reward, terminated, False, {}

# ==========================================
# 2. THE AI AGENT (Q-Learning)
# ==========================================
class QLearningAgent:
    def __init__(self):
        self.q_table = np.zeros((3, 3, 2, 3)) 
        self.alpha = 0.1       
        self.gamma = 0.95      
        self.epsilon = 1.0     
        self.epsilon_decay = 0.9995 # Slower decay so it explores more
        self.epsilon_min = 0.05

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 2) 
        return np.argmax(self.q_table[state]) 

    def learn(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# ==========================================
# 3. TRAINING LOOP
# ==========================================
def train():
    env = OrbitMindEnv()
    agent = QLearningAgent()
    
    episodes = 5000 # Increased episodes for better learning
    rewards_history = []
    
    print("Starting Training... (This will take about 30 seconds)")
    
    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.learn(state, action, reward, next_state)
            
            state = next_state
            total_reward += reward
            
        rewards_history.append(total_reward)
        
        if (ep+1) % 1000 == 0:
            print(f"Episode {ep+1}/{episodes} | Total Reward: {total_reward:.1f} | Epsilon: {agent.epsilon:.2f}")

    print("Training Complete!")
    
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history, color='blue', alpha=0.2, label='Episode Reward')
    moving_avg = np.convolve(rewards_history, np.ones(200)/200, mode='valid')
    plt.plot(moving_avg, color='red', linewidth=2, label='200-Episode Moving Average')
    plt.title("AI Learning Curve (Fixed Physics)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train()