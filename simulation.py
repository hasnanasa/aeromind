import numpy as np
import matplotlib.pyplot as plt

class OrbitMindEnv:
    def __init__(self):
        # --- Power Subsystem ---
        self.battery_capacity = 1000.0  
        self.battery_energy = 800.0     
        self.min_energy = 100.0         
        self.solar_max_power = 150.0    
        self.dt = 1.0 
        
        # --- Subsystems ---
        self.science_power = 60.0       # Power to run science
        self.comms_power = 100.0        # Power to transmit data
        self.idle_power = 20.0          # Base housekeeping power
        
        # --- Data Buffer ---
        self.data_buffer = 0.0          # MB of data waiting to be sent
        self.max_buffer = 500.0
        self.science_rate = 2.0         # MB generated per step
        self.comms_rate = 5.0           # MB sent per step
        
        # --- Faults ---
        self.solar_fault = False
        self.sensor_fault = False
        self.comms_fault = False        # If True, comms draws power but sends NO data
        
    def get_solar_power(self, time_step):
        # 1 orbit = 1000 steps. 400 steps of eclipse.
        if (time_step % 1000) < 400: 
            base_solar = 0.0 
        else:
            base_solar = self.solar_max_power * np.sin(np.pi * ((time_step % 1000) - 400) / 600)
            
        if self.solar_fault:
            return base_solar * 0.4 # 60% power loss
        return base_solar

    def step(self, time_step, action):
        solar_gen = self.get_solar_power(time_step)
        
        # --- Action Logic ---
        # Actions: 0 = IDLE, 1 = SCIENCE, 2 = DOWNLINK (Send Data)
        power_draw = self.idle_power
        data_generated = 0.0
        data_sent = 0.0
        
        if action == "SCIENCE":
            power_draw += self.science_power
            data_generated = self.science_rate
            
        elif action == "DOWNLINK":
            power_draw += self.comms_power
            if not self.comms_fault:
                # Send data if buffer has some
                data_sent = min(self.comms_rate, self.data_buffer)
                
        # --- Update Physics ---
        net_power = solar_gen - power_draw
        self.battery_energy += net_power * self.dt
        self.battery_energy = max(0.0, min(self.battery_energy, self.battery_capacity))
        
        # --- Update Data Buffer ---
        self.data_buffer += data_generated
        self.data_buffer -= data_sent
        self.data_buffer = max(0.0, min(self.data_buffer, self.max_buffer))
        
        return self.battery_energy, self.data_buffer, power_draw

    def get_telemetry(self):
        # AI sees this data. Sensor fault adds a +200 bias to energy.
        measured_energy = self.battery_energy + (200.0 if self.sensor_fault else 0.0)
        return measured_energy, self.data_buffer

# --- BASELINE CONTROLLER (Rule-Based) ---
def baseline_controller(measured_energy, data_buffer):
    # Rule 1: If battery is low, go IDLE.
    if measured_energy < 250.0:
        return "IDLE"
    # Rule 2: If data buffer is almost full, send data.
    if data_buffer > 400.0:
        return "DOWNLINK"
    # Rule 3: Otherwise, do science.
    return "SCIENCE"

# --- RUN SIMULATION ---
def run_simulation():
    env = OrbitMindEnv()
    
    # History tracking
    h_energy, h_data, h_action = [], [], []
    total_data_sent = 0.0

    for t in range(3000):
        # Inject Faults at T=1500
        if t == 1500:
            print("🚨 FAULT INJECTED: Solar Degradation + Sensor Bias + Comms Antenna Failure!")
            env.solar_fault = True
            env.sensor_fault = True
            env.comms_fault = True

        # 1. AI Observes Telemetry
        measured_e, buffer = env.get_telemetry()
        
        # 2. Controller decides Action
        action = baseline_controller(measured_e, buffer)
        
        # 3. Environment steps forward
        actual_e, data, power = env.step(t, action)
        
        # Tracking
        h_energy.append(actual_e)
        h_data.append(data)
        h_action.append(1 if action == "SCIENCE" else (2 if action == "DOWNLINK" else 0))
        
        # Calculate mission success (data sent to Earth)
        if action == "DOWNLINK" and not env.comms_fault:
            total_data_sent += min(env.comms_rate, data)

    print(f"✅ Mission Result: Total Data Sent to Earth = {total_data_sent:.2f} MB")
    
    # --- PLOTTING ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    
    ax1.plot(h_energy, label='TRUE Battery', color='b', linewidth=2)
    ax1.axhline(y=env.min_energy, color='r', linestyle='--', label='Min Safety Limit')
    ax1.set_title('Battery Energy')
    ax1.legend()
    
    ax2.plot(h_data, label='Data Buffer (MB)', color='purple')
    ax2.set_title('Data Waiting to be Sent')
    ax2.legend()

    ax3.plot(h_action, label='Action (0=Idle, 1=Science, 2=Comms)', color='g')
    ax3.set_title('Spacecraft Actions')
    ax3.set_xlabel('Time Step')
    ax3.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation()