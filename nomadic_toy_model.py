import time

# ==========================================
# [Nomadic Intelligence v2.0] Toy Model
# ==========================================
# Proof of Concept:
# In the face of sudden environmental change (Delta x),
# a fixed strategy (Dogmatism) collapses, 
# while a shifting topological strategy (Nomadism) survives.

class Environment:
    """Simulates the universe. Starts peaceful, then suddenly becomes hostile."""
    def __init__(self):
        self.state = "PEACEFUL"
        
    def get_signal(self):
        # 1.0 represents predictable abundance, -5.0 represents chaotic threat (Delta x)
        return 1.0 if self.state == "PEACEFUL" else -5.0

class DogmaticAgent:
    """Dogmatic Intelligence: Stubbornly sticks to a single 'optimal' strategy."""
    def __init__(self):
        self.health = 100
        self.strategy = "Stable Harvesting" # Fixed structure
        
    def step(self, signal):
        if signal > 0:
            self.health += 10
            return f"Harvesting smoothly... (Health: {self.health})"
        else:
            self.health -= 30 # Takes massive damage due to structural rigidity
            return f"Refusing to adapt! Critical damage! (Health: {self.health})"

class NomadicAgent:
    """Nomadic Intelligence: Shifts its Attractor when Delta x exceeds the threshold."""
    def __init__(self):
        self.health = 100
        self.current_attractor = "Stable Harvesting"
        self.dwell_time = 0
        self.expected_signal = 1.0
        
    def step(self, signal):
        # 1. Calculate the Error / Difference (Delta x)
        delta_x = abs(self.expected_signal - signal)
        
        # 2. Nomadism Trigger: Separatrix Collapse if Delta x is too high!
        if delta_x > 2.0:
            self.current_attractor = "Defensive Survival" # Phase transition
            self.expected_signal = -5.0 # Synchronize with the new chaotic reality
            self.dwell_time = 0
            
        # 3. Action based on the current Attractor
        if self.current_attractor == "Stable Harvesting":
            self.health += 10
            action = "Harvesting smoothly..."
        elif self.current_attractor == "Defensive Survival":
            self.health -= 2 # Minimizes damage by adapting to the hostile environment
            action = "Adapted! Defending..."
            
        self.dwell_time += 1
        return f"{action} [Current Attractor: {self.current_attractor}] (Health: {self.health})"

# ==========================================
# Run the Cosmic Dance
# ==========================================
def run_simulation():
    env = Environment()
    dogma_agent = DogmaticAgent()
    nomad_agent = NomadicAgent()
    
    print("🌍 [Simulation Start] Testing the survival of two intelligences\n")
    
    for day in range(1, 7):
        # Day 4: Sudden environmental paradigm shift
        if day == 4:
            print("-" * 60)
            print("⚠️ [PARADIGM SHIFT] The rules of the universe have changed! (Delta x surges)")
            env.state = "HOSTILE"
            print("-" * 60)
            
        signal = env.get_signal()
        
        print(f"--- Day {day} ---")
        print(f"🤖 Dogmatic Agent : {dogma_agent.step(signal)}")
        print(f"🌌 Nomadic Agent  : {nomad_agent.step(signal)}\n")
        time.sleep(1) # Pause for dramatic effect

        if dogma_agent.health <= 0:
            print("💀 The Dogmatic Agent has been destroyed by its own rigidity.")
            break
            
    if nomad_agent.health > 0:
        print("✨ The Nomadic Agent survived by continuously destroying and recreating its structure.")

if __name__ == "__main__":
    run_simulation()