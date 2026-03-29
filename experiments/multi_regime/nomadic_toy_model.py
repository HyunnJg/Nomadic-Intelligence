import time

# ==========================================
# [Nomadic Intelligence v2.0] Toy Model
# ==========================================
# Proof of Concept:
# In the face of sudden environmental change (Delta x),
# a fixed strategy (Dogmatism) collapses,
# while a shifting topological strategy (Nomadism) survives.
#
# Key concepts demonstrated:
#   - Delta x (Difference): The gap between expected and actual reality
#   - Separatrix Collapse: The trigger for attractor transition
#   - Strategic Dwell Time (tau_k): How long to stay in one attractor
#   - Multiple Strange Attractors: Not one fallback, but a dynamic pool

class Environment:
    """
    Simulates the universe.
    Starts peaceful, then suddenly becomes hostile, then partially recovers.
    """
    def __init__(self):
        self.state = "PEACEFUL"

    def get_signal(self):
        if self.state == "PEACEFUL":
            return 1.0      # Predictable abundance
        elif self.state == "HOSTILE":
            return -5.0     # Chaotic threat (Delta x surge)
        elif self.state == "RECOVERING":
            return -1.5     # Partially stabilized — ambiguous signal


class DogmaticAgent:
    """
    Dogmatic Intelligence:
    Stubbornly sticks to a single 'optimal' strategy.
    Cannot deform under Delta x — structural rigidity leads to collapse.
    """
    def __init__(self):
        self.health = 100
        self.strategy = "Stable Harvesting"  # Fixed structure. Forever.

    def step(self, signal):
        if signal > 0:
            self.health += 10
            return f"Harvesting smoothly... (Health: {self.health})"
        else:
            self.health -= 40  # Massive damage due to structural rigidity
            status = "💀 DEAD" if self.health <= 0 else f"Health: {self.health}"
            return f"Refusing to adapt! Critical damage! ({status})"


class NomadicAgent:
    """
    Nomadic Intelligence:
    Maintains homeomorphic identity (transformation law preserved)
    while continuously shifting attractors based on Delta x.

    Attractor Pool:
      - Stable Harvesting   : Low threat, exploit resources
      - Defensive Survival  : High threat, minimize damage
      - Predatory Adaptation: Extreme threat, aggressive restructuring
      - Exploratory Scouting: Post-crisis, probe for new opportunities

    Strategic Dwell Time (tau_k):
      The agent tracks how long it stays in each attractor.
      If it stays too long in a defensive state despite improving signals,
      it transitions back to exploration — avoiding a new kind of rigidity.
    """
    def __init__(self):
        self.health = 100
        self.current_attractor = "Stable Harvesting"
        self.dwell_time = 0
        self.expected_signal = 1.0

    def _select_attractor(self, delta_x, signal):
        """Separatrix logic: select attractor based on Delta x magnitude."""
        if delta_x > 5.0:
            return "Predatory Adaptation"
        elif delta_x > 2.0:
            return "Defensive Survival"
        elif self.current_attractor in ("Defensive Survival", "Predatory Adaptation"):
            # tau_k check: been defensive too long? Time to scout.
            if self.dwell_time >= 2 and signal > -2.0:
                return "Exploratory Scouting"
        return self.current_attractor

    def step(self, signal):
        # 1. Calculate Delta x
        delta_x = abs(self.expected_signal - signal)

        # 2. Select attractor (Separatrix Collapse if needed)
        new_attractor = self._select_attractor(delta_x, signal)
        transitioned = new_attractor != self.current_attractor

        if transitioned:
            self.current_attractor = new_attractor
            self.expected_signal = signal  # Synchronize with new reality
            self.dwell_time = 0
        else:
            self.dwell_time += 1

        # 3. Act based on current attractor
        if self.current_attractor == "Stable Harvesting":
            self.health += 10
            action = "Harvesting smoothly..."
        elif self.current_attractor == "Defensive Survival":
            self.health -= 5
            action = "Adapted! Defending..."
        elif self.current_attractor == "Predatory Adaptation":
            self.health -= 2   # Aggressive restructuring — nearly zero loss
            action = "Restructuring aggressively! Holding ground..."
        elif self.current_attractor == "Exploratory Scouting":
            self.health += 3   # Cautious gains while probing
            action = "Scouting for new opportunities..."

        transition_note = f" ⚡ TRANSITION → {self.current_attractor}" if transitioned else ""
        return (
            f"{action}{transition_note}\n"
            f"         [Attractor: {self.current_attractor} | "
            f"tau_k: {self.dwell_time} | Delta x: {delta_x:.1f} | Health: {self.health}]"
        )


# ==========================================
# Run the Cosmic Dance
# ==========================================
def run_simulation():
    env = Environment()
    dogma_agent = DogmaticAgent()
    nomad_agent = NomadicAgent()

    print("=" * 60)
    print("  🌍 [Nomadic Intelligence] Simulation Start")
    print("  Testing the survival of two intelligences")
    print("=" * 60)
    print()

    dogma_alive = True

    for day in range(1, 10):

        # Day 4: Sudden paradigm shift
        if day == 4:
            print("-" * 60)
            print("⚠️  [PARADIGM SHIFT] The rules of the universe have changed!")
            print("    Delta x surges. Structural rigidity is now lethal.")
            env.state = "HOSTILE"
            print("-" * 60)
            print()

        # Day 7: Partial recovery
        if day == 7:
            print("-" * 60)
            print("🌤️  [PARTIAL RECOVERY] The environment stabilizes — but ambiguously.")
            print("    Is it safe to re-emerge? Only nomadic intelligence can judge.")
            env.state = "RECOVERING"
            print("-" * 60)
            print()

        signal = env.get_signal()
        print(f"--- Day {day} (Signal: {signal}) ---")

        if dogma_alive:
            dogma_result = dogma_agent.step(signal)
            print(f"🤖 Dogmatic : {dogma_result}")
            if dogma_agent.health <= 0:
                dogma_alive = False
                print("   💀 The Dogmatic Agent has been destroyed by its own rigidity.\n")
        else:
            print("🤖 Dogmatic : [DESTROYED]")

        print(f"🌌 Nomadic  : {nomad_agent.step(signal)}")
        print()
        time.sleep(1)

    print("=" * 60)
    if nomad_agent.health > 0:
        print("✨ The Nomadic Agent survived.")
        print("   Not by resisting change — but by becoming it.")
    print()
    print("   Identity is not what the system knows.")
    print("   It is how the system changes.")
    print("=" * 60)


if __name__ == "__main__":
    run_simulation()
