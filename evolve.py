import subprocess
import random
import json
import time
import sys
from judge import evaluate_names

# Configuration for Evolution
POPULATION_SIZE = 6
GENERATIONS = 5      # Reduced for interactive speed
ELITISM = 2          # Number of top performers to keep
MUTATION_RATE = 0.5

# Fixed constraints
TRAIN_STEPS = 300    # Reduced to prevent timeouts on the unoptimized engine
INPUT_FILE = "input.txt"

class Genome:
    def __init__(self, emb=None, head=None, layer=None, lr=None):
        self.n_emb = emb if emb else random.choice([16, 24, 32]) # Kept small for performance
        self.n_head = head if head else random.choice([2, 4])
        # Constraint: emb must be divisible by head
        if self.n_emb % self.n_head != 0:
            self.n_head = 2
            
        self.n_layer = layer if layer else random.randint(1, 3)
        self.n_ctx = 16 
        self.lr = lr if lr else random.uniform(0.001, 0.015)
        
        self.fitness = 0.0
        self.names = []

    def mutate(self):
        choice = random.choice(['emb', 'head', 'layer', 'lr'])
        
        if choice == 'emb':
            self.n_emb = random.choice([16, 24, 32, 40])
        elif choice == 'head':
            self.n_head = random.choice([2, 4])
        elif choice == 'layer':
            self.n_layer = max(1, min(4, self.n_layer + random.choice([-1, 1])))
        elif choice == 'lr':
            self.lr = max(0.0001, min(0.1, self.lr * random.uniform(0.7, 1.3)))
            
        # Enforce constraints
        if self.n_emb % self.n_head != 0:
            self.n_emb = (self.n_emb // self.n_head) * self.n_head
            if self.n_emb == 0: self.n_emb = self.n_head

    def __str__(self):
        return f"[Emb:{self.n_emb} Head:{self.n_head} Lay:{self.n_layer} LR:{self.lr:.5f}]"

    def run(self):
        """Runs the MicroGPT binary with this genome's config and returns generated names."""
        cmd = [
            "./target/release/microgpt_rust",
            "--steps", str(TRAIN_STEPS),
            "--emb", str(self.n_emb),
            "--head", str(self.n_head),
            "--layer", str(self.n_layer),
            "--lr", str(self.lr),
            "--ctx", str(self.n_ctx),
            "--silent"
        ]
        
        try:
            t0 = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            t_run = time.time() - t0
            
            if result.returncode != 0:
                print(f"Error: {result.stderr}")
                return [], 0.0
                
            lines = result.stdout.strip().split('\n')
            names = [l.strip() for l in lines if l.strip()]
            return names, t_run
        except subprocess.TimeoutExpired:
            print("Timed out!")
            return [], 120.0
        except Exception as e:
            print(f"Exception: {e}")
            return [], 0.0

def run_evolution():
    print(f"--- Starting Aesthetic Evolution (Pop: {POPULATION_SIZE}, Gens: {GENERATIONS}) ---")
    
    # Initialize Population
    population = [Genome() for _ in range(POPULATION_SIZE)]
    
    for gen in range(GENERATIONS):
        print(f"\n=== Generation {gen+1}/{GENERATIONS} ===")
        gen_start = time.time()
        
        # 1. Evaluate Fitness
        for i, individual in enumerate(population):
            print(f"Org {i+1}: {individual} ... ", end='', flush=True)
            
            # Skip evaluation if already evaluated (for elites)
            if individual.fitness != 0.0 and individual.names:
                print(f"(Cached) -> {individual.fitness:.4f}")
                continue

            names, t_run = individual.run()
            
            t_judge_start = time.time()
            score = evaluate_names(names)
            t_judge = time.time() - t_judge_start
            
            individual.fitness = score
            individual.names = names
            
            print(f" Run: {t_run:.2f}s | Judge: {t_judge:.4f}s -> Score: {score:.4f}")
            if names:
                print(f"    Sample: {', '.join(names[:3])}")

        gen_time = time.time() - gen_start
        print(f"--- Generation Time: {gen_time:.2f}s ---")

        # 2. Sort by Fitness
        population.sort(key=lambda x: x.fitness, reverse=True)
        best = population[0]
        print(f"\n>> Gen {gen+1} Champion: {best}")
        print(f">> Score: {best.fitness:.4f}")
        print(f">> Names: {', '.join(best.names[:5])}")

        # 3. Selection & Reproduction (Elitism)
        if gen < GENERATIONS - 1:
            new_pop = population[:ELITISM] # Keep elites
            
            # Fill rest with mutated offspring of elites
            while len(new_pop) < POPULATION_SIZE:
                parent = random.choice(population[:ELITISM]) 
                child = Genome(parent.n_emb, parent.n_head, parent.n_layer, parent.lr)
                child.mutate()
                child.fitness = 0.0 # Reset fitness for new child
                new_pop.append(child)
                
            population = new_pop

    print("\n--- Evolution Complete ---")
    champion = population[0]
    print(f"Final Champion: {champion}")
    print(f"Score: {champion.fitness}")
    print("Generated Names:")
    for n in champion.names:
        print(f" - {n}")

if __name__ == "__main__":
    run_evolution()
