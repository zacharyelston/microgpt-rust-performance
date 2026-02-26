import subprocess
import random
import json
import time
import sys
import concurrent.futures
import os
from judge import evaluate_names

# Configuration for Evolution
POPULATION_SIZE = 12    # Increased since we have parallel processing
GENERATIONS = 5
ELITISM = 2
MUTATION_RATE = 0.5
MAX_WORKERS = os.cpu_count() # Use all available cores

# Fixed constraints
TRAIN_STEPS = 300
INPUT_FILE = "input.txt"

class Genome:
    def __init__(self, emb=None, head=None, layer=None, lr=None):
        self.n_emb = emb if emb else random.choice([16, 24, 32])
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
                # print(f"Error: {result.stderr}") # Silence error for cleaner parallel output
                return [], 0.0
                
            lines = result.stdout.strip().split('\n')
            names = [l.strip() for l in lines if l.strip()]
            return names, t_run
        except subprocess.TimeoutExpired:
            return [], 120.0
        except Exception:
            return [], 0.0

def evaluate_one(individual):
    """Helper function for parallel execution"""
    # Skip if cached (though in parallel map we re-eval usually, or handle in loop)
    # But for simplicity, we just run.
    names, t_run = individual.run()
    
    t_judge_start = time.time()
    score = evaluate_names(names)
    t_judge = time.time() - t_judge_start
    
    return names, score, t_run, t_judge

def run_evolution():
    print(f"--- Starting Aesthetic Evolution (Pop: {POPULATION_SIZE}, Gens: {GENERATIONS}, Workers: {MAX_WORKERS}) ---")
    
    # Initialize Population
    population = [Genome() for _ in range(POPULATION_SIZE)]
    
    for gen in range(GENERATIONS):
        print(f"\n=== Generation {gen+1}/{GENERATIONS} ===")
        gen_start = time.time()
        
        # Prepare list for evaluation (filter out cached elites if needed, but easier to just eval all or track indices)
        # We need to map only the ones that need evaluation
        to_eval = []
        indices = []
        
        for i, ind in enumerate(population):
            if ind.fitness != 0.0 and ind.names:
                print(f"Org {i+1}: {ind} ... (Cached) -> {ind.fitness:.4f}")
            else:
                to_eval.append(ind)
                indices.append(i)
        
        if to_eval:
            print(f"Evaluating {len(to_eval)} organisms in parallel...")
            with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                results = list(executor.map(evaluate_one, to_eval))
                
            # Update population with results
            for idx, (names, score, t_run, t_judge) in zip(indices, results):
                ind = population[idx]
                ind.names = names
                ind.fitness = score
                print(f"Org {idx+1}: {ind} ... Run: {t_run:.2f}s | Judge: {t_judge:.4f}s -> Score: {score:.4f}")
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
