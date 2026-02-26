import sys
import math
from collections import Counter

# Load training data to penalize memorization
try:
    with open("input.txt", "r") as f:
        TRAINING_DATA = set(line.strip().lower() for line in f)
except FileNotFoundError:
    TRAINING_DATA = set()

def score_flow(name):
    """
    Scores how 'pronounceable' the name is.
    Heuristic: Alternating vowel/consonant patterns are good.
    Long strings of consonants or vowels are bad.
    """
    vowels = set("aeiouy")
    score = 0
    consecutive_v = 0
    consecutive_c = 0
    
    for char in name:
        if char in vowels:
            consecutive_v += 1
            consecutive_c = 0
        else:
            consecutive_c += 1
            consecutive_v = 0
        
        if consecutive_v > 2 or consecutive_c > 2:
            score -= 1.0  # Penalty for unpronounceable clusters
    
    # Bonus for ideal length (4-8 chars)
    if 4 <= len(name) <= 8:
        score += 0.5
        
    return score

def score_symmetry(name):
    """
    Scores visual or phonetic symmetry.
    """
    score = 0
    
    # Palindrome check (high value)
    if name == name[::-1] and len(name) > 3:
        score += 2.0
        
    # Repeating sub-patterns (e.g. "lala", "nana")
    if len(name) >= 4:
        mid = len(name) // 2
        if name[:mid] == name[mid:mid*2]:
            score += 1.5

    # End rhyme with common soft endings
    if name.endswith("a") or name.endswith("n") or name.endswith("y"):
        score += 0.2
        
    return score

def score_creativity(name):
    """
    Scores uniqueness. 
    Penalty for memorizing training data.
    Reward for being new but valid-looking.
    """
    if name in TRAINING_DATA:
        return -5.0  # Heavy penalty for pure memorization
    return 1.0       # Reward for novelty

def evaluate_names(names):
    if not names:
        return -100.0
    
    total_score = 0
    valid_names = 0
    
    for name in names:
        name = name.strip().lower()
        if not name or len(name) < 3: 
            continue
            
        # Clean generated text (sometimes it might have artifacts)
        if not name.isalpha():
            continue
            
        s_flow = score_flow(name)
        s_sym = score_symmetry(name)
        s_creat = score_creativity(name)
        
        # Weighted sum
        # We prioritize Flow and Creativity to get "Beautiful, New Names"
        item_score = (s_flow * 1.0) + (s_sym * 1.2) + (s_creat * 2.0)
        
        total_score += item_score
        valid_names += 1
        
    if valid_names == 0:
        return -100.0
        
    return total_score / valid_names

if __name__ == "__main__":
    # Read names from stdin
    input_names = [line.strip() for line in sys.stdin if line.strip()]
    
    # If passed as arguments instead
    if len(sys.argv) > 1:
        input_names = sys.argv[1:]
        
    final_score = evaluate_names(input_names)
    print(f"{final_score:.4f}")
