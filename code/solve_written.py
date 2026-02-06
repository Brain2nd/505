import numpy as np

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def solve_q7():
    print("--- Q7 ---")
    embeddings = {
        'the': np.array([0.8, 0.2, 0.1, 0.0]),
        'a': np.array([0.8, 0.2, 0.4, 0.0]),
        'cat': np.array([0.0, 0.5, 0.8, 0.9]),
        'dog': np.array([0.0, 0.4, 1.1, 0.7])
    }
    
    target = 'dog'
    max_sim = -1
    best_token = None
    
    for token, vec in embeddings.items():
        if token == target:
            continue
        sim = cosine_sim(embeddings[target], vec)
        print(f"Sim(dog, {token}) = {sim:.4f}")
        if sim > max_sim:
            max_sim = sim
            best_token = token
            
    print(f"Highest similarity to '{target}': '{best_token}' with score {max_sim:.4f}")

def solve_q8():
    print("\n--- Q8 ---")
    # Q8a
    # Vocab: feed(0), the(1), cat(2), dog(3), eats(4)
    # Corpus: 
    # feed the cat
    # feed the dog
    # the dog eats
    # Context +/- 1
    # Occurrences of "the":
    # 1. "feed [the] cat" -> context: feed, cat
    # 2. "feed [the] dog" -> context: feed, dog
    # 3. "[the] dog eats" -> context: (Start?), dog ... wait. "the dog eats". Left of "the"? 
    # Usually padded or ignored at start.
    # Line 3 start with "the". So left context empty (or PAD). Right: "dog".
    
    # Vector for "the" (size 5): [count(feed), count(the), count(cat), count(dog), count(eats)]
    # 1: feed, cat -> +1 index 0, +1 index 2
    # 2: feed, dog -> +1 index 0, +1 index 3
    # 3: dog -> +1 index 3
    # Total: feed:2, the:0?, cat:1, dog:2, eats:0
    # Vector: [2, 0, 1, 2, 0] ?
    print("Q8a logic check inside script comments.")

    # Q8b
    # Paris | city x3, government x2, French x10
    # France | French x10
    # Italy | Italian x10
    # Rome | city x3, government x2, Italian x10
    
    # Indices: city, government, French, Italian
    v_Paris = np.array([3, 2, 10, 0])
    v_France = np.array([0, 0, 10, 0])
    v_Italy = np.array([0, 0, 0, 10])
    v_Rome = np.array([3, 2, 0, 10])
    
    result = v_Paris - v_France + v_Italy
    print(f"v_Paris: {v_Paris}")
    print(f"v_France: {v_France}")
    print(f"v_Italy: {v_Italy}")
    print(f"v_Rome: {v_Rome}")
    print(f"Calculated v_Rome = v_Paris - v_France + v_Italy = {result}")
    print(f"Matches v_Rome? {np.array_equal(result, v_Rome)}")

def solve_q9():
    print("\n--- Q9 ---")
    # c. Vectors satisfying conditions.
    # Dataset:
    # Paris France
    # Rome Italy
    # in France
    # in Italy
    
    # Pairs (undirected assumption? Or directed? Usually text is linear).
    # Lines:
    # 1. Paris France -> Pairs: (Paris, France), (France, Paris)
    # 2. Rome Italy -> Pairs: (Rome, Italy), (Italy, Rome)
    # 3. in France -> Pairs: (in, France), (France, in)
    # 4. in Italy -> Pairs: (in, Italy), (Italy, in)
    
    pos_pairs = [
        ('Paris', 'France'), ('France', 'Paris'),
        ('Rome', 'Italy'), ('Italy', 'Rome'),
        ('in', 'France'), ('France', 'in'),
        ('in', 'Italy'), ('Italy', 'in')
    ]
    
    vocab = ['Paris', 'France', 'Rome', 'Italy', 'in']
    
    # Hardcoded solution search? Or Logic?
    # Want w . c > 2.2 (approx) for pos
    # Want w . c < -2.2 (approx) for neg (assuming typo correction p(-|neg) > 0.9 => p(+|neg) < 0.1)
    
    # Structure:
    # France and Italy are "hubs".
    # Paris -> France <- in -> Italy <- Rome
    
    # Let's try:
    # France = [1, 0], Italy = [-1, 0]
    # in = [0, 1] ?
    # France . in = 0 (Neutral). We want high.
    
    # Better:
    # France = [1, 1], in = [1, 1]. Dot = 2.
    # Italy = [-1, 1].
    # in . Italy = 0. Bad.
    
    # Maybe 45 degrees?
    # France = [1, 0]
    # Italy = [0, 1]
    # in = [1, 1] (b/c close to both). 
    # Dot(in, France) = 1. Dot(in, Italy) = 1.
    # Paris = [1, -0.5]? Close to France.
    # Rome = [-0.5, 1]? Close to Italy.
    
    # Need scale.
    # Let's just output raw logic in text.
    pass

if __name__ == "__main__":
    solve_q7()
    solve_q8()
    solve_q9()
