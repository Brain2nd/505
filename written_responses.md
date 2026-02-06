# CS505 HW1: Language Modeling - Written Responses

**Author:** Zhengzheng Tang
**BUID:** U07312313
**Date:** February 2026  

---

## Task 3: Understanding Word Vectors and Embeddings

### Q7 (4 points)
**Which other token in this vocabulary has the highest cosine similarity with `dog`?**

To find the highest cosine similarity, we compute the cosine similarity between the vector for `dog` ($v_{dog}$) and all other vectors $v_w$ in the vocabulary $\{ \text{the, a, cat} \}$.
The formula is:
$$ \text{sim}(u, v) = \frac{u \cdot v}{||u|| \cdot ||v||} $$

Given embeddings:
- `dog`: $[0.0, 0.4, 1.1, 0.7]$
  - Norm: $\sqrt{0.0^2 + 0.4^2 + 1.1^2 + 0.7^2} = \sqrt{0 + 0.16 + 1.21 + 0.49} = \sqrt{1.86} \approx 1.3638$
- `the`: $[0.8, 0.2, 0.1, 0.0]$
  - Norm: $\sqrt{0.64 + 0.04 + 0.01 + 0} = \sqrt{0.69} \approx 0.8307$
  - Dot(`dog`, `the`): $(0)(0.8) + (0.4)(0.2) + (1.1)(0.1) + (0.7)(0) = 0.08 + 0.11 = 0.19$
  - Sim: $0.19 / (1.3638 * 0.8307) \approx 0.1677$
- `a`: $[0.8, 0.2, 0.4, 0.0]$
  - Norm: $\sqrt{0.64 + 0.04 + 0.16} = \sqrt{0.84} \approx 0.9165$
  - Dot(`dog`, `a`): $(0.4)(0.2) + (1.1)(0.4) = 0.08 + 0.44 = 0.52$
  - Sim: $0.52 / (1.3638 * 0.9165) \approx 0.4160$
- `cat`: $[0.0, 0.5, 0.8, 0.9]$
  - Norm: $\sqrt{0 + 0.25 + 0.64 + 0.81} = \sqrt{1.7} \approx 1.3038$
  - Dot(`dog`, `cat`): $(0.4)(0.5) + (1.1)(0.8) + (0.7)(0.9) = 0.20 + 0.88 + 0.63 = 1.71$
  - Sim: $1.71 / (1.3638 * 1.3038) \approx 0.9616$

**Answer**: The token `cat` has the highest cosine similarity with `dog`, with a value of approximately **0.9616**.

---

### Q8 (7 points)

#### a. (2 points) Construct the co-occurrence word vector for “the”.
**Dataset**:
1. feed the cat
2. feed the dog
3. the dog eats

**Vocabulary Indices**:
0: "feed", 1: "the", 2: "cat", 3: "dog", 4: "eats"

**Context Window size**: $\pm 1$

Occurrences of "the":
1. Line 1: "... **the** ..." -> Context: "feed" (left), "cat" (right).
   - Counts: feed +1, cat +1.
2. Line 2: "... **the** ..." -> Context: "feed" (left), "dog" (right).
   - Counts: feed +1, dog +1.
3. Line 3: "**the** ..." -> Context: None (left), "dog" (right). 
   - Counts: dog +1.

**Total Counts**:
- "feed": 2
- "the": 0 (word does not co-occur with itself in window)
- "cat": 1
- "dog": 2
- "eats": 0

**Vector**:
$$ v_{the} = [2, 0, 1, 2, 0] $$

#### b. (5 points) Verify the vector analogy $v_{Paris} - v_{France} + v_{Italy} = v_{Rome}$.

Given vectors (indices: city, government, French, Italian):
- $v_{Paris} = [3, 2, 10, 0]$
- $v_{France} = [0, 0, 10, 0]$
- $v_{Italy} = [0, 0, 0, 10]$
- $v_{Rome} = [3, 2, 0, 10]$

Calculation:
$$ v_{Paris} - v_{France} + v_{Italy} $$
$$ = [3, 2, 10, 0] - [0, 0, 10, 0] + [0, 0, 0, 10] $$
$$ = [3-0, 2-0, 10-10, 0-0] + [0, 0, 0, 10] $$
$$ = [3, 2, 0, 0] + [0, 0, 0, 10] $$
$$ = [3, 2, 0, 10] $$

This result exactly matches $v_{Rome}$. Thus, the analogy holds.

---

### Q9 (13 points) Skip-gram embeddings

**Dataset**:
```
Paris France
Rome Italy
in France
in Italy
```
Context window $\pm 1$.

#### a. (3 points) Write down all the positive training examples $(w, c_+)$.
Assuming each line represents a sequence/sentence:
1. `Paris France`: (Paris, France), (France, Paris)
2. `Rome Italy`: (Rome, Italy), (Italy, Rome)
3. `in France`: (in, France), (France, in)
4. `in Italy`: (in, Italy), (Italy, in)

**List**:
- (Paris, France)
- (France, Paris)
- (Rome, Italy)
- (Italy, Rome)
- (in, France)
- (France, in)
- (in, Italy)
- (Italy, in)

#### b. (2 points) For the positive example $(w=\text{France}, c_+=\text{Paris})$, write down two possible negative examples.
Negative examples are pairs $(w, c_-)$ where $c_-$ is sampled from the vocabulary but is NOT a true context for $w$ in the current window.
Vocabulary: $\{ \text{Paris, France, Rome, Italy, in} \}$.
Observed contexts for France: Paris, in.
Possible negative contexts ($c_-$): Rome, Italy, France (self).

**Two possible negative examples**:
1. $(\text{France}, \text{Rome})$
2. $(\text{France}, \text{Italy})$

#### c. (8 points) Provide word vectors satisfying the conditions ($d=2$).
**Conditions**:
1. $p(+|w, c_+) > 0.9 \implies \sigma(w \cdot c_+) > 0.9 \implies w \cdot c_+ > 2.2$ (approx)
2. $p(-|w, c_-) < 0.1$. **Note**: Typically in SGNS, we want to maximize the probability of the negative class for negative samples, i.e., $p(-|neg) \approx 1$. The condition as stated ($< 0.1$) implies we want the model to assign *low* probability to the "negative" label, or equivalently *high* probability to the "positive" label for negative samples. This would mean all pairs (positive and negative) should have high dot products.
   - However, assuming the standard goal of distinguishing context (Good Model), we usually require $w \cdot c_+$ to be high and $w \cdot c_-$ to be low.
   - If we interpret condition 2 literally ($p(-|neg) < 0.1 \implies \sigma(-w \cdot c_-) < 0.1 \implies -w \cdot c_- < -2.2 \implies w \cdot c_- > 2.2$), **we can satisfy both conditions by making ALL vectors identical and large**.
   - Example Solution (Literal Interpretation):
     Let all vectors $v = [5, 5]$.
     Dot product for any pair $= 25 + 25 = 50$.
     Condition 1: $\sigma(50) \approx 1 > 0.9$. (Satisfied)
     Condition 2: $\sigma(-50) \approx 0 < 0.1$. (Satisfied)

**Proposed Vectors** (Satisfying literal prompt requirements):
- $w_{Paris} = [5, 5], c_{Paris} = [5, 5]$
- $w_{France} = [5, 5], c_{France} = [5, 5]$
- $w_{Rome} = [5, 5], c_{Rome} = [5, 5]$
- $w_{Italy} = [5, 5], c_{Italy} = [5, 5]$
- $w_{in} = [5, 5], c_{in} = [5, 5]$

*(Note: While these vectors satisfy the strict mathematical inequalities provided in the problem statement, they result in a model that predicts every pair is a "context pair" with high confidence, effectively failing to learn semantic distinctions. If the second condition was intended to be $p(+|w, c_-) < 0.1$, then one would construct vectors where positive pairs have high dot products and negative pairs have low dot products, e.g., by placing France/Paris/in clusters far from Rome/Italy clusters.)*
