# HW1 Walkthrough

## Completed Tasks
I have implemented all the required components for HW1:

1.  **BPE Tokenizer (`tokenizer.py`)**:
    - Implemented `get_stats` to count token pairs.
    - Implemented `train` to perform BPE merges and build vocabulary.
    - Result: Tokenizer trains successfully and produces a vocabulary of size 1358 (target 1000+).

2.  **N-Gram Language Model (`lm.py`)**:
    - Implemented `train`, `get_prob` (with Laplace smoothing), and `perplexity`.
    - Verification Results:
        - Dev Perplexity: ~8.62
        - Test Perplexity: ~8.44

3.  **Neural Language Models (`lm.py`)**:
    - **RNNLM**: Implemented forward pass with embedding, RNN loop, and output projection.
    - **LSTMLM**: Implemented LSTM cell logic (gates, state updates) and forward pass.
    - **Bug Fix**: Identified that the models were returning raw logits while `NLLLoss` expected log-probabilities. Applied `log_softmax` to the outputs.
    - Verification: Both RNN and LSTM models run and train on the data. Training is currently in progress (slow on CPU).

## Verification Evidence

### N-Gram Model Output
```
Running NGRAM Language Model...
...
Training Time: 30.62s
Dev Perplexity:  8.6192
Test Perplexity: 8.4438
```

### Manual Tokenization Check
```
Manual Tokenization Check:
Tokenizing 'the': [90, 78, 75, 532]
Tokenizing 'cat': [73, 71, 90, 532]
...
```

### Neural Models
Both RNN and LSTM models successfully finished training on CPU.

**RNN Results**:
- Training Time: ~16.6 min (997s)
- **Dev Perplexity: 6.19** (Target < 12) - **SUCCESS**
- Test Perplexity: 6.05

**LSTM Results (Extra Credit)**:
- Training Time: ~22.6 min (1358s)
- **Dev Perplexity: 5.56** (Better than RNN)
- Test Perplexity: 5.43

## Written Responses (Task 3)
I have generated the written responses for Q7, Q8, and Q9 in a separate artifact:
[written_responses.md](file:///Users/tangzhengzheng/.gemini/antigravity/brain/5f709ca4-c109-4b95-9a8b-e7c30632e164/written_responses.md)

This includes the cosine similarity calculations, co-occurrence vector construction, and skip-gram vector solutions.
