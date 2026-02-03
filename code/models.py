# CS505: NLP - Spring 2026
# Author: Zhengzheng Tang
# BUID: U07312313

import torch
import numpy as np
import math
from collections import Counter, defaultdict
from nltk.tokenize import word_tokenize
import random

# Optional imports for advanced preprocessing (used in BEST model)
try:
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    NLTK_ADVANCED_AVAILABLE = True
except:
    NLTK_ADVANCED_AVAILABLE = False

class BoWFeaturizer:
    """
    This is a bag-of-words featurizer. It uses `build_vocab` to load a list of Examples
    and uses the top `max_vocab_size` words by frequency as its vocabulary.
    For a given Example, it counts the number of instances of each word in
    `self.vocab` and returns this vector of counts.
    """
    def __init__(self, max_vocab_size=10000):
        self.max_vocab_size = max_vocab_size
        self.vocab = {} # mapping word -> index
        self.inverse_vocab = {}
        self.vocab_size = 0

    def build_vocab(self, data):
        counts = Counter()
        # TODO: count the number instances of each token (here, just words and
        # punctuation) in `data`. Filter the vocab down to the `self.max_vocab_size` most
        # frequent tokens, and put these in a variable called `most_common`. HINT:
        # you can use the `word_tokenize` function that's been
        # imported above to tokenize the string.
        # STUDENT START ---------------------------------
        for ex in data:
            tokens = word_tokenize(ex.text.lower())
            counts.update(tokens)

        most_common = counts.most_common(self.max_vocab_size)
        # STUDENT END ------------------------------------

        # you might need to remove the `count` variable here, depending on how you
        # implemented the above.
        self.vocab = {word: idx for idx, (word, count) in enumerate(most_common)}
        self.inverse_vocab = {idx: word for idx, (word, count) in enumerate(most_common)}
        self.vocab_size = len(self.vocab)

        print(f"Vocabulary built with {self.vocab_size} words.")

    def get_feature_vector(self, text):
        # TODO: Return a bag-of-words feature vector. Each index in
        # the vocabulary should have a corresponding index in this vector.
        # A token's vector index should contain the frequency of that token
        # in `text`.
        # This shold return a torch tensor of size (vocab_size,).
        # STUDENT START -------------------------
        vec = torch.zeros(self.vocab_size)
        tokens = word_tokenize(text.lower())
        for token in tokens:
            if token in self.vocab:
                vec[self.vocab[token]] += 1
        return vec
        # STUDENT END ---------------------------


class BigramFeaturizer(BoWFeaturizer):
    def build_vocab(self, data):
        counts = Counter()
        for ex in data:
            tokens = word_tokenize(ex.text.lower())
            # TODO: generate bigrams
            # STUDENT START ----------------------------
            for i in range(len(tokens) - 1):
                bigram = tokens[i] + "|" + tokens[i + 1]
                counts[bigram] += 1
            # STUDENT END ------------------------------

        # TODO: build your vocabulary of the `self.max_vocab_size` most frequent bigrams.
        # STUDENT START -------------------------------------------
        most_common = counts.most_common(self.max_vocab_size)
        self.vocab = {bigram: idx for idx, (bigram, _) in enumerate(most_common)}
        self.inverse_vocab = {idx: bigram for idx, (bigram, _) in enumerate(most_common)}
        self.vocab_size = len(self.vocab)
        print(f"Bigram vocabulary built with {self.vocab_size} bigrams.")
        # STUDENT END ---------------------------------------------

    def get_feature_vector(self, text):
        tokens = word_tokenize(text.lower())
        vec = torch.zeros(self.vocab_size)

        # TODO: use the list of tokens to generate bigram features.
        # Return the bigram feature vector.
        # STUDENT START --------------------------------------
        for i in range(len(tokens) - 1):
            bigram = tokens[i] + "|" + tokens[i + 1]
            if bigram in self.vocab:
                vec[self.vocab[bigram]] += 1
        return vec
        # STUDENT END -----------------------------------------


class TFIDFFeaturizer:
    """
    TF-IDF Featurizer: Applies TF-IDF weighting to reduce the impact of common words
    and emphasize discriminative terms.
    """
    def __init__(self, max_vocab_size=8000, use_stopwords=False, use_stemming=False):
        self.max_vocab_size = max_vocab_size
        self.vocab = {}
        self.inverse_vocab = {}
        self.vocab_size = 0
        self.idf = {}  # Inverse Document Frequency for each word
        self.num_docs = 0

        # Preprocessing options (only enable if NLTK advanced features available)
        self.use_stopwords = use_stopwords and NLTK_ADVANCED_AVAILABLE
        self.use_stemming = use_stemming and NLTK_ADVANCED_AVAILABLE

        # Initialize stopwords and stemmer
        if self.use_stopwords:
            self.stopwords = set(stopwords.words('english'))
        else:
            self.stopwords = set()

        if self.use_stemming:
            self.stemmer = PorterStemmer()
        else:
            self.stemmer = None

    def _preprocess_tokens(self, tokens):
        """Apply stopwords filtering and stemming to tokens"""
        # Filter: keep only alphanumeric tokens not in stopwords
        if self.use_stopwords:
            tokens = [t for t in tokens if t.isalnum() and t not in self.stopwords]
        else:
            tokens = [t for t in tokens if t.isalnum()]

        # Apply stemming
        if self.use_stemming and self.stemmer:
            tokens = [self.stemmer.stem(t) for t in tokens]

        return tokens

    def build_vocab(self, data):
        """Build vocabulary and compute IDF values"""
        word_counts = Counter()
        doc_freq = Counter()  # Number of documents containing each word
        self.num_docs = len(data)

        # First pass: count word frequencies and document frequencies
        for ex in data:
            tokens = word_tokenize(ex.text.lower())
            tokens = self._preprocess_tokens(tokens)
            word_counts.update(tokens)
            # Count each word once per document
            unique_tokens = set(tokens)
            doc_freq.update(unique_tokens)

        # Build vocabulary from most common words
        most_common = word_counts.most_common(self.max_vocab_size)
        self.vocab = {word: idx for idx, (word, _) in enumerate(most_common)}
        self.inverse_vocab = {idx: word for idx, (word, _) in enumerate(most_common)}
        self.vocab_size = len(self.vocab)

        # Compute IDF: log(N / (df + 1)) + 1 (smoothed IDF)
        for word in self.vocab:
            df = doc_freq.get(word, 0)
            self.idf[word] = math.log(self.num_docs / (df + 1)) + 1

        print(f"TF-IDF vocabulary built with {self.vocab_size} words (stopwords={self.use_stopwords}, stemming={self.use_stemming}).")

    def get_feature_vector(self, text):
        """Return TF-IDF weighted feature vector with L2 normalization"""
        tokens = word_tokenize(text.lower())
        tokens = self._preprocess_tokens(tokens)
        token_counts = Counter(tokens)
        total_tokens = len(tokens) if len(tokens) > 0 else 1

        vec = torch.zeros(self.vocab_size)
        for token, count in token_counts.items():
            if token in self.vocab:
                tf = count / total_tokens  # Term frequency (normalized)
                idf = self.idf.get(token, 1.0)
                vec[self.vocab[token]] = tf * idf

        # L2 normalization
        norm = torch.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec


class CharNgramFeaturizer:
    """
    Character N-gram Featurizer: Uses character-level n-grams (n=3,4) instead of
    word-level features. Character n-grams capture sub-word patterns, morphological
    features, and are more robust to spelling variations and rare words.
    """
    def __init__(self, max_vocab_size=5000, n_range=(3, 4)):
        self.max_vocab_size = max_vocab_size
        self.n_range = n_range
        self.vocab = {}
        self.inverse_vocab = {}
        self.vocab_size = 0

    def _extract_char_ngrams(self, text):
        """Extract character n-grams from text"""
        text = text.lower()
        ngrams = []
        for n in range(self.n_range[0], self.n_range[1] + 1):
            for i in range(len(text) - n + 1):
                ngrams.append(text[i:i+n])
        return ngrams

    def build_vocab(self, data):
        counts = Counter()
        for ex in data:
            ngrams = self._extract_char_ngrams(ex.text)
            counts.update(ngrams)

        most_common = counts.most_common(self.max_vocab_size)
        self.vocab = {ngram: idx for idx, (ngram, _) in enumerate(most_common)}
        self.inverse_vocab = {idx: ngram for idx, (ngram, _) in enumerate(most_common)}
        self.vocab_size = len(self.vocab)
        print(f"CharNgram vocabulary built with {self.vocab_size} character n-grams.")

    def get_feature_vector(self, text):
        ngrams = self._extract_char_ngrams(text)
        vec = torch.zeros(self.vocab_size)
        total = len(ngrams) if len(ngrams) > 0 else 1
        for ng in ngrams:
            if ng in self.vocab:
                vec[self.vocab[ng]] += 1
        # Normalize by total n-grams (TF-like)
        vec = vec / total
        # L2 normalize
        norm = torch.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


class CombinedFeaturizer:
    """
    Combined Featurizer: Concatenates Unigram (TF-IDF) and Bigram features
    for richer representation.
    """
    def __init__(self, unigram_vocab=5000, bigram_vocab=10000):
        self.unigram_featurizer = TFIDFFeaturizer(max_vocab_size=unigram_vocab)
        self.bigram_featurizer = BigramFeaturizer(max_vocab_size=bigram_vocab)
        self.vocab_size = 0

    def build_vocab(self, data):
        """Build vocabularies for both unigram and bigram featurizers"""
        self.unigram_featurizer.build_vocab(data)
        self.bigram_featurizer.build_vocab(data)
        self.vocab_size = self.unigram_featurizer.vocab_size + self.bigram_featurizer.vocab_size
        print(f"Combined vocabulary: {self.unigram_featurizer.vocab_size} unigrams + {self.bigram_featurizer.vocab_size} bigrams = {self.vocab_size} total")

    def get_feature_vector(self, text):
        """Return concatenated unigram and bigram feature vectors"""
        unigram_vec = self.unigram_featurizer.get_feature_vector(text)
        bigram_vec = self.bigram_featurizer.get_feature_vector(text)

        # L2 normalize bigram vector for consistency
        norm = torch.norm(bigram_vec)
        if norm > 0:
            bigram_vec = bigram_vec / norm

        return torch.cat([unigram_vec, bigram_vec])


class BlackBoxClassifier(torch.nn.Module):
    """
    This is a logistic regression classifier using PyTorch's built-in modules.
    Only used in Task 1. You will implement something like this from scratch
    in the LogisticRegressionClassifier class.
    """
    def __init__(self, input_dim, num_classes):
        super(BlackBoxClassifier, self).__init__()
        self.linear = torch.nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        # Returns logits (unnormalized scores)
        return self.linear(x)


class LogisticRegressionClassifier:
    def __init__(self, input_dim, num_classes):
        # Initialize weights and bias
        # Weights: (input_dim, num_classes), Bias: (num_classes)
        self.weights = torch.randn(input_dim, num_classes, requires_grad=False) * 0.01
        self.bias = torch.zeros(num_classes, requires_grad=False)

    def forward(self, x):
        # TODO: implement the logistic regression as z = W^T * x + b. Return z.
        # Hint: this should only require one line of code!
        # STUDENT START ---------------------------------
        return torch.matmul(x, self.weights) + self.bias
        # STUDENT END -----------------------------------

    def softmax(self, logits):
        # TODO: implement softmax. You may *not* use torch.nn.softmax or any
        # similar function. You may use torch.exp if you wish.
        # STUDENT START --------------------------------
        # Subtract max for numerical stability
        exp_logits = torch.exp(logits - torch.max(logits))
        return exp_logits / torch.sum(exp_logits)
        # STUDENT END ----------------------------------

    def predict(self, x):
        logits = self.forward(x)
        probs = self.softmax(logits)
        return torch.argmax(probs).item()


def train_logistic_regression(train_data, dev_data, featurizer, num_classes=4, lr=0.01, epochs=5,
                              method="bow"):
    input_dim = featurizer.vocab_size
    if method == "lr":
        model = LogisticRegressionClassifier(input_dim, num_classes)
    elif method == "bow":
        model = BlackBoxClassifier(input_dim, num_classes)

    print("Training logistic regression...")

    for epoch in range(epochs):
        shuffled_train = train_data.copy()
        random.shuffle(shuffled_train)
        total_loss = 0

        for ex in shuffled_train:
            x = featurizer.get_feature_vector(ex.text) # (vocab_size,)
            y_true = ex.label

            # 1. Call the forward function and compute the probability
            # of each class according to the model.
            logits = model.forward(x)
            probs = model.softmax(logits)

            # TODO: 2. Compute the negative log likelihood loss.
            # STUDENT START ----------------------------
            loss = -torch.log(probs[y_true] + 1e-10)
            total_loss += loss.item()
            # STUDENT END ------------------------------

            # TODO: 3. Compute the gradient for the weights, and the gradient for
            # for the bias. You may not use .backward().
            # STUDENT START ----------------------------
            one_hot = torch.zeros(num_classes)
            one_hot[y_true] = 1.0
            grad_output = probs - one_hot  # (num_classes,)

            # dL/dW = outer(x, grad_output), shape: (input_dim, num_classes)
            grad_weights = torch.outer(x, grad_output)
            # dL/db = grad_output
            grad_bias = grad_output
            # STUDENT END ------------------------------

            # TODO: 4. Update the parameters by multiplying the gradients you
            # derived in the previous step by the learning rate, and then subtracting
            # them from the weights and biases. You will need at least 1 line to update the
            # weight matrix, and at least 1 line to update the bias.
            # STUDENT START ----------------------------
            model.weights = model.weights - lr * grad_weights
            model.bias = model.bias - lr * grad_bias
            # STUDENT END ------------------------------

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_data):.4f}")

    return model


def train_torch_model(train_data, dev_data, featurizer, num_classes=4, lr=0.01, epochs=5):
    """
    Pre-provided gradient descent function using PyTorch's optimizer and loss.
    """
    input_dim = featurizer.vocab_size
    model = BlackBoxClassifier(input_dim, num_classes)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    print("Training Built-in PyTorch Model...")
    
    # This is an example of a training loop. Here, we're using only black-box
    # built-in PyTorch functions. You will implement the underlying functionality
    # of these functions as part of Task 2.
    for epoch in range(epochs):
        shuffled_train = train_data.copy()
        random.shuffle(shuffled_train)
        model.train() # Set model to training mode
        total_loss = 0
        
        for ex in shuffled_train:
            x = featurizer.get_feature_vector(ex.text)
            x_tensor = x.unsqueeze(0)
            y_tensor = torch.tensor([ex.label], dtype=torch.long)
            optimizer.zero_grad()
            logits = model(x_tensor)
            loss = criterion(logits, y_tensor)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_data):.4f}")
    
    # TODO: You're given the weight matrix of your trained model, which is of
    # shape (C, V), where C is the number of classes (here, 4) and V is
    # the vocabulary size. For each class, you will get the top-5 weight indices,
    # and print out the tokens they correspond to. No need to return anything here;
    # just print out the top weights/tokens and put them in your written report.
    # STUDENT START ----------------------------------
    weights = model.linear.weight.detach().cpu()
    label_map = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Tech'}
    print("\nTop-5 highest weighted tokens for each class:")
    for i in range(num_classes):
        # Get indices of top 5 weights for this class
        top_indices = torch.topk(weights[i], k=5).indices
        # Convert indices to words using inverse_vocab
        top_words = [featurizer.inverse_vocab[idx.item()] for idx in top_indices]
        print(f"Class {i} ({label_map.get(i, 'Unknown')}): {top_words}")
    # STUDENT END ------------------------------------
        
    return model