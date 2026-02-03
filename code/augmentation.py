# CS505: NLP - Spring 2026
"""
Data Augmentation Module
- Only used for training data
- Goal: Balance class distribution using Easy Data Augmentation (EDA)
"""
import random
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

# Ensure NLTK data is downloaded
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    wordnet.synsets('test')
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)


class EDAugmenter:
    """Easy Data Augmentation (EDA) based on Wei & Zou (2019)"""

    def __init__(self, alpha=0.1, num_aug=4):
        """
        Args:
            alpha: Strength of each operation (affects ~10% of words by default)
            num_aug: Number of augmented samples to generate per original sample
        """
        self.alpha = alpha
        self.num_aug = num_aug
        self.stopwords = set(stopwords.words('english'))

    def _get_synonyms(self, word):
        """Get synonyms from WordNet"""
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                if lemma.name() != word and '_' not in lemma.name():
                    synonyms.add(lemma.name())
        return list(synonyms)

    def synonym_replacement(self, words, n):
        """Synonym Replacement: Randomly replace n non-stopwords with synonyms"""
        new_words = words.copy()
        random_words = [w for w in words if w.lower() not in self.stopwords]
        random.shuffle(random_words)

        replaced = 0
        for word in random_words:
            synonyms = self._get_synonyms(word.lower())
            if synonyms:
                synonym = random.choice(synonyms)
                new_words = [synonym if w == word else w for w in new_words]
                replaced += 1
            if replaced >= n:
                break
        return new_words

    def random_deletion(self, words, p):
        """Random Deletion: Delete each word with probability p"""
        if len(words) == 1:
            return words
        new_words = [w for w in words if random.random() > p]
        return new_words if new_words else [random.choice(words)]

    def random_swap(self, words, n):
        """Random Swap: Swap n pairs of words"""
        new_words = words.copy()
        for _ in range(n):
            if len(new_words) >= 2:
                i, j = random.sample(range(len(new_words)), 2)
                new_words[i], new_words[j] = new_words[j], new_words[i]
        return new_words

    def random_insertion(self, words, n):
        """Random Insertion: Insert n synonyms at random positions"""
        new_words = words.copy()
        for _ in range(n):
            candidates = [w for w in new_words if w.lower() not in self.stopwords]
            if candidates:
                word = random.choice(candidates)
                synonyms = self._get_synonyms(word.lower())
                if synonyms:
                    pos = random.randint(0, len(new_words))
                    new_words.insert(pos, random.choice(synonyms))
        return new_words

    def augment(self, text):
        """Augment text, returning multiple augmented versions"""
        words = word_tokenize(text)
        n = max(1, int(self.alpha * len(words)))

        augmented = []
        for _ in range(self.num_aug):
            op = random.choice([0, 1, 2, 3])
            if op == 0:
                new_words = self.synonym_replacement(words, n)
            elif op == 1:
                new_words = self.random_deletion(words, self.alpha)
            elif op == 2:
                new_words = self.random_swap(words, n)
            else:
                new_words = self.random_insertion(words, n)
            augmented.append(' '.join(new_words))
        return augmented


class Oversampler:
    """Oversampler: Balance class distribution through data augmentation"""

    def __init__(self, target_count=None):
        self.target_count = target_count

    def analyze_distribution(self, data):
        """Analyze data distribution"""
        labels = [ex.label for ex in data]
        return Counter(labels)

    def oversample_with_eda(self, data, augmenter, target_count=400):
        """
        Oversample minority classes using EDA

        Args:
            data: List of Example objects
            augmenter: EDAugmenter instance
            target_count: Target number of samples for minority classes

        Returns:
            Augmented data list
        """
        # Import Example class for creating new examples
        from dataset import Example
        LABEL_MAP_REVERSE = {0: "world", 1: "sports", 2: "business", 3: "tech"}

        dist = self.analyze_distribution(data)
        print(f"Original distribution: {dict(dist)}")

        augmented_data = list(data)  # Copy original data

        for label, count in dist.items():
            if count < target_count:
                samples_needed = target_count - count
                class_samples = [ex for ex in data if ex.label == label]

                augmented_count = 0
                while augmented_count < samples_needed:
                    for ex in class_samples:
                        if augmented_count >= samples_needed:
                            break
                        aug_texts = augmenter.augment(ex.text)
                        for aug_text in aug_texts:
                            if augmented_count >= samples_needed:
                                break
                            new_ex = Example(LABEL_MAP_REVERSE[label], aug_text)
                            augmented_data.append(new_ex)
                            augmented_count += 1

        new_dist = self.analyze_distribution(augmented_data)
        print(f"Augmented distribution: {dict(new_dist)}")

        return augmented_data
