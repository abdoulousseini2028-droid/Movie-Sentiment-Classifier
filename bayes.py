"""
Naive Bayes Sentiment Classifier
Author: Abdoul Rahim Ousseini
"""

import math
import os
import pickle

POSITIVE_DATA = "pos.dat"
NEGATIVE_DATA = "neg.dat"
NEGATIVE_FILE_PREFIX = "movies-1"
POSITIVE_FILE_PREFIX = "movies-5"


class BayesClassifier:
    """A simple Naive Bayes classifier for sentiment analysis"""

    def __init__(self):
        """Initialize the classifier with empty frequency dictionaries"""
        self.pos_freqs = {}
        self.neg_freqs = {}

    def setup(self, training_data: str):
        """Load cached model or train a new one
        
        Args:
            training_data: Path to folder containing training files
        """
        if os.path.isfile(POSITIVE_DATA) and os.path.isfile(NEGATIVE_DATA):
            print("Loading cached model...")
            self.pos_freqs = self.load_dict(POSITIVE_DATA)
            self.neg_freqs = self.load_dict(NEGATIVE_DATA)
        else:
            print("Training new model...")
            self.train(training_data)
            self.save_dict(self.pos_freqs, POSITIVE_DATA)
            self.save_dict(self.neg_freqs, NEGATIVE_DATA)
            print("Model trained and cached!")

    def save_dict(self, dict: dict[str, int], filepath: str):
        """Pickle dictionary to file"""
        with open(filepath, "wb") as f:
            pickle.Pickler(f).dump(dict)

    def load_dict(self, filepath: str) -> dict[str, int]:
        """Load pickled dictionary from file"""
        with open(filepath, "rb") as f:
            return pickle.Unpickler(f).load()

    def load_file(self, filepath: str) -> str:
        """Read text content from file"""
        with open(filepath, "r", encoding="utf8") as f:
            return f.read()

    def update_dict(self, words: list[str], freqs: dict[str, int]):
        """Update word frequency dictionary
        
        Args:
            words: List of tokens to count
            freqs: Dictionary to update with frequencies
        """
        for word in words:
            freqs[word] = freqs.get(word, 0) + 1

    def tokenize(self, text: str) -> list[str]:
        """Convert text into lowercase tokens
        
        Tokens consist of letters, digits, and the symbols ', -, _
        All other characters are treated as separators
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of lowercase tokens
        """
        tokens = []
        current_token = ""
        
        for char in text:
            if char.isalnum() or char in "'-_":
                current_token += char.lower()
            else:
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
        
        if current_token:
            tokens.append(current_token)
        
        return tokens

    def train(self, training_data: str):
        """Train classifier on movie reviews
        
        Args:
            training_data: Path to folder with review files
        """
        _, _, files = next(os.walk(training_data), (None, None, []))
        if not files:
            raise RuntimeError(f"Training data not found: {training_data}")

        for filename in files:
            filepath = os.path.join(training_data, filename)
            text = self.load_file(filepath)
            tokens = self.tokenize(text)
            
            if filename.startswith(POSITIVE_FILE_PREFIX):
                self.update_dict(tokens, self.pos_freqs)
            elif filename.startswith(NEGATIVE_FILE_PREFIX):
                self.update_dict(tokens, self.neg_freqs)

    def classify(self, text: str) -> str:
        """Classify text as positive or negative
        
        Uses Bayesian inference with log probabilities and Laplace smoothing
        
        Args:
            text: Input text to classify
            
        Returns:
            Either 'positive' or 'negative'
        """
        tokens = self.tokenize(text)
        
        pos_total = sum(self.pos_freqs.values())
        neg_total = sum(self.neg_freqs.values())
        
        pos_prob = 0
        neg_prob = 0
        
        for token in tokens:
            # Laplace smoothing: add 1 to handle unseen words
            pos_count = self.pos_freqs.get(token, 0) + 1
            neg_count = self.neg_freqs.get(token, 0) + 1
            
            # Use log probabilities to prevent underflow
            pos_prob += math.log(pos_count / pos_total)
            neg_prob += math.log(neg_count / neg_total)
        
        return 'positive' if pos_prob > neg_prob else 'negative'
