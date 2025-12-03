"""
Author: Abdoul Rahim Ousseini
Starting Date: 12/2/2025

An implementation of a Naive Bayes Classifier
"""

import math  # math.log()
import os
import pickle

"""
The name of positive dictionary cache file
"""
POSITIVE_DATA = "pos.dat"

"""
The name of negative dictionary cache file
"""
NEGATIVE_DATA = "neg.dat"

"""
The file name prefix for negative reviews
"""
NEGATIVE_FILE_PREFIX = "movies-1"

"""
The file name prefix for positive reviews
"""
POSITIVE_FILE_PREFIX = "movies-5"


class BayesClassifier:
    """A simple BayesClassifier implementation"""

    """
    A dictionary of frequencies of negative words
    """
    pos_freqs: dict[str, int]

    """
    A dictionary of frequencies of positive words
    """
    neg_freqs: dict[str, int]

    def __init__(self):
        """Constructor for the the Naive Bayes Sentiment Classifier"""
        self.pos_freqs = {}
        self.neg_freqs = {}

    def setup(self, training_data: str):
        """
        If a cache of a trained classifier is stored in the current folder, 
          it is loaded from that file,
        Otherwise the system will proceed through training

        Once setup, the classifier is ready to classify input text

        Arguments:
            training_data: the local folder name with files to train on
        """
        # check if both cached classifiers exist within the current directory
        if os.path.isfile(POSITIVE_DATA) and os.path.isfile(NEGATIVE_DATA):
            print("Data files found - loading to use cached values...")
            self.pos_freqs = self.load_dict(POSITIVE_DATA)
            self.neg_freqs = self.load_dict(NEGATIVE_DATA)
        else:
            print("Data files not found - running training...")
            self.train(training_data)
            print(self.pos_freqs)
            self.save_dict(self.pos_freqs, POSITIVE_DATA)
            self.save_dict(self.neg_freqs, NEGATIVE_DATA)

    def save_dict(self, dict: dict[str, int], filepath: str):
        """Pickle a given dictionary to a file with the given name

        Args:
            dict: A dictionary to pickle
            filepath: The relative path to file to save
        """
        print(f"dictionary saved to file: {filepath}")
        with open(filepath, "wb") as f:
            pickle.Pickler(f).dump(dict)

    def load_dict(self, filepath: str) -> dict[str, int]:
        """Load a pickled dictionary stored in given file

        Args:
            filepath: The relative path to file to load

        Returns:
            A dictionary stored in given file
        """
        print(f"Loading dictionary from file: {filepath}")
        with open(filepath, "rb") as f:
            return pickle.Unpickler(f).load()  # type: ignore

    def load_file(self, filepath: str) -> str:
        """Load text of given file.

        Args:
            filepath: The relative path to file to load.

        Returns:
            The text of the given file.
        """
        with open(filepath, "r", encoding="utf8") as f:
            return f.read()

    def update_dict(self, words: list[str], freqs: dict[str, int]):
        """Update a given (word -> frequency) dictionary with given words list

        That is, increment the count of each word in words in the dictionary
        If any word in words is not currently in the dictionary,
          it is added with an initial count of 1

        Args:
            words: The list of tokens to update frequencies of
            freqs: A dictionary of frequencies to update
        """
        for word in words:
            if word in freqs:
                freqs[word]+=1
            else:
                freqs[word] = 1


        return

    def tokenize(self, text: str) -> list[str]:
        """Split the given text into a list of the individual tokens in order

        Specifically, tokens consist of lowercase letters, digits, and the symbols ', -, and _
          such that each word in the text is a distinct string in the output string

        Words are considered to be separated by _any_ non-tokenized characters

        For example, 
        tokenize('Hello World 1234-5678') -> ['hello', 'world', '1234-5678']

        Args:
            text: The text to tokenize

        Returns:
            The tokens of given text in order
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
                    
        if current_token: #last token
            tokens.append(current_token)
        return tokens    
        
        
    def train(self, training_data: str):
        """Train the Naive Bayes Sentiment Classifier"""
        _, _, files = next(os.walk(training_data), (None, None, []))
        if not files:
            raise RuntimeError(f"Couldn't find path {training_data}")

        for filename in files:
            filepath = os.path.join(training_data, filename)
            text = self.load_file(filepath)
            tokens = self.tokenize(text)
        
            if filename.startswith(POSITIVE_FILE_PREFIX):
                self.update_dict(tokens, self.pos_freqs)
            elif filename.startswith(NEGATIVE_FILE_PREFIX):
                self.update_dict(tokens, self.neg_freqs)
           


    def classify(self, text: str) -> str:
        """Classify the given text as negative or positive

        This is done by calculating the most likely document class 
          as described in the assignment 7 writeup

        Args:
            text: The text to classify

        Returns:
            The classification string, either 'positive' or 'negative'
        """
        tokens = self.tokenize(text)
        
        # Calculate total counts for probabilities
        pos_total = sum(self.pos_freqs.values())
        neg_total = sum(self.neg_freqs.values())
        
        # Calculate log probabilities
        pos_prob = 0
        neg_prob = 0
        
        for token in tokens:
            # Get counts, add 1 for smoothing
            pos_count = self.pos_freqs.get(token, 0) + 1
            neg_count = self.neg_freqs.get(token, 0) + 1
            
            # Calculate log probabilities
            pos_prob += math.log(pos_count / pos_total)
            neg_prob += math.log(neg_count / neg_total)
        
        # Return the class with higher probability
        if pos_prob > neg_prob:
            return 'positive'
        else:
            return 'negative'
