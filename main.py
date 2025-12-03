"""
Author: Dietrich Geisler
Last Modified: 11/27/2025

A simple menu for training and inspecting 
  the Naive Bayes Classifier implemented in bayes.py
"""

import bayes

"""
The relative path to training directory
"""
TRAINING_DATA_DICTIONARY = "movie_reviews/"


def inspect_probabilities(classifier: bayes.BayesClassifier):
    """Prints out probabilities related to various words for manual inspection

    Arguments:
        classifier: a Naive Bayes Classifier that has already been trained
    """
    request = 'Give a single word to view probabilities of (or no input to exit):\n'
    pos_denominator = sum(classifier.pos_freqs.values())
    neg_denominator = sum(classifier.neg_freqs.values())

    word = input(request)
    while word != '':
        print(f"P('{word}' | pos) = {(classifier.pos_freqs.get(word.lower(), -1)+1)/pos_denominator}")
        print(f"P('{word}' | neg) = {(classifier.neg_freqs.get(word.lower(), -1)+1)/neg_denominator}")
        word = input(request)
    print("Exiting the probability viewer")


def test_classifier(classifier: bayes.BayesClassifier):
    """Manually tests the behavior of the classifier (i.e. how it classifies sentences)

    Arguments:
        classifier: a Naive Bayes Classifier that has already been trained
    """
    request = 'Give a sentence to classify (or no input to exit): '
    sentence = input(request)
    while sentence != '':
        result = classifier.classify(sentence)
        print(f'The classifier rates your sentence as {result}')
        sentence = input(request)
    print("Exiting the classifier")


def main():
    """
    Trains the classifier if not already trained
    Then calls into menu functions to inspect results
    """
    classifier = bayes.BayesClassifier()
    classifier.setup(TRAINING_DATA_DICTIONARY)
    choice = input('What should we do with our classifier?\n 1 to inspect probabilities\n 2 to classify text\n')
    if choice == '1':
        inspect_probabilities(classifier)
    elif choice == '2':
        test_classifier(classifier)
    else:
        print('Invalid option, quitting')


if __name__ == "__main__":
    main()
