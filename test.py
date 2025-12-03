"""
Author: Abdoul Rahim
Starting Date: 12/2/2025

Test cases for the Naive Bayes Classifier implemented in bayes.py
"""

import bayes


def test_classifier_constructor():
    """Test cases for the Bayes Classifier constructor"""

    fn_name = 'Bayes Classifier constructor'

    classifier = bayes.BayesClassifier()

    # Test that each attribute was set correctly by the constructor

    result = classifier.pos_freqs
    expected = {}
    error_message = f'while testing {fn_name}, expected {expected} got {result}'
    assert expected == result, error_message

    result = classifier.neg_freqs
    expected = {}
    error_message = f'while testing {fn_name}, expected {expected} got {result}'
    assert expected == result, error_message

    print(f'{fn_name} tests passed')


def test_classifier_update_dict():
    """Test cases for the Bayes Classifier update dictionary"""

    fn_name = 'Bayes Classifier update dictionary'

    classifier = bayes.BayesClassifier()

    lst = ['a', 'a', 'b', 'b', 'b', 'a', 'a']
    result = {'a': 2, 'b': 2}
    classifier.update_dict(lst, result)
    expected = {'a': 6, 'b': 5}
    error_message = f'while testing {fn_name}, expected {expected} got {result}'
    assert expected == result, error_message

    lst = ['testing', 'testing', '1234']
    result = {'1234': 2}
    classifier.update_dict(lst, result)
    expected = {'testing': 2, '1234': 3}
    error_message = f'while testing {fn_name}, expected {expected} got {result}'
    assert expected == result, error_message

    lst = ['I', 'really', 'like', 'this', 'movie',
           '.', 'I', 'hope', 'you', 'like', 'it', 'too']
    result = {}
    classifier.update_dict(lst, result)
    expected = {'I': 2, 'really': 1, 'like': 2, 'this': 1,
                'movie': 1, '.': 1, 'hope': 1, 'you': 1, 'it': 1, 'too': 1}
    error_message = f'while testing {fn_name}, expected {expected} got {result}'
    assert expected == result, error_message
    
    #Testing with 1 word

    lst = ['hello']
    result = {}
    classifier.update_dict(lst, result)
    expected = {'hello': 1}
    error_message = f'while testing {fn_name}, expected {expected} got {result}'
    assert expected == result, error_message

    # Testing with empty list
    lst = []
    result = {'existing': 5}
    classifier.update_dict(lst, result)
    expected = {'existing': 5}
    error_message = f'while testing {fn_name}, expected {expected} got {result}'
    assert expected == result, error_message

    print(f'{fn_name} tests passed')


def test_classifier_tokenize():
    """Test cases for the Bayes Classifier tokenizer"""

    fn_name = 'Bayes Classifier tokenizer'

    classifier = bayes.BayesClassifier()

    result = classifier.tokenize('Hello World 1234-5678')
    expected = ['hello', 'world', '1234-5678']
    error_message = f'while testing {fn_name}, expected {expected} got {result}'
    assert expected == result, error_message

    result = classifier.tokenize("It's a test")
    expected = ["it's", 'a', 'test']
    error_message = f'while testing {fn_name}, expected {expected} got {result}'
    assert expected == result, error_message

    result = classifier.tokenize('tabs\tand\nnewlines   are_fine')
    expected = ['tabs', 'and', 'newlines', 'are_fine']
    error_message = f'while testing {fn_name}, expected {expected} got {result}'
    assert expected == result, error_message

    # Test with empty string

    result = classifier.tokenize('')
    expected = []
    error_message = f'while testing {fn_name}, expected {expected} got {result}'
    assert expected == result, error_message

    print(f'{fn_name} tests passed')

    # Test with punctuation only
    result = classifier.tokenize('!!!  ???  ...')
    expected = []
    error_message = f'while testing {fn_name}, expected {expected} got {result}'
    assert expected == result, error_message

    print(f'{fn_name} tests passed')


def test_classifier_train():
    """Test cases for the Bayes Classifier train"""

    fn_name = 'Bayes Classifier train'

    # Negative only
    classifier1 = bayes.BayesClassifier()
    classifier1.train('train_test1')

    # test both positive and negative dictionaries
    result = classifier1.pos_freqs
    expected = {}
    error_message = f'while testing {fn_name}, expected {expected} got {result}'
    assert expected == result, error_message

    result = classifier1.neg_freqs
    expected = {'crap': 1, 'very': 1, 'much': 1, 'terrible': 2}
    error_message = f'while testing {fn_name}, expected {expected} got {result}'
    assert expected == result, error_message

    # Simple positive and negatives
    classifier2 = bayes.BayesClassifier()
    classifier2.train('train_test2')

    # test both positive and negative dictionaries
    result = classifier2.pos_freqs
    expected = {'great': 3, 'movie': 1, '5': 1, 'stars': 1, 'pretty': 1}
    error_message = f'while testing {fn_name}, expected {expected} got {result}'
    assert expected == result, error_message

    result = classifier2.neg_freqs
    expected = {'bad': 1, 'movie': 1, '1': 1, 'star': 1}
    error_message = f'while testing {fn_name}, expected {expected} got {result}'
    assert expected == result, error_message

    # Weird symbols
    classifier3 = bayes.BayesClassifier()
    classifier3.train('train_test3')

    # test both positive and negative dictionaries
    result = classifier3.pos_freqs
    expected = {'one': 1, 'two': 2, 'three': 1, "'''comment": 1, "test'''": 1, 'lines': 1}
    error_message = f'while testing {fn_name}, expected {expected} got {result}'
    assert expected == result, error_message

    result = classifier3.neg_freqs
    expected = {"it's": 1, 'a_movie': 1, '__test__': 1, '__test2__': 1}
    error_message = f'while testing {fn_name}, expected {expected} got {result}'
    assert expected == result, error_message

    # NOTE: you do not need to add any more tests to this function

    print(f'{fn_name} tests passed')


def test_classifier_classify():
    """Test cases for the Bayes Classifier classify"""

    fn_name = 'Bayes Classifier classify'

    classifier = bayes.BayesClassifier()

    # Testing positive result (a appears relatively more in pos_freqs)
    classifier.pos_freqs = {'a': 2, 'b': 1}
    classifier.neg_freqs = {'a': 1, 'b': 2}
    result = classifier.classify('a a')
    expected = 'positive'
    error_message = f'while testing {fn_name}, expected {expected} got {result}'
    assert expected == result, error_message

    # Testing negative result (b appears relatively more in neg_freqs)
    classifier.pos_freqs = {'a': 2, 'b': 1}
    classifier.neg_freqs = {'a': 1, 'b': 2}
    result = classifier.classify('b b')
    expected = 'negative'
    error_message = f'while testing {fn_name}, expected {expected} got {result}'
    assert expected == result, error_message

    # Testing b appearing just barely more in the string, which should give a negative
    classifier.pos_freqs = {'a': 2, 'b': 1}
    classifier.neg_freqs = {'a': 1, 'b': 2}
    result = classifier.classify('a a a b b b b')
    expected = 'negative'
    error_message = f'while testing {fn_name}, expected {expected} got {result}'
    assert expected == result, error_message

    # Test with word not in either dictionary (should handle smoothing)
    classifier.pos_freqs = {'good': 10}
    classifier.neg_freqs = {'bad': 10}
    result = classifier.classify('unknown')
    
    # Should still return positive or negative (not crash)
    assert result in ['positive', 'negative'], f'Expected positive or negative, got {result}'

    # Test with empty string
    classifier.pos_freqs = {'a': 5}
    classifier.neg_freqs = {'b': 5}
    result = classifier.classify('')
    assert result in ['positive', 'negative'], f'Expected positive or negative, got {result}'
 
    print(f'{fn_name} tests passed')


def test_all():
    test_classifier_constructor()
    test_classifier_update_dict()
    test_classifier_tokenize()
    test_classifier_train()
    test_classifier_classify()

    print('All tests passed!')


if __name__ == "__main__":
    test_all()
