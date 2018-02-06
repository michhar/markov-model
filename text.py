import re
import string
from collections import defaultdict
from markov_model import MarkovModel

class TextMarkovModel(MarkovModel):
    """
    A HMM that can be trained with a text and that is able to generate sentences from it.
    Here the states are the words in the vocabulary of the text.
    """

    def __init__(self, text):
        # We split the text into words
        self.words = self._lex(text)
        # The vocabulary is the set of different states
        self.states = set(self.words)
        super().__init__(self.states)

    def train(self):
        super().train(self.words)

    def _lex(self, text):
        """
        Splits the text into words, removing stopwords and infrequent words
        :param text: the text
        :return: a list of words
        """
        # Let's process the corpus and get the richest information we can

        # Split at each character or sequence of character that is not a valid word character (in the \w regex class)
        stoplist = set('for a of the and to in'.split())
        text = [re.sub(r'[^\w]', '', word) for word in \
                    text.lower().split() if word not in stoplist]
        
        # Get frequencies of words in corpus
        frequency = defaultdict(int)
        for word in text:
            frequency[word] += 1

        # Remove infrequent words (occurs only once) and return results
        text = [word for word in text if frequency[word] > 1]
        return text

        