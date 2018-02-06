import random
from scipy.sparse import dok_matrix
import numpy as np
from collections import Counter


class MarkovModel:
    """
    A simple discrete-time, discrete space first-order Markov model.
    The probability matrix is a square matrix represented this way:
    ```
          +-----+-----+-----+
          |  A  |  B  |  C  |
    +-----+-----+-----+-----+
    |  A  |  a  |  b  |  c  |
    +-----+-----+-----+-----+
    |  B  |  d  |  e  |  f  |
    +-----+-----+-----+-----+
    |  C  |  i  |  j  |  k  |
    +-----+-----+-----+-----+
    ```
    with:
     - `a` the probability for the state A to got to state A
     - `b` the probability for the state A to got to state B
     - `c` the probability for the state A to got to state C
     - ...
    Here we use a sparse dictionary of keys matrix to represent 
    the above.
    """

    def __init__(self, states):
        """
        Create a markov chain
        :param states: a set of all the different states
        """
        self.states = list(states)
        # We create the matrix
        # self.matrix = {state: Counter() for state in self.states}

        # We create a sparse scipy matrix
        n_states = len(states)
        self.matrix = dok_matrix((n_states, n_states), dtype=np.float32)
        self.dictionary = {state: elem_id for (state, elem_id) in 
                               zip(self.states, range(len(self.states))) } 

    def next_state(self, current_state):
        """
        Generate a next state according to the matrix's probabilities
        :param current_state: the state to start with
        :return: a next state
        """
        # Get the lookup id (current state numerically)
        elem_id = self.dictionary[current_state]
        # We get the row associated with the current state
        row = np.array(self.matrix[elem_id, :].todense())[0]
        max_val = max(row)
        # We'll start at a random point in array so create random num
        n = np.random.randint(0, len(row))
        # We'll introduce some noise to the comparisons
        noise = np.random.normal(0, 0.2, 1)[0]
        # Search for a higher than max value (with noise) starting at n
        # This is mainly to "jump" out of local minima
        for i in range(n, len(row)-1):
            num = row[n] + noise
            if num >= max_val:
                return self.states[n]
        # If all else fails, return simply the max val for row
        return self.states[np.argsort(row)[-1]]

    def probability_of_chain(self, chain):
        """
        Compute the probability for a given chain of text to occur.
        :param chain: the chain of states as an ordered list
        :return: the probability for it to happen
        """
        # If the chain is empty, we return a null probability
        if len(chain) == 0:
            return 0

        # If the chain is made of a single state, we return 1 if the state 
        # exists, 0 otherwise
        if len(chain) == 1:
            if chain[0] in self.matrix:
                return 1
            else:
                return 0

        probability = 1.0
        for state, next_state in zip(chain, chain[1:]):
            row = self.matrix[state]  # The row associated with the state

            # If the transition between state and next_state is impossible, 
            # the probability of the chain is 0
            if next_state not in row:
                return 0

            probability *= row[next_state]
        return probability

    def generate_chain(self, start_state, size):
        """
        Generate of probable chain of state, respecting the probabilities 
        in the matrix
        :param start_state: the starting state of the chain
        :param size: the size of the chain
        :return: the chain as an ordered list
        """
        chain = [start_state]
        state = start_state
        for n in range(0, size):
            state = self.next_state(state)
            chain.append(state)
        return chain

    def train(self, chain):
        """
        Train the model on an example chain
        :param chain: the chain of state as an ordered list
        """
        # We read the text two words by two words
        for s1, s2 in zip(chain, chain[1:]):
            self.matrix[self.dictionary[s1], self.dictionary[s2]] += 1

        # Normalize by dividing each row by its sum
        for i in range(len(self.states)):
            row_sum = sum(np.array(self.matrix[i, :].todense()))
            # Calculate marginal probabilities
            self.matrix[i, :] = self.matrix[i, :] / \
                                self.matrix[i, :].sum()
