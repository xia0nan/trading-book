import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")

import numpy as np
import random as rand
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

from util import get_data, plot_data


class QLearner(object):
    """
    Model-free
    Rewards function is unknown
    Transition function is unknown

    """

    def __init__(self,
                 num_states=100,
                 num_actions=4,
                 alpha=0.2,
                 gamma=0.9,
                 rar=0.5,
                 radr=0.99,
                 dyna=0,
                 verbose=False):
        """
        Params:
        @num_states: integer, the number of states to consider, 10 x 10 = 100
        @num_actions: integer, the number of actions available
        @alpha: float, the learning rate used in the update rule.
                        Should range between 0.0 and 1.0 with 0.2 as a typical value
        @gamma: float, the discount rate used in the update rule. For furture rewards weight.
                        Should range between 0.0 and 1.0 with 0.9 as a typical value
        @rar: float, random action rate: the probability of selecting a random action at each step.
                        Early on we use random value to explore (high rar),
                        but should decay overtime and eventually become 0
                        Should range between 0.0 (no random actions) to 1.0 (always random action)
                        with 0.5 as a typical value.
        @radr: float, random action decay rate, after each update, rar = rar * radr.
                        The decay rate of random action rate, make rar shrink a bit by each step
                        Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.
        @dyna: integer, conduct this number of dyna updates for each regular update.
                        When Dyna is used, 200 is a typical value.
        @verbose: boolean, if True, your class is allowed to print debugging statements,
                        if False, all printing is prohibited.

        """

        self.verbose = verbose
        self.num_actions = num_actions
        self.num_states = num_states
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount rate
        self.rar = rar  # random actions rate
        self.radr = radr  # random actions decay rate
        self.dyna = dyna

        self.s = 0  # state
        self.a = 0  # action

        # save all experience tuples <s,a,s',r> for Dyna
        self.experience_tuples = []

        # Q table: 100 x 4 (number of states) x (number of actions)
        self.Q = np.zeros((num_states, num_actions))  # Policy:  Q[s, a] for the number of states and actions

    def querysetstate(self, s):
        """
        @summary: Update the state and action without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        # Ref. https://youtu.be/X9UhB953TDA
        # decide if we're going to ignore the action and choose a random one instead
        if rand.uniform(0., 1.) <= self.rar:  # going rogue
            # take a random action
            action = rand.randint(0, self.num_actions - 1)
        else:
            # take the action that maximizes the return
            action = np.argmax(self.Q[s])

            # Update state
        self.s = s
        # Update action
        self.a = action

        # if self.verbose: print "s =", s, "a =", action
        return action

    def query(self, s_prime, r):
        """
        1. Init Q table
        2. Observe s
        3. Execute a, obvserve s', r
        4. Update Q with <s,a,s',r>

        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: a real valued immediate reward
        @returns: The selected action
        """
        # Update Q table
        self.Q_learn(state=self.s, action=self.a, reward=r, next_state=s_prime)

        # Update experience_tuples
        self.experience_tuples.append((self.s, self.a, r, s_prime))

        # Check Dyna
        if self.dyna:
            self.DynaQ()

        # Get action based on state
        self.querysetstate(s=s_prime)

        # Update rar
        self.rar *= self.radr

        # if self.verbose: print "s =", self.s, "a =", self.a, "r =", r

        action = self.a
        return action

    def Q_learn(self, state, action, reward, next_state):
        """
        Update Q-table with experience tuple (s, a, r, s')

        Update Formula: Q' = (1-alpha) * Q + alpha * (R + gamma * later_rewards)
        """
        # get previous Q value to be updated
        q_prev = self.Q[state][action]
        # get maximum future reward
        later_rewards = np.max(self.Q[next_state])
        # calculate current reward with discount rate gamma
        q_update = reward + self.gamma * later_rewards
        # update Q table with learning rate alpha
        self.Q[state][action] = (1 - self.alpha) * q_prev + self.alpha * q_update

    def DynaQ(self):
        """ Implement Dyna methods, Dyna-Q improve model convergence
        Ref. https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume4/kaelbling96a-html/node29.html
        Ref. http://www-anw.cs.umass.edu/~barto/courses/cs687/Chapter%209.pdf

        From lecture:
        1. Learn model (T, R)
            - T'[s,a,s'] prob(s,a -> s')
                - init T_c[]=0.00001
                - while executing, observe s,a,s'
                - increment T_c[s,a,s']
                - After enough iterations, T_c[s,a,s'] = T_c[s,a,s'] / SUMi(T_c[s,a,i])
            - R'[s,a]
                - R'[s,a] = (1-alpha) * R[s,a] + alpha * r
                - r: immediate reward
                - R[s,a]: expected reward for s,a
        2. Halucinate experience
            - s = random
            - a = random
            - s' = infer from T
            - r = R[s,a]
        3. Update Q
            - Update Q with <s,a,s',r>
            - Go back to step 2
            - for about 200 loops

        Use random past experience tuples to update Q table
        """
        for i in range(self.dyna):
            # get a random sample from previous experiences
            random_index = rand.randint(0, len(self.experience_tuples) - 1)
            # get experience tuple (s, a, r, s')
            state, action, reward, next_state = self.experience_tuples[random_index]
            # update Q table with experience tuple
            self.Q_learn(state, action, reward, next_state)


def test_code():
    learner = QLearner(num_states=100,
                       num_actions=4,
                       alpha=0.2,
                       gamma=0.9,
                       rar=0.98,
                       radr=0.999,
                       dyna=0,
                       verbose=False)

    s = 99  # our initial state

    a = learner.querysetstate(s)  # action for state s

    s_prime = 5  # the new state we end up in after taking action a in state s

    r = 0  # reward for taking action a in state s

    next_action = learner.query(s_prime, r)


if __name__ == "__main__":
    test_code()
