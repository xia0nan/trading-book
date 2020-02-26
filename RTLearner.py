from copy import deepcopy

import numpy as np
import pandas as pd
from scipy import stats


class RTLearner(object):

    def __init__(self, leaf_size=5, prev_tree=None, verbose=False):
        """
        Param:
        @leaf_size: is the maximum number of samples to be aggregated at a leaf.
        @prev_tree: if there is any previous tree, deepcopy it and new tree will be appended to it
        @verbose: if verbose, print information about DTLearner
        """
        self.leaf_size = leaf_size
        self.prev_tree = deepcopy(prev_tree)
        self.verbose = verbose
        if verbose:
            self.learner_info()

    def author(self):
        return 'nxiao30'

    def learner_info(self):
        print("Verbose?", self.verbose)
        print("Leaf Size:", self.leaf_size)
        if self.prev_tree is not None:
            # Check existing tree
            np.set_printoptions(suppress=True)
            print("Factor", "SplitVal", "Left", "Right")
            print("Prev tree\n", self.prev_tree)

    def build_tree(self, data):
        """ Build tree from data

        Return: (append(root, lefttree, righttree))
        """
        # Retrive x, y from data
        dataX = data[:, :-1]
        dataY = data[:, -1:]

        num_samples = data.shape[0]
        num_features = dataX.shape[1]

        # define leaf node
        # print(dataY.mean())
        # print(stats.mode(dataY, axis=None)[0][0])

        # change to classification by change np.mean to stats.mode
        # to avoid index error, we manually check and assign nan value
        try:
            y_mode = stats.mode(dataY, axis=None)[0][0]
        except IndexError:
            y_mode = np.nan

        leaf = np.array([-1, y_mode, np.nan, np.nan])

        if (num_samples <= self.leaf_size) or (len(np.unique(dataY)) == 1):
            return leaf
        else:
            # determine random feature i to split on
            random_i = np.random.randint(0, num_features)
            # SplitVal = np.median(data[:, random_i])
            SplitVal = self.find_random_SplitVal(num_samples, data[:, random_i])

            # check whether it is possible to split
            is_splittable_flag = self.is_splittable(data, random_i, SplitVal)

            if not is_splittable_flag:
                return leaf

            lefttree = self.build_tree(data[data[:, random_i] <= SplitVal])
            righttree = self.build_tree(data[data[:, random_i] > SplitVal])

            # if 1d array, tree.shape[0] will be num of elements. So fix it to 1
            lefttree_shape = 1 if lefttree.ndim == 1 else lefttree.shape[0]
            root = np.array([random_i, SplitVal, 1, lefttree_shape + 1])

            return np.vstack((root, lefttree, righttree))

    @staticmethod
    def is_splittable(data, i, SplitVal):
        """ Check if tree is splittable based on best feature index and SplitVal"""
        left_tree_size = (data[data[:, i] <= SplitVal]).shape[0]
        right_tree_size = (data[data[:, i] > SplitVal]).shape[0]
        total_size = data.shape[0]
        if (left_tree_size == total_size) or (right_tree_size == total_size):
            return False
        else:
            return True

    @staticmethod
    def find_random_SplitVal(num_samples, feature_col):
        """ Find random feature i to split on

        Return: index of random feature i
        """
        rand_rows = np.random.randint(0, num_samples, size=2)

        val1 = feature_col[rand_rows[0]]
        val2 = feature_col[rand_rows[1]]

        return (val1 + val2) / 2

    def addEvidence(self, dataX, dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        # create data using [dataX, dataY]
        data = np.column_stack((dataX, dataY))
        tree = self.build_tree(data)

        if self.prev_tree:
            # append if there is previous tree
            self.prev_tree = np.vstack((self.prev_tree, tree))
        else:
            self.prev_tree = tree

        # if not matrix, reshape
        if self.prev_tree.ndim == 1:
            self.prev_tree = np.reshape(self.prev_tree, (1, -1))

        # Check tree after build tree
        if self.verbose:
            self.learner_info()

    def query(self, points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """

        Ytest = np.array([])
        for point in points:
            # search tree to get y value for each row of input x
            Ytest = np.append(Ytest, self.search(point))

        # flat to 1d array
        Ytest = Ytest.ravel()
        return Ytest

    def search(self, row):
        """ Search tree and return leaf value """
        current_row = 0
        factor = self.prev_tree[current_row, 0]
        factor = factor.astype(int)
        while int(factor) != -1:  # loop until reach leaf
            SplitVal = self.prev_tree[int(current_row), 1]
            if row[int(factor)] <= SplitVal:
                # go through left tree
                current_row += 1
            else:
                # go through right tree
                right = self.prev_tree[int(current_row), 3]
                current_row += right
            factor = self.prev_tree[int(current_row), 0]
        return self.prev_tree[int(current_row), 1]


if __name__ == "__main__":
    learner = RTLearner()

"""
Reference:
Project website: http://quantsoftware.gatech.edu/Summer_2019_Project_3:_Assess_Learners
Lecture slides: http://quantsoftware.gatech.edu/images/4/4e/How-to-learn-a-decision-tree.pdf
Lecture Video & Quiz
JR Quinlan's paper: https://link.springer.com/content/pdf/10.1007/BF00116251.pdf
"""
