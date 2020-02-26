import numpy as np
import pandas as pd

import RTLearner as rt


class BagLearner(object):
    def __init__(self, learner, kwargs, bags, boost, verbose):
        self.bags = bags
        self.learners = []
        for i in range(0, self.bags):
            self.learners.append(learner(**kwargs))

        if verbose:
            print("learner:", learner)
            print("kwargs:", kwargs)
            print("bags:", bags)
            print("boost:", boost)

    def addEvidence(self, dataX, dataY):
        """ Random select same size data to add to each learner """
        data = np.column_stack((dataX, dataY))
        num_samples = data.shape[0]
        for learner in self.learners:
            # random sample with replacement
            row_index = np.random.choice(num_samples,
                                         size=num_samples,
                                         replace=True)
            bagX = dataX[row_index]
            bagY = dataY[row_index]
            # add a bag of random data to each learner
            learner.addEvidence(bagX, bagY)

    def query(self, points):
        """Return query result by averaging all learner's results"""
        Ytests = []
        for learner in self.learners:
            Ytest = learner.query(points)
            Ytests.append(Ytest)
        average_Ytest = np.mean(Ytests, axis=0)
        return average_Ytest


if __name__ == "__main__":
    learner = BagLearner(learner=rt.RTLearner,
                         kwargs={"leaf_size":5},
                         bags=10,
                         boost=False,
                         verbose=False)
