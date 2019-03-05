# Apply the RANSAC algorithm to robustly estimate parameters from data
#
#   1. Randomly select a subset of data (hin)
#   2. Generate a model based on hin
#   3. Find the subset of remaining points that agree with the model (cset)
#   4. If enough points agree with the model, recalculate model on hin2 = (hin OR cset)
#   5. Check model error; if it beats the current best model, replace it
#   6. Repeat steps 1-5 until max_iters is reached
#
#   data        array where each row is a data point
#   mdl_fun     function returning a model from a subset of the data
#   ptw_cost    function returning pointwise error against model (checked against thresh for agreement)
#   glb_cost    function returning global error against model
#
#   Example: To robustly estimate mean for 1D data:
#       mdl_fun  = [mean, stdev]
#       ptw_cost = absolute value of z-score (pointwise)
#       glb_cost = mean squared error (over all data)
#
#   Kellan Moorse 2019-02-25

import numpy as np
import random
import warnings
import matplotlib.pyplot as plt


def ransac(data, mdl_fun, ptw_cost, glb_cost, d=None, thresh=1, max_iters=int(1e4)):

    if type(max_iters) != int:
        warnings.warn('\'max_iters\' has type %s; converting to int' % type(max_iters).__name__)
        max_iters = int(max_iters)

    # Initialize best model parameters
    best_mdl = []
    best_err = np.inf

    if not d:
        d = int(len(data)/2)

    for i in range(max_iters):

        # Select a random subset of data (hypothetical inliers)
        hin = np.random.choice([True, False], data.shape[0])
        while np.sum(hin) < 2:
            hin[np.random.randint(len(hin))] = True  # Ensure hin has at least 2 points (std > 0)

        # Generate a tentative model
        mdl = mdl_fun(data[hin])

        # Find points that agree with the model (consensus set)
        cset = np.zeros(hin.shape).astype(bool)
        cset[np.bitwise_and(~hin, ptw_cost(data, mdl) <= thresh)] = True

        # Combine hin and cset, recalculate model, and check against current best
        if np.sum(cset) >= d:
            hin2 = np.bitwise_or(hin, cset)
            mdl = mdl_fun(data[hin2])
            err = glb_cost(data[hin2], mdl)
            if err < best_err:
                best_mdl = mdl
                best_err = err

    return best_mdl


# Calculate mean and stdev of data
def meanstd(data):

    return [np.mean(data), np.std(data)]


# Calculate absolute value of z-score
# mdl = [mean, stdev]
def azscore(x, mdl):

    return np.abs(x-mdl[0])/mdl[1]


# Calculated mean squared error
# mdl = [mean, stdev]
def mse(data, mdl):

    return np.mean((data - mdl[0])**2)


if __name__ == '__main__':

    data = np.random.randn(50)
    data = np.append(data, np.arange(10, 15))
    random.shuffle(data)
    ix = np.arange(len(data))

    mdl = ransac(data, meanstd, azscore, mse, thresh=4)
    print(mdl)
