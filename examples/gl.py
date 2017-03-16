#!/usr/bin/env python

import numpy as np
from minimal.estimators import GroupLasso

def main():
    np.random.seed(42)
    # The number of samples is defined as:
    n = 200

    # The number of features per group is defined as:
    d_group = 15

    # The number of groups
    n_group = 4

    # The final dimension of the data
    d = d_group * n_group

    # Each group of features is sampled from a Gaussian distribution with different mean and std
    means = np.hstack([np.ones(d_group)*(4*np.random.rand(1)-2) for i in range(n_group)])
    stds = np.hstack([np.ones(d_group)*(np.random.rand(1)) for i in range(n_group)])

    # Define X randomly as simulated data with group structure
    X = stds*np.random.randn(n, d)+means
    # X -= np.mean(X, axis=0)

    # Define beta_star a 0-1 vector with group structure:
    # the relevant groups will be the g0 and g2
    beta_star = np.zeros(d)
    beta_star[:d_group] = 1 + np.hstack([np.random.randn(1)]*d_group)
    beta_star[2*d_group:3*d_group] = -1 + np.hstack([np.random.randn(1)]*d_group)

    # Define y as X*beta + noise
    noise = np.random.randn(n)
    # y = np.sign(X.dot(beta_star) + noise)
    y = X.dot(beta_star) + noise

    # Evaluate the chance probability
    # chance = 0.5 +  abs(y.sum())/(2.0*n)
    # print("Chance: {:2.3f}".format(chance))

    # Best model error
    print("Best model error: {:2.3f}".format(np.mean(abs(y - X.dot(beta_star)))))

    # Define the groups variable as in
    # parsimony.functions.nesterov.gl.linear_operator_from_groups
    groups = [map(lambda x: x+i, range(d_group)) for i in range(0, d, d_group)]
    print(groups)
    print(beta_star)

    mdl = GroupLasso(alpha=1, groups=groups)
    mdl.fit(X, y)
    print(mdl.coef_)

if __name__ == '__main__':
    main()
