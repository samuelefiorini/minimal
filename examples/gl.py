#!/usr/bin/env python

import numpy as np
from minimal.estimators import GroupLasso
from sklearn import metrics

def main():
    # np.random.seed(42)
    # The number of samples is defined as:
    n = 200

    # The number of features per group is defined as:
    d_group = 10

    # The number of groups
    n_group = 10

    # The final dimension of the data
    d = d_group * n_group

    # Create covariance matrix
    rho = 0.5
    THETA = np.zeros((d_group, d_group))
    for i in range(d_group):
        for j in range(d_group):
            THETA[i, j] = rho**np.abs(i-j)

    # Define X randomly as simulated data with group structure
    X = np.hstack([
        np.random.multivariate_normal(mean=np.ones(d_group)*(4*np.random.rand(1)-2),
                                      cov=THETA, size=(n)) for i in range(n_group)])/np.sqrt(n)


    # Define beta_star a 0-1 vector with group structure:
    # the relevant groups will be the g0 and g2
    beta_star = np.zeros(d)
    beta_star[:d_group] = 1 + np.hstack([np.random.randn(1)]*d_group)
    beta_star[2*d_group:3*d_group] = -1 + np.hstack([np.random.randn(1)]*d_group)

    # Define y as X*beta + noise
    noise = np.random.randn(n)
    # y = np.sign(X.dot(beta_star) + noise)
    y = X.dot(beta_star) + noise

    print(X.shape)
    print(beta_star.shape)
    print(y.shape)
    print('----------------------')
    # Evaluate the chance probability
    # chance = 0.5 +  abs(y.sum())/(2.0*n)
    # print("Chance: {:2.3f}".format(chance))

    # Best model error
    print("Best model error: {:2.3f}".format(np.mean(abs(y - X.dot(beta_star)))))

    # Define the groups variable as in
    # parsimony.functions.nesterov.gl.linear_operator_from_groups
    groups = [map(lambda x: x+i, range(d_group)) for i in range(0, d, d_group)]
    # print(groups)
    print(beta_star)

    # mdl = GroupLasso(alpha=0.1, groups=groups)  # square
    mdl = GroupLasso(alpha=0.01, groups=groups)
    mdl.fit(X, y)
    print(mdl.coef_)

    print("Estimated prediction error = {:2.3f}".format(
        metrics.mean_absolute_error(mdl.predict(X), y)))

if __name__ == '__main__':
    main()
