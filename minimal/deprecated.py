@deprecated('Use minimal.loss_functions.square_loss() instead.')
def square_loss(X, Y, W):
    """Compute the value of the square loss on W.

    Parameters
    ----------
    X : (n, d) float ndarray
        data matrix
    Y : (n, T) float ndarray
        labels matrix
    W : (d, T) float ndarray
        weights

    Returns
    ----------
    obj : float
        The value of the objective function on W.
    """
    return 1.0 / X.shape[0] * np.linalg.norm(np.dot(X, W) - Y, ord='fro') ** 2


@deprecated('Use minimal.loss_functions.square_loss() instead.')
def square_loss_grad(X, Y, W):
    """Compute the square loss gradient at W.

    Parameters
    ----------
    X : (n, d) float ndarray
        data matrix
    Y : (n, T) float ndarray
        labels matrix
    W : (d, T) float ndarray
        weights

    Returns
    ----------
    G : (d, T) float ndarray
        square loss gradient evaluated on the current iterate W.
    """
    return 2.0 / X.shape[0] * np.dot(X.T, (np.dot(X, W) - Y))


@deprecated('Use minimal.penalties.soft_thresholding() instead.')
def soft_thresholding(w, alpha):
    """Compute the element-wise soft-thresholding operator on the vector w.

    Parameters
    ----------
    w : (d,) or (d, 1) ndarray
        input vector
    alpha : float
        threshold

    Returns
    ----------
    wt : (d,) or (d, 1) ndarray
        soft-thresholded vector
    """
    return np.sign(w) * np.clip(np.abs(w) - alpha, 0.0, np.inf)


@deprecated('Use minimal.penalties.trace_norm_bound() instead.')
def trace_norm_bound(X, Y, loss='square'):
    """Compute maximum value for the trace norm regularization parameter.

    Parameters
    ----------
    data : (n, d) float ndarray
        data matrix
    labels : (n, T) float ndarray
        labels matrix
    loss : string
        the selected loss function in {'square', 'logit'}. Default is 'square'

    Returns
    ----------
    max_tau : float
        maximum value for the trace norm regularization parameter
    """
    if loss.lower() == 'square':
        # In this case max_tau := 2/n * max_sing_val(X^T * Y)
        return np.linalg.norm(np.dot(X.T, Y), ord=2) * (2.0/X.shape[0])
    else:
        print('Only square loss implemented so far.')
        sys.exit(-1)


@deprecated('Use minimal.penalties.l21_norm_bound() instead.')
def l21_norm_bound(X, Y, loss='square'):
    """Compute maximum value for the l12-norm regularization parameter.

    Parameters
    ----------
    data : (n, d) float ndarray
        data matrix
    labels : (n, T) float ndarray
        labels matrix
    loss : string
        the selected loss function in {'square', 'logit'}. Default is 'square'

    Returns
    ----------
    max_tau : float
        maximum value for the l21-norm regularization parameter
    """
    if loss.lower() == 'square':
        # In this case max_tau := 2/n * max(||[X^T * Y]s||_2)
        # First compute the 2-norm of each row of X^T * Y
        norm2 = map(lambda x: np.linalg.norm(x, ord=2), X.T.dot(Y))
        return np.max(norm2) * (2.0/X.shape[0])
    else:
        print('Only square loss implemented so far.')
        sys.exit(-1)


@deprecated('Use minimal.penalties.trace_norm_prox() instead.')
def trace_norm_prox(W, alpha):
    """Compute trace norm proximal operator on W.

    This function returns: prox           (W)
                             alpha ||.||_*

    Parameters
    ----------
    W : (n1, n2) float ndarray
        proximal operator input
    alpha : float
        proximal threshold

    Returns
    ----------
    Wt : (n1, n2) float ndarray
        trace norm prox result
    """
    d, T = W.shape
    U, s, V = np.linalg.svd(W, full_matrices=True)
    # U ~ (d, d)
    # s ~ (min(d, T), min(d, T))
    # V ~ (T, T)
    s = soft_thresholding(s, alpha)
    # make the output n1 x n2
    if d >= T:
        st_S = np.vstack((np.diag(s), np.zeros((np.abs(d-T), T))))
    else:
        st_S = np.hstack((np.diag(s), np.zeros((d, np.abs(d-T)))))
    return np.dot(U, np.dot(st_S, V))


@deprecated('Use minimal.penalties.l21_norm_prox() instead.')
def l21_norm_prox(W, alpha):
    """Compute l2,1-norm proximal operator on W.

    This function returns: prox             (W)
                             alpha ||.||_2,1

    Parameters
    ----------
    W : (n1, n2) float ndarray
        proximal operator input
    alpha : float
        proximal threshold

    Returns
    ----------
    Wt : (n1, n2) float ndarray
        l2,1-norm prox result
    """
    d, T = W.shape

    # Compute the soft-thresholding operator for each row of an unitary matrix
    ones = np.ones(T)
    Wst = np.empty(W.shape)
    for i, Wi in enumerate(W):
        thresh = alpha / np.sqrt(Wi.T.dot(Wi))
        Wst[i, :] = soft_thresholding(ones, thresh)

    # Return the Hadamard-product between Wst and W
    return W * Wst
