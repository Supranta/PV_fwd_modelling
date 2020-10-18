import numpy as np

def sample_autocorr(x, time_arr):
    """
    Function to compute the autocorrelation of a given sample.
    :param x: Chain of the sample. Must be of the shape (n_dim, n_chain)
    :param time_arr: An array of time arrat for which the auto correlation is calculated
    """
    n_dim = x.shape[1]
    auto_corr_arr = np.zeros((n_dim, len(time_arr)))
    for d in range(n_dim):
        for t in time_arr:
            auto_corr_arr[d,t] = np.corrcoef(np.array(x[:len(x)-t, d]), np.array(x[t:len(x), d]))[0,1]
    return auto_corr_arr

def effective_sample_size(samples, t_max=1000):
    """
    """
    t = np.arange(t_max)
    return len(samples)/(1.0 + 2*np.max(np.sum(sample_autocorr(samples, t), axis=1)))

def coordinate_transform(x, eig_vecs):
    """
    """
    return np.matmul(eig_vecs.T, x)

def inverse_transform(x_transformed, eig_vecs):
    """
    """
    return np.matmul(eig_vecs, x_transformed)

def add_array(x, direction, value):
    """
    """
    add_arr = np.zeros(len(x))
    add_arr[direction] = value
    x = x + add_arr
    return x

def update_step_size(n, x, mu, cov, non_diagonal):
    """
    """
    mu = ((n-1) / n)*mu + x/n
    delta_mu = -(1/(n-1))*(mu + x)
    if(non_diagonal):
        cov = (n-1)/n *(cov + row_wise_self_multiply(delta_mu)) + (1/n)*row_wise_self_multiply(x - mu)
    else:
        s = (x - mu)
        cov = cov * (n-1)/n + s*s/n
    return mu, cov

def row_wise_self_multiply(x):
    """
    """
    return np.multiply(np.tile(x, (len(x),1)).T, x)
