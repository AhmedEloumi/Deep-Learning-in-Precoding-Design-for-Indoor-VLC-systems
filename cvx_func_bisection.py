import numpy as np
import cvxpy as cp

def cvx_func_bisection(t, M, K, sigma, p, H):
    ll = M * K
    V = np.kron(np.eye(M), np.ones((1, K)))
    c = 10 ** 10
    w = cp.Variable(ll)
    d = cp.Variable(ll)

    constraints = [
        w <= d,
        w >= -d,
        V @ d <= p
    ]

    objective = cp.Minimize(t)

    for k in range(K):
        h_k = H[:, k]
        I_k = np.eye(K)
        I_k[k, k] = 0
        A_k = np.vstack((np.kron(h_k.T, I_k), np.zeros((1, ll))))
        sigma_k = np.vstack((np.zeros((K, 1)), np.sqrt(sigma[k])))
        e_k = np.zeros((K, 1))
        e_k[k] = 1
        T_k = np.kron(np.eye(M), e_k.T)
        constraints.append(c * (t * (cp.norm(A_k @ w + sigma_k.reshape((len(sigma_k),))))) <= (h_k.T @ T_k @ w) * c)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CVXOPT, verbose=True)

    status = prob.status
    W = np.reshape(w.value, (K, M)).T

    return W, d.value, status


def cvx_func_bisection_2(t, M, K, sigma, p, H):
    ll = M * K
    V = np.kron(np.eye(M), np.ones((1, K)))
    c = 10 ** 10
    w = cp.Variable(ll)
    d = cp.Variable(ll)
    status = 'Failed'
    aa = np.zeros(K)
    bb = np.zeros(K)

    constraints = [
        w <= d,
        w >= -d,
        V @ d <= p
    ]

    objective = cp.Minimize(t)

    for k in range(K):
        h_k = H[:, k]
        I_k = np.eye(K)
        I_k[k, k] = 0
        A_k = np.vstack((np.kron(h_k.T, I_k), np.zeros((1, ll))))
        sigma_k = np.vstack((np.zeros((K, 1)), np.sqrt(sigma[k])))
        e_k = np.zeros((K, 1))
        e_k[k] = 1
        T_k = np.kron(np.eye(M), e_k.T)
        constraints.append(c * (t * (cp.norm(A_k @ w + sigma_k.reshape((len(sigma_k),))))) <= (h_k.T @ T_k @ w) * c)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CPLEX, verbose=True)

    for k in range(K):
        h_k = H[:, k]
        I_k = np.eye(K)
        I_k[k, k] = 0
        A_k = np.vstack((np.kron(h_k.T, I_k), np.zeros((1, ll))))
        sigma_k = np.vstack((np.zeros((K, 1)), np.sqrt(sigma[k])))
        e_k = np.zeros((K, 1))
        e_k[k] = 1
        T_k = np.kron(np.eye(M), e_k.T)
        constraints.append(c * (t * (cp.norm(A_k @ w + sigma_k.reshape((len(sigma_k),))))) <= (h_k.T @ T_k @ w) * c)
        aa[k] = (t * cp.norm(A_k @ w + sigma_k.reshape((len(sigma_k),)))).value
        bb[k] = (h_k.T @ T_k @ w).value

    if (np.sum(aa > bb) == 0) and (np.sum((p < (V @ d).value)) == 0) and (np.sum(np.isnan(np.concatenate((aa, bb, w.value), axis=0))) == 0):
        status = 'Solved'

    W = np.reshape(w.value, (K, M)).T

    return W, d.value, status


