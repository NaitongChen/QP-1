import autograd.numpy as np
from SURE_ridge import SURE_ridge

########################
## independent features
########################

n = 200
d = 2
X = np.random.normal(size=(n, d))
X = np.hstack((np.ones(n).reshape(n, 1), X))
beta = np.array([2., 3., -1.]).reshape(d+1, 1)
y = (X @ beta).ravel() + np.random.normal(size=n)

# beta hat with optimized regularization weight
sr = SURE_ridge(X, y, 1)
l = sr.solve(0., 1., 10000)
print(l)
beta_hat_l = sr.beta_hat(l)
print(beta_hat_l)
print(np.linalg.norm(beta_hat_l - beta))

# beta hat without regularization
hat = X.T @ X
L = np.linalg.cholesky(hat)
L_inv = np.linalg.inv(L)
hat_inv = L_inv.T @ L_inv
beta_hat = hat_inv @ X.T @ y
print(beta_hat)
print(np.linalg.norm(beta_hat - beta))

##############################
## almost collinear features
##############################

n = 200
d = 3
X = np.random.normal(size=(n, d-1))
X_collinear = 5 * X[:,0] - X[:,1] + np.random.normal(scale = 0.8, size=n)
X = np.hstack((X, X_collinear.reshape(n, 1)))
X = np.hstack((np.ones(n).reshape(n, 1), X))
beta = np.array([2., 3., -1., 0.3]).reshape(d+1, 1)
y = (X @ beta).ravel() + np.random.normal(size=n)

# beta hat with optimized regularization weight
sr = SURE_ridge(X, y, 1)
l = sr.solve(0., 0.01, 10000)
print(l)
beta_hat_l = sr.beta_hat(l)
print(beta_hat_l)
print(np.linalg.norm(beta_hat_l - beta))

# beta hat without regularization
hat = X.T @ X
L = np.linalg.cholesky(hat)
L_inv = np.linalg.inv(L)
hat_inv = L_inv.T @ L_inv
beta_hat = hat_inv @ X.T @ y
print(beta_hat)
print(np.linalg.norm(beta_hat - beta))