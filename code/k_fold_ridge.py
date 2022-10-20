import autograd.numpy as np
from autograd import grad

class k_fold_ridge():
  def __init__(self, X, y, sigma, k, seed):
    self.X = X # including column of 1's.
    self.y = y
    self.sigma = sigma
    self.n, self.p = X.shape
    self.seed = seed
    self.k = k
    self.X_folds = []
    self.y_folds = []
    # assign folds
    np.random.seed(self.seed)
    self.ii = np.random.choice(np.arange(self.n) % self.k, size=self.n, replace=False)

  def _hat_l_inv(self, X_curr, l):
    hat_l = X_curr.T @ X_curr + l * np.eye(self.p)
    L = np.linalg.cholesky(hat_l)
    L_inv = np.linalg.inv(L)
    return L_inv.T @ L_inv

  def beta_hat(self, X_curr, y_curr, l):
    return self._hat_l_inv(X_curr, l) @ X_curr.T @ y_curr

  def _loss(self, l):
    obj = 0
    for i in np.arange(self.k):
      # fit beta using all observations not in kth fold
      X_curr = self.X[self.ii != i]
      y_curr = self.y[self.ii != i]
      bhat = self.beta_hat(X_curr, y_curr, l)
      # compute prediction error on observations in kth fold
      residual_pred = self.y[self.ii == i] - (self.X[self.ii == i] @ bhat)
      obj +=  np.power(np.linalg.norm(residual_pred), 2)
    return obj

  def solve(self, ls):
    # l_unc = l_unc_0
    # loss_grad = grad(self._loss)
    # for i in np.arange(iter):
    #   l_unc -= loss_grad(l_unc) * stepsize
    #   print(np.exp(l_unc))
    # return np.exp(l_unc)
    ls_size = ls.shape[0]
    objs = np.zeros(ls_size)
    for i in np.arange(ls_size):
      objs[i] = self._loss(ls[i])
    return objs

  def predict(self, l, X_test):
    return X_test @ self.beta_hat(self.X, self.y, l)