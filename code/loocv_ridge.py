import autograd.numpy as np
from autograd import grad

class loocv_ridge():
  def __init__(self, X, y, sigma, seed):
    self.X = X # including column of 1's.
    self.y = y
    self.sigma = sigma
    self.n, self.p = X.shape
    self.seed = seed

  def _leverages(self, l):
    hat_mat = self.X @ self._hat_l_inv(l) @ self.X.T
    self.ds = np.linalg.svd(hat_mat, compute_uv = False)
    self.ds2 = np.power(self.ds, 2)
    return (self.ds2 / (self.ds2 + l)).ravel()

  def _hat_l_inv(self, l):
    hat_l = self.X.T @ self.X + l * np.eye(self.p)
    L = np.linalg.cholesky(hat_l)
    L_inv = np.linalg.inv(L)
    return L_inv.T @ L_inv

  def beta_hat(self, l):
    return self._hat_l_inv(l) @ self.X.T @ self.y

  def _loss(self, l_unc):
    l = np.exp(l_unc)
    return (1 / self.n) * np.ones(self.n) @ np.power((self.y - self.X @ self.beta_hat(l)) / (1 - self._leverages(l)), 2)

  def solve(self, l_unc_0, stepsize, iter):
    l_unc = l_unc_0
    loss_grad = grad(self._loss)
    for i in np.arange(iter):
      l_unc -= loss_grad(l_unc) * stepsize
      print(np.exp(l_unc))
    return np.exp(l_unc)

  def predict(self, l, X_test):
    return X_test @ self.beta_hat(l)