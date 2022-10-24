import autograd.numpy as np
from autograd import grad

class SURE_ridge():
  def __init__(self, X, y, sigma):
    self.X = X # including column of 1's.
    self.y = y
    self.sigma = sigma
    self.n, self.p = X.shape
    self.ds = np.linalg.svd(self.X, compute_uv = False)
    self.ds2 = np.power(self.ds, 2)
    # self.l_unc = 0 # l = exp(l_unc)

  def _hat_l_inv(self, l):
    hat_l = self.X.T @ self.X + l * np.eye(self.p)
    L = np.linalg.cholesky(hat_l)
    L_inv = np.linalg.inv(L)
    return L_inv.T @ L_inv

  def beta_hat(self, l):
    return self._hat_l_inv(l) @ self.X.T @ self.y

  def _loss(self, l_unc):
    l = np.exp(l_unc)
    residual = self.y - self.X @ self.beta_hat(l)
    return -self.n * self.sigma**2 + residual.T @ residual + 2 * self.sigma**2 * sum(self.ds2 / (self.ds2 + l))
    # return -self.n * self.sigma**2 + residual.T @ residual + 2 * self.sigma**2 * np.trace(self.X.T @ self.X * self._hat_l_inv(l))

  def _rss(self, l_unc):
    l = np.exp(l_unc)
    residual = self.y - self.X @ self.beta_hat(l)
    return residual.T @ residual
  
  def _edf(self, l_unc):
    l = np.exp(l_unc)
    return 2 * self.sigma**2 * sum(self.ds2 / (self.ds2 + l))

  def _shifted_loss(self, l_unc):
    l = np.exp(l_unc)
    residual = self.y - self.X @ self.beta_hat(l)
    return residual.T @ residual + 2 * self.sigma**2 * sum(self.ds2 / (self.ds2 + l))

  def solve(self, l_unc_0, stepsize, iter):
    l_unc = l_unc_0
    loss_grad = grad(self._loss)
    for i in np.arange(iter):
      l_unc -= loss_grad(l_unc) * stepsize
      if i % 1000 == 0:
        print("-----")
        print(i)
        print(np.exp(l_unc))
        print(self._loss(l_unc))
        print("-----")
    return np.exp(l_unc)

  def predict(self, l, X_test):
    return X_test @ self.beta_hat(l)

  def get_scores(self, ls):
    ls_size = ls.shape[0]
    objs = np.zeros(ls_size)
    for i in np.arange(ls_size):
      objs[i] = self._loss(np.log(ls[i]))
    return objs

  def get_rss(self, ls):
    ls_size = ls.shape[0]
    objs = np.zeros(ls_size)
    for i in np.arange(ls_size):
      objs[i] = self._rss(np.log(ls[i]))
    return objs

  def get_edf(self, ls):
    ls_size = ls.shape[0]
    objs = np.zeros(ls_size)
    for i in np.arange(ls_size):
      objs[i] = self._edf(np.log(ls[i]))
    return objs
  
  def get_shifted_loss(self, ls):
    ls_size = ls.shape[0]
    objs = np.zeros(ls_size)
    for i in np.arange(ls_size):
      objs[i] = self._shifted_loss(np.log(ls[i]))
    return objs