import autograd.numpy as np
from SURE_ridge import SURE_ridge
from k_fold_ridge import k_fold_ridge
from loocv_ridge import loocv_ridge
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import RidgeCV
import matplotlib.pyplot as plt

##############################
## almost collinear features
##############################

n = 200
d = 3
X = np.random.normal(loc=[-5., -6.], size=(n, d-1))
X_collinear = 5 * X[:,0] - X[:,1] + np.random.normal(scale = 0.01, size=n)
X = np.hstack((X, X_collinear.reshape(n, 1)))
# scaler = StandardScaler()
# scaler.fit(X)
# X = scaler.transform(X)
X = np.hstack((np.ones(n).reshape(n, 1), X))
beta = np.array([2., 3., -1., 0.3]).reshape(d+1, 1)
y = (X @ beta).ravel() + np.random.normal(size=n)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# beta hat with optimized regularization weight
sr = SURE_ridge(X_train, y_train, 1)
l = sr.solve(0., 0.1, 10000)
print(l)
beta_hat_l = sr.beta_hat(l)
print(beta_hat_l)
mse_sr = np.power(np.linalg.norm(beta_hat_l.ravel() - beta.ravel()), 2)
test_error_SURE = (1 / X_test.shape[0]) * np.power(np.linalg.norm(sr.predict(l, X_test) - y_test), 2)

# beta hat without regularization
hat = X_train.T @ X_train
L = np.linalg.cholesky(hat)
L_inv = np.linalg.inv(L)
hat_inv = L_inv.T @ L_inv
beta_hat = hat_inv @ X_train.T @ y_train
print(beta_hat)
mse = np.power(np.linalg.norm(beta_hat.ravel() - beta.ravel()), 2)
test_error = (1 / X_test.shape[0]) * np.power(np.linalg.norm(X_test @ beta_hat - y_test), 2)

# beta hat from k fold CV
alphas = np.arange(1,501,1)/100.
kfold = RidgeCV(alphas = alphas, cv = 10, fit_intercept=False).fit(X_train, y_train)
l_kfold = kfold.alpha_
beta_hat_l_kfold = sr.beta_hat(l_kfold)
print(beta_hat_l_kfold)
mse_kfold = np.power(np.linalg.norm(beta_hat_l_kfold.ravel() - beta.ravel()), 2)
test_error_kfold = (1 / X_test.shape[0]) * np.power(np.linalg.norm(kfold.predict(X_test) - y_test), 2)

# beta hat from LOOCV
loocv = RidgeCV(alphas = alphas, fit_intercept=False).fit(X_train, y_train)
l_loocv = loocv.alpha_
beta_hat_l_loocv = sr.beta_hat(l_loocv)
print(beta_hat_l_loocv)
mse_loocv = np.power(np.linalg.norm(beta_hat_l_loocv.ravel() - beta.ravel()), 2)
test_error_loocv = (1 / X_test.shape[0]) * np.power(np.linalg.norm(loocv.predict(X_test) - y_test), 2)

print("-----")
print([l, 0., l_kfold, l_loocv])
print([mse_sr, mse, mse_kfold, mse_loocv])
print([test_error_SURE, test_error, test_error_kfold, test_error_loocv])
print("-----")