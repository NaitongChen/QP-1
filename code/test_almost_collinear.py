import autograd.numpy as np
from SURE_ridge import SURE_ridge
from k_fold_ridge import k_fold_ridge
from loocv_ridge import loocv_ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
import matplotlib.pyplot as plt
import pickle as pk
import os
import sys

########################
## independent features
########################

np.random.seed(1)
sigma = 1.
n = 100
d = 3
X = np.random.normal(loc=[-5., -6., 3.], scale=[1., 1., 1.],  size=(n, d-1))
X_collinear = -5 * X[:,1] + np.random.normal(scale = 0.01, size=n)
X = np.hstack((X, X_collinear.reshape(n, 1)))
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
beta = np.array([0., 3., -1., 2.]).reshape(d, 1)
y = (X @ beta).ravel() + np.random.normal(scale = sigma, size=n)

test_errors = np.zeros((100,5))
lambdas = np.zeros((100,5))
mses = np.zeros((100,5))

for i in np.arange(100):
    print("-----")
    print(i)
    print("-----")
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=i)

    # beta hat with optimized regularization weight
    sr = SURE_ridge(X_train, y_train, sigma)
    l = sr.solve(0., 0.05, 20000)
    print(l)
    beta_hat_l = sr.beta_hat(l)
    print(beta_hat_l)
    mse_sr = np.power(np.linalg.norm(beta_hat_l.ravel() - beta.ravel()), 2)
    test_error_SURE = (1 / X_test.shape[0]) * np.power(np.linalg.norm(sr.predict(l, X_test) - y_test), 2)

    # alphas = np.arange(1,2001,1)/100.
    # cv_scores = sr.get_scores(alphas)

    # rss = sr.get_rss(alphas)
    # edf = sr.get_edf(alphas)
    # shifted_loss = sr.get_shifted_loss(alphas)

    # plt.plot(alphas, cv_scores)

    # plt.plot(alphas, rss)
    # # plt.plot(alphas, edf)
    # plt.plot(alphas, shifted_loss)
    # plt.show()

    # beta hat without regularization
    hat = X_train.T @ X_train
    L = np.linalg.cholesky(hat)
    L_inv = np.linalg.inv(L)
    hat_inv = L_inv.T @ L_inv
    beta_hat = hat_inv @ X_train.T @ y_train
    print(beta_hat)
    mse = np.power(np.linalg.norm(beta_hat.ravel() - beta.ravel()), 2)
    test_error = (1 / X_test.shape[0]) * np.power(np.linalg.norm(X_test @ beta_hat - y_test), 2)

    alphas = np.arange(1,2001,1)/100.
    kfold = k_fold_ridge(X_train, y_train, sigma, 5, i)
    cv_scores = kfold.solve(alphas)
    best_index = np.argmin(cv_scores)
    l_5fold = alphas[best_index]
    beta_hat_l_5fold = sr.beta_hat(l_5fold)
    print(beta_hat_l_5fold)
    mse_5fold = np.power(np.linalg.norm(beta_hat_l_5fold.ravel() - beta.ravel()), 2)
    test_error_5fold = (1 / X_test.shape[0]) * np.power(np.linalg.norm(kfold.predict(l_5fold, X_test) - y_test), 2)

    alphas = np.arange(1,2001,1)/100.
    kfold = k_fold_ridge(X_train, y_train, sigma, 10, i)
    cv_scores = kfold.solve(alphas)
    best_index = np.argmin(cv_scores)
    l_10fold = alphas[best_index]
    beta_hat_l_10fold = sr.beta_hat(l_10fold)
    print(beta_hat_l_10fold)
    mse_10fold = np.power(np.linalg.norm(beta_hat_l_10fold.ravel() - beta.ravel()), 2)
    test_error_10fold = (1 / X_test.shape[0]) * np.power(np.linalg.norm(kfold.predict(l_10fold, X_test) - y_test), 2)

    # plt.plot(alphas, cv_scores)

    # beta hat from LOOCV
    alphas = np.arange(1,2001,1)/100.
    loocv = RidgeCV(alphas = alphas, fit_intercept=False, store_cv_values=True).fit(X_train, y_train)
    l_loocv = loocv.alpha_
    beta_hat_l_loocv = sr.beta_hat(l_loocv)
    print(beta_hat_l_loocv)
    mse_loocv = np.power(np.linalg.norm(beta_hat_l_loocv.ravel() - beta.ravel()), 2)
    test_error_loocv = (1 / X_test.shape[0]) * np.power(np.linalg.norm(loocv.predict(X_test) - y_test), 2)

    # cv_scores = np.sum(loocv.cv_values_, 0)
    # plt.plot(alphas, cv_scores)

    # print("-----")
    lambdas[i,:] = ([l, 0., l_5fold, l_10fold, l_loocv])
    mses[i,:] = ([mse_sr, mse, mse_5fold, mse_10fold, mse_loocv])
    test_errors[i,:] = np.array([test_error_SURE, test_error, test_error_5fold, test_error_10fold, test_error_loocv])
    # print("-----")

path = os.path.normpath(os.path.join(os.path.dirname( __file__ ), "..", "data", "collinear"))
fi = open(path, 'wb')
pk.dump((lambdas, mses, test_errors), fi)
fi.close()