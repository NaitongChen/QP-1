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

plt.rcParams['font.size'] = '20'
plt.rcParams['figure.autolayout'] = True

np.random.seed(1)
sigma = 1.
n = 100
d = 5
X = np.random.normal(loc=[-5., 0., 3.], scale=[1., 1., 1.],  size=(n, d-2))
X_collinear = -5 * X[:,1] + np.random.normal(scale = 0.01, size=n)
X = np.hstack((X, X_collinear.reshape(n, 1)))
X_collinear = 1 * X[:,2] + np.random.normal(scale = 0.01, size=n)
X = np.hstack((X, X_collinear.reshape(n, 1)))
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
beta = np.array([0., 3., -1., 1., 2.]).reshape(d, 1)
y = (X @ beta).ravel() + np.random.normal(scale = sigma, size=n)

for i in np.arange(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=i)

    ##########
    # SURE
    ##########
    sr = SURE_ridge(X_train, y_train, sigma)
    l = sr.solve(0., 0.05, 30000)
    print(l)
    beta_hat_l = sr.beta_hat(l)
    print(beta_hat_l)
    mse_sr = np.power(np.linalg.norm(beta_hat_l.ravel() - beta.ravel()), 2)
    test_error_SURE = (1 / X_test.shape[0]) * np.power(np.linalg.norm(sr.predict(l, X_test) - y_test), 2)
    alphas = np.arange(1,2001,1)/100.
    cv_scores_sure = sr.get_scores(alphas)

    rss = sr.get_rss(alphas)
    edf = sr.get_edf(alphas)

    plt.plot(alphas, rss, label="rss")
    plt.plot(alphas, edf, label="edf")
    plt.plot(alphas, cv_scores_sure, label="loss")
    plt.legend()
    plt.yscale("log")
    path = os.path.normpath(os.path.join(os.path.dirname( __file__ ), "..", "plots", "collinear_tradeoff_" + str(i) + ".png"))
    plt.ylabel("loss")
    plt.xlabel("lambda")
    plt.savefig(path)
    plt.clf()

    plt.plot(alphas, cv_scores_sure)
    path = os.path.normpath(os.path.join(os.path.dirname( __file__ ), "..", "plots", "collinear_sure_obj_" + str(i) + ".png"))
    plt.ylabel("SURE")
    plt.xlabel("lambda")
    plt.savefig(path)
    plt.clf()

    ##########
    # 5 fold
    ##########
    alphas = np.arange(1,2001,1)/100.
    kfold = k_fold_ridge(X_train, y_train, sigma, 5, i)
    cv_scores_5_fold = kfold.solve(alphas)

    plt.plot(alphas, cv_scores_5_fold)
    path = os.path.normpath(os.path.join(os.path.dirname( __file__ ), "..", "plots", "collinear_5fold_obj_" + str(i) + ".png"))
    plt.ylabel("CV error")
    plt.xlabel("lambda")
    plt.savefig(path)
    plt.clf()

    ##########
    # 10 fold
    ##########
    alphas = np.arange(1,2001,1)/100.
    kfold = k_fold_ridge(X_train, y_train, sigma, 10, i)
    cv_scores_10_fold = kfold.solve(alphas)

    plt.plot(alphas, cv_scores_10_fold)
    path = os.path.normpath(os.path.join(os.path.dirname( __file__ ), "..", "plots", "collinear_10fold_obj_" + str(i) + ".png"))
    plt.ylabel("CV error")
    plt.xlabel("lambda")
    plt.savefig(path)
    plt.clf()

    ##########
    # LOOCV
    ##########
    alphas = np.arange(1,2001,1)/100.
    loocv = RidgeCV(alphas = alphas, fit_intercept=False, store_cv_values=True).fit(X_train, y_train)
    cv_scores_loocv = np.sum(loocv.cv_values_, 0)

    plt.plot(alphas, cv_scores_loocv)
    path = os.path.normpath(os.path.join(os.path.dirname( __file__ ), "..", "plots", "collinear_loocv_obj_" + str(i) + ".png"))
    plt.ylabel("CV error")
    plt.xlabel("lambda")
    plt.savefig(path)
    plt.clf()