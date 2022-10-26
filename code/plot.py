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
import pandas as pd

plt.rcParams['font.size'] = '20'
plt.rcParams['figure.autolayout'] = True

# collinear
path = os.path.normpath(os.path.join(os.path.dirname( __file__ ), "..", "data", "collinear"))
collinear = pk.load(open(path, 'rb'))

collinear_lambdas = collinear[0]
collinear_mses = collinear[1]
collinear_test_errors = collinear[2]

collinear_lambdas_df = pd.DataFrame(collinear_lambdas[:, np.array([0,2,3,4,])], columns=["SURE", "5 fold", "10 fold", "LOO"])
collinear_mses_df = pd.DataFrame(collinear_mses, columns=["SURE", "OLS", "5 fold", "10 fold", "LOO"])
collinear_test_errors_df = pd.DataFrame(collinear_test_errors, columns=["SURE", "OLS", "5 fold", "10 fold", "LOO"])

path = os.path.normpath(os.path.join(os.path.dirname( __file__ ), "..", "plots", "collinear_lambda.png"))
plt.boxplot(collinear_lambdas_df, labels=["SURE", "5 fold", "10 fold", "LOO"])
plt.yscale("log")
plt.ylabel("lambda")
plt.savefig(path)
plt.clf()

path = os.path.normpath(os.path.join(os.path.dirname( __file__ ), "..", "plots", "collinear_mse.png"))
plt.boxplot(collinear_mses_df, labels=["SURE", "OLS", "5 fold", "10 fold", "LOO"])
plt.yscale("log")
plt.ylabel("beta MSE")
plt.savefig(path)
plt.clf()

path = os.path.normpath(os.path.join(os.path.dirname( __file__ ), "..", "plots", "collinear_mspe.png"))
plt.boxplot(collinear_test_errors_df, labels=["SURE", "OLS", "5 fold", "10 fold", "LOO"])
plt.ylabel("MSPE")
plt.savefig(path)
plt.clf()