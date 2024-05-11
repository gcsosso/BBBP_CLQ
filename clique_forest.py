from base64 import encode
from collections import Counter
from json import encoder
import numpy as np
import pandas as pd # for dataframe and CSV handling
import argparse

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import matthews_corrcoef, roc_auc_score, r2_score, mean_squared_error
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import umap
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from imblearn.over_sampling import SMOTE 

from tqdm import tqdm

from cliques.cliques import get_clique_decomposition

import matplotlib.pyplot as plt
import seaborn as sns


parser = argparse.ArgumentParser()
parser.add_argument('--data', help='data csv file', default='./BBBP_clean.csv')
parser.add_argument('--smote', help='use smote', default=False, action='store_true')
parser.add_argument('--n_fold', help='number of folds for kfold cross validation', default=5 ,type=int)
parser.add_argument('--n_repeats', help='maximum number of repeats to run our experiments', default=10 ,type=int)
parser.add_argument('--num_workers', help='number of workers for dataloader', default=4, type=int)
args = parser.parse_args()

regression = False
data=pd.read_csv(args.data)
fig_name = 'clique_forest'
smiles = data.SMILES

if 'regression' in args.data:
    regression = True
    assert not args.smote, 'Cannot use SMOTE with regression, only classification'
    y = data.logBB.to_numpy().astype(float)
else:
    y = (data['BBB+/BBB-'] == 'BBB+').to_numpy().astype(int)


cliques, vocab = get_clique_decomposition(smiles)
X = cliques.to_numpy().astype(int)

if args.smote:
    fig_name = fig_name + '_smote'

cm = np.zeros((args.n_repeats, args.n_fold, 2, 2))
if regression:
    r2 = np.zeros((args.n_repeats, args.n_fold,))
    rmse = np.zeros((args.n_repeats, args.n_fold,))
    preds_all = np.zeros((args.n_repeats, len(y)))
else:
    mcc = np.zeros((args.n_repeats, args.n_fold,))
    auc = np.zeros((args.n_repeats, args.n_fold,))

for i in range(args.n_repeats):
    # Setting the random state here to i gives us reproducibility
    # but ensures we get a new random set of kfold indices each time
    kf = KFold(n_splits=args.n_fold, shuffle=True, random_state=i)
    for j, (train_index, test_index) in enumerate(kf.split(X)):
        print('Training Repeat {} Fold {}'.format(i, j))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if args.smote:
            print('Using SMOTE')
            sm = SMOTE(random_state=42)

            X_train, y_train = sm.fit_resample(X_train, y_train)
            X_train = np.clip(np.round(X_train), 0, np.inf)

            print('SMOTE Resampled dataset shape %s' % Counter(y_train))

        y_train = y_train[np.any(X_train > 0, axis=1)]
        X_train = X_train[np.any(X_train > 0, axis=1)]

        if regression:
            clf = RandomForestRegressor(n_estimators=64, random_state=42)
        else:
            clf = RandomForestClassifier(n_estimators=64, random_state=42)
        clf.fit(X_train, y_train)

        #ax = plt.gca()
        #rfc_disp = RocCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax, alpha=0.8)
        #plt.savefig(fig_name + '_roc.png')

        pred_test = clf.predict(X_test)
        if regression:
            r2[i, j] = r2_score(y_test, pred_test)
            rmse[i, j] = mean_squared_error(y_test, pred_test, squared=False)
            preds_all[i, test_index] = pred_test
        else:
            pred_proba_test = clf.predict_proba(X_test)[:, 1]
            cm[i, j] = confusion_matrix(y_test, pred_test)
            mcc[i, j] = matthews_corrcoef(y_test, pred_test)
            auc[i, j] = roc_auc_score(y_test, pred_proba_test)

if regression:
    print('Overall Mean R2 Score', np.mean(r2))
    print('Overall Mean RMSE', np.mean(rmse))

    fig, ax = plt.subplots(figsize=(10, 8))
    r2_dict = dict([(str(i), r2[:i].flatten()) for i in range(1, args.n_repeats + 1)])
    ax.boxplot(r2_dict.values())
    ax.set_xticklabels(r2_dict.keys())
    ax.set_xlabel('Number of repeats')
    ax.set_ylabel('R2 Score')
    plt.savefig(fig_name + '_r2.png')

    fig, ax = plt.subplots(figsize=(10, 8))
    rmse_dict = dict([(str(i), rmse[:i].flatten()) for i in range(1, args.n_repeats + 1)])
    ax.boxplot(rmse_dict.values())
    ax.set_xticklabels(rmse_dict.keys())
    ax.set_xlabel('Number of repeats')
    ax.set_ylabel('RMSE')
    plt.savefig(fig_name + '_rmse.png')

    fig, ax = plt.subplots(figsize=(10, 8))
    plot_x = y.tolist() * args.n_repeats
    plot_y = preds_all.tolist() # This works because we have repeats along the first axis
    ax.scatter(plot_x, plot_y, alpha=0.1)
    ax.plot([np.min(y), np.max(y)], [np.min(y), np.max(y)], color='black')
    ax.set_xlabel('True logBB')
    ax.set_ylabel('Predicted logBB')
    ax.grid()
    plt.savefig(fig_name + '_scatter.png')

else:
    cm = np.sum(cm, axis=1)
    cm = cm / np.unique(y, return_counts=True)[1][:, None]
    cm_mean = np.mean(cm, axis=0)
    cm_std = np.std(cm, axis=0)
    print(cm_mean)
    print(cm_std)

    plt.figure(figsize=(10, 8))
    sns.set(font_scale = 2.0)
    cmap = sns.color_palette("Blues", as_cmap=True)

    cm_labels = np.asarray([['{0:.4f}'.format(cm_mean[i,j]) + '\n+- ' + '{0:.4f}'.format(cm_std[i,j]) for j in range(2)] for i in range(2)])
    ax = sns.heatmap(cm_mean, annot=cm_labels, fmt='', cmap=cmap, vmin=0, vmax=1)
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')

    ax.xaxis.set_ticklabels(['BBBP-', 'BBBP+'])
    ax.yaxis.set_ticklabels(['BBBP-', 'BBBP+'])
    plt.savefig(fig_name + '_cm.png')


    fig, ax = plt.subplots(figsize=(10, 8))
    mcc_dict = dict([(str(i), mcc[:i].flatten()) for i in range(1, args.n_repeats + 1)])
    ax.boxplot(mcc_dict.values())
    ax.set_xticklabels(mcc_dict.keys())
    ax.set_xlabel('Number of repeats')
    ax.set_ylabel('Matthews Correlation Coefficient')
    plt.savefig(fig_name + '_mcc.png')

    print('Overall Mean Matthews Correlation Coefficient', np.mean(mcc), '+-', np.std(np.mean(auc, axis=1)))

    fig, ax = plt.subplots(figsize=(10, 8))
    auc_dict = dict([(str(i), auc[:i].flatten()) for i in range(1, args.n_repeats + 1)])
    ax.boxplot(auc_dict.values())
    ax.set_xticklabels(auc_dict.keys())
    ax.set_xlabel('Number of repeats')
    ax.set_ylabel('ROC AUC')
    plt.savefig(fig_name + '_auc.png')

    print('Overall Mean ROC AUC', np.mean(auc), '+-', np.std(np.mean(auc, axis=1)))

#plt.show()
