from collections import Counter
import numpy as np
import pandas as pd # for dataframe and CSV handling
import argparse

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import umap
from sklearn.manifold import TSNE

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
args = parser.parse_args()

data=pd.read_csv(args.data)
fig_name = 'clique_forest'
smiles = data.SMILES

y = (data['BBB+/BBB-'] == 'BBB+').to_numpy().astype(int)

print('Getting clique decompositions')
cliques, vocab = get_clique_decomposition(smiles)
X = cliques.to_numpy().astype(int)
print('Done getting clique decompositions')

if args.smote:
    fig_name = fig_name + '_smote'

cm = np.zeros((args.n_repeats, args.n_fold, 2, 2))
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

        clf = RandomForestClassifier(n_estimators=64, random_state=42)
        clf.fit(X_train, y_train)

        #ax = plt.gca()
        #rfc_disp = RocCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax, alpha=0.8)
        #plt.savefig(fig_name + '_roc.png')

        pred_test = clf.predict(X_test)
        pred_proba_test = clf.predict_proba(X_test)[:, 1]
        cm[i, j] = confusion_matrix(y_test, pred_test)
        mcc[i, j] = matthews_corrcoef(y_test, pred_test)
        auc[i, j] = roc_auc_score(y_test, pred_proba_test)

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

# Do feature importance analysis
clf = RandomForestClassifier(n_estimators=64, random_state=42)
clf.fit(X, y)

feature_importance = clf.feature_importances_
feature_importance_std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)

feature_importance = pd.DataFrame({'feature': vocab, 'importance': feature_importance, 'std': feature_importance_std, 'frequency': np.sum(X > 0, axis=0)})
feature_importance = feature_importance[feature_importance['frequency'] > 50]
feature_importance = feature_importance.sort_values('importance', ascending=False)

feature_importance.to_csv(fig_name + '_feature_importance.csv', index=False)

# Plot the feature importance of the top 20 features
plt.figure(figsize=(16, 9))
sns.set(font_scale = 2.0)
ax = sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
ax.set_xlabel('Feature Importance')
ax.set_ylabel('Clique')
ax.set_title('Feature Importance Analysis by Scikit-learn using Mean Decrease in Impurity')
plt.savefig(fig_name + '_feature_importance.png', bbox_inches = "tight")

#plt.show()
