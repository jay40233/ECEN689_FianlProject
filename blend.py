

from __future__ import division
import numpy as np
from rw_csv import getTraindata, getTestdata, getSolution
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import time



if __name__ == '__main__':
    start = time.time()
    np.random.seed(0) # seed to shuffle the train set

    n_folds = 10
    verbose = True
    shuffle = False

    print "Loading data..."
    # Reading training data from CSV file as input feature vector matrix X and output labels Y"    
    X,y = getTraindata("train.csv")
    # Reading testing data from CSV file
    X_submission = getTestdata("test.csv")
    y_true = getSolution("solution.csv")

    if shuffle:
        idx = np.random.permutation(y.size)
        X = X[idx]
        y = y[idx]

    skf = list(StratifiedKFold(y, n_folds))

    clfs = [RandomForestClassifier(n_estimators=1000, n_jobs=-1, criterion='gini'),
            RandomForestClassifier(n_estimators=1000, n_jobs=-1, criterion='entropy'),
            ExtraTreesClassifier(n_estimators=1000, n_jobs=-1, criterion='gini'),
            ExtraTreesClassifier(n_estimators=1000, n_jobs=-1, criterion='entropy'),
            GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=7, n_estimators=100)]

    print "Creating train and test sets for blending."
    
    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))
    
    for j, clf in enumerate(clfs):
        print j, clf
        dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
        for i, (train, test) in enumerate(skf):
            print "Fold", i
            X_train = X[train]
            y_train = y[train]
            X_test = X[test]
            y_test = y[test]
            clf.fit(X_train, y_train)
            y_submission = clf.predict_proba(X_test)[:,1]
            dataset_blend_train[test, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)[:,1]
        dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)

    print
    print "Blending."
    clf = LogisticRegression()
    clf.fit(dataset_blend_train, y)
    y_submission = clf.predict_proba(dataset_blend_test)[:,1]
    y_scores = y_submission
    print "Linear stretch of predictions to [0,1]"
    y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())
    y_pred = clf.predict(dataset_blend_test)
    print "Saving Results."
    np.savetxt(fname='result.csv', X=y_submission, fmt='%0.9f')
    AUC_value = roc_auc_score(y_true, y_scores)
    acc = accuracy_score(y_true, y_pred)
    print "Result:"
    print "AUC Score: " , AUC_value
    print "Accuracy: ", acc
    print "Coefficients (class weights):", clf.coef_
    end = time.time()
    print "Execution time: ",(end - start)