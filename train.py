r"""
This file defines the method required to training the classification models for predicting the gesture class.
"""
from copy import deepcopy
from typing import Sequence

from sklearn.model_selection import KFold
# Import basic models for classification
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, 
    BaggingClassifier, 
    AdaBoostClassifier, 
    GradientBoostingClassifier)


# import some other utils for training/inference
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


# collect model classes in an array
model_classes = [
    SVC,
    GaussianNB,
    BaggingClassifier,
    LogisticRegression,
    AdaBoostClassifier,
    KNeighborsClassifier,
    DecisionTreeClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
    ]

# in case one needs to be specific in hyper-parameters
model_args = {model_class.__name__: dict() for model_class in model_classes}



def train(X: Sequence, y: Sequence):
    """Method to train the classifier for predicting the gesture class.

    Args:
        X (Sequence): feaature data
        y (Sequence): label data

    Returns: a callable that predict the gesture class
    """
    # model selection
    kf = KFold(n_splits=4)
    acc = {
        model_class.__name__: [] for model_class in model_classes}
    f1score, precision, recall = deepcopy(acc), deepcopy(acc), deepcopy(acc)

    for model_class in model_classes:  
        model_name = model_class.__name__
        # TODO: organize logging before
        print('training {}'.format(model_name))
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model = model_class(**model_args[model_name])
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc[model_name].append(accuracy_score(y_test, y_pred))
            f1, prec, rec, _ = precision_recall_fscore_support(
                y_test, y_pred, average='micro')

            f1score[model_name].append(f1)
            precision[model_name].append(prec)
            recall[model_name].append(rec)
    # find the max accros key, mean(values) of f1scores, 
    # compare and get the best key
    class_of_max_f1 = max(f1score, key=lambda k, d=f1score: d[k])
    # train on whole data
    model_class_dict = {
        model_class.__name__: model_class for model_class in model_classes}
    best_model_class = model_class_dict[class_of_max_f1]
    # TODO: add some logging here
    print('best model was {}'.format(class_of_max_f1))
    print('kfold f1 acc {}'.format(f1score[class_of_max_f1]))

    # train the best model with the whole data
    model = best_model_class(**model_args[class_of_max_f1])
    print('training on the whole data')
    model.fit(X, y)
    # return the model
    return model
    




            


    


