import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.base import clone
from sklearn.model_selection import KFold
import numpy as np

def show_predict_and_real(X, y):
    fig, axes = plt.subplots(nrows=1, ncols=len(y), figsize = (15, 5))
    for ax, image, label in zip(axes, X, y):
        ax.set_axis_off()
        ax.imshow(image.reshape((28, 28)), cmap = plt.cm.gray_r)
        ax.set_title("Predicted: %s" % str(label))

def plot_confusion_matrix(predicted, y):
    # Compute confusion matrix to evaluate the accuracy of a classification.
    confusion_matr = confusion_matrix(y, predicted)
    # Confusion Matrix visualization.
    confusion_matrix_display = ConfusionMatrixDisplay(confusion_matr)
    fig, ax = plt.subplots(figsize = (10, 10))
    confusion_matrix_display.plot(ax = ax)

# we need not only score thats why we don't use skleark cross_val_score
def cross_val_score_mod(classifier, X, y, folds, shuffle = False):
    #K-Folds cross-validator
    #Provides train/test indices to split data in train/test sets. Split dataset into k consecutive folds (without shuffling by default).
    #Each fold is then used once as a validation while the k - 1 remaining folds form the training set.
    kf = KFold(n_splits = folds, shuffle = shuffle)
    trained_classifier = []
    accuracy_list = []
    for train_index, test_index in kf.split(X):
        X_train_kfold, X_test_kfold = X[train_index], X[test_index]
        y_train_kfold, y_test_kfold = y[train_index], y[test_index]
        classifier_fold = clone(classifier)
        classifier_fold.fit(X_train_kfold, y_train_kfold)
        prediction = classifier_fold.predict(X_test_kfold)
        accuracy = accuracy_score(y_test_kfold, prediction)
        trained_classifier.append(classifier_fold)
        accuracy_list.append(accuracy)
    return trained_classifier, accuracy_list

def check_classifier(classifier, X_train, y_train, X_val, y_val):
    trained_classifier, accuracy_list = cross_val_score_mod(classifier, X_train, y_train, 5)
    best_classifier = trained_classifier[np.argmax(accuracy_list)]
    accuracy = 0.0
    for i in range(len(accuracy_list)):
        accuracy += accuracy_score(y_val, trained_classifier[i].predict(X_val))
    print(accuracy_list)
    print("Accuracy %.4f" %  (accuracy / len(accuracy_list)))

    return best_classifier

def show_classifier_result(clf, X_train, y_train, X_val, y_val, show_number = True):
    best_clf = check_classifier(clf, X_train, y_train, X_val, y_val)
    Z = best_clf.predict(X_val)
    plot_confusion_matrix(y_val, Z)
    wrong_predictions = [i for i in np.arange(len(Z)) if Z[i] != y_val[i]]
    if show_number:
        show_predict_and_real(X_val[wrong_predictions][:8], Z[wrong_predictions][:8])