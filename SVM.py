import pandas as pd
import sklearn.model_selection as model_selection
from sklearn.model_selection import PredefinedSplit
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, balanced_accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np

def gridsearch(X_train, X_validation, y_train, y_validation, param_grid):

    X_train_val = pd.concat([X_train, X_validation])
    y_train_val = pd.concat([y_train, y_validation])

    split_index = [-1] * len(X_train) + [0] * len(X_validation)
    predefined_split = PredefinedSplit(test_fold=split_index)

    svc = svm.SVC(class_weight='balanced', probability=True)
    grid = model_selection.GridSearchCV(estimator=svc, param_grid=param_grid, cv=predefined_split, scoring='accuracy')

    grid.fit(X_train_val, y_train_val)

    cv_results_df = pd.DataFrame(grid.cv_results_)

    print("Best Parameters:", grid.best_params_)
    print("Best Score:", grid.best_score_)

    return grid.best_estimator_

def svm_test(best_model, X_test, y_test):

    y_pred = best_model.predict(X_test)
    y_pred_prob = best_model.predict_proba(X_test)

    y_test =y_test.values.ravel()
    label_encoder = LabelEncoder()
    label_encoder.fit(np.concatenate([y_test, y_pred]))
    y_test_encoded = label_encoder.transform(y_test)


    print(f"Test Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC: {roc_auc_score(y_test_encoded, y_pred_prob, multi_class='ovr', average='weighted'):.4f}")
    print(f"Classification Report: \n{classification_report(y_test, y_pred, digits=4)}")

    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    unique_classes = sorted(np.unique(np.concatenate([y_test, y_pred])))

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=unique_classes, yticklabels=unique_classes)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix (Best Kernel)")
    plt.show()