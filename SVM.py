import pandas as pd
import sklearn.model_selection as model_selection
from sklearn.model_selection import PredefinedSplit
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, balanced_accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
import json
import os

def gridsearch(X_train, y_train, param_grid, cv=5):

    svc = svm.SVC(class_weight='balanced', probability=True)
    grid = model_selection.GridSearchCV(estimator=svc, param_grid=param_grid, cv=cv, scoring='accuracy')

    grid.fit(X_train, y_train)

    print("Best Parameters:", grid.best_params_)
    print("Best Score:", grid.best_score_)

    return grid.best_estimator_

def svm_test(best_model, X_test, y_test, output_path):

    y_pred = best_model.predict(X_test)
    y_pred_prob = best_model.predict_proba(X_test)

    y_test =y_test.values.ravel()

    balanced_acc = balanced_accuracy_score(y_test, y_pred)


    label_encoder = LabelEncoder()
    label_encoder.fit(np.concatenate([y_test, y_pred]))
    y_test_encoded = label_encoder.transform(y_test)
    auc_roc = roc_auc_score(y_test_encoded, y_pred_prob, multi_class='ovr', average='weighted')

    report_dict = classification_report(y_test, y_pred, digits=4, output_dict=True)

    results = {
        'balanced_accuracy': balanced_acc,
        'auc_roc': auc_roc,
        'classification_report': report_dict
    }

    metrics_path = os.path.join(output_path, 'SVM_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=4)

    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    class_labels = sorted(np.unique(np.concatenate([y_test, y_pred])))

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")

    cm_path = os.path.join(output_path, 'SVM_confusion_matrix.png')
    plt.savefig(cm_path, dpi=300)
    plt.close()
