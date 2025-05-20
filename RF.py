from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os

def grid_search(X_train, y_train, param_grid, cv=5, scoring='accuracy'):

    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring=scoring)
    grid_search.fit(X_train, y_train)
    print("Best parameters:", grid_search.best_params_)
    print(f'Best grid Search Score: {grid_search.best_score_:.2f}')
    
    return grid_search.best_params_

def train_model(X_train, y_train, best_params):

    model = RandomForestClassifier(**best_params, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, output_path):

    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)

    y_test = y_test.values.ravel()
    y_test_pred = y_test_pred.ravel()

    balanced_acc = balanced_accuracy_score(y_test, y_test_pred)

    label_encoder = LabelEncoder()
    label_encoder.fit(np.concatenate([y_test, y_test_pred]))
    y_test_encoded = label_encoder.transform(y_test)
    auc_roc = roc_auc_score(y_test_encoded, y_test_prob, multi_class='ovr', average='weighted')

    report_dict = classification_report(y_test, y_test_pred, digits=4, output_dict=True)

    results = {
        'balanced_accuracy': balanced_acc,
        'auc_roc': auc_roc,
        'classification_report': report_dict
    }

    metrics_path = os.path.join(output_path, 'RF_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=4)

    cm = confusion_matrix(y_test, y_test_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    class_labels = sorted(np.unique(np.concatenate([y_test, y_test_pred])))

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")

    cm_path = os.path.join(output_path, 'RF_confusion_matrix.png')
    plt.savefig(cm_path, dpi=300)
    plt.close()
