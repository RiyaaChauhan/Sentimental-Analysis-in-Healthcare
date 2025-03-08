import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_curve,
    roc_auc_score, precision_recall_curve, auc
)
import numpy as np
import os

# Function to create output directory if it doesn't exist
def create_output_directory(directory='output'):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to save confusion matrix as an image
def plot_confusion_matrix(y_test, y_pred, directory='output'):
    create_output_directory(directory)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', 
                xticklabels=['Negative', 'Neutral', 'Positive'], 
                yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(directory, 'confusion_matrix.png'))
    plt.close()

# Function to plot and save the ROC curve
def plot_roc_curve(y_test, y_proba, directory='output'):
    create_output_directory(directory)
    fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(directory, 'roc_curve.png'))
    plt.close()

# Function to plot and save precision-recall curve
def plot_precision_recall_curve(y_test, y_proba, directory='output'):
    create_output_directory(directory)
    precision, recall, _ = precision_recall_curve(y_test, y_proba, pos_label=1)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, color='b', lw=2, label=f'Precision-Recall curve (area = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(directory, 'precision_recall_curve.png'))
    plt.close()

# Function to plot and save accuracy graph
def plot_accuracy(y_test, y_pred, directory='output'):
    create_output_directory(directory)
    accuracy = accuracy_score(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    plt.bar(['Accuracy'], [accuracy], color='green')
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.savefig(os.path.join(directory, 'accuracy_graph.png'))
    plt.close()

# Function to save classification report and accuracy to a text file
def save_evaluation_to_text(y_test, y_pred, y_proba, directory='output'):
    create_output_directory(directory)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label=1)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, _ = precision_recall_curve(y_test, y_proba, pos_label=1)
    pr_auc = auc(recall, precision)
    
    with open(os.path.join(directory, 'evaluation_report.txt'), 'w') as file:
        # Write Accuracy
        file.write(f"Accuracy on test data: {accuracy:.4f}\n\n")
        
        # Write Classification Report
        file.write("Classification Report:\n")
        file.write(report)
        file.write("\n")

        # Write Confusion Matrix
        file.write("Confusion Matrix:\n")
        file.write(np.array2string(cm))
        file.write("\n\n")

        # Write ROC-AUC Information
        file.write(f"ROC-AUC Score: {roc_auc:.4f}\n")
        file.write(f"FPR: {fpr}\n")
        file.write(f"TPR: {tpr}\n\n")

        # Write Precision-Recall Information
        file.write(f"Precision-Recall AUC Score: {pr_auc:.4f}\n")
        file.write(f"Precision: {precision}\n")
        file.write(f"Recall: {recall}\n")

# Function to evaluate the model's performance and save outputs
def evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Probability estimates for positive class

    # Evaluate model accuracy and classification report
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on test data: {accuracy}")
    print("Classification Report:")
    report = classification_report(y_test, y_pred)
    print(report)

    # Save confusion matrix plot
    plot_confusion_matrix(y_test, y_pred)

    # Save ROC curve plot
    plot_roc_curve(y_test, y_proba)

    # Save Precision-Recall curve plot
    plot_precision_recall_curve(y_test, y_proba)

    # Save accuracy graph
    plot_accuracy(y_test, y_pred)

    # Save evaluation report to a text file
    save_evaluation_to_text(y_test, y_pred, y_proba)

# Example usage (assuming you have a model, X_test, and y_test ready)
# evaluate_model(your_model, X_test, y_test)
