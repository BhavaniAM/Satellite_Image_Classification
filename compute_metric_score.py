from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

def compute_metrics(y_test, preds):
    print("Accuracy score: ", accuracy_score(y_test, np.round_(preds)))
    print("Classification report:")
    print(classification_report(y_test, np.round_(preds)))
    print("Accuracy of CNN model is: ", accuracy_score(y_test,np.round_(preds))*100)
