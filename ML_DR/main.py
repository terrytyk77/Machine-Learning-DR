# Import the necessary modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, accuracy_score
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Create training and test sets
    dr_df = pd.read_csv('dr_dataset.csv')
    X = dr_df.drop(columns=["class"])
    y = dr_df["class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create the classifier:
    # model = DecisionTreeClassifier()
    model = LogisticRegression(solver='lbfgs', max_iter=2000)

    # Fit the classifier to the training data
    model.fit(X_train, y_train)

    # Predict the labels of the test set
    y_pred = model.predict(X_test)

    # Predicted Output
    pred_df = X_test.copy()
    pred_df["class"] = y_pred
    print("Predicted Data: \n", pred_df)
    print("Accuracy Score: ", accuracy_score(y_test, y_pred))
    print(y_test, y_pred)

    # Compute predicted probabilities
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # Generate ROC curve values
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

    # Plot ROC curve
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()
