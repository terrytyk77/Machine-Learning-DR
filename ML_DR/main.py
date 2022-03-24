# Import the necessary modules
import math
import functions as my_func

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm as SupportVectorMachine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, accuracy_score
from sklearn.model_selection import train_test_split

# Create data frame from the .csv file
DR_DF = pd.read_csv('dr_dataset.csv')

# Create training sets
X = DR_DF.drop(columns=['class'])
Y = DR_DF['class']
x_train = x_test = y_train = y_test = 0

# Create the classifiers
MODEL = {
    'NAIVE_BAYES': GaussianNB(),
    'DECISION_TREE': DecisionTreeClassifier(),
    'SUPPORT_VECTOR_MACHINE': SupportVectorMachine.SVC(probability=True),
    'LOGISTIC_REGRESSION': LogisticRegression(solver='lbfgs', max_iter=DR_DF.shape[0] * 2)
}

# Constants
NUMBER_OF_ITERATIONS = 10  # How many train/test sets are we giving to each algorithm
TEST_SIZE_SCALE = 0.1   # How much from 0 to 1, is attributed to the test set from all the data

# Create train samples
TRAIN_TEST = []
for index in range(NUMBER_OF_ITERATIONS):
    TRAIN_TEST.append(train_test_split(X, Y, test_size=TEST_SIZE_SCALE))

# Test all the models
for key, model in MODEL.items():
    average_score, minimum_score, maximum_score, index_max = 0, math.inf, -math.inf, 0
    for index in range(NUMBER_OF_ITERATIONS):
        # Get one of the training sample
        x_train, x_test, y_train, y_test = TRAIN_TEST[index]

        # Fit the classifier to the training data
        model.fit(x_train, y_train)

        # Predict the labels of the test set
        y_pred = model.predict(x_test)

        # Predicted Output
        score = accuracy_score(y_test, y_pred)
        average_score += score

        # Find Minimum
        if score < minimum_score:
            minimum_score = score

        # Find Maximum
        if score > maximum_score:
            maximum_score = score
            index_max = index

    # Outputting the results
    print('\033[93m', key, '\033[0m')
    average_score /= NUMBER_OF_ITERATIONS
    my_func.print_model_accuracy_results(maximum_score, minimum_score, average_score)

    # Get the set that gave the maximum result
    x_train, x_test, y_train, y_test = TRAIN_TEST[index_max]

    # Compute predicted probabilities
    y_pred_prob = model.predict_proba(x_test)[:, 1]

    # Generate ROC curve values
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

    # Plot ROC curve
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(key + ' ROC Curve')
    plt.show()
