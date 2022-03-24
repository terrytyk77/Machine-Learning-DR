# Import the necessary modules
import math
import functions as my_func

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm as SupportVectorMachine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split

# Create data frame from the .csv file
DR_DF = pd.read_csv('dr_dataset.csv')
print('\033[93m', "Default data shape:", DR_DF.shape, '\033[0m')

# Pre-processing the data: Drop bad quality images
DR_DF.drop(DR_DF[DR_DF['q'] == 0].index, inplace=True)
# Drop quality column, because it's irrelevant because, they are all good quality
DR_DF = DR_DF.drop(columns=['q'])
print('\033[92m', "After pre-processing:", DR_DF.shape, '\033[0m', "\n")

# Correlation Matrix
my_func.plot_correlation_matrix(plt, DR_DF)

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
TEST_SIZE_SCALE = 0.25  # How much from 0 to 1, is attributed to the test set from all the data

# Create train samples
TRAIN_TEST = []
for index in range(NUMBER_OF_ITERATIONS):
    TRAIN_TEST.append(train_test_split(X, Y, test_size=TEST_SIZE_SCALE))

# Test all the models
for key, model in MODEL.items():
    # Set all evaluation metrics variable to default
    average_accuracy_score, minimum_accuracy_score, maximum_accuracy_score, index_max = 0, math.inf, -math.inf, 0
    average_sensitivity_score, minimum_sensitivity_score, maximum_sensitivity_score = 0, math.inf, -math.inf
    average_specificity_score, minimum_specificity_score, maximum_specificity_score = 0, math.inf, -math.inf
    average_precision_score, minimum_precision_score, maximum_precision_score = 0, math.inf, -math.inf,
    average_f1_score, minimum_f1_score, maximum_f1_score = 0, math.inf, -math.inf

    for index in range(NUMBER_OF_ITERATIONS):
        # Get one of the training sample
        x_train, x_test, y_train, y_test = TRAIN_TEST[index]

        # Fit the classifier to the training data
        model.fit(x_train, y_train)

        # Predict the labels of the test set
        y_pred = model.predict(x_test)

        # Predicted output and calculate evaluation metrics scores
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp)
        f1 = (2 * precision * sensitivity) / (precision + sensitivity)

        # Add evaluation metrics scores for average result
        average_accuracy_score += accuracy
        average_sensitivity_score += sensitivity
        average_specificity_score += specificity
        average_precision_score += precision
        average_f1_score += f1

        # Find minimum evaluation metrics scores
        minimum_accuracy_score = my_func.get_minimum(minimum_accuracy_score, accuracy)
        minimum_sensitivity_score = my_func.get_minimum(minimum_sensitivity_score, sensitivity)
        minimum_specificity_score = my_func.get_minimum(minimum_specificity_score, specificity)
        minimum_precision_score = my_func.get_minimum(minimum_precision_score, precision)
        minimum_f1_score = my_func.get_minimum(minimum_f1_score, f1)

        # Find maximum evaluation metrics scores
        maximum_accuracy_score = my_func.get_maximum(maximum_accuracy_score, accuracy)
        maximum_sensitivity_score = my_func.get_maximum(maximum_sensitivity_score, sensitivity)
        maximum_specificity_score = my_func.get_maximum(maximum_specificity_score, specificity)
        maximum_precision_score = my_func.get_maximum(maximum_precision_score, precision)
        maximum_f1_score = my_func.get_maximum(maximum_f1_score, f1)

    # Outputting the algorithm used
    print('\033[93m', key, '\033[0m')

    # Calculate the average of the evaluation metrics scores
    average_accuracy_score /= NUMBER_OF_ITERATIONS
    average_sensitivity_score /= NUMBER_OF_ITERATIONS
    average_specificity_score /= NUMBER_OF_ITERATIONS
    average_precision_score /= NUMBER_OF_ITERATIONS
    average_f1_score /= NUMBER_OF_ITERATIONS

    # Outputting the results
    my_func.print_model_accuracy_results("Accuracy", maximum_accuracy_score, minimum_accuracy_score, average_accuracy_score)
    my_func.print_model_accuracy_results("Sensitivity", maximum_sensitivity_score, minimum_sensitivity_score, average_sensitivity_score)
    my_func.print_model_accuracy_results("Specificity", maximum_specificity_score, minimum_specificity_score, average_specificity_score)
    my_func.print_model_accuracy_results("Precision", maximum_precision_score, minimum_precision_score, average_precision_score)
    my_func.print_model_accuracy_results("F1", maximum_f1_score, minimum_f1_score, average_f1_score)

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
