def convert_to_percentage(value):
    return str(round(value * 100, 2)) + '%'


def print_model_accuracy_results(maximum_score, minimum_score, average_score):
    red, reset = '\033[91m', '\033[0m'
    print(red, 'Maximum Accuracy Score: ', reset, convert_to_percentage(maximum_score))
    print(red, 'Minimum Accuracy Score: ', reset, convert_to_percentage(minimum_score))
    print(red, 'Average Accuracy Score: ', reset, convert_to_percentage(average_score))
    print('\n')
