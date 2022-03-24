def convert_to_percentage(value):
    return str(round(value * 100, 2)) + '%'


def print_model_accuracy_results(metric, maximum_score, minimum_score, average_score):
    red, reset = '\033[91m', '\033[0m'
    print(red, 'Maximum ', metric, ' Score: ', reset, convert_to_percentage(maximum_score))
    print(red, 'Minimum ', metric, ' Score: ', reset, convert_to_percentage(minimum_score))
    print(red, 'Average ', metric, ' Score: ', reset, convert_to_percentage(average_score))
    print('\n')


def get_minimum(value, new_value):
    return new_value if new_value < value else value


def get_maximum(value, new_value):
    return new_value if new_value > value else value


def plot_correlation_matrix(plt, df):
    figure = plt.figure(figsize=(10, 8))
    plt.matshow(df.corr(), fignum=figure.number)
    plt.xticks(range(
        df.select_dtypes(['number']).shape[1]),
        df.select_dtypes(['number']).columns,
        fontsize=14,
        rotation=45)
    plt.yticks(range(
        df.select_dtypes(['number']).shape[1]),
        df.select_dtypes(['number']).columns,
        fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16)
    plt.show()
