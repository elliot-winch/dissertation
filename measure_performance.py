def mean_average_precision(self, matrix):

    if len(matrix) is not len(matrix[0]):
        print("Error: MAP cannot be calculated. Confusion matrix must be square")
        exit()

    total = 0
    total_correct = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            total += matrix[i][j]

            if i is j:
                total_correct += matrix[i][j]

    return total_correct / float(total)
