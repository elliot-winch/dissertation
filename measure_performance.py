def accuracy(matrix):

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

def error_rate(matrix):
    return 1 - accuracy(matrix)

def recall(matrix, row, _class):
    total = 0
    total_correct = 0
    for i in range(len(matrix[row])):
        total += matrix[row][i]

        if i == _class:
            total_correct += matrix[row][i]

    return total_correct / float(total)

def precision(matrix, column, _class):
    total = 0
    total_correct = 0
    for i in range(len(matrix)):
        total += matrix[i][column]

        if i == _class:
            total_correct += matrix[i][column]

    return total_correct / float(total)

def f1(matrix, _class):
    prec = precision(matrix, _class)
    rec = recall(matrix, _class)
    return 2 * prec * rec / (prec + rec)

if __name__ == '__main__':
    matrix = [[1,2],[3,4]]

    print(accuracy(matrix))
    print(error_rate(matrix))
    print(recall(matrix, 1))
    print(precision(matrix, 1))
    print(f1(matrix, 1))
