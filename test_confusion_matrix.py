import confusion_matrix

true = [0, 0, 0, 1, 1, 1, 1]
predicted = [1, 0, 0, 1, 0, 0, 0]
num_classes = 2
result = [[2, 1], [3, 1]]

if __name__ == "__main__":
    print('True: {} Predicted: {} Result: {} Calculated: {}'.format(true, predicted, result, confusion_matrix.get_confusion_matrx(true, predicted, num_classes)))
