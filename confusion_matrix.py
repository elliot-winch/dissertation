#Arrays with matching indicies showing the predicted / true class index
def get_confusion_matrx(true, pred, num_classes):

    confusion_matrix = [ [0]*num_classes for i in range(num_classes)]

    for i in range(len(true)):
        confusion_matrix[true[i]][pred[i]] += 1

    return confusion_matrix
