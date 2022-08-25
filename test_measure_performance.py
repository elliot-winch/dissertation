import measure_performance

TN = 1
FP = 100
FN = 50
TP = 200

matrix = [[TN, FP], [FN, TP]]

accuracy = (TN + TP) / (TN + FP + FN + TP)
precision = TP / (FP + TP)
recall = TP / (TP + FN)
false_positive_rate = FP / (FP + TN)

if __name__ == '__main__':
    print(matrix)
    print("True: Accuracy {} Precision {} Recall {} FPR {}".format(accuracy, precision, recall, false_positive_rate))
    print("Calculated: Accuracy {} Precision {} Recall {} FPR {}".format(measure_performance.accuracy(matrix), measure_performance.precision(matrix, 1, 1), measure_performance.recall(matrix, 1, 1), measure_performance.recall(matrix, 0, 1)))
