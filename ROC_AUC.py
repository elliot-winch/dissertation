import argparse
from matplotlib import pyplot as plt

import handle_json
import measure_performance

def trapeziod_area(base, h1, h2):
    base = abs(base)
    square = base * h1
    triangle = 0.5 * base * (h2 - h1)
    return square + triangle

def roc(results):

    false_positives = []
    true_positives = []

    for i in range(0, len(results)):
        print("{}: {}".format(i, results[i].confusion_matrix))

        false_positives.append(measure_performance.recall(results[i].confusion_matrix, row=0, _class=1))
        true_positives.append(measure_performance.recall(results[i].confusion_matrix, row=1, _class=1))

    return false_positives, true_positives, result_names

#Assuming ascending ordered by false_positives
#Else overlapping areas might be counted
def auc(false_positives, true_positives):
    auc = 0

    for i in range(0, len(false_positives)):
        if i > 0:
            trapeziod = trapeziod_area(false_positives[i] - false_positives[i - 1], true_positives[i - 1], true_positives[i])
        else:
            trapeziod = trapeziod_area(false_positives[i] - 0, 0, true_positives[i])

        auc += trapeziod

    return auc

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder_name", help="path to results files")
    args = parser.parse_args()

    order_by = lambda results : [measure_performance.recall(result.confusion_matrix, row=0, _class=1) for result in results]
    results, file_names = handle_json.load_jsons(args.folder_name, order_by=order_by)
    fps, tps = roc(results)

    auc = auc(fps, tps)

    print("ROC: Area under curve is {}\nFile Names: {}\nFalse Positives: {}\nTrue Positives:{}".format(auc, file_names, fps, tps))
    plt.plot(fps, tps, '-x')

    for i in range(0, len(file_names)):
        plt.annotate(file_names[i], [fps[i], tps[i]])

    plt.plot([0, 1], [0, 1], '--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.show()
