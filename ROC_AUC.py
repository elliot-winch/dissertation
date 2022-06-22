import handle_json
import measure_performance

from os import listdir
from os.path import isfile, join

import argparse
from matplotlib import pyplot as plt


json_extension = ".json"

def load_results(resultsFolder):
    json_file_names = [f for f in listdir(resultsFolder) if isfile(join(resultsFolder, f)) and f.endswith(json_extension)]

    results = []
    for json_file in json_file_names:
        results.append(handle_json.json_file_to_obj(join(resultsFolder, json_file)))

    return results, json_file_names

def trapeziod_area(base, h1, h2):
    base = abs(base)
    square = base * h1
    triangle = 0.5 * base * (h2 - h1)
    return square + triangle

def roc(results):

    false_positives = []
    true_positives = []
    result_names = []

    for i in range(0, len(results)):
        print("{}: {}".format(i, results[i].confusion_matrix))

        false_positives.append(measure_performance.recall(results[i].confusion_matrix, row=0, _class=1))
        true_positives.append(measure_performance.recall(results[i].confusion_matrix, row=1, _class=1))
        result_names.append(results[i].name)

    #Sorts the list and returns the index order
    indices = sorted(range(len(false_positives)), key=false_positives.__getitem__)
    false_positives[:] = [false_positives[i] for i in indices]
    true_positives[:] = [true_positives[i] for i in indices]
    result_names[:] = [result_names[i] for i in indices]

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

    results, file_names = load_results(args.folder_name)
    fps, tps, result_names = roc(results)

    auc = auc(fps, tps)

    print("ROC: Area under curve is {}\nFile Names: {}\nFalse Positives: {}\nTrue Positives:{}".format(auc, file_names, fps, tps))
    plt.plot(fps, tps, '-x')

    for i in range(0, len(result_names)):
        plt.annotate(result_names[i], [fps[i], tps[i]])

    plt.plot([0, 1], [0, 1], '--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.show()
