import load_results
from lerp import lerp_vector

import argparse
from matplotlib import pyplot as plt

def plot_loss(result, show=True, loss_color='blue', val_color='orange'):
    losses = [l.loss for l in result.epochs]
    val_losses = [l.validation_loss for l in result.epochs]

    plt.plot(losses, label="{} loss".format(result.name), color=loss_color)
    plt.plot(val_losses, label="{} Validation loss".format(result.name), color=val_color)

    if show:
        show()

def show(ax):
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.show()

if __name__ == '__main__':

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder_name", help="path to results files")
    args = parser.parse_args()
    """

    results, _ = load_results.load_results("experiment_upsampling_2106") #args.folder_name)

    fig = plt.figure()
    ax = plt.subplot(111)

    loss_color_start = [0,0,1]
    loss_color_end = [0,0.75,1]
    val_color_start = [1,0,0]
    val_color_end = [1,0.75,0]

    for i in range(0, len(results)):
        color_level = i / (len(results) - 1)

        plot_loss(
            results[i],
            show=False,
            loss_color=lerp_vector(loss_color_start, loss_color_end, color_level),
            val_color=lerp_vector(val_color_start, val_color_end, color_level),
        )

    show(ax)

    """
    results, file_names = load_results.load_results(args.folder_name)
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
    """
