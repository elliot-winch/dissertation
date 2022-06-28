import argparse
from matplotlib import pyplot as plt

import handle_json
from lerp import lerp_vector

class PlotLoss(object):

    def __init__(self):
        super(PlotLoss, self).__init__()

    def plot_loss(self, name, epochs, loss_color='blue', val_color='orange', plot_learning_rate=False):

        plt.plot([epoch.loss for epoch in epochs], label="{} Train".format(name), color=loss_color)
        plt.plot([epoch.validation_loss for epoch in epochs], label="{} Val".format(name), color=val_color)

        if plot_learning_rate:
            #TODO: plot on second Y axis
            plt.plot([epoch.learning_rate for epoch in epochs], label="{} Val".format(name), color='red', linestyle="--")

        plt.pause(0.1)

    def show(self, block):

        plt.legend(bbox_to_anchor=(1, 1))

        plt.xlabel("Epochs")
        plt.ylabel("Loss")

        plt.ion()
        plt.tight_layout()
        plt.show(block = block)
        plt.pause(0.1)

    def hide(self):
        plt.close()

    def set_epoch_count(self, epoch_count):
        plt.xlim([0, epoch_count - 1])

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder_name", help="path to results files")
    args = parser.parse_args()

    order_by = lambda results : [result.config.order for result in results]
    results, file_names = handle_json.load_jsons(args.folder_name, order_by=order_by)

    plot = PlotLoss()

    loss_color_start = [0,0,1]
    loss_color_end = [0,0.75,1]
    val_color_start = [1,0,0]
    val_color_end = [1,0.75,0]

    for i in range(0, len(results)):
        color_level = i / (len(results) - 1)

        plot.plot_loss(
            name=results[i].name,
            epochs=results[i].epochs,
            loss_color=lerp_vector(loss_color_start, loss_color_end, color_level),
            val_color=lerp_vector(val_color_start, val_color_end, color_level),
            plot_learning_rate=False
        )

        plot.show(block = False)

    input("Press ENTER to close")
