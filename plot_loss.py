import load_results
from lerp import lerp_vector

import argparse
from matplotlib import pyplot as plt


class PlotLoss(object):

    def __init__(self):
        super(PlotLoss, self).__init__()

    def plot_loss(self, name, epochs, loss_color='blue', val_color='orange'):
        losses = [l.loss for l in epochs]
        val_losses = [l.validation_loss for l in epochs]

        plt.plot(losses, label="{} Training Loss".format(name), color=loss_color)
        plt.plot(val_losses, label="{} Validation Loss".format(name), color=val_color)

        plt.pause(0.01)

    def show(self, block):
        plt.xlabel("Epochs")
        plt.ylabel("Loss")

        plt.ion()
        plt.pause(0.01)
        plt.show(block = block)

    def hide(self):
        plt.close()

    def set_epoch_count(self, epoch_count):
        plt.xlim([0, epoch_count - 1])

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder_name", help="path to results files")
    args = parser.parse_args()

    results, _ = load_results.load_results(args.folder_name)

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
            val_color=lerp_vector(val_color_start, val_color_end, color_level)
        )

        plot.show(block = False)
    input("Press ENTER to close")
