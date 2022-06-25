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

    def show(self):
        #box = self.ax.get_position()
        #self.ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        #self.ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.xlabel("Epochs")
        plt.ylabel("Loss")

        plt.ion()
        plt.pause(0.01)
        plt.show(block = False)

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

        print(str(results[i].name))
        print(results[i].epochs)
        print(plot.ax)

        plot.plot_loss(
            name=results[i].name,
            epochs=results[i].epochs,
            loss_color=lerp_vector(loss_color_start, loss_color_end, color_level),
            val_color=lerp_vector(val_color_start, val_color_end, color_level)
        )

    plot.show()
