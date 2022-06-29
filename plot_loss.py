import argparse
import matplotlib
from matplotlib import pyplot as plt

import handle_json
from lerp import lerp_vector

class PlotLoss(object):

    fig = None
    ax1 = None
    ax2 = None

    lines = {}

    def __init__(self):
        super(PlotLoss, self).__init__()

        matplotlib.use('TkAgg')
        plt.ion()
        self.fig = plt.figure()
        self.ax1 = plt.subplot(1,1,1)
        self.ax2 = self.ax1.twinx()

    def plot_loss(self, name, epochs, loss_color='blue', val_color='orange', plot_learning_rate=False):

        self.plot_line("{} Train".format(name), [epoch.loss for epoch in epochs], self.ax1, loss_color)
        self.plot_line("{} Val".format(name), [epoch.validation_loss for epoch in epochs], self.ax1, val_color)

        if plot_learning_rate:
            self.plot_line("{} Learning Rate".format(name), [epoch.learning_rate for epoch in epochs], self.ax2, 'red')

        self.ax1.legend(bbox_to_anchor=(1, 1))

        self.ax1.relim()                  # recompute the data limits
        self.ax1.autoscale_view()         # automatic axis scaling
        self.ax2.relim()                  # recompute the data limits
        self.ax2.autoscale_view()         # automatic axis scaling
        self.fig.canvas.flush_events()   # update the plot and take care of window events (like resizing etc.)


    def plot_line(self, name, data, ax, color):
        if name in self.lines:
            line_index = self.lines[name]
            ax.lines[line_index].set_data(range(0, len(data)), data)
        else:
            self.lines[name] = len(ax.lines)
            ax.plot(data, label=name, color=color)

    def show(self):
        self.ax1.set_xlabel("Epochs")
        self.ax1.set_ylabel("Loss")
        self.ax2.set_ylabel("Learning Rate")
        self.fig.show()

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
    plot.show()

    loss_color_start = [0,0,1]
    loss_color_end = [0,0.75,1]
    val_color_start = [1,0,0]
    val_color_end = [1,0.75,0]

    for i in range(0, len(results)):
        color_level = i / max(1, (len(results) - 1))

        plot.plot_loss(
            name=results[i].config.name,
            epochs=results[i].epochs,
            loss_color=lerp_vector(loss_color_start, loss_color_end, color_level),
            val_color=lerp_vector(val_color_start, val_color_end, color_level),
            plot_learning_rate=False
        )

    input("Press ENTER to close")
