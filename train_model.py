import argparse
import msvcrt
import threading
import time

from neural_network import NeuralNetwork
import handle_json
import plot_loss


def train_model(config):
    neural_network = NeuralNetwork(config)
    loss_graph = plot_loss.PlotLoss()
    loss_graph.set_epoch_count(config.epochs)
    loss_graph.show(block=False)

    #TODO: might be worth writing a Event class
    neural_network.on_epoch_finished.append(lambda: plot_during_training(neural_network, loss_graph))

    cancel_thread = threading.Thread(target=check_cancel, args=(neural_network,))
    cancel_thread.daemon = True
    cancel_thread.start()
    neural_network.train_from_config()
    cancel_thread.join()

    loss_graph.hide()

    neural_network.test_from_config()

    return neural_network.output

def check_cancel(neural_network):
    while neural_network.finished_training == False:
        if msvcrt.kbhit():
            pressedKey = msvcrt.getch()

            if pressedKey == b'x':
                print("Training cancelled. After the next epoch, the model will be saved and training will stop")
                neural_network.cancel = True
            elif pressedKey == b'c':
                print("Training cancellation prevented.")
                neural_network.cancel = False
        else:
            time.sleep(1)

def plot_during_training(neural_network, loss_graph):
    loss_graph.plot_loss('Current', neural_network.output.epochs)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", help="path to config file")
    parser.add_argument("-r", "--rearrange", help="does the data set need to be rearranged for processing?",
                        action="store_true")
    parser.add_argument("-o", "--output_file", help="path to output file")
    parser.add_argument("-n", "--model_name", help="name for a model")
    args = parser.parse_args()

    config = handle_json.json_file_to_obj(args.config_file)

    if args.rearrange:
        arrange_files.arrange_files(self.config)

    output = train_model(config)

    if hasattr(args, 'model_name'):
        output.name = args.model_name

    handle_json.obj_to_json_file(output, args.output_file)
