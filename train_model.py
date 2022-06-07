from neural_network import NeuralNetwork
import handle_json
import argparse
import sys

def train_model(filename, rearrange):
        neural_network = NeuralNetwork(handle_json.json_file_to_obj(filename))
        neural_network.train(needs_arrange=rearrange)

        neural_network = NeuralNetwork(handle_json.json_file_to_obj(filename))
        confusion_matrix = neural_network.test()

        print(confusion_matrix)
        print(neural_network.mean_average_precision(confusion_matrix))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", help="path to config file")
    parser.add_argument("-r", "--rearrange", help="does the data set need to be rearranged for processing?",
                        action="store_true")
    args = parser.parse_args()

    train_model(args.filename, args.rearrange)
