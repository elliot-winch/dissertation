from neural_network import NeuralNetwork
import handle_json
import argparse
import sys

def train_model(config_file_name, rearrange, output_file_name):
    config = handle_json.json_file_to_obj(config_file_name)

    neural_network = NeuralNetwork(config)
    neural_network.train(needs_arrange=rearrange)
    neural_network.test()

    handle_json.obj_to_json_file(neural_network.output, output_file_name)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", help="path to config file")
    parser.add_argument("-r", "--rearrange", help="does the data set need to be rearranged for processing?",
                        action="store_true")
    parser.add_argument("-o", "--output_file", help="path to output file")
    args = parser.parse_args()

    train_model(args.config_file, args.rearrange, args.output_file)
