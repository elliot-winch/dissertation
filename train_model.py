from neural_network import NeuralNetwork
import handle_json
import argparse
import sys

def train_model(config, output_file_name):
    neural_network = NeuralNetwork(config)
    neural_network.train_from_config()
    neural_network.test_from_config()

    handle_json.obj_to_json_file(neural_network.output, output_file_name)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", help="path to config file")
    parser.add_argument("-r", "--rearrange", help="does the data set need to be rearranged for processing?",
                        action="store_true")
    parser.add_argument("-o", "--output_file", help="path to output file")
    args = parser.parse_args()

    config = handle_json.json_file_to_obj(args.config_file)

    if args.rearrange:
        arrange_files.arrange_files(self.config)

    train_model(config, args.output_file)
