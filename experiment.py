import argparse

import train_model
import handle_json

import time
from os.path import splitext

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_folder_name", help="path to read config files")
    parser.add_argument("-o", "--output_folder_name", help="path to write output files")
    args = parser.parse_args()

    order_by = lambda configs : [config.order for config in configs]
    configs, names = handle_json.load_jsons(args.config_folder_name, order_by=order_by)

    timestr = time.strftime("%m%d%H%M")

    for i in range(0, len(configs)):
        output_file = "{}_{}/{}".format(args.output_folder_name, timestr, names[i])

        print("Running training for {}".format(output_file))

        train_model.train_model(configs[i], output_file)
