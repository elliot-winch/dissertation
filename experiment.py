import argparse

import train_model
import handle_json

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_folder_name", help="path to read config files")
    parser.add_argument("-o", "--output_folder_name", help="path to write output files")
    args = parser.parse_args()

    configs, names = handle_json.load_jsons(args.config_folder_name)

    for i in range(0, len(configs)):
        output_file = "{}_{}/{}".format(i, args.output_folder_name, names[i])
        print("Running training for {}".format(output_file))
        output = train_model.train_model(configs[i])
        output.name = names[i]
        handle_json.obj_to_json_file(output, output_file)
