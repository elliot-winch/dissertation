import train_model
import handle_json

class_balances = [[0.25, 1], [0.5, 1], [0.75, 1], [1,1]]

config_file_name = 'experiment_config.json'
output_file_name = 'experiment_upsampling_2106'
output_file_extension = 'json'

config = handle_json.json_file_to_obj(config_file_name)

for class_balance in class_balances:
    config.class_balance = class_balance
    output_file = "{}/{}.{}".format(output_file_name, str(class_balance), output_file_extension)
    print(output_file)
    output = train_model.train_model(config)
    output.name = str(class_balance)
    handle_json.obj_to_json_file(output, output_file_name)
