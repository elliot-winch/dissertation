from main import NeuralNetwork
import handle_json

neural_network = NeuralNetwork(handle_json.json_file_to_obj("model_config.json"))
confusion_matrix = neural_network.test()

print(confusion_matrix)
print(neural_network.mean_average_precision(confusion_matrix))
