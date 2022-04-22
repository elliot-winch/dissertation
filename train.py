from neural_network import NeuralNetwork
import handle_json

neural_network = NeuralNetwork(handle_json.json_file_to_obj("model_config.json"))
neural_network.train(needs_arrange=False)
