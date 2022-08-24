import smote
import handle_json

seed = 1
oversampling_factor_by_class = [2]
test_file_location = 'Testing/test_encoded_images.json'

if __name__ == "__main__":

    file_contents = handle_json.json_file_to_obj(test_file_location)

    synthetic_vectors = smote.generate_synthetic(file_contents.encoded_images, oversampling_factor_by_class, seed = seed)
    print(synthetic_vectors)
