import os

def load_images(image_folder, return_full_path=False, image_type=".png", search_subfolders=False):
    file_names = []
    for file_name in os.listdir(image_folder):
        full_path = os.path.join(image_folder, file_name)
        if os.path.isfile(full_path) and file_name.endswith(image_type):
            file_names.append(full_path if return_full_path else file_name)

    if search_subfolders:
        for folder_name in os.listdir(image_folder):
            full_path = os.path.join(image_folder, file_name)
            if os.path.isdir(full_path):
                file_names = file_names + load_images(full_path, return_full_path, image_type, search_subfolders)

    return file_names
