import os

def list_jpg_files(folder_path):
    jpg_files = [f"{folder_path}/{file}" for file in os.listdir(folder_path)]
    if not jpg_files:
        raise FileNotFoundError("No JPG files found in the folder.")
    return jpg_files
