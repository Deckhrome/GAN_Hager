import os
import re
import shutil
from zipfile import ZipFile, BadZipFile

def unzip_all_files(input_folder='downloads', output_folder='extracted'):
    os.makedirs(output_folder, exist_ok=True)

    # List all files in the input folder
    file_list = os.listdir(input_folder)

    # Filter files based on the pattern file.zip.number
    zip_files = [file for file in file_list if re.match(r'^.*.zip.\d+$', file)]

    # Combine all parts into a single file
    combined_file_path = os.path.join(output_folder, 'combined.zip')
    with open(combined_file_path, 'wb') as combined_file:
        for zip_file in sorted(zip_files):
            zip_file_path = os.path.join(input_folder, zip_file)
            with open(zip_file_path, 'rb') as part_file:
                shutil.copyfileobj(part_file, combined_file)

    # Unzip the combined file
    try:
        with ZipFile(combined_file_path, 'r') as zip_ref:
            zip_ref.extractall(output_folder)
        print(f"Successfully extracted combined file to {output_folder}")

    except BadZipFile:
        print("Error: Combined file is not a valid zip file")

    finally:
        # Optional: Remove the combined file if you no longer need it
        os.remove(combined_file_path)

if __name__ == '__main__':
    unzip_all_files()