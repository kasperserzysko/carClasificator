import shutil
import zipfile
import os

zip_path = '../../over-20-car-brands-dataset.zip'
extract_path = '../data'

os.makedirs(extract_path, exist_ok=True)

#rozpakowanie pobranego pliku zip
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print(f"Plik rozpakowany do {extract_path}")

#rozpakowany plik zip bedzie zawieral podwojny folder audi usuwamy jeden i zdj przerzucamy do drugiego
audi_folder_path = os.path.join(extract_path, 'Audi', 'Audi')


if os.path.exists(audi_folder_path):

    for file_name in os.listdir(audi_folder_path):
        src_file = os.path.join(audi_folder_path, file_name)
        dest_file = os.path.join(extract_path, 'Audi', file_name)
        shutil.move(src_file, dest_file)

    os.rmdir(audi_folder_path)

print("Zdjęcia zostały przeniesione, a pusty folder usunięty.")

#usuniecie wszytkich zdjec bez rozszerzenia .jpg
def remove_non_jpg_files(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if not file.lower().endswith('.jpg'):
                os.remove(os.path.join(root, file))

remove_non_jpg_files(extract_path)

print("Usunięto wszystkie pliki bez rozszerzenia .jpg.")