import os
import pandas as pd


base_dir = os.path.dirname(os.path.abspath(__file__))

folder_path = os.path.join(base_dir, '../resource', 'car-brands')

car_brands = []

for brand_folder in os.listdir(folder_path):
    brand_folder_path = os.path.join(folder_path, brand_folder)
    if os.path.isdir(brand_folder_path):
        car_brands.append(brand_folder)

df = pd.DataFrame(car_brands, columns=['Brand'])
print(df)
print("co sie tu dzieje")
