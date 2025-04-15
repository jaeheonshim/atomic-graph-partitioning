import csv
import random

NUM_MATERIALS = 100
MATERIAL_IDS_FILE = 'material_ids.txt'

material_ids = []

with open(MATERIAL_IDS_FILE, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        material_ids.append(row)
        
# choose a random subset
material_ids = random.choices(material_ids, k=NUM_MATERIALS)

with open('material_ids_rand_subset.csv', 'w') as file:
    reader = csv.writer(file)
    for row in material_ids:
        print(row)
        reader.writerow(row)