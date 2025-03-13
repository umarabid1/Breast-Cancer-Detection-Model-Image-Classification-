import os
import shutil
import random
from collections import defaultdict

# Paths to the original dataset and the new dataset directories
original_dataset_dir = r"C:\Users\eyita\Downloads\archive (1)"  # Replace with your path
new_dataset_dir = r"C:\Users\eyita\Downloads\dataset"  # This will be created

# List of tumor subtypes
subtypes = ['A', 'F', 'PT', 'TA', 'DC', 'LC', 'MC', 'PC']

# Create directories for train and validation sets
for split in ['train', 'validation']:
    for subtype in subtypes:
        dir_path = os.path.join(new_dataset_dir, split, subtype)
        os.makedirs(dir_path, exist_ok=True)

# Dictionary to hold patient IDs for each subtype
patients_per_subtype = defaultdict(set)

# Function to parse filename and extract subtype and patient ID
def parse_filename(filename):
    # Example filename: SOB_B_TA-14-4659-40-001.png
    parts = filename.split('_')
    tumor_class = parts[1]  # 'B' or 'M'
    subtype_patient = parts[2]  # 'TA-14-4659-40-001.png'
    subtype = subtype_patient.split('-')[0]
    patient_id = '-'.join(subtype_patient.split('-')[1:3])  # '14-4659'
    return subtype, patient_id

# Collect patient IDs for each subtype
for root, dirs, files in os.walk(original_dataset_dir):
    for file in files:
        if file.endswith('.png'):
            subtype, patient_id = parse_filename(file)
            patients_per_subtype[subtype].add(patient_id)

# Split patients into training and validation sets
train_patients = defaultdict(set)
validation_patients = defaultdict(set)

for subtype, patients in patients_per_subtype.items():
    patients = list(patients)
    random.shuffle(patients)
    split_idx = int(len(patients) * 0.8)  # 80% training, 20% validation
    train_patients[subtype] = set(patients[:split_idx])
    validation_patients[subtype] = set(patients[split_idx:])

# Now copy images to the respective directories
for root, dirs, files in os.walk(original_dataset_dir):
    for file in files:
        if file.endswith('.png'):
            subtype, patient_id = parse_filename(file)
            src_path = os.path.join(root, file)
            if patient_id in train_patients[subtype]:
                dest_dir = os.path.join(new_dataset_dir, 'train', subtype)
            elif patient_id in validation_patients[subtype]:
                dest_dir = os.path.join(new_dataset_dir, 'validation', subtype)
            else:
                # Should not happen
                continue
            shutil.copy(src_path, dest_dir)

print("Dataset has been split into training and validation sets.")
