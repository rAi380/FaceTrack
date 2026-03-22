import os
import shutil
import random

source_path = r"C:\Users\risha\.cache\kagglehub\datasets\jessicali9530\lfw-dataset\versions\4\lfw-deepfunneled\lfw-deepfunneled"
target_path = "dataset_small"

os.makedirs(target_path, exist_ok=True)

# select only 10 random people
people = os.listdir(source_path)
selected_people = random.sample(people, 10)

for person in selected_people:
    src_folder = os.path.join(source_path, person)
    dst_folder = os.path.join(target_path, person)

    if not os.path.isdir(src_folder):
        continue

    os.makedirs(dst_folder, exist_ok=True)

    images = os.listdir(src_folder)[:10]  # take only 10 images

    for img in images:
        shutil.copy(
            os.path.join(src_folder, img),
            os.path.join(dst_folder, img)
        )

print("Small dataset created!")