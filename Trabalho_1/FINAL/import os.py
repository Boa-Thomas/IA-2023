import os
import shutil
import random

# Define the paths to your original dataset and the directory where you want to save your new splits
original_dataset_dir = 'IMAGES'
base_dir = 'IMAGES_SPLIT'

# Define the names of the new directories for the training, validation, and testing sets
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# Create the new directories for the training, validation, and testing sets
os.mkdir(base_dir)
os.mkdir(train_dir)
os.mkdir(validation_dir)
os.mkdir(test_dir)

# Define the maximum number of images in each set per class
max_images_per_class = 100

# Loop over each class in the original dataset and split its images into the new directories
for class_name in os.listdir(original_dataset_dir):
    class_dir = os.path.join(original_dataset_dir, class_name)
    if not os.path.isdir(class_dir):
        continue
    image_filenames = os.listdir(class_dir)
    random.shuffle(image_filenames)

    # Define the number of images in each set for this class
    num_images = min(len(image_filenames), 3 * max_images_per_class)
    num_train_images = min(num_images, max_images_per_class)
    num_val_images = min(num_images - num_train_images, max_images_per_class)
    num_test_images = num_images - num_train_images - num_val_images

    # Ensure that we have exactly 100 images from each class in each set
    if class_name == 'OVELHA':
        num_train_images = min(num_train_images, 100)
        num_val_images = min(num_val_images, 100 - num_train_images)
        num_test_images = min(num_test_images, 100 - num_train_images - num_val_images)
    elif class_name == 'VACA':
        num_train_images = min(num_train_images, 100)
        num_val_images = min(num_val_images, 100 - num_train_images)
        num_test_images = min(num_test_images, 100 - num_train_images - num_val_images)

    # Copy images to the new directories
    for i, image_filename in enumerate(image_filenames[:num_train_images]):
        src = os.path.join(class_dir, image_filename)
        dst = os.path.join(train_dir, class_name, image_filename)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
    for i, image_filename in enumerate(image_filenames[num_train_images:num_train_images+num_val_images]):
        src = os.path.join(class_dir, image_filename)
        dst = os.path.join(validation_dir, class_name, image_filename)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
    for i, image_filename in enumerate(image_filenames[num_train_images+num_val_images:num_images]):
        src = os.path.join(class_dir, image_filename)
        dst = os.path.join(test_dir, class_name, image_filename)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        
print("Done!")
