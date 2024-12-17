import os
from PIL import Image
import random


def resize_and_save(input_dir, output_dir, image_files, size=(256, 256)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in image_files:
        img = Image.open(os.path.join(input_dir, filename))
        img = img.resize(size, Image.LANCZOS)
        img.save(os.path.join(output_dir, filename))


def main():
    base_dir = "E:/PHD/UIEB/base/Datasets/LSUI"
    input_dir = os.path.join(base_dir, "input")
    gt_dir = os.path.join(base_dir, "GT")

    train_input_dir = os.path.join(base_dir, "train_256/input")
    test_input_dir = os.path.join(base_dir, "test_256/input")
    train_target_dir = os.path.join(base_dir, "train_256/target")
    test_target_dir = os.path.join(base_dir, "test_256/target")

    # Get all image filenames
    all_images = os.listdir(input_dir)
    random.shuffle(all_images)  # Shuffle to randomize selection for train/test

    # Split into train and test sets
    train_images = all_images[:1500]
    test_images = all_images[1500:1700]

    # Resize and save images
    resize_and_save(input_dir, train_input_dir, train_images)
    resize_and_save(input_dir, test_input_dir, test_images)
    resize_and_save(gt_dir, train_target_dir, train_images)
    resize_and_save(gt_dir, test_target_dir, test_images)


if __name__ == "__main__":
    main()

# import os
# from PIL import Image
#
# def resize_images(input_dir, output_dir, size=(256, 256)):
#     """
#     Resize all images in the specified directory to the given size and save them to the output directory with modified filenames.
#
#     Args:
#     input_dir (str): Path to the directory containing the original images.
#     output_dir (str): Path to the directory where resized images will be saved.
#     size (tuple): A tuple of two integers, the new size of the images.
#     """
#     # Create the output directory if it does not exist
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#
#     # List all files in the input directory
#     files = os.listdir(input_dir)
#
#     # Process each file
#     for file in files:
#         if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
#             # Construct full file path
#             input_path = os.path.join(input_dir, file)
#
#             # Parse the filename to extract the desired number
#             # Assuming the format is set_f100.png and number is after '_f'
#             parts = file.split('_f')
#             if len(parts) > 1:
#                 # Extract the number and remove any trailing file type or extra characters
#                 num = parts[1].split('.')[0]
#                 new_filename = num + '.png'
#             else:
#                 new_filename = file  # Use original filename if the expected pattern is not found
#
#             # Construct output file path
#             output_path = os.path.join(output_dir, new_filename)
#
#             # Open and resize the image
#             img = Image.open(input_path)
#             img_resized = img.resize(size, Image.LANCZOS)
#
#             # Save the resized image
#             img_resized.save(output_path)
#
#
# if __name__ == "__main__":
#     input_directory = r'E:\PHD\UIEB\base\Datasets\UFO-120\Test\lrd'
#     output_directory = r'E:\PHD\UIEB\base\Datasets\UFO-120\test_256\input'
#     resize_images(input_directory, output_directory)