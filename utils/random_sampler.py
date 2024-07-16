import os
import shutil
import random


def sample_images(input_dir, output_dir, num_images):
    # Ensure the input directory exists
    if not os.path.exists(input_dir):
        raise ValueError(
            f"The specified input directory does not exist: {input_dir}")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get a list of all files in the input directory
    all_files = [f for f in os.listdir(
        input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    # Filter out files that are not images or specific image formats if necessary
    image_files = [f for f in all_files if f.lower().endswith(
        ('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    # Check if the number of images requested is greater than available images
    if num_images > len(image_files):
        raise ValueError(
            "Requested number of images exceeds the number available in the directory.")

    # Randomly select images
    sampled_images = random.sample(image_files, num_images)

    # Copy selected images to the output directory
    for image in sampled_images:
        shutil.copy(os.path.join(input_dir, image),
                    os.path.join(output_dir, image))

    print(f"Successfully copied {num_images} images to {output_dir}")


if __name__ == "__main__":
    # Usage example
    input_directory = '/home/pneuma/Desktop/ML/Deep Learning/PulmonaryDiseaseClassifier/PNEUMONIA_FULL'
    output_directory = '/home/pneuma/Desktop/ML/Deep Learning/PulmonaryDiseaseClassifier/dataset/train/PNEUMONIA'
    number_of_images = 1300  # Number of images to sample
    sample_images(input_directory, output_directory, number_of_images)
