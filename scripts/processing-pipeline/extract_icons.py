import os
import pandas as pd
from PIL import Image
import glob

def extract_icons_from_csv(samples_dir, output_dir):
    """
    Extract icons from all start screenshots in samples based on bounding boxes defined in CSV files.
    Automatically selects appropriate CSV based on sample folder name.

    Args:
        samples_dir (str): Directory to search for start images (data/samples).
        output_dir (str): Directory to save extracted icons.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Find all start images in samples directory
    start_images = []
    for root, dirs, files in os.walk(samples_dir):
        for file in files:
            if 'match_start' in file and file.endswith('.png'):
                start_images.append(os.path.join(root, file))

    print(f"Found {len(start_images)} start images to process")

    # Process each start image
    for image_path in start_images:
        try:
            img = Image.open(image_path)
            image_name = os.path.basename(image_path)
            sample_folder = os.path.basename(os.path.dirname(image_path))

            # Determine config based on folder name
            if 'unranked' in sample_folder:
                csv_path = os.path.join(os.path.dirname(samples_dir), 'unranked_bounding_boxes.csv')
            elif 'ranked' in sample_folder:
                csv_path = os.path.join(os.path.dirname(samples_dir), 'ranked_bounding_boxes.csv')
            else:
                print(f"Skipping {image_path}: Could not determine if ranked or unranked")
                continue

            if not os.path.exists(csv_path):
                print(f"Warning: CSV file not found: {csv_path}")
                continue

            # Read the CSV file
            df = pd.read_csv(csv_path)
            # Handle duplicates (keep last)
            if 'label_name' in df.columns:
                 df = df.drop_duplicates(subset=['label_name'], keep='last')

            # Process each bounding box
            for _, row in df.iterrows():
                label_name = row['label_name']
                bbox_x = int(row['bbox_x'])
                bbox_y = int(row['bbox_y'])
                bbox_width = int(row['bbox_width'])
                bbox_height = int(row['bbox_height'])

                # Crop the image
                cropped_img = img.crop((bbox_x, bbox_y, bbox_x + bbox_width, bbox_y + bbox_height))

                # Create a safe filename
                safe_label = label_name.replace(' ', '_').replace('-', '_').replace('/', '_')
                output_filename = f"{sample_folder}_{safe_label}_{os.path.splitext(image_name)[0]}.png"
                output_path = os.path.join(output_dir, output_filename)

                # Save the cropped image
                cropped_img.save(output_path)
                print(f"Extracted and saved: {output_path}")

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")

if __name__ == "__main__":
    # Define paths
    samples_dir = os.path.join('data', 'samples')
    output_dir = os.path.join('data', 'target-icons')

    # Run the extraction
    extract_icons_from_csv(samples_dir, output_dir)