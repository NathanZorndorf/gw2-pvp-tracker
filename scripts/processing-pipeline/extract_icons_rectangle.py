import os
import pandas as pd
from PIL import Image

def extract_icons_from_rectangles(csv_path, samples_dir, output_dir):
    """
    Extract icons from rectangles that contain multiple icons, splitting each rectangle into 5 evenly spaced squares.

    Args:
        csv_path (str): Path to the CSV file containing rectangle bounding box data.
        samples_dir (str): Directory to search for start images (data/samples).
        output_dir (str): Directory to save extracted icons.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the CSV file
    df = pd.read_csv(csv_path)

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

            # Process each rectangle
            for _, row in df.iterrows():
                label_name = row['label_name']
                bbox_x = int(row['bbox_x'])
                bbox_y = int(row['bbox_y'])
                bbox_width = int(row['bbox_width'])
                bbox_height = int(row['bbox_height'])

                # Crop the rectangle
                rect_img = img.crop((bbox_x, bbox_y, bbox_x + bbox_width, bbox_y + bbox_height))

                # Split into 5 evenly spaced squares
                num_icons = 5
                icon_height = bbox_height // num_icons
                square_size = min(bbox_width, icon_height)  # Make it square

                team = "Red" if "Red" in label_name else "Blue"

                for i in range(num_icons):
                    # Calculate vertical position
                    y_start = i * icon_height
                    y_end = y_start + square_size

                    # Center horizontally
                    x_start = (bbox_width - square_size) // 2
                    x_end = x_start + square_size

                    # Crop the square
                    icon_img = rect_img.crop((x_start, y_start, x_end, y_end))

                    # Create filename
                    player_num = i + 1
                    safe_label = f"{team}_Player_{player_num}___Class"
                    output_filename = f"{sample_folder}_{safe_label}_{os.path.splitext(image_name)[0]}.png"
                    output_path = os.path.join(output_dir, output_filename)

                    # Save the icon
                    icon_img.save(output_path)
                    print(f"Extracted and saved: {output_path}")

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")

if __name__ == "__main__":
    # Define paths
    csv_path = os.path.join('data', 'ranked-icon-bounding-boxes-rectangle.csv')
    samples_dir = os.path.join('data', 'samples')
    output_dir = os.path.join('data', 'target-icons')

    # Run the extraction
    extract_icons_from_rectangles(csv_path, samples_dir, output_dir)