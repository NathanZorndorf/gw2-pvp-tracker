import pandas as pd
import cv2
import os
from pathlib import Path
import numpy as np

def visualize_bounding_boxes(csv_path, samples_dir, output_dir):
    """
    Reads bounding boxes from CSV and draws them on ALL match_start images found in samples_dir.
    This helps verify if the bounding boxes align across different samples.
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Extract unique labels and their boxes as a reference set
    # We assume the CSV contains one "truth" set we want to test against all images
    reference_boxes = []
    seen_labels = set()
    
    for _, row in df.iterrows():
        label = row['label_name']
        if label not in seen_labels:
            reference_boxes.append({
                'label': label,
                'x': int(row['bbox_x']),
                'y': int(row['bbox_y']),
                'w': int(row['bbox_width']),
                'h': int(row['bbox_height'])
            })
            seen_labels.add(label)
    
    print(f"Loaded {len(reference_boxes)} reference bounding boxes from CSV.")

    # Find all match_start images
    samples_path = Path(samples_dir)
    print(f"Scanning {samples_dir} for match_start images...")
    images_to_process = list(samples_path.rglob("match_start*.png"))
    
    print(f"Found {len(images_to_process)} images to process.")

    for image_path in images_to_process:
        print(f"Processing {image_path.name}...")
        
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Error: Failed to load image {image_path}")
            continue

        # Draw ALL reference boxes on this image
        for box in reference_boxes:
            x, y, w, h = box['x'], box['y'], box['w'], box['h']
            label = box['label']
            
            # Draw rectangle 
            # Use red color (BGR)
            color = (0, 0, 255) 
            thickness = 2
            cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
            
            # Draw label
            font_scale = 0.6
            font = cv2.FONT_HERSHEY_SIMPLEX
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw background for text (semi-transparent if possible, but solid for now)
            # Ensure text doesn't go off screen at top
            text_y = y - 5
            if text_y < 20: 
                text_y = y + h + 20
                
            cv2.rectangle(img, (x, text_y - text_height - 5), (x + text_width, text_y + 5), color, -1)
            
            # Draw text
            cv2.putText(img, label, (x, text_y), font, font_scale, (255, 255, 255), thickness)

        # Use the parent folder name and type (start/end) for the filename
        # e.g., ranked-1-start.png
        parent_name = image_path.parent.name
        
        if "match_start" in image_path.name:
            suffix = "start"
        elif "match_end" in image_path.name:
            suffix = "end"
        else:
            suffix = "other"
            
        output_filename = f"{parent_name}-{suffix}.png"
        output_path = os.path.join(output_dir, output_filename)
        
        cv2.imwrite(output_path, img)
        print(f"Saved labeled image to {output_path}")

    print(f"Finished. Check results in {output_dir}")

if __name__ == "__main__":
    csv_file = "data/ranked_bounding_boxes.csv"
    samples_directory = "data/samples"
    output_directory = "data/debug/bounding_boxes_vis"
    
    visualize_bounding_boxes(csv_file, samples_directory, output_directory)
