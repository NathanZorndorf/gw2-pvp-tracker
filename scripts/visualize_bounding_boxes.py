import pandas as pd
import cv2
import os
from pathlib import Path
import numpy as np

def visualize_bounding_boxes(samples_dir, output_dir):
    """
    Reads bounding boxes from appropriate CSVs and draws them on match_start images.
    Uses ranked CSV for ranked samples, unranked CSV for unranked samples.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load both CSVs
    ranked_csv = "data/ranked_bounding_boxes.csv"
    unranked_csv = "data/unranked_bounding_boxes.csv"
    
    ranked_df = pd.read_csv(ranked_csv)
    unranked_df = pd.read_csv(unranked_csv)
    
    # Extract boxes
    def get_boxes(df, source_name):
        boxes_dict = {}
        duplicates = []
        for _, row in df.iterrows():
            label = row['label_name']
            if label in boxes_dict:
                duplicates.append(label)
            
            boxes_dict[label] = {
                'label': label,
                'x': int(row['bbox_x']),
                'y': int(row['bbox_y']),
                'w': int(row['bbox_width']),
                'h': int(row['bbox_height'])
            }
        
        if duplicates:
            print(f"Warning: Duplicate labels found in {source_name}: {set(duplicates)}. Using last occurrence.")
            
        return list(boxes_dict.values())
    
    ranked_boxes = get_boxes(ranked_df, ranked_csv)
    unranked_boxes = get_boxes(unranked_df, unranked_csv)
    
    print(f"Loaded {len(ranked_boxes)} ranked and {len(unranked_boxes)} unranked bounding boxes.")


    # Find all match_start images
    samples_path = Path(samples_dir)
    print(f"Scanning {samples_dir} for match_start images...")
    images_to_process = list(samples_path.rglob("match_start*.png"))
    
    print(f"Found {len(images_to_process)} images to process.")

    for image_path in images_to_process:
        print(f"Processing {image_path.name}...")
        
        # Determine which boxes to use
        parent_name = image_path.parent.name
        if 'unranked' in parent_name:
            boxes = unranked_boxes
            csv_type = "unranked"
        elif 'ranked' in parent_name:
            boxes = ranked_boxes
            csv_type = "ranked"
        else:
            print(f"Unknown sample type for {parent_name}, skipping")
            continue
        
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Error: Failed to load image {image_path}")
            continue

        # Draw boxes
        for box in boxes:
            x, y, w, h = box['x'], box['y'], box['w'], box['h']
            label = box['label']
            
            # Draw rectangle 
            color = (0, 0, 255) 
            thickness = 2
            cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
            
            # Draw label
            font_scale = 0.6
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            text_y = y - 5
            if text_y < 20: 
                text_y = y + h + 20
                
            cv2.rectangle(img, (x, text_y - text_height - 5), (x + text_width, text_y + 5), color, -1)
            
            cv2.putText(img, label, (x, text_y), font, font_scale, (255, 255, 255), thickness)

        # Output filename
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize bounding boxes on sample images')
    parser.add_argument('--samples', default='data/samples', help='Directory containing sample images')
    parser.add_argument('--output', default='data/debug/bounding_boxes_vis', help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    visualize_bounding_boxes(args.samples, args.output)
