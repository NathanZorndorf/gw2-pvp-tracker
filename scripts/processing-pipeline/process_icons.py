import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

# Take a set of "target" icon samples
# take a set of "Reference" icon samples
# Run them through a pre-processing pipeline to normalize size, color inversion, etc.
# For each target icon, compare it with all reference icons, find best match
# At the end, compare all ground-truth matches with detected matches to compute accuracy

REFERENCE_ICON_SIZE = (32, 32)

# load a single image from data/extracted-icons
def load_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    print(f"Loaded image from {image_path}, shape: {img.shape if img is not None else 'None'}")
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return img

def show_image(img, title = "Target Icon") -> None:
    # show image using matplotlib
    if img is not None:
        # Convert BGR to RGB for matplotlib
        # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title("Target Icon")
        plt.axis('off')
        plt.show()
    else:
        print("Failed to load image")


def letterbox_image(img, target_size=REFERENCE_ICON_SIZE) -> np.ndarray:
    h, w = img.shape[:2]
    target_w, target_h = target_size

    # 1. Calculate the scaling factor
    # We want to fit the image inside the box, so we take the smaller ratio
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # 2. Resize using the scale (keeps aspect ratio)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 3. Create a black canvas of the target size
    canvas = np.zeros((target_h, target_w, 3) if len(img.shape)==3 else (target_h, target_w), dtype=np.uint8)

    # 4. Center the resized image on the canvas
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return canvas

def boost_contrast_color(img):
    # Check if image is grayscale (1 channel) or color (3 channels)
    if len(img.shape) == 2 or img.shape[2] == 1:
        # Apply CLAHE directly to grayscale image
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2,2))
        return clahe.apply(img)
    else:
        # Original color processing
        # Convert BGR to LAB
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to the L-channel (Lightness)
        # This makes lighters lighter and darkers darker locally
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2,2))
        cl = clahe.apply(l)

        # Merge channels back and convert to BGR
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def target_icon_pre_processing_pipeline(img: np.ndarray) -> None:
    print(f'Original img.shape: {img.shape}')

    # 1. Apply Greyscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # brighten 
    # img = boost_contrast_color(img)
    # img = cv2.convertScaleAbs(img, alpha=1.5, beta=30)  # increase contrast and brightness

    # 2. Resize 
    img = cv2.resize(img, REFERENCE_ICON_SIZE, interpolation=cv2.INTER_AREA)
    img = letterbox_image(img, REFERENCE_ICON_SIZE)

    # 3. Enhance Contrast (CLAHE is better than global equalization)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2,2))
    img = clahe.apply(img)

    # 4. Extract Edges (Optional but recommended for icons)
    # This makes the "shape" the only thing that matters
    edges = cv2.Canny(img, 50, 150)
    img = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    return img


# loop through each image in target icons folder
paths = os.listdir("data/target-icons")
paths = [os.path.join("data/target-icons", path) for path in paths if path.endswith(".png")]

# path = paths[-3]
# img = load_image(path)
# img = target_icon_pre_processing_pipeline(img)
# show_image(img, "Reference Icon")
# exit()


for path in paths:
    img = load_image(path)
    img = target_icon_pre_processing_pipeline(img)
    # show_image(img, "Target Icon")
    output_path = os.path.join("scripts/processing-pipeline/processed-target-icons", os.path.basename(path))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"Loaded image: {path}, shape: {img.shape if img is not None else 'None'}")

    
# # do the same for the reference icons folder
paths = os.listdir("data/reference-icons/icons-white")
paths = [os.path.join("data/reference-icons/icons-white", path) for path in paths if path.endswith(".png")]

# path = paths[0]
# img = load_image(path)
# edges = cv2.Canny(img, 50, 150)
# img = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
# show_image(img, "Reference Icon")
# exit()

for path in paths:
    img = load_image(path)
    img = target_icon_pre_processing_pipeline(img)
    # show_image(img, "Reference Icon")
    output_path = os.path.join("scripts/processing-pipeline/processed-reference-icons", os.path.basename(path))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"Loaded image: {path}, shape: {img.shape if img is not None else 'None'}")

# Template Matching for Icon Comparison
print("\nPerforming template matching...")

# Load processed images
target_images = {}
for filename in os.listdir("scripts/processing-pipeline/processed-target-icons"):
    if filename.endswith('.png'):
        path = os.path.join("scripts/processing-pipeline/processed-target-icons", filename)
        target_images[filename] = cv2.imread(path)

ref_images = {}
for filename in os.listdir("scripts/processing-pipeline/processed-reference-icons"):
    if filename.endswith('.png'):
        path = os.path.join("scripts/processing-pipeline/processed-reference-icons", filename)
        ref_images[filename] = cv2.imread(path)

results = []

print(f"Matching {len(target_images)} targets against {len(ref_images)} references...")

for target_name, target_img in target_images.items():
    best_match = None
    highest_score = -1
    
    for ref_name, ref_img in ref_images.items():
        # Ensure both images are grayscale for matching
        if len(target_img.shape) == 3:
            target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        else:
            target_gray = target_img
            
        if len(ref_img.shape) == 3:
            ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        else:
            ref_gray = ref_img
        
        # Perform template matching
        result = cv2.matchTemplate(target_gray, ref_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        if max_val > highest_score:
            highest_score = max_val
            best_match = ref_name
    
    results.append({
        'target': target_name,
        'best_match': best_match,
        'similarity': highest_score
    })
    print(f"{target_name} -> {best_match} (score: {highest_score:.4f})")

# Save results to CSV
import csv
with open('scripts/processing-pipeline/template_matching_results.csv', 'w', newline='') as csvfile:
    fieldnames = ['target', 'best_match', 'similarity']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print("\nTemplate matching complete! Results saved to template_matching_results.csv")

# Evaluate accuracy using ground truth mappings
print("\nEvaluating accuracy...")

# Load ground truth mappings
ground_truth = {}
mappings_file = "data/target-icons/mappings.csv"
if os.path.exists(mappings_file):
    import csv
    with open(mappings_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            ground_truth[row['file_name']] = row['mapping_icon_name']

    # Load matching results
    results_file = "scripts/processing-pipeline/template_matching_results.csv"
    if os.path.exists(results_file):
        predictions = {}
        with open(results_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Extract profession from best_match filename (remove .png)
                predicted_profession = row['best_match'].replace('.png', '') if row['best_match'] else None
                predictions[row['target']] = {
                    'predicted': predicted_profession,
                    'similarity': float(row['similarity'])
                }

        # Calculate accuracy
        correct = 0
        total = 0
        evaluation_results = []

        for target_file, pred_data in predictions.items():
            if target_file in ground_truth:
                total += 1
                actual = ground_truth[target_file]
                predicted = pred_data['predicted']
                similarity = pred_data['similarity']
                
                is_correct = actual == predicted
                if is_correct:
                    correct += 1
                
                evaluation_results.append({
                    'target': target_file,
                    'actual': actual,
                    'predicted': predicted,
                    'similarity': similarity,
                    'correct': is_correct
                })
                
                status = "✓" if is_correct else "✗"
                print(f"{status} {target_file}: {actual} -> {predicted} (sim: {similarity:.4f})")

        accuracy = correct / total * 100 if total > 0 else 0
        print(f"\nAccuracy: {correct}/{total} = {accuracy:.2f}%")

        # Save detailed evaluation results
        eval_file = "scripts/processing-pipeline/evaluation_results.csv"
        with open(eval_file, 'w', newline='') as csvfile:
            fieldnames = ['target', 'actual', 'predicted', 'similarity', 'correct']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(evaluation_results)
        
        print(f"Detailed results saved to {eval_file}")
    else:
        print(f"Results file not found: {results_file}")
else:
    print(f"Mappings file not found: {mappings_file}")