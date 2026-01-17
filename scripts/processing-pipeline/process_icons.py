import cv2
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
import os
import numpy as np
import argparse
import csv
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
try:
    import torch
    import torchvision.transforms as transforms
    from torchvision.models import resnet18
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Take a set of "target" icon samples
# take a set of "Reference" icon samples
# Run them through a pre-processing pipeline to normalize size, color inversion, etc.
# For each target icon, compare it with all reference icons, find best match
# At the end, compare all ground-truth matches with detected matches to compute accuracy

REFERENCE_ICON_SIZE = (32, 32)

parser = argparse.ArgumentParser(description='Icon matching algorithms comparison')
parser.add_argument('--algorithms', nargs='+', choices=['template', 'sift', 'orb', 'cnn', 'chamfer', 'all'], default=['template'],
                    help='Algorithms to run (default: template)')
parser.add_argument('--template-method', choices=['TM_CCOEFF_NORMED', 'TM_CCORR_NORMED', 'TM_SQDIFF_NORMED'], 
                    default='TM_CCOEFF_NORMED', help='Template matching method')
parser.add_argument('--grid-search', action='store_true', 
                    help='Perform grid search hyperparameter optimization for template matching')
parser.add_argument('--bilateral', action='store_true', help='Apply bilateral filtering to smooth noise while preserving edges')
parser.add_argument('--mask-circular', action='store_true', help='Apply circular mask to remove corner artifacts')
parser.add_argument('--morph-closing', action='store_true', help='Apply morphological closing to bridge gaps in contours')
parser.add_argument('--debug', action='store_true', help='Save visualization images of matches to scripts/processing-pipeline/debug/icon_matching_vis/')
parser.add_argument('--top-n', type=int, default=1, help='Number of top matches to show per target (default: 1)')
args = parser.parse_args()

def extract_icons_from_rectangles(samples_dir, output_dir):
    """
    Extract icons from rectangles that contain multiple icons, splitting each rectangle into 5 evenly spaced squares.
    Uses appropriate CSV based on ranked/unranked samples.
    """
    if not PIL_AVAILABLE or not PANDAS_AVAILABLE:
        print("PIL or pandas not available, cannot extract icons")
        return
    
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

            # Determine which CSV to use based on sample folder
            if 'unranked' in sample_folder:
                csv_path = os.path.join('data', 'unranked_bounding_boxes.csv')
            elif 'ranked' in sample_folder:
                csv_path = os.path.join('data', 'ranked_bounding_boxes.csv')
            else:
                print(f"Unknown sample type for {sample_folder}, skipping")
                continue

            # Read the CSV file and handle potential duplicate labels (take last)
            df = pd.read_csv(csv_path)
            if 'label_name' in df.columns:
                 df = df.drop_duplicates(subset=['label_name'], keep='last')

            # Process each rectangle
            for _, row in df.iterrows():
                label_name = row['label_name']
                if 'Icons' not in label_name:
                    continue  # Only process icon rectangles
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

if 'all' in args.algorithms:
    args.algorithms = ['template', 'sift', 'orb', 'cnn', 'chamfer']

# Grid search parameters for template matching optimization
TEMPLATE_METHODS = ['TM_CCOEFF_NORMED', 'TM_CCORR_NORMED', 'TM_SQDIFF_NORMED']
CLAHE_CLIPS = [1.0, 2.0, 3.0]
CLAHE_TILES = [(2,2), (4,4), (8,8)]
CANNY_MINS = [30, 50, 100]
CANNY_MAXS = [150, 200, 255]
INTERPOLATIONS = [cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC]

def run_template_matching_with_params(target_images, ref_images, template_method, clahe_clip, clahe_tile, canny_min, canny_max, interpolation):
    """Run template matching with specific parameters and return results"""
    results = []
    
    for target_name, target_img in target_images.items():
        best_match = None
        highest_score = -1
        
        for ref_name, ref_img in ref_images.items():
            # Preprocess with parameters
            target_proc = target_icon_pre_processing_pipeline(target_img, clahe_clip, clahe_tile, canny_min, canny_max, interpolation)
            ref_proc = target_icon_pre_processing_pipeline(ref_img, clahe_clip, clahe_tile, canny_min, canny_max, interpolation)
            method = getattr(cv2, template_method)
            score = match_template(target_proc, ref_proc, method)
            
            if score > highest_score:
                highest_score = score
                best_match = ref_name
        
        results.append({
            'target': target_name,
            'best_match': best_match,
            'similarity': highest_score
        })
    
    return results

def evaluate_accuracy(results, ground_truth):
    """Calculate accuracy for given results"""
    predictions = {}
    for row in results:
        predicted_profession = row['best_match'].replace('.png', '') if row['best_match'] else None
        predictions[row['target']] = predicted_profession

    correct = 0
    total = 0
    for target_file, actual in ground_truth.items():
        if target_file in predictions:
            total += 1
            if predictions[target_file] == actual:
                correct += 1
    
    accuracy = correct / total * 100 if total > 0 else 0
    return accuracy, correct, total

# load a single image from data/extracted-icons
def load_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    print(f"Loaded image from {image_path}, shape: {img.shape if img is not None else 'None'}")
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return img

def show_image(img, title = "Target Icon") -> None:
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available, cannot show image")
        return
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

def target_icon_pre_processing_pipeline(img: np.ndarray, clahe_clip=1.0, clahe_tile=(2,2), canny_min=50, canny_max=150, interpolation=cv2.INTER_AREA,
                                      use_bilateral=False, use_mask_circular=False, use_morph_closing=False) -> np.ndarray:
    # print(f'Original img.shape: {img.shape}')

    # 1. Apply Greyscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Resize 
    img = cv2.resize(img, REFERENCE_ICON_SIZE, interpolation=interpolation)
    img = letterbox_image(img, REFERENCE_ICON_SIZE)

    # NEW: Bilateral Filter (Smooths noise while preserving edges)
    if use_bilateral:
        # d=5, sigmaColor=75, sigmaSpace=75 are common defaults
        img = cv2.bilateralFilter(img, 5, 75, 75)

    # 3. Enhance Contrast (CLAHE is better than global equalization)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
    img = clahe.apply(img)

    # 4. Extract Edges (Optional but recommended for icons)
    # This makes the "shape" the only thing that matters
    edges = cv2.Canny(img, canny_min, canny_max)

    # NEW: Morphological Closing (Bridge gaps)
    if use_morph_closing:
        kernel = np.ones((2,2), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # NEW: Circular Mask (Remove corner artifacts)
    if use_mask_circular:
        h, w = edges.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        # Assuming icon is centered, radius is half of width (minus a bit of margin if needed)
        cv2.circle(mask, (w//2, h//2), w//2, 255, -1)
        edges = cv2.bitwise_and(edges, edges, mask=mask)

    img = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    return img


import cv2
import numpy as np

def preprocess_reference_icon_for_feature_matching(img: np.ndarray, clahe_clip=1.0, clahe_tile=(2,2)) -> np.ndarray:
    # 1. Grayscale and Resize
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, REFERENCE_ICON_SIZE, interpolation=cv2.INTER_AREA)

    return img 

def preprocess_target_icon_for_feature_matching(img: np.ndarray, clahe_clip=1.0, clahe_tile=(2,2)) -> np.ndarray:
    # 1. Grayscale and Resize
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, REFERENCE_ICON_SIZE, interpolation=cv2.INTER_AREA)

    # 2. Initial Contrast Enhancement
    # clahe_clip = 1.0
    # clahe_tile = (4,4)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
    enhanced = clahe.apply(img)


    # 3. Create a Binary Mask (Otsu's method works well for dark backgrounds)
    # We use a slight Gaussian blur to reduce noise before thresholding
    blurred = cv2.GaussianBlur(enhanced, (1, 1), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # return mask # 71 % accuracy 

    # 6. Apply mask to the enhanced image
    # This makes everything outside the contour 0 (black)
    result = cv2.bitwise_and(enhanced, enhanced, mask=mask)

    # return result # 76 % accuracy 

    # Normalize the masked icon to make it "whiter" (maps brightest pixel to 255)
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    return result 


    # 5. Extract the Icon (Largest Contour)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Assume the icon is the largest object found
        largest_contour = max(contours, key=cv2.contourArea)
    
        # Draw the contour: 
        # -1 means 'draw the specific contour provided'
        # 255 is the color (white)
        # 2 is the thickness of the line
        # contour_canvas = np.zeros_like(enhanced) # Create a black canvas of the same size as your image
        # cv2.drawContours(contour_canvas, [largest_contour], -1, 255, thickness=cv2.FILLED)
        # return contour_canvas

        # Create a final clean black mask
        final_mask = np.zeros_like(img)
        cv2.drawContours(final_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        
        # 6. Apply mask to the enhanced image
        # This makes everything outside the contour 0 (black)
        result = cv2.bitwise_and(enhanced, enhanced, mask=final_mask)

        # return result

        # 7. Normalize Brightness (Stretch the icon's intensity to the full 0-255 range)
        # We only look at pixels where the mask is active
        icon_pixels = enhanced[final_mask > 0]
        if icon_pixels.size > 0:
            min_val = np.min(icon_pixels)
            max_val = np.max(icon_pixels)
            
            # Linear stretch: (Pixel - min) / (max - min) * 255
            # This ensures the darkest part of the icon is black and the brightest is white
            if max_val > min_val:
                stretched = (enhanced.astype(float) - min_val) * (255.0 / (max_val - min_val))
                stretched = np.clip(stretched, 0, 255).astype(np.uint8)
                
                # Re-mask to keep the background black
                result = cv2.bitwise_and(stretched, stretched, mask=final_mask)
                return result
            
        return result

    return enhanced # Fallback to original if no contours found


def preprocess_for_cnn(img: np.ndarray) -> torch.Tensor:
    """Preprocess image for CNN (RGB, 32x32, normalized)"""
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Resize to 32x32 to match reference icon size
    img = cv2.resize(img, REFERENCE_ICON_SIZE, interpolation=cv2.INTER_LINEAR)
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img)
    return img_tensor.unsqueeze(0)  # Add batch dimension


def match_template(target_img, ref_img, method=cv2.TM_CCOEFF_NORMED):
    """Template matching similarity"""
    if len(target_img.shape) == 3:
        target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    else:
        target_gray = target_img
        
    if len(ref_img.shape) == 3:
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    else:
        ref_gray = ref_img
    
    result = cv2.matchTemplate(target_gray, ref_gray, method)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return max_val


def match_sift(target_img, ref_img):
    """SIFT feature matching similarity"""
    sift = cv2.SIFT_create()
    
    kp1, des1 = sift.detectAndCompute(target_img, None)
    kp2, des2 = sift.detectAndCompute(ref_img, None)
    
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return 0.0
    
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # Sort by distance and take top matches
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:min(10, len(matches))]  # Top 10 matches
    
    if len(good_matches) == 0:
        return 0.0
    
    # Average distance (lower is better, so invert)
    avg_distance = np.mean([m.distance for m in good_matches])
    similarity = 1.0 / (1.0 + avg_distance)  # Normalize to 0-1
    return similarity


def match_orb(target_img, ref_img):
    """ORB feature matching similarity"""
    orb = cv2.ORB_create()
    
    kp1, des1 = orb.detectAndCompute(target_img, None)
    kp2, des2 = orb.detectAndCompute(ref_img, None)
    
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return 0.0
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:min(10, len(matches))]
    
    if len(good_matches) == 0:
        return 0.0
    
    avg_distance = np.mean([m.distance for m in good_matches])
    similarity = 1.0 / (1.0 + avg_distance)
    return similarity


def match_cnn(target_tensor, ref_tensor, model):
    """CNN feature similarity using cosine similarity"""
    with torch.no_grad():
        target_feat = model(target_tensor)
        ref_feat = model(ref_tensor)
    
    # Cosine similarity
    target_feat = target_feat.squeeze()
    ref_feat = ref_feat.squeeze()
    
    cos_sim = torch.nn.functional.cosine_similarity(target_feat, ref_feat, dim=0)
    return cos_sim.item()

def match_chamfer(target_proc, ref_dist_map):
    """Chamfer distance matching similarity"""
    # 1. Convert target to binary edges
    if len(target_proc.shape) == 3:
        target_gray = cv2.cvtColor(target_proc, cv2.COLOR_RGB2GRAY)
    else:
        target_gray = target_proc
    
    _, binary = cv2.threshold(target_gray, 1, 255, cv2.THRESH_BINARY)
    
    # 2. Sum distances for all edge pixels in target
    edge_pixels = binary > 0
    num_edge_pixels = np.sum(edge_pixels)
    
    if num_edge_pixels == 0:
        return 0.0
        
    total_distance = np.sum(ref_dist_map[edge_pixels])
    avg_distance = total_distance / num_edge_pixels
    
    # Normalize to 0.0-1.0 (matching ProfessionDetector logic)
    similarity = 1.0 / (1.0 + (avg_distance * 0.5))
    return float(similarity)

def visualize_match(target_name, target_img, best_match_name, ref_img, algorithm, output_dir, target_proc_in=None, ref_proc_in=None):
    """Visualize and save the match comparison"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Get processed versions
    if target_proc_in is not None:
        target_proc = target_proc_in
    else:
        target_proc = preprocess_target_icon_for_feature_matching(target_img)

    if ref_proc_in is not None:
        ref_proc = ref_proc_in
    else:
        ref_proc = preprocess_reference_icon_for_feature_matching(ref_img)
    
    # helper to ensure BGR and size
    def ensure_bgr_32(img):
        if img is None: return np.zeros((32,32,3), np.uint8)
        
        # If float (from some preproc?), convert
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
            
        resize = cv2.resize(img, REFERENCE_ICON_SIZE, interpolation=cv2.INTER_NEAREST)
        if len(resize.shape) == 2:
            resize = cv2.cvtColor(resize, cv2.COLOR_GRAY2BGR)
        return resize

    t_orig = ensure_bgr_32(target_img)
    r_orig = ensure_bgr_32(ref_img)
    t_proc = ensure_bgr_32(target_proc)
    r_proc = ensure_bgr_32(ref_proc)
    
    # Visualization Layout
    scale = 4
    # Scale individual images first
    def scale_up(img, s):
        return cv2.resize(img, (img.shape[1]*s, img.shape[0]*s), interpolation=cv2.INTER_NEAREST)
        
    t_orig_s = scale_up(t_orig, scale)
    r_orig_s = scale_up(r_orig, scale)
    t_proc_s = scale_up(t_proc, scale)
    r_proc_s = scale_up(r_proc, scale)
    
    sub_h, sub_w = t_orig_s.shape[:2]
    
    # Margins and spacing
    margin_left = 100
    margin_top = 40
    gap = 10
    
    total_w = margin_left + sub_w + gap + sub_w + gap
    total_h = margin_top + sub_h + gap + sub_h + gap + 40 # extra bottom margin for text
    
    canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    
    # Place images
    # Top-Left (Target Original)
    y1 = margin_top
    x1 = margin_left
    canvas[y1:y1+sub_h, x1:x1+sub_w] = t_orig_s
    
    # Top-Right (Ref Original)
    y1 = margin_top
    x2 = margin_left + sub_w + gap
    canvas[y1:y1+sub_h, x2:x2+sub_w] = r_orig_s
    
    # Bottom-Left (Target Processed)
    y2 = margin_top + sub_h + gap
    x1 = margin_left
    canvas[y2:y2+sub_h, x1:x1+sub_w] = t_proc_s
    
    # Bottom-Right (Ref Processed)
    y2 = margin_top + sub_h + gap
    x2 = margin_left + sub_w + gap
    canvas[y2:y2+sub_h, x2:x2+sub_w] = r_proc_s
    
    # Draw Labels
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    font_scale = 0.7
    color = (255, 255, 255)
    thickness = 1
    
    # Column Headers
    cv2.putText(canvas, "TARGET", (margin_left + 10, margin_top - 10), font, font_scale, color, thickness)
    cv2.putText(canvas, "MATCH", (margin_left + sub_w + gap + 10, margin_top - 10), font, font_scale, color, thickness)
    
    # Row Headers
    cv2.putText(canvas, "ORIGINAL", (5, margin_top + sub_h // 2), font, font_scale, color, thickness)
    cv2.putText(canvas, "PROCESSED", (5, margin_top + sub_h + gap + sub_h // 2), font, font_scale, color, thickness)
    
    # File Infos at bottom
    cv2.putText(canvas, f"T: {target_name[:25]}", (5, total_h - 25), font, 0.6, (180, 180, 180), 1)
    cv2.putText(canvas, f"M: {best_match_name}", (5, total_h - 10), font, 0.6, (180, 180, 180), 1)

    out_path = os.path.join(output_dir, f"{algorithm}_{target_name}")
    cv2.imwrite(out_path, canvas)


def load_ground_truth_from_yamls(samples_dir):
    """Load ground truth data from YAML files in samples directory"""
    ground_truth = {}
    if not YAML_AVAILABLE:
        print("PyYAML not available, cannot load ground truth from YAMLs")
        return ground_truth

    print(f"Loading ground truth from {samples_dir}...")
    
    for sample_folder in os.listdir(samples_dir):
        folder_path = os.path.join(samples_dir, sample_folder)
        if not os.path.isdir(folder_path):
            continue
            
        yaml_path = os.path.join(folder_path, "ground_truth.yaml")
        if not os.path.exists(yaml_path):
            continue
            
        try:
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
                
            # Find the match_start sample
            match_start_sample = None
            if 'samples' in data:
                for sample in data['samples']:
                    if 'match_start' in sample['filename']:
                        match_start_sample = sample
                        break
            
            if match_start_sample:
                image_name = match_start_sample['filename']
                image_base = os.path.splitext(image_name)[0]
                
                # Helper to map teams
                def map_team(team_list, team_name):
                    if not team_list: return
                    for i, player in enumerate(team_list):
                        profession = player.get('profession')
                        if profession:
                            # Construct filename key
                            # Format: {sample_folder}_{team}_Player_{player_num}___Class_{image_base}.png
                            key = f"{sample_folder}_{team_name}_Player_{i+1}___Class_{image_base}.png"
                            ground_truth[key] = profession
                
                map_team(match_start_sample.get('red_team'), "Red")
                map_team(match_start_sample.get('blue_team'), "Blue")
                
        except Exception as e:
            print(f"Error reading {yaml_path}: {e}")
            
    print(f"Loaded {len(ground_truth)} ground truth entries")
    return ground_truth


def perform_grid_search():
    """Perform grid search over template matching hyperparameters"""
    print("Starting grid search for template matching hyperparameters...")
    
    # Load images
    target_images_orig = {}
    target_icons_dir = os.path.join('scripts', 'processing-pipeline', 'target-icons')
    if os.path.exists(target_icons_dir):
        for filename in os.listdir(target_icons_dir):
            if filename.endswith('.png'):
                path = os.path.join(target_icons_dir, filename)
                target_images_orig[filename] = cv2.imread(path)

    ref_images_orig = {}
    ref_icons_dir = os.path.join("data", "reference-icons", "icons-white") 
    if os.path.exists(ref_icons_dir):
        for filename in os.listdir(ref_icons_dir):
            if filename.endswith('.png'):
                path = os.path.join(ref_icons_dir, filename)
                ref_images_orig[filename] = cv2.imread(path)

    # Load ground truth
    ground_truth = load_ground_truth_from_yamls(os.path.join("data", "samples"))

    best_accuracy = 0
    best_params = None
    results_summary = []

    total_combinations = len(TEMPLATE_METHODS) * len(CLAHE_CLIPS) * len(CLAHE_TILES) * len(CANNY_MINS) * len(CANNY_MAXS) * len(INTERPOLATIONS)
    print(f"Testing {total_combinations} parameter combinations...")
    
    combination_count = 0
    for template_method in TEMPLATE_METHODS:
        for clahe_clip in CLAHE_CLIPS:
            for clahe_tile in CLAHE_TILES:
                for canny_min in CANNY_MINS:
                    for canny_max in CANNY_MAXS:
                        for interpolation in INTERPOLATIONS:
                            combination_count += 1
                            
                            # Run matching with these parameters
                            results = run_template_matching_with_params(
                                target_images_orig, ref_images_orig, 
                                template_method, clahe_clip, clahe_tile, 
                                canny_min, canny_max, interpolation
                            )
                            
                            # Evaluate accuracy
                            accuracy, correct, total = evaluate_accuracy(results, ground_truth)
                            
                            params = {
                                'template_method': template_method,
                                'clahe_clip': clahe_clip,
                                'clahe_tile': clahe_tile,
                                'canny_min': canny_min,
                                'canny_max': canny_max,
                                'interpolation': interpolation,
                                'accuracy': accuracy,
                                'correct': correct,
                                'total': total
                            }
                            
                            results_summary.append(params)
                            
                            if accuracy > best_accuracy:
                                best_accuracy = accuracy
                                best_params = params
                            
                            print(f"[{combination_count}/{total_combinations}] {template_method} | CLAHE({clahe_clip},{clahe_tile}) | Canny({canny_min},{canny_max}) | Acc: {accuracy:.2f}%")
    
    # Save results
    with open('grid_search_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['template_method', 'clahe_clip', 'clahe_tile', 'canny_min', 'canny_max', 'interpolation', 'accuracy', 'correct', 'total']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_summary)
    
    print("\nGrid search complete!")
    print(f"Best accuracy: {best_accuracy:.2f}%")
    print(f"Best parameters: {best_params}")
    
    return best_params

# Run grid search if requested
if args.grid_search:
    best_params = perform_grid_search()
    exit(0)

# Extract target icons if not present
target_icons_dir = os.path.join('scripts', 'processing-pipeline', 'target-icons')
print("Extracting target icons...")
samples_dir = os.path.join('data', 'samples')
extract_icons_from_rectangles(samples_dir, target_icons_dir)

# Load original images
target_images_orig = {}
for filename in os.listdir(target_icons_dir):
    if filename.endswith('.png'):
        path = os.path.join(target_icons_dir, filename)
        target_images_orig[filename] = cv2.imread(path)

ref_images_orig = {}
for filename in os.listdir("data/reference-icons/icons-white"):
    if filename.endswith('.png'):
        path = os.path.join("data/reference-icons/icons-white", filename)
        ref_images_orig[filename] = cv2.imread(path)

# Prepare CNN model if needed
cnn_model = None
device = None
if 'cnn' in args.algorithms and TORCH_AVAILABLE:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnn_model = resnet18(pretrained=True)
    cnn_model = torch.nn.Sequential(*list(cnn_model.children())[:-1])  # Remove final layers
    cnn_model.to(device)
    cnn_model.eval()
elif 'cnn' in args.algorithms and not TORCH_AVAILABLE:
    print("PyTorch not available, skipping CNN")
    if 'cnn' in args.algorithms:
        args.algorithms.remove('cnn')

# Run matching for each algorithm
all_results = {}

for algorithm in args.algorithms:
    print(f"\nPerforming {algorithm} matching...")
    
    # Precompute CNN features if using CNN
    if algorithm == 'cnn':
        print("Precomputing CNN features...")
        target_features = {}
        ref_features = {}
        
        for name, img in target_images_orig.items():
            tensor = preprocess_for_cnn(img).to(device)
            with torch.no_grad():
                feat = cnn_model(tensor).squeeze().cpu().numpy()
            target_features[name] = feat
            
        for name, img in ref_images_orig.items():
            tensor = preprocess_for_cnn(img).to(device)
            with torch.no_grad():
                feat = cnn_model(tensor).squeeze().cpu().numpy()
            ref_features[name] = feat
        
        print("CNN features computed!")

    # Precompute Distance Maps if using Chamfer
    ref_dist_maps = {}
    if algorithm == 'chamfer':
        print("Precomputing distance maps for Chamfer matching...")
        for name, img in ref_images_orig.items():
            # Match the Canny pipeline used in ProfessionDetector
            proc = target_icon_pre_processing_pipeline(img, use_mask_circular=True)
            gray = cv2.cvtColor(proc, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            dist_map = cv2.distanceTransform(cv2.bitwise_not(binary), cv2.DIST_L2, 3)
            ref_dist_maps[name] = dist_map

    results = []
    
    for target_name, target_img in target_images_orig.items():
        matches = []
        
        # Determine target preprocessing once per algorithm
        if algorithm == 'template':
            # Use translation-robust style if template is selected
            # Extract a 36x36 window for target to allow 32x32 template to slide
            target_proc = target_icon_pre_processing_pipeline(
                target_img, 
                clahe_clip=1.0, clahe_tile=(2,2),
                interpolation=cv2.INTER_AREA,
                use_bilateral=args.bilateral, 
                use_mask_circular=args.mask_circular, 
                use_morph_closing=args.morph_closing
            )
            # Re-resize target to 36x36 for search area
            target_proc_search = cv2.resize(target_proc, (36, 36), interpolation=cv2.INTER_AREA)
        elif algorithm == 'chamfer':
            target_proc = target_icon_pre_processing_pipeline(
                target_img, 
                use_mask_circular=True
            )
        elif algorithm in ['sift', 'orb']:
            target_proc = preprocess_target_icon_for_feature_matching(target_img)
        
        for ref_name, ref_img in ref_images_orig.items():
            if algorithm == 'template':
                ref_proc = target_icon_pre_processing_pipeline(
                    ref_img, 
                    use_bilateral=args.bilateral, 
                    use_mask_circular=args.mask_circular, 
                    use_morph_closing=args.morph_closing
                )
                method = getattr(cv2, args.template_method)
                # Use the 36x36 search area for the target
                score = match_template(target_proc_search, ref_proc, method)
            elif algorithm == 'chamfer':
                score = match_chamfer(target_proc, ref_dist_maps[ref_name])
            elif algorithm == 'sift':
                ref_proc = preprocess_reference_icon_for_feature_matching(ref_img)
                score = match_sift(target_proc, ref_proc)
            elif algorithm == 'orb':
                ref_proc = preprocess_reference_icon_for_feature_matching(ref_img)
                score = match_orb(target_proc, ref_proc)
            elif algorithm == 'cnn':
                # Use precomputed features
                target_feat = target_features[target_name]
                ref_feat = ref_features[ref_name]
                score = np.dot(target_feat, ref_feat) / (np.linalg.norm(target_feat) * np.linalg.norm(ref_feat))
            
            matches.append((ref_name, score))
        
        matches.sort(key=lambda x: x[1], reverse=True)
        top_matches = matches[:args.top_n]
        
        print(f"{target_name} -> {', '.join([f'{name} ({score:.4f})' for name, score in top_matches])}")
        
        best_match = top_matches[0][0] if top_matches else None
        
        # Prepare processed ref for visualization if template
        ref_proc_vis = None
        if algorithm == 'template' and best_match:
            ref_proc_vis = target_icon_pre_processing_pipeline(
                ref_images_orig[best_match], 
                use_bilateral=args.bilateral, 
                use_mask_circular=args.mask_circular, 
                use_morph_closing=args.morph_closing
            )
        elif algorithm == 'chamfer' and best_match:
            ref_proc_vis = target_icon_pre_processing_pipeline(
                ref_images_orig[best_match], 
                use_mask_circular=True
            )

        for rank, (match_name, score) in enumerate(top_matches, 1):
            results.append({
                'target': target_name,
                'match': match_name,
                'similarity': score,
                'rank': rank
            })

        if args.debug and best_match:
            visualize_match(
                target_name, 
                target_img, 
                best_match, 
                ref_images_orig[best_match], 
                algorithm, 
                "scripts/processing-pipeline/debug/icon_matching_vis",
                target_proc_in=target_proc if algorithm in ['template', 'chamfer'] else None,
                ref_proc_in=ref_proc_vis
            )
    
    
    # Save results to CSV
    results_file = f'scripts/processing-pipeline/{algorithm}_matching_results.csv'
    with open(results_file, 'w', newline='') as csvfile:
        fieldnames = ['target', 'match', 'similarity', 'rank']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n{algorithm.capitalize()} matching complete! Results saved to {results_file}")
    all_results[algorithm] = results

# Evaluate accuracy using ground truth mappings for each algorithm
print("\nEvaluating accuracy for all algorithms...")

# Load ground truth mappings
ground_truth = load_ground_truth_from_yamls(os.path.join('data', 'samples'))

if ground_truth:
    summary_results = []
    
    for algorithm in args.algorithms:
        print(f"\n--- Evaluating {algorithm} ---")
        
        results_file = f"scripts/processing-pipeline/{algorithm}_matching_results.csv"
        if os.path.exists(results_file):
            predictions = {}
            with open(results_file, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    predicted_profession = row['match'].replace('.png', '') if row['match'] else None
                    target = row['target']
                    rank = int(row['rank'])
                    sim = float(row['similarity'])
                    if target not in predictions:
                        predictions[target] = []
                    predictions[target].append({
                        'predicted': predicted_profession,
                        'similarity': sim,
                        'rank': rank
                    })

            # Calculate accuracy
            correct = 0
            total = 0
            evaluation_results = []

            for target_file, preds in predictions.items():
                if target_file in ground_truth:
                    total += 1
                    actual = ground_truth[target_file]
                    # Sort preds by rank
                    preds.sort(key=lambda x: x['rank'])
                    top_preds = preds[:args.top_n]
                    
                    # Check if best is correct
                    best_pred = top_preds[0]['predicted'] if top_preds else None
                    is_correct = actual == best_pred
                    if is_correct:
                        correct += 1
                    
                    # For evaluation_results, use the best
                    best_sim = top_preds[0]['similarity'] if top_preds else 0.0
                    evaluation_results.append({
                        'target': target_file,
                        'actual': actual,
                        'predicted': best_pred,
                        'similarity': best_sim,
                        'correct': is_correct
                    })
                    
                    status = "✓" if is_correct else "✗"
                    pred_str = ', '.join([f"{p['predicted']} ({p['similarity']:.4f})" for p in top_preds])
                    print(f"{status} {target_file}: {actual} -> {pred_str}")

            accuracy = correct / total * 100 if total > 0 else 0
            print(f"\n{algorithm.capitalize()} Accuracy: {correct}/{total} = {accuracy:.2f}%")
            
            summary_results.append({
                'algorithm': algorithm,
                'accuracy': accuracy,
                'correct': correct,
                'total': total
            })

            # Save detailed evaluation results
            eval_file = f"scripts/processing-pipeline/{algorithm}_evaluation_results.csv"
            with open(eval_file, 'w', newline='') as csvfile:
                fieldnames = ['target', 'actual', 'predicted', 'similarity', 'correct']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(evaluation_results)
            
            print(f"Detailed results saved to {eval_file}")
        else:
            print(f"Results file not found: {results_file}")

    # Print summary table and save to CSV
    if summary_results:
        print("\n" + "="*60)
        print("ALGORITHM ACCURACY SUMMARY")
        print("="*60)
        print(f"{'Algorithm':<20} | {'Accuracy':<10} | {'Correct/Total':<15}")
        print("-" * 60)
        
        for res in summary_results:
            print(f"{res['algorithm'].capitalize():<20} | {res['accuracy']:.2f}%     | {res['correct']}/{res['total']}")
        
        print("="*60 + "\n")

        # Save summary CSV
        summary_file = "scripts/processing-pipeline/all_algorithms_summary.csv"
        with open(summary_file, 'w', newline='') as csvfile:
            fieldnames = ['algorithm', 'accuracy', 'correct', 'total']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_results)
        print(f"Summary saved to {summary_file}")
else:
    print(f"No ground truth data found in data/samples")