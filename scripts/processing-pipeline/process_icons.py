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
    import torch
    import torchvision.transforms as transforms
    from torchvision.models import resnet18
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Take a set of "target" icon samples
# take a set of "Reference" icon samples
# Run them through a pre-processing pipeline to normalize size, color inversion, etc.
# For each target icon, compare it with all reference icons, find best match
# At the end, compare all ground-truth matches with detected matches to compute accuracy

REFERENCE_ICON_SIZE = (32, 32)

parser = argparse.ArgumentParser(description='Icon matching algorithms comparison')
parser.add_argument('--algorithms', nargs='+', choices=['template', 'sift', 'orb', 'cnn', 'all'], default=['template'],
                    help='Algorithms to run (default: template)')
parser.add_argument('--template-method', choices=['TM_CCOEFF_NORMED', 'TM_CCORR_NORMED', 'TM_SQDIFF_NORMED'], 
                    default='TM_CCOEFF_NORMED', help='Template matching method')
parser.add_argument('--grid-search', action='store_true', 
                    help='Perform grid search hyperparameter optimization for template matching')
parser.add_argument('--bilateral', action='store_true', help='Apply bilateral filtering to smooth noise while preserving edges')
parser.add_argument('--mask-circular', action='store_true', help='Apply circular mask to remove corner artifacts')
parser.add_argument('--morph-closing', action='store_true', help='Apply morphological closing to bridge gaps in contours')
args = parser.parse_args()

if 'all' in args.algorithms:
    args.algorithms = ['template', 'sift', 'orb', 'cnn']

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


def preprocess_for_feature_matching(img: np.ndarray) -> np.ndarray:
    """Preprocess image for SIFT/ORB (grayscale, resize)"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, REFERENCE_ICON_SIZE, interpolation=cv2.INTER_AREA)
    return img


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


def perform_grid_search():
    """Perform grid search over template matching hyperparameters"""
    print("Starting grid search for template matching hyperparameters...")
    
    # Load images
    target_images_orig = {}
    for filename in os.listdir("../../data/target-icons"):
        if filename.endswith('.png'):
            path = os.path.join("../../data/target-icons", filename)
            target_images_orig[filename] = cv2.imread(path)

    ref_images_orig = {}
    for filename in os.listdir("../../data/reference-icons/icons-white"):
        if filename.endswith('.png'):
            path = os.path.join("../../data/reference-icons/icons-white", filename)
            ref_images_orig[filename] = cv2.imread(path)

    # Load ground truth
    ground_truth = {}
    mappings_file = "../../data/target-icons/mappings.csv"
    with open(mappings_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            ground_truth[row['file_name']] = row['mapping_icon_name']

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


# Load original images
target_images_orig = {}
for filename in os.listdir("data/target-icons"):
    if filename.endswith('.png'):
        path = os.path.join("data/target-icons", filename)
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
    
    results = []
    
    for target_name, target_img in target_images_orig.items():
        best_match = None
        highest_score = -1
        
        for ref_name, ref_img in ref_images_orig.items():
            if algorithm == 'template':
                # Preprocess for template
                target_proc = target_icon_pre_processing_pipeline(
                    target_img, 
                    use_bilateral=args.bilateral, 
                    use_mask_circular=args.mask_circular, 
                    use_morph_closing=args.morph_closing
                )
                ref_proc = target_icon_pre_processing_pipeline(
                    ref_img, 
                    use_bilateral=args.bilateral, 
                    use_mask_circular=args.mask_circular, 
                    use_morph_closing=args.morph_closing
                )
                method = getattr(cv2, args.template_method)
                score = match_template(target_proc, ref_proc, method)
            elif algorithm == 'sift':
                target_proc = preprocess_for_feature_matching(target_img)
                ref_proc = preprocess_for_feature_matching(ref_img)
                score = match_sift(target_proc, ref_proc)
            elif algorithm == 'orb':
                target_proc = preprocess_for_feature_matching(target_img)
                ref_proc = preprocess_for_feature_matching(ref_img)
                score = match_orb(target_proc, ref_proc)
            elif algorithm == 'cnn':
                # Use precomputed features
                target_feat = target_features[target_name]
                ref_feat = ref_features[ref_name]
                score = np.dot(target_feat, ref_feat) / (np.linalg.norm(target_feat) * np.linalg.norm(ref_feat))
            
            if score > highest_score:
                highest_score = score
                best_match = ref_name
        
        results.append({
            'target': target_name,
            'best_match': best_match,
            'similarity': highest_score
        })
        print(f"{target_name} -> {best_match} (score: {highest_score:.4f})")
    
    # Save results to CSV
    results_file = f'scripts/processing-pipeline/{algorithm}_matching_results.csv'
    with open(results_file, 'w', newline='') as csvfile:
        fieldnames = ['target', 'best_match', 'similarity']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n{algorithm.capitalize()} matching complete! Results saved to {results_file}")
    all_results[algorithm] = results

# Evaluate accuracy using ground truth mappings for each algorithm
print("\nEvaluating accuracy for all algorithms...")

# Load ground truth mappings
ground_truth = {}
mappings_file = "data/target-icons/mappings.csv"
if os.path.exists(mappings_file):
    with open(mappings_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            ground_truth[row['file_name']] = row['mapping_icon_name']

    for algorithm in args.algorithms:
        print(f"\n--- Evaluating {algorithm} ---")
        
        results_file = f"scripts/processing-pipeline/{algorithm}_matching_results.csv"
        if os.path.exists(results_file):
            predictions = {}
            with open(results_file, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
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
            print(f"\n{algorithm.capitalize()} Accuracy: {correct}/{total} = {accuracy:.2f}%")

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
else:
    print(f"Mappings file not found: {mappings_file}")