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

def target_icon_pre_processing_pipeline(img: np.ndarray) -> None:
    print(f'Original img.shape: {img.shape}')

    # 1. Apply Greyscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Resize 
    # img = cv2.resize(img, REFERENCE_ICON_SIZE, interpolation=cv2.INTER_AREA)
    # img = letterbox_image(img, REFERENCE_ICON_SIZE)

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

# path = paths[0]
# img = load_image(path)
# img = target_icon_pre_processing_pipeline(img)
# show_image(img, "Reference Icon")
    
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
    

for path in paths:
    img = load_image(path)
    img = target_icon_pre_processing_pipeline(img)
    # show_image(img, "Reference Icon")
    output_path = os.path.join("scripts/processing-pipeline/processed-reference-icons", os.path.basename(path))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"Loaded image: {path}, shape: {img.shape if img is not None else 'None'}")