
import time
import cv2
import numpy as np
import pytesseract
from pathlib import Path

def test_tesseract():
    # Create mock text image
    img = np.zeros((60, 400, 3), dtype=np.uint8)
    cv2.putText(img, "Player Name", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Preprocess (Tesseract likes black text on white bg)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    print("Tesseract (Individual calls):")
    start = time.time()
    for _ in range(10):
        pytesseract.image_to_string(thresh, config='--psm 7')
    print(f"10 individual calls: {time.time() - start:.3f}s")

if __name__ == "__main__":
    test_tesseract()
