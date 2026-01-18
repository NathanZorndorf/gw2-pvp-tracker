
import time
import cv2
import numpy as np
import easyocr
from pathlib import Path

def test_batching():
    reader = easyocr.Reader(['en'], gpu=False) # Force CPU
    
    # Create 10 mock text images (or use actual ones if we had them easily)
    # Better: just use one image and replicate it 10 times to simulate batching
    img = np.zeros((60, 400, 3), dtype=np.uint8)
    cv2.putText(img, "Player Name", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    print("Individual calls:")
    start = time.time()
    for _ in range(10):
        reader.readtext(img)
    print(f"10 individual calls: {time.time() - start:.3f}s")
    
    print("\nBatched (concatenated):")
    batched_img = np.vstack([img for _ in range(10)])
    start = time.time()
    results = reader.readtext(batched_img)
    print(f"1 batched call: {time.time() - start:.3f}s")
    print(f"Found {len(results)} items")

if __name__ == "__main__":
    test_batching()
