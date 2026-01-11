import cv2
import os
import sys
import csv
import numpy as np

try:
    import pytesseract
except Exception:
    pytesseract = None
try:
    import easyocr
except Exception:
    easyocr = None
try:
    from thefuzz import process as fuzzprocess
except Exception:
    fuzzprocess = None

def load_names(names_path):
    if not names_path or not os.path.exists(names_path):
        return None
    with open(names_path, encoding='utf-8') as f:
        names = [line.strip() for line in f if line.strip()]
    return names


def sort_boxes(boxes, row_tol=0.6):
    # boxes: list of (x,y,w,h)
    if not boxes:
        return []
    # sort by y then x with row grouping
    boxes_sorted = sorted(boxes, key=lambda b: (b[1], b[0]))
    rows = []
    for b in boxes_sorted:
        x, y, w, h = b
        placed = False
        for row in rows:
            # compare y to first box in row
            ry = row[0][1]
            if abs(y - ry) <= int(h * row_tol) or abs(y - ry) <= 10:
                row.append(b)
                placed = True
                break
        if not placed:
            rows.append([b])
    # sort boxes within each row by x
    result = []
    for row in rows:
        result.extend(sorted(row, key=lambda b: b[0]))
    return result


def main():
    # CLI: optional args: input_image, names_file, out_dir
    img_path = sys.argv[1] if len(sys.argv) > 1 else "profession icons - greyscale - cropped.png"
    names_path = sys.argv[2] if len(sys.argv) > 2 else None
    out_dir = sys.argv[3] if len(sys.argv) > 3 else "templates"

    os.makedirs(out_dir, exist_ok=True)

    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read image: {img_path}")
        sys.exit(2)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Improve contrast and reduce noise
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    # Otsu threshold to adapt to brightness
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert if background is dark/bright (we want shapes as white)
    white_area = np.sum(thresh == 255)
    black_area = np.sum(thresh == 0)
    if black_area < white_area:
        thresh = cv2.bitwise_not(thresh)

    # Morphology to close small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if w < 16 or h < 16:
            continue
        if area < 100:
            continue
        # filter extreme aspect ratios
        ar = w / float(h)
        if ar < 0.4 or ar > 2.5:
            continue
        boxes.append((x, y, w, h))

    boxes = sort_boxes(boxes)

    names = load_names(names_path)
    # Known spec names (common GW2 elite specs / professions)
    known_specs = [
        "Elementalist","Necromancer","Mesmer","Ranger","Engineer","Thief",
        "Revenant","Guardian","Warrior","Tempest","Reaper","Chronomancer",
        "Druid","Scrapper","Daredevil","Herald","Dragonhunter","Berserker",
        "Weaver","Scourge","Mirage","Soulbeast","Holosmith","Deadeye",
        "Renegade","Firebrand","Spellbreaker","Catalyst","Harbinger","Virtuoso",
        "Untamed","Mechanist","Specter","Vindicator","Willbender","Bladesworn"
    ]
    mapping = []
    pad = 6
    for i, (x, y, w, h) in enumerate(boxes):
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(img.shape[1], x + w + pad)
        y1 = min(img.shape[0], y + h + pad)
        roi = img[y0:y1, x0:x1]
        # Attempt to OCR label below the icon
        label_y0 = y + h + 2
        label_y1 = min(img.shape[0], y + h + int(h * 0.7) + 12)
        label = None
        ocr_text = None
        if label_y0 < label_y1:
            label_roi = img[label_y0:label_y1, max(0, x - int(w*0.2)):min(img.shape[1], x + w + int(w*0.2))]
            gray_label = cv2.cvtColor(label_roi, cv2.COLOR_BGR2GRAY)
            _, lab_thresh = cv2.threshold(gray_label, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # invert if necessary
            if np.sum(lab_thresh == 255) < np.sum(lab_thresh == 0):
                lab_thresh = cv2.bitwise_not(lab_thresh)
            # try pytesseract first
            if pytesseract is not None:
                try:
                    ocr_text = pytesseract.image_to_string(lab_thresh, config='--psm 7').strip()
                except Exception:
                    ocr_text = None
            # fallback to easyocr
            if (not ocr_text or ocr_text.strip() == '') and easyocr is not None:
                try:
                    reader = easyocr.Reader(['en'], gpu=False)
                    res = reader.readtext(lab_thresh, detail=0)
                    if res:
                        ocr_text = ' '.join(res).strip()
                except Exception:
                    ocr_text = ocr_text or None

        # decide name: explicit names file overrides OCR
        if names and i < len(names):
            name = names[i]
        else:
            name = None
            if ocr_text:
                # clean common noise
                cleaned = ''.join(ch for ch in ocr_text if ch.isalnum() or ch in (' ','-','_')).strip()
                # try fuzzy matching to known specs
                if fuzzprocess and known_specs:
                    best = fuzzprocess.extractOne(cleaned, known_specs)
                    if best and best[1] >= 65:
                        name = best[0]
                    else:
                        name = cleaned
                else:
                    name = cleaned
            if not name or name == '':
                name = f"spec_{i+1}"
        # sanitize name
        safe = "".join(c for c in name if c.isalnum() or c in ("-","_"," ")).rstrip()
        # ensure unique filename
        fname = f"{safe}.png"
        base, ext = os.path.splitext(fname)
        k = 1
        while os.path.exists(os.path.join(out_dir, fname)):
            fname = f"{base}_{k}{ext}"
            k += 1
        out_path = os.path.join(out_dir, fname)
        cv2.imwrite(out_path, roi)
        mapping.append((fname, x, y, w, h, ocr_text or ''))
        print(f"Saved: {out_path} (ocr='{ocr_text}')")

    # write mapping CSV
    csv_path = os.path.join(out_dir, "mapping.csv")
    with open(csv_path, "w", newline='', encoding='utf-8') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["filename","x","y","w","h","ocr_text"])
        for row in mapping:
            writer.writerow(row)
    print(f"Wrote mapping to {csv_path}")


if __name__ == '__main__':
    main()