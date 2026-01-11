import easyocr
from pathlib import Path

reader = easyocr.Reader(['en','es','fr','pt','de'], gpu=True, verbose=False)

files = [
    "data/debug/match_start_20260110_102550_full.png_red_name_2.png",
    "data/debug/match_start_20260110_102550_full.png_blue_name_2.png",
]

for f in files:
    p = Path(f)
    print('\nFILE:', p)
    if not p.exists():
        print('  MISSING')
        continue
    results = reader.readtext(str(p))
    for res in results:
        bbox, text, conf = res
        print(f"  text='{text}', conf={conf:.3f}, bbox={bbox}")
