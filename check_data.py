from pathlib import Path
import cv2

RAW_DIR = Path('data/Combined Dataset/train')
PROCESSED_DIR = Path('data/processed/train')

print("RAW TRAIN counts:")
for class_name in ['Non Demented', 'Very mild DementIa', 'Mild Dementia', 'Moderate DementIa']:
    count = len(list((RAW_DIR / class_name).glob('*.jpg')))
    print(f"  {class_name}: {count} images")

print("\nPROCESSED TRAIN counts:")
for class_path in PROCESSED_DIR.iterdir():
    count = len(list(class_path.glob('*.jpg')))
    print(f"  {class_path.name}: {count} images")
