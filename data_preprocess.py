import os
import numpy as np
import cv2
from pathlib import Path
import shutil

RAW_TRAIN = Path('data/Combined Dataset/train')
RAW_TEST = Path('data/Combined Dataset/test')
IMG_SIZE = (128, 128)

CLASSES = {
    'Non Demented': 'Non_Demented',
    'Very mild DementIa': 'Very_mild_Dementia', 
    'Mild Dementia': 'Mild_Dementia',
    'Moderate DementIa': 'Moderate_Dementia'
}

def safe_preprocess(img_path, out_path):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"⚠️  Skip {img_path}")
        return False
    
    # Grayscale + resize + center crop
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMG_SIZE)
    h, w = resized.shape
    center = resized[int(0.15*h):int(0.85*h), int(0.15*w):int(0.85*w)]
    final = cv2.resize(center, IMG_SIZE)
    
    cv2.imwrite(str(out_path), (final * 255 / final.max()).astype(np.uint8))
    return True

# CLEAN START
for split in ['train', 'test']:
    (Path('data/processed') / split).mkdir(parents=True, exist_ok=True)
    
    raw_dir = RAW_TRAIN if split == 'train' else RAW_TEST
    print(f"\n🔄 Processing {split}...")
    
    for old_name, new_name in CLASSES.items():
        raw_class = raw_dir / old_name
        out_class = Path('data/processed') / split / new_name
        
        if not raw_class.exists():
            print(f"❌ Missing: {raw_class}")
            continue
            
        out_class.mkdir(exist_ok=True)
        imgs = list(raw_class.glob('*.jpg'))
        
        success = 0
        for img_path in imgs:
            out_path = out_class / img_path.name
            if safe_preprocess(img_path, out_path):
                success += 1
        
        print(f"  {new_name}: {success}/{len(imgs)} processed")

print("✅ COMPLETE!")


import nibabel as nib

def extract_3views_nifti(nii_path, output_dir):
    img = nib.load(str(nii_path)).get_fdata()
    # Axial (z-slices), Coronal (y), Sagittal (x) - middle slices for hippocampus
    mid_z, mid_y, mid_x = img.shape[1]//2, img.shape[2]//2, img.shape[0]//2
    views = {
        'axial.png': img[:, :, mid_z],
        'coronal.png': img[:, mid_y, :],
        'sagittal.png': img[mid_x, :, :]
    }
    for name, view in views.items():
        view = (view / view.max() * 255).astype(np.uint8)
        cv2.imwrite(str(output_dir / name), view)
# Example usage:
# extract_3views_nifti(Path('data/raw/ADNI_001_S_0027.nii'), OUTPUT_DIR / 'ADNI_001_S_0027')  
