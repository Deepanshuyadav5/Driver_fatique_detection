"""
Script to organize MRL Eyes dataset images into closed_eyes and open_eyes folders.

Filename format: s{subject}_{image}_{gender}_{glasses}_{eye_state}_{reflections}_{lighting}_{sensor}.png
Eye state: 0 = closed, 1 = open
"""

import os
import shutil
import glob

# Paths
source_root = r"c:\Users\dy229\OneDrive\Desktop\AI and ML\Machine learning\full project\Driver_fatique_detection\mrlEyes_2018_01 (4)\mrlEyes_2018_01 (4)\mrlEyes_2018_01"
dest_closed = r"c:\Users\dy229\OneDrive\Desktop\AI and ML\Machine learning\full project\Driver_fatique_detection\train_dataset\closed_eyes"
dest_open = r"c:\Users\dy229\OneDrive\Desktop\AI and ML\Machine learning\full project\Driver_fatique_detection\train_dataset\open_eyes"

# Create destination folders if they don't exist
os.makedirs(dest_closed, exist_ok=True)
os.makedirs(dest_open, exist_ok=True)

closed_count = 0
open_count = 0
skipped_count = 0

# Iterate through all subject folders (s0001, s0002, ...)
for subject_dir in sorted(glob.glob(os.path.join(source_root, "s*"))):
    if not os.path.isdir(subject_dir):
        continue
    
    subject_name = os.path.basename(subject_dir)
    print(f"Processing {subject_name}...")
    
    for img_file in os.listdir(subject_dir):
        if not img_file.endswith(".png"):
            continue
        
        # Parse filename: s{subject}_{image}_{gender}_{glasses}_{eye_state}_{reflections}_{lighting}_{sensor}.png
        parts = img_file.replace(".png", "").split("_")
        
        if len(parts) < 8:
            skipped_count += 1
            continue
        
        # Eye state is at index 4 (5th field)
        eye_state = parts[4]
        
        src_path = os.path.join(subject_dir, img_file)
        
        if eye_state == "0":
            # Closed eyes
            dst_path = os.path.join(dest_closed, img_file)
            shutil.copy2(src_path, dst_path)
            closed_count += 1
        elif eye_state == "1":
            # Open eyes
            dst_path = os.path.join(dest_open, img_file)
            shutil.copy2(src_path, dst_path)
            open_count += 1
        else:
            skipped_count += 1

print(f"\n{'='*50}")
print(f"Done!")
print(f"Closed eyes images copied: {closed_count}")
print(f"Open eyes images copied:   {open_count}")
print(f"Skipped (unrecognized):     {skipped_count}")
print(f"Total processed:            {closed_count + open_count + skipped_count}")
