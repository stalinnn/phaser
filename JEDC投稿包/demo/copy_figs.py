import shutil
import glob
import os

source_dir = "JEDC_Submission_Package/figures"
target_dir = "JEDC_Submission_Package/demo/figures"

os.makedirs(target_dir, exist_ok=True)

for file in glob.glob(os.path.join(source_dir, "*.png")):
    shutil.copy(file, target_dir)
    print(f"Copied {file} to {target_dir}")
