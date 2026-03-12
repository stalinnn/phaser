import os
import sys
import numpy as np
import warnings

# 尝试导入必要的库
try:
    import nilearn
    from nilearn import datasets
    from nilearn import input_data
    import nibabel as nib
    print("Nilearn version:", nilearn.__version__)
except ImportError:
    print("Error: Nilearn is not installed. Please run: pip install nilearn nibabel")
    sys.exit(1)

# 配置路径
DATA_DIR = os.path.join("data", "propofol_dataset")
os.makedirs(DATA_DIR, exist_ok=True)

def check_and_download_atlas():
    print("\n[1/3] Fetching Schaefer 2018 Atlas...")
    try:
        # 下载 100 ROI 图谱
        dataset = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, data_dir=os.path.join("data", "nilearn_data"))
        print(f"Atlas maps loaded: {dataset.maps}")
        return dataset
    except Exception as e:
        print(f"Failed to fetch atlas: {e}")
        return None

def check_ds003171():
    print("\n[2/3] Checking OpenNeuro ds003171 data...")
    # 检查目录下是否有数据
    subjects = [d for d in os.listdir(DATA_DIR) if d.startswith("sub-")]
    if len(subjects) > 0:
        print(f"Found {len(subjects)} subjects in {DATA_DIR}.")
        print("Data check PASSED.")
        return True
    else:
        print(f"No subjects found in {DATA_DIR}.")
        print("WARNING: Automatic download of full fMRI datasets is heavy.")
        print("Please manually download 'ds003171' from OpenNeuro using DataLad:")
        print(f"  datalad install https://github.com/OpenNeuroDatasets/ds003171.git {DATA_DIR}")
        print("Or place your preprocessed .nii.gz files in the data folder structure.")
        return False

def generate_placeholder_readme():
    readme_path = os.path.join(DATA_DIR, "README_INSTRUCTIONS.txt")
    with open(readme_path, "w") as f:
        f.write("To reproduce Figure 4a (fMRI), please download ds003171 from OpenNeuro.\n")
        f.write("Structure should be:\n")
        f.write("data/propofol_dataset/\n")
        f.write("  sub-01/\n")
        f.write("    func/\n")
        f.write("      sub-01_task-restawake_bold.nii.gz\n")
        f.write("      sub-01_task-restdeep_bold.nii.gz\n")
    print(f"\n[3/3] Instructions saved to {readme_path}")

if __name__ == "__main__":
    print(">>> Setting up Real Data Environment <<<")
    check_and_download_atlas()
    has_data = check_ds003171()
    generate_placeholder_readme()
    
    if not has_data:
        print("\n[ACTION REQUIRED] Please download the real fMRI data to proceed with strict replication.")
    else:
        print("\n[READY] Environment is ready for real data analysis.")
