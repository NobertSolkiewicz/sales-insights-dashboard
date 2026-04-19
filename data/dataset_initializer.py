import os
import shutil
from pathlib import Path
import kagglehub

# folder główny projektu: sales_dashboard_app/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# folder docelowy na dane
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# pobranie datasetu
path = kagglehub.dataset_download("vivek468/superstore-dataset-final")
print("Path to dataset files:", path)

files = os.listdir(path)
print("Files in directory:", files)

source_file = Path(path) / files[0]
target_file = DATA_DIR / "sales.csv"

shutil.copy(source_file, target_file)

print("Copied file:", source_file)
print("Saved as:", target_file)