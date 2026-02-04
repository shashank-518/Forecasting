import os
import logging
from pathlib import Path

project_name = "forecasting"

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s]: %(message)s:')


list_of_files = [

    f"src/{project_name}/__init__.py",

    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_validation.py",
    f"src/{project_name}/components/preprocessing.py",
    f"src/{project_name}/components/feature_engineering.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_evaluator.py",
    f"src/{project_name}/components/model_selector.py",

    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/training_pipeline.py",
    f"src/{project_name}/pipeline/prediction_pipeline.py",

    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/utils/logger.py",

    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/config.yaml",

   
    "artifacts/models/.gitkeep",
    "artifacts/best_model_per_state.csv",

    "data/raw/.gitkeep",
    "data/processed/.gitkeep",

    
    "app.py",               # FastAPI backend
    "streamlit_app.py",     # UI dashboard
    "Dockerfile",
    ".dockerignore",
    "requirements.txt",
    "setup.py",

    "README.md",
    "main.py",
]


for filepath in list_of_files:

    filepath = Path(filepath)

    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating file: {filepath}")

    else:
        logging.info(f"{filename} already exists")
