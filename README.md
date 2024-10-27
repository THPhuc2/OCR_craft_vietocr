# OCR Project: CRAFT and VietOCR

## Introduction
This project provides a solution for text recognition from tables in images by combining two prominent OCR models: **CRAFT** (Character Region Awareness for Text Detection) and **VietOCR** (an OCR model tailored for Vietnamese text). The goal is to identify and extract text from tables within PNG image files.

## Requirements
- Python 3.x
- Necessary libraries:
  - OpenCV
  - Hugging Face Transformers
  - PyTorch
  - other dependencies listed in `requirements.txt`

## Installation
1. **Clone this repository:**
   ```bash
   git clone [https://github.com/THPhuc2/OCR_dbnet_craft_vietocr.git]
   cd OCR_dbnet_craft_vietocr
   ```

2. **Install the necessary libraries:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the models from Hugging Face:**
   - [Models Download Link](https://huggingface.co/THP2903/ocr_db_craft_vietocr/tree/main)

   After downloading, ensure to integrate them into the appropriate directories in the project.

## Usage
1. Ensure you have installed the required libraries and downloaded the models.
2. Run the `main.py` file located in the `src/` directory:
   ```bash
   python src/main.py
   ```

3. Enter the path to the directory containing the PNG image files when prompted:
   ```
   Enter the directory path containing images (you can use / or \\), images must be .png, do not input other formats: 
   ```

The results will be saved in a directory corresponding to the input image file, which will include extracted tables and recognized text.

## Directory Structure
- `src/`: Contains the main source code and necessary modules for processing.
  - `main.py`: The main script to execute the OCR processing.
  - `VietOCR_model.py`: The VietOCR model for text recognition.
  - `Craft_model.py`: The CRAFT model for detecting tables in images.
  - `extract_table.py`: Functions to extract and save tables from images.
  - `rotation.py`: Function to recognize text in a folder.
  - `utils.py`: Utility functions supporting the processing workflow.

## Additional Information
This project utilizes two powerful models in the field of text recognition and information extraction from tables. You can refer to [Hugging Face](https://huggingface.co) for more information on using and customizing these models for your project.

## Contact
For any questions or issues, feel free to contact the project maintainer at PhucTH290303@gmail.com
