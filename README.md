# Information Extraction Project
*This small project can extract some essential information from an administrative document such as:*
* Country’s official name and motto
* The issuing entity’s name
* Document’s numbers or codes
* Name of place and date of issue
* Document type and abstract or summary of document content
* Title, full name and signature of the competent person
* Seal, signature of the entity
* Recipient
## Installation
### First of all, you have to create a virtual environment and set up Tesseract:
*If you run this project on Windows, please follow this step:*
* `python -m venv env`
* `pip install -r requirements.txt`

*If you run this project on Ubuntu (Linux):*
* You can search on Google to create virtual environment  
* After that, you can set up Tesseract module following steps in **how_to_run_tesseract_on_ubuntu.txt**
### After that, you have to clone this repository to your computer using Git
## Inference
### Please run this command:
* `python main.py`
## Modules used
1. **YOLOv5**: detect regions of essential information.
2. **Tesseract**: read content of the cropped images and return strings.
3. **FastAPI**: deploy application on web