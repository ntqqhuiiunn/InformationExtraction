import fitz
import os


def convert(file: str, pdf_folder: str, img_folder: str):
    doc = fitz.open(os.path.join(pdf_folder, file))
    file_name = file[0: len(file) - 4] + ".png"
    img = doc[0].get_pixmap()
    img.save(os.path.join(img_folder, file_name))
    return file_name
