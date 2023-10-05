import os
import shutil
from converter import convert
import mimetypes


def selectType(file, big_folder):
    mime_type, encoding = mimetypes.guess_type(file.filename)
    fileType = mime_type
    if fileType == "application/pdf":
        file_path = "./" + big_folder + "/pdf/iter.pdf"
        file_dir = "./" + big_folder + "/pdf"
        with open(file_path, 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)
        pdfFile = os.listdir(file_dir)[0]
        convert(pdfFile, file_dir, "./" + big_folder + "/image")
        return ""
    else:
        typeOfInput, typeOfFormat = str(fileType).split("/")
        if typeOfFormat in ["png", "jpg", "jpeg"]:
            file_dir = "./" + big_folder + "/image"
            file_path = file_dir + "/iter." + typeOfFormat
            with open(file_path, 'wb') as buffer:
                shutil.copyfileobj(file.file, buffer)
                print("Successful")
            return typeOfFormat
        else:
            return None
