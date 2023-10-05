import os
import shutil
import cv2
from PIL import Image
import time
import subprocess
from model import delete_space, split_rows, replace_words
from preprocessing import runProgram
from selectFileType import selectType


class Detect:
    def __init__(self) -> None:
        pass

    def split_text_content(self, label_path: str, formatType, big_folder):
        try:
            for element in os.listdir(label_path):
                element_path = os.path.join(label_path, element)
                _, label, value = element.split('$')
                detected_list = split_rows(element_path)
                os.remove(element_path)
                for index, splitted in enumerate(detected_list):
                    piece_name = str(index) + "." + formatType
                    path_to_piece = os.path.join(
                        "./" + big_folder + "/Result2/" + label, piece_name)
                    cv2.imwrite(path_to_piece, splitted)
                    delete_space(path_to_piece)
                    ####################
        except FileNotFoundError:
            print("An error in {0}".format(label_path))

    def read_text(self, image_directory: str, image_name: str):
        img_path = image_directory + "/" + image_name
        name, formatType = image_name.split(".")
        output_path = image_directory + "/" + name
        txt_path = output_path + ".txt"
        cmd_arg = "tesseract " + img_path + " " + output_path + " -l nonItalic12"
        subprocess.run(cmd_arg, shell=True)
        with open(txt_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content

    def detect_text(self, big_folder):
        dicti = {}
        for label in os.listdir("./" + big_folder + "/Result2"):
            stringOfLabel = ""
            for partPath in sorted(os.listdir("./" + big_folder + "/Result2/" + label)):
                content = self.read_text(
                    "./" + big_folder + "/Result2/" + label, partPath)
                # content = self.runTesseract(os.path.join("./ProXgram/Result2/" + label, partPath))
                if label == "tieuDe" and content == content.upper():
                    dicti["loaiVanBan"] = content.replace('\n', '')
                    out_content = ""
                else:
                    out_content = replace_words(content)
                stringOfLabel += out_content + " "
            dicti[label] = stringOfLabel.replace('\n', '')
        return dicti

    def run(self, files, model, big_folder):
        start = time.time()
        error = ""
        for file in files:
            try:
                formatType = selectType(file, big_folder)
                print(formatType)
                if len(formatType) == 0:
                    formatType = "png"
            except TypeError:
                print("Uploaded files must be in PDF")
            except FileNotFoundError:
                print("File not found")
        try:
            runProgram(model, formatType, big_folder)
            self.split_text_content(
                "./" + big_folder + "/Result", formatType, big_folder)
            v = self.detect_text(big_folder)
            end = time.time()
            elapsed = "{0:.2f}".format(end - start)
            v["Elapsed time"] = elapsed
            if len(v["so"]) == 0:
                error = "Uploaded file must be an administrative document // Van ban tai len phai la van ban hanh chinh va co so hieu"
            else:
                error = ""
        except:
            error = "Uploaded file must be in PDF or images(.png, .jpg or .jpeg) // File tai len phai co dinh dang pdf hoac la anh (jpg, png hoac jpeg)"
            v = {}
        return v, error
