from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image
import os, numpy as np
error = 0
images_dataset = "C:\\Users\\scien\\Desktop\\api\\New-dataset\\data_line\\InkData_line_processed"
text_dataset = "C:\\Users\\scien\\Desktop\\api\\New-dataset\\data_line\\train_line_annotation.txt"
version2_text = "D:\\DataSet-Viet\\Data_version_2\\train_line_annotation.txt"
version2_images = "D:\\DataSet-Viet\\Data_version_2\\InkData_line_processed"
# with open(version2_text, mode= 'r', encoding= 'utf-8') as dataset:
#     content = dataset.readlines()

def get_image_name(list):
    out = []
    for line in list:
        if line.find('\t') >= 0:
            a = line.split('\t')
            b = a[0]
            c = b[23 : len(b)]
            out.append(c)
        else:
            print("Error: ", line)
    return out
# images = get_image_name(content)
def matchLists(imgs, texts):
    for i in imgs:
        if texts.index(i) < 0:
            return False
    return True
# print(matchLists(images, os.listdir(images_dataset)))
def checkImageInFolder(folder : str, text : str):
    t = 0
    images = np.array(os.listdir(folder))
    used_images = []
    with open(text, mode= 'r', encoding= 'utf-8') as annotation:
        content = annotation.readlines()
    for line in content:
        image_name = line[23 : 55]
        used_images.append(image_name)
    used_images = np.array(used_images)
    for img in images:
        if img not in used_images:
            path = os.path. join(folder, img)
            os.remove(path)   
    return t

def matchFolders(folder_1 : str, folder_2 : str):
    t = 0
    f = np.array(os.listdir(folder_1))
    for i in os.listdir(folder_2):
        if i not in f:
            t += 1
    return t

print(matchFolders(images_dataset, version2_images))