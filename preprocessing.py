import os
import shutil
import cv2
# from model import delete_space, split_rows, create_number, replace_words
from yolo import RunModel
from converter import convert
from PIL import Image
import shutil


def makeDirectory(big_folder):
    if os.path.exists("./" + big_folder):
        shutil.rmtree("./" + big_folder)
    os.mkdir("./" + big_folder)
    labels = ['coQuan', 'ngayThang', 'nguoiKy', 'so', 'tieuDe', 'trichYeu']
    os.mkdir("./" + big_folder + "/Result2")
    for l in labels:
        os.mkdir(os.path.join("./" + big_folder + "/Result2", l))
    os.mkdir("./" + big_folder + "/Result")
    os.mkdir("./" + big_folder + "/image")
    os.mkdir("./" + big_folder + "/pdf")


def clearDirectory(big_folder):
    labels = ['coQuan', 'ngayThang', 'nguoiKy', 'so', 'tieuDe', 'trichYeu']
    for l in labels:
        for i in os.listdir(os.path.join("./" + big_folder + "/Result2", l)):
            os.remove(os.path.join("./" + big_folder + "/Result2", l, i))
    listDir = ["./" + big_folder + "/pdf", "./" +
               big_folder + "/image", "./" + big_folder + "/Result"]
    for dir in listDir:
        for i in os.listdir(dir):
            os.remove(os.path.join(dir, i))


def runYoloDetection(model, images_folder, image_name):
    """
    Chạy YOLO trả về tọa độ bounding box, label và conf score
    Args:
        model (_type_): _description_
        source (_type_): _description_

    Returns:
        _type_: _description_
    """
    detection = RunModel(model)
    detectedInformation, detectedLabels = detection.inference(
        os.path.join(images_folder, image_name))
    eliminateErrorInfor(detectedInformation)
    return detectedInformation, image_name, detectedLabels


def eliminateErrorInfor(detectedInformation):
    """
    Hàm chỉnh sửa và lấy confidence score cao nhất nếu có trường thông tin bị gán nhãn 2 lần
    Args:
        detectedInformation (_type_): _description_
    """
    pass


def splitInfoFromYoloToDictionary(information, images_folder, image_name):
    """
    Từ thông tin detect được ở trên, tách và chia các ảnh trường thông tin, lưu vào dictionary
    information bao gồm (boundingBoxes, labels, confScores)
    Args:
        information (_type_): _description_

    Returns:
        _type_: _description_
    """
    def crop(xywh, image):
        x, y, w, h = xywh[0], xywh[1], xywh[2], xywh[3]
        x_topleft = int(x - float(w / 2))
        y_topleft = int(y - float(h / 2))
        x_bottomright = int(x + float(w / 2))
        y_bottomright = int(y + float(h / 2))
        cropped = image[y_topleft: y_bottomright, x_topleft: x_bottomright]
        return cropped
    image = cv2.imread(os.path.join(images_folder, image_name))
    dictionary = {}
    for info in information:
        xywh = info[0]
        label = info[1]
        conf = info[2]
        cropped = crop(xywh, image)
        dictionary[label] = {}
        dictionary[label]["image"] = cropped
        dictionary[label]["conf"] = conf
    return dictionary


def splitTextInLines(information):
    """
    Tách từng dòng trong cả khung ảnh và lưu vào thư mục
    """
    pass


def runProgram(model, formatType, big_folder):
    all_indx = 0
    fileName = os.listdir("./" + big_folder + "/image")[0]
    print(fileName)
    infor, nameOfImage, detectedLabels = runYoloDetection(
        model, "./" + big_folder + "/image", fileName)
    result = splitInfoFromYoloToDictionary(
        infor, "./" + big_folder + "/image", fileName)
    labels = ['coQuan', 'ngayThang', 'nguoiKy', 'so', 'tieuDe', 'trichYeu']
    for label in labels:
        if label in detectedLabels:
            part = result[label]["image"]
            cv2.imwrite("./" + big_folder + "/Result/img" + "$" +
                        str(label) + "$" + str(all_indx) + "." + formatType, part)
            all_indx += 1
    print(0)
