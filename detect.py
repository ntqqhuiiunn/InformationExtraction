import os, shutil, cv2, fitz, random 
import numpy as np
from PIL import Image
import time
from model import delete_space, split_rows, create_number, replace_words
from yolov7 import YOLOv7
from create import Create

class Detect:
    def __init__(self) -> None:
        yolo = YOLOv7()
        self.yolo = yolo
        cr = Create()
        self.cr = cr
    def get_4_corners(self, x_center, y_center, w, h):
        min_width = int(x_center - float(w/2))
        max_width = int(x_center + float(w/2))
        min_height = int(y_center - float(h/2))
        max_height = int(y_center + float(h/2))
        top_left = [min_width, min_height]
        top_right = [max_width, min_height]
        bottom_left = [min_width, max_height]
        bottom_right = [max_width, max_height]
        return [top_left, top_right, bottom_left, bottom_right]

    def detect_text_position(self, path : str): #, indx_of_folder : int, save_path : str):
        image = cv2.imread(path)
        res = self.yolo.run(image)
        return res
    def eliminate_pdf(self, path : str, indx_of_folder : int, save_path : str):
        res = self.detect_text_position(path) 
        output_boxes = res[0]
        image = cv2.imread(path)  
        im0 = image.copy()
        labels = []
        out_data = []
        all_indx = 0
        page_position = None
        text_position = []
        save_path_pdf = os.path.join(save_path, "pdf_" + str(indx_of_folder))
        if os.path.exists(save_path_pdf) != True:
            os.mkdir(save_path_pdf)
        count = len(os.listdir(save_path_pdf))
        for element in output_boxes:
            labels.append(element[1])
        labels = np.array(labels)
        if (('time' in labels and 'number' in labels and 'heading' in labels) or ('to' in labels)):
            out_data = output_boxes
            page_position = path[len(path) - 6 : len(path) - 4]
        if len(out_data) != 0: 
            data_labels = [] 
            new_folder = os.path.join(save_path_pdf, str(count))
            if os.path.exists(new_folder) == False:
                os.mkdir(new_folder)
            for data in out_data:
                data_labels.append(data[1])
            if 'number' in data_labels and 'to' not in data_labels:
                group1 = ['heading', 'number', 'time', 'title_1', 'title_2']
                for data in out_data:
                    if data[1] not in group1:
                        out_data.remove(data)
            if 'to' in data_labels and 'number' not in data_labels: 
                group2 = ['to', 'author', 'signature']
                for data in out_data:
                    if data[1] not in group2:
                        out_data.remove(data)
            for data in out_data:
                box = data[0]
                label = data[1]
                x_center = box[0]
                y_center = box[1]
                width = box[2]
                height = box[3]
                    #######################
                start_point_x = int(x_center - float(width / 2))
                start_point_y = int(y_center - float(height / 2))
                top_right = int(x_center + float(width / 2))
                bottom_left = int(y_center + float(height / 2))
                ############################### 
                text_position.append([self.get_4_corners(x_center, y_center, width, height), label])
                    #######################
                img_part = im0[start_point_y : bottom_left, start_point_x : top_right]
                img_name = str(indx_of_folder) + "$" + str(label) + "$" + str(all_indx) + ".png"
                    ##########################
                path_to_img = os.path.join(new_folder, img_name)
                all_indx += 1
                cv2.imwrite(path_to_img, img_part)
        last_page_checked = 'to' in labels
        return [last_page_checked, page_position, text_position]    
    def eliminate_error_info(self, pdfi_path : str, labels_big_folder : str):
        group1 = np.array(['author', 'to', 'signature'])
        group2 = np.array(['number', 'heading', 'time'])
        for folder in os.listdir(pdfi_path):
            items = os.listdir(os.path.join(pdfi_path, folder))
            labels = []
            for item in items:
                indx, label, name = item.split('$')
                labels.append(label)
            labels = set(labels)
            i = 0
            j = 0
            for label in labels:
                if label in group1:
                    i += 1
                if label in group2:
                    j += 1
            if (i + j) != 3 and (i + j) != 6:
                shutil.rmtree(os.path.join(pdfi_path, folder)) 
        for folder_i in os.listdir(pdfi_path):
            folder_i_path = os.path.join(pdfi_path, folder_i)
            items = os.listdir(folder_i_path)
            for item_name in items:
                new_path = os.path.join(pdfi_path, item_name)
                item = cv2.imread(os.path.join(folder_i_path, item_name))
                cv2.imwrite(new_path, item)
            shutil.rmtree(folder_i_path)
        for element in os.listdir(pdfi_path):
            element_path = os.path.join(pdfi_path, element)
            label = element.split('$')[1]
            label_path = os.path.join(labels_big_folder, label)
            new_path = os.path.join(label_path, element)
            element_img = cv2.imread(element_path)
            cv2.imwrite(new_path, element_img) 
    def split_text_content(self, label_path : str, res):
        cont_list = []
        try:
            for element in os.listdir(label_path):
                element_path = os.path.join(label_path, element)
                indx_of_pdf, label, value = element.split('$')
                indx_of_pdf = int(indx_of_pdf)
                detected_list = split_rows(element_path)
                os.remove(element_path)
                for index, splitted in enumerate(detected_list):
                    piece_name = create_number(index) + "$" + element
                    path_to_piece = os.path.join(label_path, piece_name)
                    cv2.imwrite(path_to_piece, splitted)
                    delete_space(path_to_piece)
                    ####################
                    cont_list.append([indx_of_pdf, label, path_to_piece])
            for i in cont_list:
                res.append(i)
        except FileNotFoundError:
            print("An error in {0}".format(label_path))
    def detect_text(self, cont_list, detector):
        dicti = {}
        fields = self.cr.fields 
        for element in cont_list:
            pdf_index = element[0]
            dicti["pdf" + str(pdf_index)] = {}
        for element in cont_list:
            name = element[1]
            pdf_index = element[0]
            if name != 'signature':
                dicti["pdf" + str(pdf_index)][fields[name]] = ""
        for element in cont_list:
            path = element[2]
            name = element[1]
            pdf_index = element[0]
            PIL_img = Image.open(path)
            content = detector.predict(PIL_img)
            out_content = replace_words(content)
            if name != "signature":
                dicti["pdf" + str(pdf_index)][fields[name]] += out_content + " "
            else: 
                if fields['signature'] in dicti["pdf" + str(pdf_index)]:
                    del dicti["pdf" + str(pdf_index)][fields['signature']]
                else: 
                    print("")
        return dicti      
    def mkdir(self, path):
        if os.path.exists(path) == False:
            os.mkdir(path)
    def run(self, files):
        drive_path = "./ResultFromProgramf"
        save_path = "./ResultFromProgramf/save_path"
        pdf_folder = "./ResultFromProgramf/pdf_folder"
        images_folder = "./ResultFromProgramf/images_folder"
        labels_big_folder = "./ResultFromProgramf/labels_folder"
        if os.path.exists(drive_path):
            shutil.rmtree(drive_path)
        self.mkdir(drive_path)
        self.mkdir(save_path)
        self.mkdir(pdf_folder)
        self.mkdir(images_folder)
        self.mkdir(labels_big_folder)
        self.cr.create_label_folder(labels_big_folder) 
        i = 0
        indx = 0
        cont_list = []
        page_position = []
        page_info = []
        detector = self.yolo.vietocr
        indx_of_pdf = 0
        list_names = []
        mat = fitz.Matrix(2.0, 2.0) 
        start = time.time()   
        for file in files:
            try:
                file_path = pdf_folder + "/" + "iter" + str(i) + ".pdf"
                list_names.append(file.filename)
                with open(file_path, 'wb') as buffer:
                    shutil.copyfileobj(file.file, buffer)
                i += 1  
            except TypeError:
                print("Uploaded files must be in PDF")
            except FileNotFoundError:
                print("File not found")
        for file in os.listdir(pdf_folder):
            each_pdf_folder = os.path.join(images_folder, str(indx_of_pdf))
            if os.path.exists(each_pdf_folder) == False:
                os.mkdir(each_pdf_folder)
            pdf = fitz.open(os.path.join(pdf_folder, file))
            for page in pdf:
                pix = page.get_pixmap(matrix=mat)
                #pix = page.getPixmap(matrix= mat, alpha= False)
                pix.save(os.path.join(each_pdf_folder, "pdf_" + str(indx_of_pdf) + "_image_" + create_number(indx) + ".png"))
                indx += 1
            indx_of_pdf += 1
        for indx_of_pdf_folder, each_images_folder in enumerate(sorted(os.listdir(images_folder))):
            each_images_folder_path = os.path.join(images_folder, each_images_folder)
            for image_name in os.listdir(each_images_folder_path):
                    image_p = os.path.join(each_images_folder_path, image_name)
                    outp = self.eliminate_pdf(image_p, indx_of_pdf_folder, save_path)
                    checked = outp[0]
                    if outp[1] != None:
                        page_position.append(outp[1])
                        page_info.append(outp[2])
                    if checked == True:
                        break
        for each_pdf_folder in os.listdir(save_path):
            pdf_folder_path = os.path.join(save_path, each_pdf_folder)
            self.eliminate_error_info(pdf_folder_path, labels_big_folder)
        labels_list = self.cr.create_label_variable(labels_big_folder)
        for t in range(len(labels_list)):
            self.split_text_content(labels_list[t], cont_list)
        v = self.detect_text(cont_list, detector)    
        # for i in range(len(page_position)):
        #     v["page " + str(page_position[i])] = str(page_info[i])
        end = time.time()
        print("Elapsed time: {:.2f} s".format(end - start))
        return v

def clear():
    shutil.rmtree("./ResultFromProgramf")