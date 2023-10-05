import torch
from module.models.common import DetectMultiBackend
from module.utils.torch_utils import select_device
from module.utils.general import non_max_suppression, scale_boxes, xyxy2xywh
from module.utils.dataloaders import LoadImages


class Model:
    def __init__(self, weights: str, device: str, yaml: str) -> None:
        self.weights = weights
        self.device = select_device(device)
        self.yaml = yaml

    def create(self):
        model = DetectMultiBackend(self.weights, self.device, self.yaml)
        imgsz = [640, 640]
        bs = 1
        pt = model.pt
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
        return model


class RunModel:
    def __init__(self, model) -> None:
        self.model = model

    def inference(self, source):
        outputBoxes = []
        detectedLabels = []
        stride, names, pt = self.model.stride, self.model.names, self.model.pt
        imgsz = [640, 640]
        dataset = LoadImages(source, img_size=imgsz,
                             stride=stride, auto=pt, vid_stride=1)
        for path, im, im0s, _, s in dataset:
            im = torch.from_numpy(im).to(self.model.device)
            im = im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]
            pred = self.model(im, augment=False, visualize=False)
            pred = non_max_suppression(
                pred, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, max_det=1000)

            for det in pred:
                im0 = im0s.copy()
                if len(det):
                    det[:, :4] = scale_boxes(
                        im.shape[2:], det[:, :4], im0.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))
                                ).view(-1).tolist()
                        label = names[int(cls)]
                        # print("xyxy: ", xyxy, " with label: ", label)

                        outputBoxes.append([xywh, label, conf])
                        detectedLabels.append(label)
        return outputBoxes, detectedLabels
