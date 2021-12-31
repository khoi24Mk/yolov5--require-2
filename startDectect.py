
import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync





class Detector:
    def __init__(self):

        self.webcam = None
        self.source= None
        self.imgsz= None
        self.stride= None
        self.pt= None
        self.model= None
        self.half= None
        self.device= None
        self.save_dir= None
        self.augment= None
        self.conf_thres= None
        self.iou_thres= None
        self.classes= None
        self.agnostic_nms= None
        self.max_det= None
        self.save_crop= None
        self.line_thickness= None
        self.names= None
        self.save_txt= None
        self.save_conf= None
        self.save_img= None
        self.hide_labels= None
        self.hide_conf= None
        self.update= None
        self.weights = None
        self.visualize = None
        self.view_img = None

        self.data_dir = None
        self.img_info = []


    def initArguments(self,
        webcam, source, imgsz, stride, pt,model, half, device,\
        save_dir, augment, conf_thres, iou_thres, classes, agnostic_nms,\
        max_det, save_crop, line_thickness, names, save_txt, save_conf,\
        save_img, hide_labels, hide_conf, update, weights,  visualize,  view_img
    ):
        self.webcam, self.source, self.imgsz, self.stride, self.pt, self.model, self.half, self.device,\
        self.save_dir, self.augment, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,\
        self.max_det, self.save_crop, self.line_thickness, self.names, self.save_txt, self.save_conf,\
        self.save_img, self.hide_labels, self.hide_conf, self.update, self.weights,  self.visualize ,self.view_img = webcam, source, imgsz, stride, pt,model, half, device,\
        save_dir, augment, conf_thres, iou_thres, classes, agnostic_nms,\
        max_det, save_crop, line_thickness, names, save_txt, save_conf,\
        save_img, hide_labels, hide_conf, update, weights,  visualize,  view_img\


    def startDections(self, data_dir):
        self.data_dir = data_dir
        # Dataloader
        if self.webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(data_dir, img_size=self.imgsz, stride=self.stride, auto=self.pt)
            bs = len(dataset)  # batch_size
        else:
            dataset = LoadImages(data_dir, img_size=self.imgsz, stride=self.stride, auto=self.pt)
            bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        self.model.warmup(imgsz=(1, 3, *self.imgsz), half=self.half)  # warmup
        dt, seen = [0.0, 0.0, 0.0], 0
        for path, im, im0s, vid_cap, s in dataset:
            t1 = time_sync()
            im = torch.from_numpy(im).to(self.device)
            im = im.half() if self.half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            self.visualize = increment_path(self.save_dir / Path(path).stem, mkdir=True) if self.visualize else False
            pred = self.model(im, augment=self.augment, visualize= self.visualize)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
            dt[2] += time_sync() - t3

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                str_img=''
                if self.webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    # s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(self.save_dir / p.name)  # im.jpg
                txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                str_img+='%gx%g ' % im.shape[2:]
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if self.save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                        str_img+= f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if self.save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if self.save_img or self.save_crop or self.view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            if self.save_crop:

                                save_one_box(xyxy, imc, file=self.save_dir / 'crops' / self.names[c] / f'{p.stem}.jpg', BGR=True)
                self.img_info.append(str_img)
                print("NEXT")
                print(f'{s}Done. ({t2 - t1:.3f}s)')
                # Print time (inference-only)
                LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

                # Stream results
                im0 = annotator.result()
                if self.view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if self.save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)



        print("RESURESULT")  # +++++++++++++++++++++++++++++++++++
        # Print results
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, * self.imgsz)}' % t)
        if self.save_txt or self.save_img:
            s = f"\n{len(list(self.save_dir.glob('labels/*.txt')))} labels saved to {self.save_dir / 'labels'}" if self.save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
        if self.update:
            strip_optimizer(self.weights)  # update model (to fix SourceChangeWarning)



        # ++++++++++++++++++++++++++++++++++++++++++++++++++













