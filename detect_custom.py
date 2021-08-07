'''
프레젠테이션 실행 파일
'''
from pathlib import Path
import sys
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())

import cv2
import time
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams
from utils.general import check_requirements, check_img_size, check_imshow, set_logging, increment_path, non_max_suppression, \
    scale_coords
from utils.torch_utils import select_device, time_sync

# custom module
from utils.ppt import output_to_detect

@torch.no_grad()
def run(weights='runs_done/v5m_results/weights/best.pt'):
    imgsz = 416
    conf_th = 0.25
    iou_th = 0.45
    max_detect = 1000
    device = ''

    half = False

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'

    w = weights
    stride, names = 64, [f'class{i}' for i in range(10000)]

    model = attempt_load(w, map_location=device)
    stride = int(model.stride.max())
    names = models.module.names if hasattr(model, 'module') else model.names
    if half:
        model.half()
    
    # Resize image
    imgsz = check_img_size(imgsz, s=stride)

    # Dataloader
    view_img = check_imshow()
    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams('0', img_size=imgsz, stride=stride)
    bs = len(dataset)

    video_path, video_writer = [None] * bs, [None] * bs

    # Run Interface
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters()))) # run once
    
    t0 = time.time()
    for path, img, im0s, video_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()   # uint8 to fp16/32
        img /= 255.0                                # normalize : 0~255 to 0~1
        
        if len(img.shape) == 3:
            img = img[None] # expand for batch dim

        t1 = time_sync() 
        pred = model(img, augment=False, visualize=False)[0]

        # NMS
        pred = non_max_suppression(pred, conf_thres=conf_th, iou_thres=iou_th, classes=None, agnostic=False, max_det=max_detect)
        t2 = time_sync()

        for i, detect in enumerate(pred):
            p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            p = Path(p) # to Path
            # save_path
            # txt_path

            s += '%gx%g ' % img.shape[2:]
            # gain = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalize gain whwh
            # im_cp = im0.copy()    # for save_crop

            if len(detect):
                # Rescale boxes from img_size to im0_size
                detect[:, :4] = scale_coords(img.shape[2:], detect[:, :4], im0.shape).round()

                # Print results
                for c in detect[:, -1].unique():
                    n = (detect[:, -1] == c).sum()  # detections per class
                    s += f'{n} {names[int(c)]} '    # add to string

                # Write results
                # pass

        # Print time (inference + NMS)
        print(f'{s} Done. ({t2 - t1:.3f}s)', end='\t')

        # Get bbox list
        hand_detected = output_to_detect(s)
        print(hand_detected)

        # preprocess detected hand signal
        ####################################
        # your code
        ####################################

        # Action according to the command
        ####################################
        # your code
        ####################################

        # Stream
        cv2.imshow(str(p), im0)
        cv2.waitKey(1)

def main():
    check_requirements(exclude=('tensorboard', 'thop'))
    run()

if __name__ == "__main__":
    main()