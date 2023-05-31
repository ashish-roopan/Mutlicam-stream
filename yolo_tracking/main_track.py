from pathlib import Path
import torch
import argparse
import numpy as np
import cv2
import cv2
cv2.startWindowThread()
import time
from rich import print

from trackers.multi_tracker_zoo import create_tracker
from ultralytics.yolo.engine.model import YOLO, TASK_MAP

from ultralytics.yolo.utils import ops
from ultralytics.yolo.utils.checks import check_imgsz, print_args
from ultralytics.yolo.engine.results import Boxes


class Tracker:
    def __init__(self,opt):
        self.tracked_objects = {}
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(50)]

    def on_predict_start(self, predictor):
        predictor.trackers = []
        predictor.tracker_outputs = [None] * predictor.dataset.bs
        predictor.args.tracking_config = \
            Path('trackers') /\
            opt.tracking_method /\
            'configs' /\
            (opt.tracking_method + '.yaml')
        for i in range(predictor.dataset.bs):
            tracker = create_tracker(
                predictor.args.tracking_method,
                predictor.args.tracking_config,
                predictor.args.reid_model,
                predictor.args.device,
                predictor.args.half
            )
            predictor.trackers.append(tracker)

    @torch.no_grad()
    def setup_model(self, 
        yolo_model,
        reid_model,
        tracking_method='strongsort',
        source = '0',
        imgsz = [640, 640],
        save_dir=False,
        vid_stride = 1,
        verbose = True,
        project = None,
        exists_ok = False,
        name = None,
        save = True,
        save_txt = True,
        visualize=False,
        plotted_img = False,
        augment = False,
        conf = 0.5,
        device = '',
        show = False,
        half = True,
        classes = None
    ):
        
        print('YOLO model path: ', yolo_model)
        print('ReID model path: ', reid_model)
        print('Tracking method: ', tracking_method)
        model = YOLO(yolo_model)
        overrides = model.overrides.copy()
        model.predictor = TASK_MAP[model.task][3](overrides=overrides, _callbacks=model.callbacks)
        predictor = model.predictor
        predictor.args.reid_model = reid_model
        predictor.args.tracking_method = tracking_method
        predictor.args.conf = 0.5
        predictor.args.project = project
        predictor.args.name = name
        predictor.args.conf = conf
        predictor.args.half = half
        predictor.args.classes = classes
        predictor.args.imgsz = imgsz
        predictor.args.vid_stride = vid_stride
        predictor.args.save_txt = True
        predictor.args.save = True
        if not predictor.model:
            predictor.setup_model(model=model.model, verbose=False)
        
        predictor.setup_source(source if source is not None else predictor.args.source)
        
        dataset = predictor.dataset
        model = predictor.model
        imgsz = check_imgsz(imgsz, stride=model.model.stride, min_dim=2)  # check image size
        source_type = dataset.source_type
        preprocess = predictor.preprocess
        postprocess = predictor.postprocess
        run_callbacks = predictor.run_callbacks
        
        if predictor.args.save or predictor.args.save_txt:
            (predictor.save_dir / 'labels' if predictor.args.save_txt else predictor.save_dir).mkdir(parents=True, exist_ok=True)
        # Warmup model
        if not predictor.done_warmup:
            predictor.model.warmup(imgsz=(1 if predictor.model.pt or predictor.model.triton else predictor.dataset.bs, 3, *predictor.imgsz))
            predictor.done_warmup = True
        predictor.seen, predictor.windows, predictor.batch, predictor.profilers = 0, [], None, (ops.Profile(), ops.Profile(), ops.Profile(), ops.Profile())
        predictor.add_callback('on_predict_start', self.on_predict_start)
        run_callbacks('on_predict_start')

        return [predictor, dataset, run_callbacks, model, preprocess, postprocess, source_type, augment, visualize, save_dir, source, imgsz, vid_stride, verbose, project, exists_ok, name, save, save_txt, plotted_img, conf, device, show, half, classes]

    @torch.no_grad()
    def run(self, setup_data):
        predictor, dataset, run_callbacks, model, preprocess, postprocess, source_type, augment, visualize, save_dir, source, imgsz, vid_stride, verbose, project, exists_ok, name, save, save_txt, plotted_img, conf, device, show, half, classes = setup_data
        
        for frame_idx, batch in enumerate(dataset):
            start = time.time()
            run_callbacks('on_predict_batch_start')
            predictor.batch = batch
            path, im0s, vid_cap, s = batch

            # Preprocess
            with predictor.profilers[0]:
                im = preprocess(im0s)

            # Inference
            with predictor.profilers[1]:
                preds = model(im, augment=augment, visualize=visualize)

            # Postprocess
            with predictor.profilers[2]:
                predictor.results = postprocess(preds, im, im0s)
            run_callbacks('on_predict_postprocess_end')
            
            i = 0
            # Visualize, save, write results
            p, im0 = path[i], im0s[i].copy()
            p = Path(p)
            
            with predictor.profilers[3]:
                # get raw bboxes tensor
                dets = predictor.results[i].boxes.data
                # get predictions
                predictor.tracker_outputs[i] = predictor.trackers[i].update(dets.cpu().detach(), im0)
            
            # overwrite bbox results with tracker predictions
            if predictor.tracker_outputs[i].size != 0:
                predictor.results[i].boxes = Boxes(
                    # xyxy, (track_id), conf, cls
                    boxes=torch.from_numpy(predictor.tracker_outputs[i]).to(dets.device),
                    orig_shape=im0.shape[:2],  # (height, width)
                )


            print('FPS: ', 1/(time.time()-start))
            print('predictor.results[i].boxes = ', predictor.results[i].boxes.data)
            print('predictor.results[i].boxes.shape = ', predictor.results[i].boxes.data.shape)
            print()

            boxes = predictor.results[i].boxes.data
            for box in boxes:
                print('box = ', box)
                x1, y1, x2, y2 = box[:4]
                obj_id = int(box[4])
                score = box[5]
                class_id = int(box[6])
                centeroid = (int((x1+x2)/2), int(y2))
                if obj_id not in self.tracked_objects:
                    self.tracked_objects[obj_id] = [centeroid]
                else:   
                    self.tracked_objects[obj_id].append(centeroid)

                cv2.rectangle(im0, (int(x1), int(y1)), (int(x2), int(y2)), self.colors[obj_id], 2)
                for centeroid in self.tracked_objects[obj_id][-100:]:
                    cv2.circle(im0, centeroid, 3, self.colors[obj_id], -1)
            
            print('showing image')
            cv2.imshow('im0', im0)
            cv2.waitKey(1)
        

def parse_opt():
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]  # root dir
    WEIGHTS = ROOT / 'weights'
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', type=str, default=WEIGHTS / 'yolov8n.pt', help='model.pt path(s)')
    parser.add_argument('--reid-model', type=Path, default=WEIGHTS / 'mobilenetv2_x1_4_dukemtmcreid.pt')
    parser.add_argument('--tracking-method', type=str, default='deepocsort', help='deepocsort, botsort, strongsort, ocsort, bytetrack')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show', action='store_true', help='display tracking video results')
    parser.add_argument('--save', action='store_true', help='save video tracking results')
    # # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exists-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    tracker = Tracker(opt)    
    setup_data = tracker.setup_model(**vars(opt))
    tracker.run(setup_data)