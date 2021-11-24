import logging
from logging.handlers import RotatingFileHandler

import panoramasdk
import numpy as np
import cv2

from yolox_postprocess import demo_postprocess, multiclass_nms

class Application(panoramasdk.node):
    def __init__(self, logger):
        super().__init__()
        self.logger = logger
        self.MODEL_NODE = 'model_node'
        self.MODEL_INPUT_NAME = 'images'
        self.MODEL_INPUT_SIZE = (640, 640)
        self.threshold = 0.
        
        try:
            # Get parameter values
            self.logger.info('Getting parameters')
            self.threshold = self.inputs.threshold.get()
        except:
            self.logger.exception('Error during initialization.')
        finally:
            self.logger.info('Initialiation complete.')
            self.logger.info('Threshold: {}'.format(self.threshold))

    def process_streams(self):
        """Processes one frame of video from one or more video streams."""
        # Loop through attached video streams
        streams = self.inputs.video_in.get()
        for stream in streams:
            self.process_media(stream)
        self.outputs.video_out.put(streams)

    def process_media(self, stream):
        """Runs inference on a frame of video."""
        # Preprocess frame
        image_data, ratio = self.preprocess(stream.image, self.MODEL_INPUT_SIZE)

        # Run inference
        inference_results = self.call({self.MODEL_INPUT_NAME: image_data}, self.MODEL_NODE)

        # Process results
        self.process_results(inference_results, stream, ratio)

    def preprocess(self, img, input_size, swap=(2, 0, 1)):
        # source: https://github.com/Megvii-BaseDetection/YOLOX/blob/dd5700c24693e1852b55ce0cb170342c19943d8b/yolox/data/data_augment.py#L144
        if len(img.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(input_size, dtype=np.uint8) * 114

        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r

    def process_results(self, inference_results, stream, ratio):
        media_height, media_width, _ = stream.image.shape
        media_scale = np.asarray([media_width, media_height, media_width, media_height])
        for output in inference_results:
            self.logger.debug(f'output shape: {output.shape}')
            boxes, scores, class_indices = self.postprocess(output, (self.MODEL_DIM, self.MODEL_DIM), ratio)
            self.logger.debug(f'boxes shape: {boxes.shape}')
            self.logger.debug(f'scores shape: {scores.shape}')
            self.logger.debug(f'class_indices shape: {class_indices.shape}')
            for box, score, class_idx in zip(boxes, scores, class_indices):
                if score * 100 > self.threshold and class_idx in self.classids:
                    (left, top, right, bottom) = np.clip(box / media_scale, 0, 1)
                    self.logger.debug(f'box: {(left, top, right, bottom)}')
                    stream.add_rect(left, top, right, bottom)

    def postprocess(self, result, input_shape, ratio):
        self.logger.debug(f'postprocess was called, with a numpy array of shape {result.shape}')
        self.logger.debug(f'ratio: {ratio}')
        
        # source: https://github.com/Megvii-BaseDetection/YOLOX/blob/2c2dd1397ab090b553c6e6ecfca8184fe83800e1/demo/ONNXRuntime/onnx_inference.py#L73
        input_size = input_shape[-2:]
        predictions = demo_postprocess(result, input_size)
        predictions = predictions[0] # TODO: iterate through eventual batches
        self.logger.debug(f'predictions shape: {predictions.shape}')
        
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= ratio
        
        self.logger.debug(f'boxes_xyxy shape: {boxes_xyxy.shape}')
        self.logger.debug(f'scores shape: {scores.shape}')
        
        # TODO: get nms params from application interface
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        
        boxes = final_boxes
        scores = final_scores
        class_indices = final_cls_inds.astype(int)
        self.logger.debug(f'objects found: {len(boxes)}')
        return boxes, scores, class_indices

# Utility functions

def get_logger(name=__name__, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = RotatingFileHandler("/opt/aws/panorama/logs/app.log", maxBytes=100000000, backupCount=2)
    formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def main():
    logger = get_logger(level=logging.DEBUG)
    try:
        logger.info('INITIALIZING APPLICATION')
        app = Application(logger)
        logger.info('PROCESSING STREAMS')
        while True:
            app.process_streams()
    except:
        logger.exception('Exception during processing loop.')

main()
