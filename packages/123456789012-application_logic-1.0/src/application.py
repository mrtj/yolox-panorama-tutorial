import logging
from logging.handlers import RotatingFileHandler

import panoramasdk
import numpy as np

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
        self.postprocess(inference_results, stream, ratio)

    def preprocess(self, img, size):
        return ((np.ones((size[0], size[1], 3), dtype=np.uint8) * 114), 1.0)

    def postprocess(self, inference_results, stream, ratio):
        pass

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
