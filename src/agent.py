import cv2
import numpy as np
import onnxruntime as rt


class AI:
    def __init__(self, config: dict):
        self.path = config['model']['path']
        self.img_size = config['model']['img_size']
        self.sess = rt.InferenceSession(self.path, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
 
        self.output_name = self.sess.get_outputs()[0].name
        self.input_name = self.sess.get_inputs()[0].name

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        #print(img.shape)
        img  = cv2.resize(img, (self.img_size, self.img_size)) # maybe this should be added to the config
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        img = np.expand_dims(img, 0) / 255
        img = img.astype(np.float32)
        #print(img.min(), img.max())
        #print(img.shape)

        return img.astype(np.float32)

    def postprocess(self, detections: np.ndarray) -> np.ndarray:
        #print(detections.shape)
        detections = detections[0] # it will have a shape (1, 2) at the input
        detections = np.minimum(detections, np.array([1.0, 1.0]))
        detections = np.maximum(detections, np.array([-1.0,-1.0]))
        #print(detection)
        return detections.astype(np.float32)

    def predict(self, img: np.ndarray) -> np.ndarray:
        inputs = self.preprocess(img)
        assert inputs.dtype == np.float32
        assert inputs.shape == (1, 3, self.img_size, self.img_size)
        
        detections = self.sess.run([self.output_name], {self.input_name: inputs})[0]
        outputs = self.postprocess(detections)
        assert outputs.dtype == np.float32
        assert outputs.shape == (2,)
        assert outputs.max() <= 1.0
        assert outputs.min() >= -1.0
        #print(outputs)
        return outputs
