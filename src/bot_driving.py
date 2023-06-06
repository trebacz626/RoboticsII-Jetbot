import cv2
import yaml

from agent import AI
from PUTDriver import PUTDriver, gstreamer_pipeline


def main():
    with open("config.yml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    driver = PUTDriver(config=config)
    ai = AI(config=config)

    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0, display_width=224, display_height=224), cv2.CAP_GSTREAMER)

    # model warm-up
    ret, image = video_capture.read()
    if not ret:
        print(f'No camera')
        return
    
    _ = ai.predict(image)

    input('Robot is ready to ride. Press Enter to start...')

    BUFFER_SIZE = 3
    forward, left = [0] * BUFFER_SIZE, [0] * BUFFER_SIZE
    while True:
        print(f'Forward: {forward[0]:.4f}\tLeft: {left[0]:.4f}')
        driver.update(forward[0], left[0])

        ret, image = video_capture.read()
        if not ret:
            print(f'No camera')
            break
        new_forward, new_left = ai.predict(image)
        if new_left < 0 and new_left > -0.45:
            new_left = 0
        new_left = 0.6 * new_left + 0.4 * left[-1]
        left.append(new_left)
        forward.append(new_forward)
        left = left[1:]
        forward = forward[1:]


if __name__ == '__main__':
    main()
