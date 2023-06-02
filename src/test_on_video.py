import argparse
from pathlib import Path

import cv2
import pandas as pd
import yaml

from agent import AI

VIDEO_PATH = Path("videos")
VIDEO_PATH.mkdir(exist_ok=True)


def main(csv_path: str, stream: bool):
    with open("config.yml", "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
            return
        
    # Initialize video writer object
    output = cv2.VideoWriter(str(VIDEO_PATH / f"{len(list(VIDEO_PATH.iterdir())):04}.avi"),
                            cv2.VideoWriter_fourcc('M','J','P','G'), 15, (224, 224))

    df = pd.read_csv(csv_path, names=["id", "forward", "left"])
    df['predicted_forward'] = 0
    df['predicted_left'] = 0
    folder = Path(csv_path[:-4])
    
    ai = AI(config=config)
    
    for index, row in df.iterrows():
        img_id, true_forward, true_left = int(row['id']), row['forward'], row['left']
        img_path = folder / f'{img_id:04}.jpg'
        img = cv2.imread(str(img_path))
        forward, left = ai.predict(img)
        df.loc[index, 'predicted_forward'] = forward
        df.loc[index, 'predicted_left'] = left
        
        text1 = f"True: {true_forward:.5f}, {true_left:.5f}"
        text2 = f"Pred: {forward:.5f}, {left:.5f}"
        cv2.putText(img, text1, (5,20), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(img, text2, (5,35), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (0, 0, 255), 1, cv2.LINE_AA)
        
        output.write(img)
        if stream:
            cv2.imshow('Drive', img)
            key = cv2.waitKey(20)
        
            if key == ord('q'):
                break

    filename = csv_path.split('/')[-1][:-4]
    df.to_csv('../predictions/' + filename+'_predicted.csv', index=False)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Tool for debuggin our model')
    parser.add_argument('-f', '--csv_path', type=str, help='path to csv with drive data',
                        default= "..\\put_jetbot_dataset\\dataset\\1652875851.3497071.csv")
    parser.add_argument('-s', '--stream', action="store_true",
                        help='whether to stream video to the screen')
    args = parser.parse_args()

    main(args.csv_path, args.stream)
