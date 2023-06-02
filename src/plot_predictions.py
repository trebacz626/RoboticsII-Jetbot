import argparse

import pandas as pd
import plotly.express as px


def main(csv_path):
    df = pd.read_csv(csv_path)
    #line plot forward and forward_predicted
    fig = px.line(df, x='id', y=['forward', 'predicted_forward'], title='Forward')
    fig.show()
    #line plot left and left_predicted
    fig = px.line(df, x='id', y=['left', 'predicted_left'], title='Left')
    fig.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Tool for debuggin our model')
    parser.add_argument('-f', '--csv_path', type=str, help='path to csv with drive data',
                        default= "..\\put_jetbot_dataset\\dataset\\1652875851.3497071.csv")
    args = parser.parse_args()

    main(args.csv_path)
