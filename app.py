import streamlit as st
import pandas as pd
import argparse 
from prediction import Prediction
import warnings
warnings.filterwarnings('ignore')
import time


def readcsv(csv):
    df = pd.read_csv(csv)
    return df

def convert_df(df):
   return df.to_csv().encode('utf-8')

def combine_df(df1, df2):
    return pd.concat([df1, df2], axis=1)

def progress_bar(timesleep):
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(timesleep)
        my_bar.progress(percent_complete + 1)

def main(config_path):

    st.title('   Money Laundering Detection   ')
    st.image('image.jpg',width=None, use_column_width='always')
    st.markdown('Application to detect the irregularities in day to day transactions')
    file = st.file_uploader('Upload your csv file', type='csv')

    if file is not None:
        
        df = readcsv(file)
        initial_df = df.copy(deep = False) 
    
        predictor = Prediction(df, config_path)
        pred_data = predictor.predict()
        final_data = combine_df(initial_df, pred_data)

        target_file = convert_df(final_data)

        st.caption('Generating the Prediction File, Please wait')
        progress_bar(0.01)
        st.download_button( "Press to Download",target_file, "file.csv", "text/csv", key='download-csv')


if __name__ == '__main__':
    
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config.yaml")
    parsed_args = args.parse_args()
    main(config_path=parsed_args.config)