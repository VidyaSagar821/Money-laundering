import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))
import pandas as pd
from data_ingestion.data_loader_training import TrainingDataGetter
from data_preprocessing.preprocessing import Preprocess
from FeatureGeneration.generateFeatures import GenerateNewFeatures
from src.utils.clustering import Cluster
import argparse 
from src.utils.models import Trainmodel
from bestmodels.bestmodel import Getmodel
from log.logger import Logger
from DataValidation.TrainingDataValidation import ValidateFile
import pickle


class Training:

   def __init__(self, file_data, config_path):

      self.file_data = file_data
      self.log_obj = Logger('Generatedlogs')
      self.logger = self.log_obj.logging()
      self.validator = ValidateFile(self.file_data)
      self.config_path = config_path
      self.loader = TrainingDataGetter(self.file_data, self.config_path)

   def train(self):
      
      if self.loader.validate_before_load():
         #self.loader.load_data_in_db()
         data = self.loader.get_data()         

         num_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest',
                     'newbalanceDest']

         MONEY_OUT = ['PAYMENT','TRANSFER','CASH_OUT','DEBIT']
         MONEY_IN = ['CASH_IN']

         ## creating new features
         featuresgen = GenerateNewFeatures(data)
         featuresgen.make_date_columns('step')
         featuresgen.make_diff_columns(MONEY_IN, MONEY_OUT)

         ## Preprocessing data
         preprocess = Preprocess(data)
         preprocess.drop_columns(['isFlaggedFraud', 'step'])     
         data = preprocess.encode_categorical_columns()
         preprocess.drop_columns(['nameOrig','nameDest'])      
         data = preprocess.scalenumericvalues(num_cols)

         ## splitting and sampling
         X_train, X_val, y_train, y_val = preprocess.split_data(0.3, 'isFraud')
         X_train_resampled, y_train_resampled = preprocess.resampledata(0.5, X_train, y_train)
         
         X_train_resampled['target'] = y_train_resampled
         X_val['target'] = y_val

         # clustering the sampled data
         cluster_obj = Cluster(X_train_resampled, 10, self.config_path)
         cluster_numbers = cluster_obj.getclusternumber('target')
         cluster_obj.tagclusternumbers(cluster_numbers, 'target')
         
         for cluster in range(0, cluster_numbers):
            
            cluster_train_x = X_train_resampled[X_train_resampled['cluster'] == cluster].drop(['cluster','target'], axis = 1)
            cluster_train_y = X_train_resampled[X_train_resampled['cluster'] == cluster]['target']

            model_obj = Trainmodel(10, cluster, cluster_train_x, cluster_train_y, self.config_path)
            rf = model_obj.RandomForestClassifierTrain()
            xgb = model_obj.XGBoostClassifierTrain()

         bestmodel = Getmodel(X_val, self.config_path)
         bestmodel.storebestmodel('target')

      
if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config.yaml")
    parsed_args = args.parse_args()
    Train = Training('data_ingestion/Dataset.csv', config_path=parsed_args.config)
    Train.train()


