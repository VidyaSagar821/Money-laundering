import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))
import pandas as pd
import argparse 
from data_ingestion.data_loader_prediction import PredictionDataGetter
from data_preprocessing.preprocessing import Preprocess
from src.utils.common import read_config
from log.logger import Logger
from FeatureGeneration.generateFeatures import GenerateNewFeatures
from src.utils.clustering import Cluster
from src.utils.models import Trainmodel
import pickle


class Prediction:

   def __init__(self, file_data, config_path):

      self.data = file_data
      self.log_obj = Logger('Generatedlogs')
      self.logger = self.log_obj.logging()
      self.config_path = config_path
      self.loader = PredictionDataGetter(self.data, self.config_path)

      
   def predict(self):

      try:
         
         if self.loader.validate_before_load():
            self.loader.load_data_in_db()

            num_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest','newbalanceDest']
            MONEY_OUT = ['PAYMENT','TRANSFER','CASH_OUT','DEBIT']
            MONEY_IN = ['CASH_IN']

            ## creating new features
            featuresgen = GenerateNewFeatures(self.data)
            featuresgen.make_date_columns('step')
            featuresgen.make_diff_columns(MONEY_IN, MONEY_OUT)

            ## Preprocessing data
            preprocess = Preprocess(self.data)
            preprocess.drop_columns(['isFlaggedFraud', 'step'])     
            self.data = preprocess.encode_categorical_columns()
            preprocess.drop_columns(['nameOrig','nameDest'])      
            self.data = preprocess.scalenumericvalues(num_cols)

            with open('models/kmeans_model.pkl', 'rb') as f:
               k = pickle.load(f)
               clusters = k.predict(self.data)
               self.data['cluster'] = clusters
               cluster_numbers = len(self.data['cluster'].unique())
            
            final_target = pd.DataFrame(columns = ['Target'])

            with open('bestmodels/bestmodels.txt', 'r') as bm:
               modelfile = bm.read()

            model_list = modelfile.split('\n')

            for cluster in range(0, cluster_numbers):

               for model in model_list:
                  if str(cluster) in model:
                     with open(f'models/{model}', 'rb') as model:
                        best_model = pickle.load(model)
               
               clustered_data = self.data[self.data['cluster'] == cluster].drop('cluster', axis = 1)
               cluster_prediction = best_model.predict(clustered_data)
               clustered_data['target'] = cluster_prediction
               final_target = pd.concat([final_target, clustered_data['target']], axis = 1)
               
            final_target = final_target.drop('Target', axis = 1)
            return final_target

      except Exception as e:
         self.logger.error('Error while generating prediction data: ' + str(e))
         sys.exit(1)
     