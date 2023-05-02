# Basic import 
import os
import sys
import numpy as np
import pandas as np

from dataclasses import dataclass

from sklearn.linear_model import LinearRegression , Ridge , Lasso , ElasticNet
from src.utils import save_object
from src.exception import CustomException
from src.logger import logging 
from src.utils import evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts' , 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_training(self , train_array , test_array):
        try:
            logging.info('Splitting Dependent and Inpendent Variables from train and test')
            x_train , y_train , x_test , y_test = (
                train_array[:,:-1],
                train_array[:, -1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                'LinearRegression' : LinearRegression() ,
                'Lasso' : Lasso(),
                'Ridge':Ridge(),
                'ElasticNet' : ElasticNet()
            }
            model_report:dict=evaluate_model(x_train, y_train , x_test , y_test , models)
            print(model_report)
            print('\n=========================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # to get best model score from dictionar
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f'Best model found , model name : {best_model_name} , R2 score : {best_model_score}')
            print('\n==================================================================\n')
            logging.info(f'Best model found , model name : {best_model_name} , R2 score : {best_model_score}')
            logging.info('hyperparameter tuning started for catboost')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )


        except Exception as e :
            logging.info('Exception occured at model training')
            raise CustomException (e , sys)
        