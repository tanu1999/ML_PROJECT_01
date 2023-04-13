import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from src.components.dataingestion import DataIngestion
from src.components.datatransform import DataTransformation
from src.components.modeltrainer import ModelTrainer


if __name__=='__main__':
    obj=DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr,test_arr,_=data_transformation.initaite_data_transformation(train_data_path,test_data_path)
    model_trainer=ModelTrainer()
    model_trainer.initate_model_training(train_arr,test_arr)

