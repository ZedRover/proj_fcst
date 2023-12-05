from autogluon.tabular import TabularDataset, TabularPredictor
import utils 
import pandas as pd 
import numpy as np 


predictor = TabularPredictor(label='y_1').fit(train_data=utils.df_train.drop([
    'y_2','benchmark_yhat'
],axis=1))
# predictions = predictor.predict(utils.df_test )