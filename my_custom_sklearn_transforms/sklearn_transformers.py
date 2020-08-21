from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd


# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')

class TransformNulls(BaseEstimator, TransformerMixin):
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        data = X.copy()
        data.loc[data.NOTA_MF>10, 'NOTA_MF'] = 10
        media_nota_go = data['NOTA_GO'].mean()
        data.update(data['NOTA_GO'].fillna(media_nota_go))
        return data
    
class Balanceamento(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X,y):
        smt = SMOTE()
        X, y = smt.fit_sample(X, y)
        return X, y 