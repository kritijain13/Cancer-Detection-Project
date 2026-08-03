from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import pandas as pd

class Data:
    #data read karna hain
    def preprocess_data(self,x):
        
        #seprating dependet and independent columns
        # x=df.drop(['diagnosis','id','Unnamed: 32'],axis=1)
        # y=df['diagnosis']

        # sepreat the categorical or numerical columns
        cat_col=x.select_dtypes(include='object').columns
        num_col=x.select_dtypes(exclude='object').columns

        #creating pipeline for the cat and num col
        cat_pipeline=Pipeline(
            steps=[('encoder',OneHotEncoder(handle_unknown='ignore')),
                   ('imputer',SimpleImputer(strategy='most_frequent'))]
        )

        num_pipeline=Pipeline(
            steps=[('imputer',SimpleImputer(strategy='median')),
                   ('scaler',StandardScaler())]
        )

        #columns transformers for both pipelines
        preprocessor=ColumnTransformer(
            transformers=[('numeric',num_pipeline,num_col),
                          ('categorical',cat_pipeline,cat_col)]
        )
        
        return preprocessor





    # missing value
    #  scaling
    # encoding
    