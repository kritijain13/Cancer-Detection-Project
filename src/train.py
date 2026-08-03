# import os 
# import sys
# import mlflow
# mlflow.set_experiment("Cancer Detection Model Experiments")
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.naive_bayes import MultinomialNB
# from xgboost import XGBClassifier
# from sklearn.model_selection import train_test_split
# import pandas as pd
# from datapreprocessing.preprocess import Data
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score



# Path=r'C:\Users\kriti\CancerDetectionProject\data\Cancer_Data.csv'
# df=pd.read_csv(Path)

# x=df.drop(['diagnosis','id','Unnamed: 32'],axis=1)
# y=df['diagnosis']
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# preprocessor=Data()
# preprocessor_transformer=preprocessor.preprocess_data(x_train)

# x_train=preprocessor_transformer.fit_transform(x_train)

# model={
#     'logistic_regression': LogisticRegression(max_iter=100),
#     'decision_tree': DecisionTreeClassifier(),
#     'random_forest': RandomForestClassifier(),
#     'naive_bayes': MultinomialNB(),
#     'xgboost': XGBClassifier()
# }
# x_test=preprocessor_transformer.transform(x_test)
# for name,algo in model.items():

# # model=LogisticRegression()
# # model.fit(x_train,y_train)

# # x_test=preprocessor_transformer.transform(x_test)
# # y_pred=model.predict(x_test)

# # print('accuracy_score',accuracy_score(y_test,y_pred))



#     with mlflow.start_run():
#         model=LogisticRegression()
#         algo.fit(x_train,y_train)
      
#         y_pred=algo.predict(x_test)
#         acc = accuracy_score(y_test, y_pred)
#         precision = precision_score(y_test, y_pred, pos_label='M')
#         recall = recall_score(y_test, y_pred, pos_label='M')
#         f1 = f1_score(y_test, y_pred, pos_label='M')

#         #storing the model metrics
#         mlflow.log_metric('accuracy',acc)
#         mlflow.log_metric('prescison',precision)
#         mlflow.log_metric('recall',recall)
#         mlflow.log_metric('f1',f1)


#         #logging the model
#         mlflow.sklearn.log_model(algo,name=name)

#         #message
#         print(f'model save succesfully{name}')

    



    #  http://127.0.0.1:5000 
from sklearn.model_selection import train_test_split 
import pandas as pd 
import os 
import sys 
import mlflow

mlflow.set_experiment("Cancer Detection Model Experiment")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datapreprocessing.preprocess import Data 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
#pip install xgboost
Path=r'C:\Users\kriti\CancerDetectionProject\data\Cancer_Data.csv'
# df = pd.read_csv(PATH)
data =pd.read_csv(Path)

X = data.drop(['diagnosis', 'id', 'Unnamed: 32'], axis=1)
y = data['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = Data()
preprocessor_transformer = preprocessor.preprocess_data(X_train)

X_train = preprocessor_transformer.fit_transform(X_train)


models = {
    "Logistic Regression": LogisticRegression(max_iter=100),
    "Decision Tree" : DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(), 
    
}
X_test = preprocessor_transformer.transform(X_test)
for name, algo in models.items():
    # From here the model experiment is starting..
    with mlflow.start_run(run_name=name):
        
        algo.fit(X_train, y_train)
        
        y_pred = algo.predict(X_test)
        # Here we are calculatiing the metrics of the model 
        acc_score = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label='M')
        recall = recall_score(y_test, y_pred, pos_label='M')
        score = f1_score(y_test, y_pred, pos_label='M')

        # Storing the model metrics 
        mlflow.log_metric("accuracy", acc_score)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("F1 Score", score)

        # Logging the model 
        mlflow.sklearn.log_model(algo, name=name)

        # message 
        print(f'{name} has been saved successfully!')