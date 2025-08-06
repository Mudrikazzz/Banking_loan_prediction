import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pickle

df=pd.read_csv(r"C:\Users\hp\Downloads\classes\credit_risk_dataset.csv")
df["loan_grade"] = df["loan_grade"].map({"A":1,"B":2,"C":3,"D":4,"E":5,"F":6,"G":7})
df=df.drop(index=[81, 183, 575, 747, 32297])
df=df.drop(index = [0, 210])
df=df.dropna()
X=df.drop("loan_status",axis = 1)
y = df["loan_status"]
num_col = X.select_dtypes(include = ["int","float"]).drop("loan_grade",axis = 1).columns
cat_col = X.select_dtypes(include = ["object","category"]).columns
num_tranformer = StandardScaler()
cat_transformer = OneHotEncoder(drop = "first")
preprocessor = ColumnTransformer(transformers = [("num",num_tranformer,num_col),\
                                 ("cat",cat_transformer,cat_col)],
                                remainder='passthrough')
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,stratify=y,random_state = 13)
model_pipeline = Pipeline([("preprocesser",preprocessor),("model",RandomForestClassifier(random_state=42))])
model_pipeline.fit(X_train,y_train)
with open(r"C:\Users\hp\Downloads\classes\Demo\loan_predict.pkl","wb") as file:
    pickle.dump(model_pipeline,file)