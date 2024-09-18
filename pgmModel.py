import networkx as nx
import pandas as pd
from torch_geometric.utils.convert import from_networkx
import matplotlib.pyplot as plt
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)


df = pd.read_csv('full_dataset.csv',delimiter=',')
print(df.columns)
X = df[['T3','TT4','TSH','goitre']]
y = df['classes']

print(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(y_train.shape)


model_NB = GaussianNB()

model = RandomForestClassifier(max_depth=5,criterion='gini')
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_pred, y_test)

confu = confusion_matrix(y_pred,y_test)
print(accuracy)
print(confu)





# y = df[8]
# X = df.drop(8,axis=1)
#
# X = pd.get_dummies(X)
# X = X*1
# y[0] = y[0].map({'y': 1, 'n': 0})
#
# print(y.columns)
# print(X)
# print(y)

# data = pd.DataFrame(data={'A':[0,0,1,0,0,0,0],'B':[0,0,1,0,0,0,0],'C':[0,0,0,1,1,0,0],'D':[0,0,0,0,0,1,0],'E':[0,0,0,0,0,0,0],'F':[0,0,0,0,0,0,1],'G':[0,0,0,0,0,0,0]})
#
# model = BayesianNetwork([('A','C'),('B','C'),('C','D'),('C','E'),('D','F'),('F','G')])
# model.fit(data)
#
#
# print(model.is_dconnected('A','B',{'C'}))
#
# print(model.active_train_nodes('A',observed={'C'}))
#

