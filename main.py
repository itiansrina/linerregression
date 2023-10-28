import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

df = pd.read_csv("https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv")
#print(df.head())
#Since we can see that there are yes and no in target we will 
# change it to 0,1
df.variety.replace(['Setosa','Virginica','Versicolor'],value=[0,1,2],inplace=True)
print(df.head())


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(df.drop('variety',axis=1))
scaled_features = scaler.transform(df.drop('variety',axis=1))
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
print(df_feat.head())
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['variety'],test_size=0.30)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
print(pred)
print(("Accuracy is"), accuracy_score(y_test,pred)*100)
