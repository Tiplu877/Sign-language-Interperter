import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import pickle

df = pd.read_csv('data.csv' , header=None)
y = df[0]
x = df.iloc[:,1:]
model = KNeighborsClassifier(n_neighbors=7)
model.fit(x,y)
pickle.dump(model, open('model.pkl', 'wb'))