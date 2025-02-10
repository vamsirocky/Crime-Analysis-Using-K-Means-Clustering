#import libraries-------------------------------------------------
import pandas as pd
import warnings 
warnings.filterwarnings("ignore")
from knn import *
from DT import *
from RF import *


##1.data slection---------------------------------------------------
#def main():
dataframe=pd.read_csv("Dataset11.csv")
print("---------------------------------------------")
print()
print("Data Selection")
print("Samples of our input data")
print(dataframe.head(10))
print("----------------------------------------------")
print()
    
    
 #2.pre processing--------------------------------------------------
#checking  missing values 
print("---------------------------------------------")
print()
print("Before Handling Missing Values")
print()
print(dataframe.isnull().sum())
print("----------------------------------------------")
print() 
    
print("-----------------------------------------------")
print("After handling missing values")
print()
dataframe_2=dataframe.fillna(0)
print(dataframe_2.isnull().sum())
print()
print("-----------------------------------------------")
    
#label encoding
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder() 
print("--------------------------------------------------")
print("Before Label Handling ")
print()
print(dataframe_2.head(10))
print("--------------------------------------------------")
print()
dataframe_2['Area_Name']= label_encoder.fit_transform(dataframe_2['Area_Name']) 
dataframe_2['Group_Name']= label_encoder.fit_transform(dataframe_2['Group_Name']) 
dataframe_2['Sub_Group_Name']= label_encoder.fit_transform(dataframe_2['Group_Name']) 
print("-------------------------------------------------------")
print("After Label Encoding")
print()
print(dataframe_2.head(10))
print("---------------------------------------------------------")
print()
    
#3.Data splitting--------------------------------------------------- 
x=dataframe_2.drop('Loss_of_Property_Above_100_Crores',axis=1)
y=dataframe_2.Loss_of_Property_Above_100_Crores
    
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 42)

##4.feature selection------------------------------------------------
##kmeans
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

x, y_true = make_blobs(n_samples=300, centers=4,cluster_std=0.60, random_state=0)
plt.scatter(x[:, 0], x[:, 1], s=50);

kmeans = KMeans(n_clusters=4)
kmeans.fit(x)
y_kmeans = kmeans.predict(x)

plt.scatter(x[:, 0], x[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

plt.title("k-means")
plt.show()


dataframe=pd.read_csv("Dataset11.csv")
dataframe_2=dataframe.fillna(0)
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
dataframe_2['Area_Name']= label_encoder.fit_transform(dataframe_2['Area_Name']) 
dataframe_2['Group_Name']= label_encoder.fit_transform(dataframe_2['Group_Name']) 
dataframe_2['Sub_Group_Name']= label_encoder.fit_transform(dataframe_2['Group_Name'])


x=dataframe_2.drop('Loss_of_Property_Above_100_Crores',axis=1)
y=dataframe_2.Loss_of_Property_Above_100_Crores

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.29,random_state = 10)

#---------------------------------------------------------------------------------------------
import numpy as np

x_train=np.expand_dims(x_train, axis=2)
x_test=np.expand_dims(x_test, axis=2)
y_train=np.expand_dims(y_train,axis=1)
y_test=np.expand_dims(y_test,axis=1)


"Hybird CNN and LSTM "
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

nb_out = 1
model = Sequential()
model.add(LSTM(input_shape=(8, 1), units=100, return_sequences=True))
model.add(Dropout(0.2))
model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=nb_out))
model.add(Activation("linear"))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

print(model.summary())
# fit the model
model.fit(x_train, y_train, epochs=10, batch_size=1, verbose=1)
Result_3=model.evaluate(x_train,y_train,verbose=1)[1]*100
#from sklearn.metrics import accuracy_score
from sklearn import metrics

CNNLSTM_prediction = model.predict(x_test)
#Result_3=accuracy_score(y_test, CNNLSTM_prediction.round())*100
from sklearn.metrics import confusion_matrix

print()
print("---------------------------------------------------------------------")
print("Hybird CNNLSTM")
print()
print(metrics.classification_report(y_test,CNNLSTM_prediction.round()))
print()
print("CNNLSTM Accuracy is:",Result_3,'%')
print()
print("Confusion Matrix:")
cm2=confusion_matrix(y_test, CNNLSTM_prediction.round())
print(cm2)
print("-------------------------------------------------------")
print()
import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(cm2, annot = True, cmap ='plasma',
        linecolor ='black', linewidths = 1)
plt.show()
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, CNNLSTM_prediction.round())
plt.plot(fpr, tpr, marker='.', label='Hybird CNNLSTM')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
