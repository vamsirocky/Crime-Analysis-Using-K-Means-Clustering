#import libraries-------------------------------------------------
import pandas as pd
import warnings 
warnings.filterwarnings("ignore")
from knn import *
from DT import *
from RF import *
from sklearn.datasets import make_blobs


##1.data slection---------------------------------------------------
#def main():
dataframe=pd.read_csv("Dataset1.csv")
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


KNN()
RF()
DT()






























































