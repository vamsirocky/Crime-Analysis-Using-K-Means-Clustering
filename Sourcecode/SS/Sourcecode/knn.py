def KNN():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    from sklearn import metrics
    import pandas as pd

    dataframe=pd.read_csv("Dataset1.csv")
    dataframe_2=dataframe.fillna(0)
    from sklearn import preprocessing
    label_encoder = preprocessing.LabelEncoder()
    dataframe_2['Area_Name']= label_encoder.fit_transform(dataframe_2['Area_Name']) 
    dataframe_2['Group_Name']= label_encoder.fit_transform(dataframe_2['Group_Name']) 
    dataframe_2['Sub_Group_Name']= label_encoder.fit_transform(dataframe_2['Group_Name'])


    x=dataframe_2.drop('Loss_of_Property_Above_100_Crores',axis=1)
    y=dataframe_2.Loss_of_Property_Above_100_Crores

    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 42)
    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)
    knn_prediction = knn.predict(x_test)
    Result_4=metrics.accuracy_score(y_test,knn_prediction)*100
    print()
    print("---------------------------------------------------------------------")
    print("KNN ")
    print()
    print("Knn Acuracy is :",Result_4,'%')
    print(metrics.classification_report(y_test , knn_prediction))
    from sklearn.metrics import confusion_matrix
    print("Confusion Matrix:")
    cm=confusion_matrix(y_test, knn_prediction)
    print(cm)
    print("-------------------------------------------------------")
    print()
    import matplotlib.pyplot as plt
    plt.imshow(cm, cmap='binary')
    import seaborn as sns
    sns.heatmap(cm, annot = True, cmap ='plasma',
            linecolor ='black', linewidths = 1)
    plt.show()
    from sklearn.metrics import roc_curve

    fpr, tpr,thersholds = roc_curve(y_test, knn_prediction,pos_label=1)
    plt.plot(fpr, tpr, marker='.', label='KNN')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    