def RF():
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
    from sklearn.ensemble import RandomForestClassifier

    rf= RandomForestClassifier(n_estimators = 100)  
    rf.fit(x_train, y_train)
    rf_prediction = rf.predict(x_test)
    Result_3=accuracy_score(y_test, rf_prediction)*100
    from sklearn.metrics import confusion_matrix

    print()
    print("---------------------------------------------------------------------")
    print("Random Forest")
    print()
    print(metrics.classification_report(y_test,rf_prediction))
    print()
    print("Random Forest Accuracy is:",Result_3,'%')
    print()
    print("Confusion Matrix:")
    cm2=confusion_matrix(y_test, rf_prediction)
    print(cm2)
    print("-------------------------------------------------------")
    print()
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.heatmap(cm2, annot = True, cmap ='plasma',
            linecolor ='black', linewidths = 1)
    plt.show()
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_test, rf_prediction)
    plt.plot(fpr, tpr, marker='.', label='RF')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()
