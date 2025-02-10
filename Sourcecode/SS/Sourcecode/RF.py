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
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt

    x, y_true = make_blobs(n_samples=300, centers=4,cluster_std=0.60, random_state=0)
    plt.scatter(x[:, 0], x[:, 1], s=50);

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(x)
    y_kmeans = kmeans.predict(x)

    plt.scatter(x[:, 0], x[:, 1], c=y_kmeans, s=50, cmap='viridis')

    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

    plt.title("k-means")
    plt.show()
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(x,y_kmeans,test_size = 0.25,random_state = 42)
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
    # y_test=y_test>0.5
    # rf_prediction=rf_prediction>0.5
    fpr, tpr, _ = roc_curve(y_test, rf_prediction)
    plt.plot(fpr, tpr, marker='.', label='RF')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()
