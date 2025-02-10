def DT():
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
    from sklearn.tree import DecisionTreeClassifier 
    dt = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5)
    dt.fit(x_train, y_train)
    dt_prediction=dt.predict(x_test)
    print()
    print("---------------------------------------------------------------------")
    print("Decision Tree")
    print()
    Result_2=accuracy_score(y_test, dt_prediction)*100
    print(metrics.classification_report(y_test,dt_prediction))
    print()
    print("DT Accuracy is:",Result_2,'%')
    print()
    print("Confusion Matrix:")
    from sklearn.metrics import confusion_matrix
    cm1=confusion_matrix(y_test, dt_prediction)
    print(cm1)
    print("-------------------------------------------------------")
    print()
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.heatmap(cm1, annot = True, cmap ='plasma',
            linecolor ='black', linewidths = 1)
    plt.show()
    #ROC graph
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_test, dt_prediction)
    plt.plot(fpr, tpr, marker='.', label='DT')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()
