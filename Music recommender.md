# *MUSIC RECOMMENDATION SYSTEM* 

`*Table of Contents*`

* _*Introduction*_
* _*Flow Chart*_
* _*Ideas for Recommendation*_
* _*Implementation using Python*_

## *Introduction*
Now a days there are millions of songs to choose from. The number of songs available exceeds the listening capacity of single individual. People sometimes feel difficult with this problem. Moreover, music service providers need an efficient way to manage songs and help their costumers to discover music by giving quality recommendation. Thus, there is a strongneed of a good recommendation system.

In this project, we have designing, implementing and analyzing a music recommendation system. We used Million Song Dataset to find correlations between users and songs and to learn from the previous listening history ofusers to provide recommendations for songs which users would prefer to listen most in future. 

We have implemented various algorithms such as popularity based model that recommends the popular songs, collaborative filtering based model which match people with similar interests as a basis for    recommendation and content based model which recommends the songs of the artists and also based on text(or) keywords that user previously listened to.

#### *What it is?*
This project is basically a Music Recommender which is built using Machine Leraning, Deep Learning and system is implemented in Jupyter Notebook using Python. In this we designed Music Recommendation system which predict songs and recommend them to the user. 

#### *Why it is?*
Too many choices can overwhelm users. If offered too many options, the user may not buy anything. Streaming services like Spotify have massive catalogs. Identifying the tracks a user might like and recommending the product they might like is crucial for their business. T
his project helps 
* people to make decisions in listening music.
* Makes work easy and saves time. 
* Recommend based on different modalities(User feedback, Text,Audio..etc).

## *Flow Chart*
![flowchart.png](https://github.com/DeekshithaKusupati/content/blob/main/flowchart.png)

## *Ideas for Recommendation*

#### *Popularity Based Recommender*
* Songs sorted according to their popularity.
* For each user, recommend the songs in order of  popularity, except those already in the user’s profile.
* Non personalized Recommender.
* Easy to implement.

### *Personalized Recommenders*
* Collaborative Filtering
* Content Based Filtering
![personalized recommender.png]()

#### *1.Collaborative Filtering*


In order to understand let's consider a example:  
We are given two groups(A and B), where A consists of circles and B consists of squares.  
Now a new data point as a triangle is introduced into the groups. So now we have to decide which group(A or B) the triangle belongs to.

![flowchart.png](https://github.com/DeekshithaKusupati/Intern-Work/blob/main/int-ml-3/KNN/images/Picture1.png)

Let the value of ‘K’ which is the number of nearest neighbors be taken as 4, that is it will take 4 neighbors from the two groups which are closest to the triangle.  
By using the Euclidean or Manhattan distance, we have to calculate the closeness or distance between the triangles and the 4 closest neighbors. 
After finding the 4 neighbors, suppose we get 3 circles(i.e group A) and 1 square(i.e group B). This shows that the new datapoint(i.e triangle) will be classified into group A of circles. 

![Picture2.png](https://github.com/DeekshithaKusupati/Intern-Work/blob/main/int-ml-3/KNN/images/Picture2.png)

#### *KNN for Classification*
A class label which is assigned to the majority of K Nearest Neighbors from the dataset is classified as a predicted class for the new data point.
#### *KNN for Regression*
The calculated Mean or median of continuous values which is  assigned to K Nearest Neighbors from dataset is the predicted continuous value for the new data point.

## *How to select the value ok K in KNN- Algorithm:*
There is no optimal number of neighbors that suits all kind of data sets since each dataset has their own requirements. In the case of a large number of neighbors it is computationally expensive and a small number of neighbors, the noise will have a higher influence on the result.

![knn.PNG](https://github.com/DeekshithaKusupati/Intern-Work/blob/main/int-ml-3/KNN/images/knn.png)

In real life time scenarios, KNN is widely used as it does not make any underlying assumptions about the distributions of data. So it is non-parametric.

As K-NN algorithm makes highly accurate predictions it can be used for applications which require high accuracy. The quality of predictions is dependent on the distance measure. Thus, this algorithm is suitable for applications for which you have sufficient domain knowledge so that it can help you select an appropriate measure. Domain knowledge is very useful in choosing the K value.

To select the right value of K for your data, we have to run the KNN algorithm several times with different values of K and select the K that reduces the number of errors .

## *Advantages of KNN- Algorithm* 
* It is easy to implement.
* For noisy training data it is robust.
* If the training data is large it can be more effective.

## *Disadvantages of KNN- Algorithm* 
* It may be complex some time as it always needs to determine the value of K .
* Because of calculating the distance between the data points for all the training samples, the computation cost is high.
* KNN works well with a small number of input variables, but struggles when the number of inputs is very large.

## *Python Implementation*
Implementation of the K Nearest Neighbor algorithm using Python’s scikit-learn library:
#### *Step 1: Get and prepare data*
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import metrics   
```
After loading important libraries, we create our data using sklearn.datasets with 200 samples, 8 features, and 2 classes. Then data is split into the train(80%) and test(20%) data and scaled using StandardScaler.
```python
X,Y=make_classification(n_samples= 200,n_features=8,n_informative=8,n_redundant=0,n_repeated=0,n_classes=2,random_state=14)
X_train, X_test, y_train, y_test= train_test_split(X, Y, test_size= 0.2,random_state=32)
sc= StandardScaler()
sc.fit(X_train)
X_train= sc.transform(X_train)
sc.fit(X_test)
X_test= sc.transform(X_test)
X.shape
```
```python
output = (200, 8)
 ```
 #### *Step 2: Find the value for K*
 For choosing the K value, we use error curves and K value with optimal variance, and bias error is chosen as K value for prediction purposes. With the error curve plotted below, we choose K=7 for the prediction
 ```python
error1= []
error2= []
for k in range(1,15):
    knn= KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred1= knn.predict(X_train)
    error1.append(np.mean(y_train!= y_pred1))
    y_pred2= knn.predict(X_test)
    error2.append(np.mean(y_test!= y_pred2))
# plt.figure(figsize(10,5))
plt.plot(range(1,15),error1,label="train")
plt.plot(range(1,15),error2,label="test")
plt.xlabel('k Value')
plt.ylabel('Error')
plt.legend()
```

![error.png](https://github.com/DeekshithaKusupati/Intern-Work/blob/main/int-ml-3/KNN/images/error.png)

#### *Step 3: Predict*
In step 2, we have chosen the K value to be 7. Now we substitute that value and get the accuracy score = 0.9 for the test data.
```python
knn= KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,y_train)
y_pred= knn.predict(X_test)
metrics.accuracy_score(y_test,y_pred)
```
```python
output = 0.9
```
## *Pseudocode for K Nearest Neighbor (classification)* 
This is pseudocode for implementing the KNN algorithm from scratch:

1. Load the training data.
2. Prepare data by scaling, missing value treatment, and dimensionality reduction as required.
3. Find the optimal value for K:
4. Predict a class value for new data:
    * Calculate distance(X, Xi) from i=1,2,3,….,n.  
      where X= new data point, Xi= training data, distance as per your chosen distance metric.
    * Sort these distances in increasing order with corresponding train data.
    * From this sorted list, select the top ‘K’ rows.
    * Find the most frequent class from these chosen ‘K’ rows. This will be your predicted class.
   
## *Practical Applications of K-NN*
Now that we have we have seen how KNN works, let us look into some of the practical applications of KNN.

* Recommending products to people with similar interests, recommending movies and TV shows as per viewer’s choice and interest.
* Recommending hotels and other accommodation facilities while you are travelling based on your previous bookings.
* Some advanced examples could include handwriting detection (like OCR), image recognition and even video recognition.

### *By : Kusupati Deekshitha , Subham Nanda*
 
  
