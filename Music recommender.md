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
![personalized recommender.png](https://github.com/DeekshithaKusupati/content/blob/main/personalized%20recommender.png)

#### *1.Collaborative Filtering*
Item-item filtering approach involves defining a co-occurrence matrix based on a song a user likes. We are seeking to answer a question, for each song, what a number of time a user, who have listened to that song, will also listen to another set of other songs. To further simplify this, based on what you like in the past, what other similar song that you will like based on what other similar user have liked. Let’s apply this to our code. First we create an instance item similarity based recommender class and feed it with our training data.
* Match people with similar interests as a basis for recommendation. 
* User based collaborative recommendation model is designed.
* Users who listen to the same songs in the past tend to have similar interests and will probably listen to the same songs in future.

#### *1.Content Based Filtering*
Recommendations done using content-based recommenders can be seen as a user-specific classification problem. This classifier learns the user’s likes and dislikes from the features of the song.
* Based on music description and user’s preference profile.
* Not based on choices of other users with similar interests.
* We make recommendations by looking for music whose features are very similar to the tastes of the user.
* Majority of songs have too few listeners, so difficult to “collaborate”. So we can use content based filtering.

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
 
  
