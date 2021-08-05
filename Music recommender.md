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

#### *I.Popularity Based Recommender*
* Songs sorted according to their popularity.
* For each user, recommend the songs in order of  popularity, except those already in the user’s profile.
* Non personalized Recommender.
* Easy to implement.

#### *II.Personalized Recommenders*
* Collaborative Filtering
* Content Based Filtering
![personalized recommender.png](https://github.com/DeekshithaKusupati/content/blob/main/personalized%20recommender.png)

#### *1.Collaborative Filtering*
Item-item filtering approach involves defining a co-occurrence matrix based on a song a user likes. We are seeking to answer a question, for each song, what a number of time a user, who have listened to that song, will also listen to another set of other songs. To further simplify this, based on what you like in the past, what other similar song that you will like based on what other similar user have liked. Let’s apply this to our code. First we create an instance item similarity based recommender class and feed it with our training data.
* Match people with similar interests as a basis for recommendation. 
* User based collaborative recommendation model is designed.
* Users who listen to the same songs in the past tend to have similar interests and will probably listen to the same songs in future.

#### *2.Content Based Filtering*
Recommendations done using content-based recommenders can be seen as a user-specific classification problem. This classifier learns the user’s likes and dislikes from the features of the song.
* Based on music description and user’s preference profile.
* Not based on choices of other users with similar interests.
* We make recommendations by looking for music whose features are very similar to the tastes of the user.
* Majority of songs have too few listeners, so difficult to “collaborate”. So we can use content based filtering.

## *Implementation*

### This repo is divided into the following two packages that contains the following files:

#### I. Content-based recommendation system:

a. A jupyter notebook named ```content_based_music_recommender``` that contains the code and analysis for the recommedation system.
b. A CSV file named ```abcdata``` and ```song_data``` containing the data for the songs used in the system.

#### II. Collaborative recommendation system:

a. A jupyter notebook named ```collaborative_based_music_recommender``` that contains the code and analysis for the recommedation system.
b. A CSV file named ```song_data``` containing the data for the songs used in the system.



### *By : Kusupati Deekshitha , Subham Nanda*
 
  
