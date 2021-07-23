# *Underfitting* 

`*Table of Contents*`

* _*Introduction*_
* _*What is Underfitting ?*_
* _*Underfitting vs Overfitting*_
* _*How to avoid Underfitting ?*_
* _*Conclusion*_


## *Introduction*
   If we consider we are designing a machine learning model. A model is said to be a good machine learning model if it generalizes the input data to get the outcome in a proper way. This helps us in classification and also to make predictions in the future data that the data model has never seen.

   Now, if we want to check how well the machine learning model trains and generalizes to the new data. For that we have overfitting and underfitting, which are responsible for the poor performance of the machine learning algorithms.

## *What is Underfitting*
   Underfitting occurs in data science where a data model is unable to capture the underlying trend of the data. It results in generating a high error rate on both the training set and unseen data. It occurs when a model is very simple(the input features are not expressive enough), which results in model needing more time for training, less regularization and more input features.  when a model is underfitted, it will have training errors and poor performance. If a model is unable to generalize well to new data, then it cannot be good for classification or prediction tasks. A model is a good machine learning model if it generalizes any new input data from the problem every day to make predictions and classify data.
   Low variance and high bias are good indicators of underfitting.


## *Underfitting vs Overfitting*
  * To be simple, overfitting is the opposite of underfitting. It occurs when the model is overtrained or it contains too much complexity, because of this it results in high error rates on test data. Overfitting a model is more common than underfitting one.
  * As mentioned above, the model is underfitting when it performs poorly on the training data. This is because the model is unable to capture the relationship between the input examples and the target values accurately.
  * The model is overfitting your training data when you see that the model performs well on the training data but does not perform well on the evaluation data. This is because the model is unable to generalize to unseen examples and memorizing the data it has seen.
  * Underfitted models are usually easier to identify compared to overfitted ones as their behaviour can be seen while using training data set.
  
   Below is an illustration of the different ways a regression can potentially fit against unseen data:
   ![UNDERFITTING.PNG](https://github.com/DeekshithaKusupati/content/blob/main/UNDERFITTING.png)
   
## *How to avoid underfitting*
Since we can detect underfitting while using the training set, we can assist in establishing accurate relationship between the input and target variables. we can avoid underfitting and make more accurate predictions by maintaining required complexity.

Below are a few techniques that can be used to reduce underfitting:
 #### *1. Decrease regularization*
   Regularization is usually used to reduce the variance with a model by applying a penalty to the input parameters with the larger coefficients. There are a number of different methods in machine learning which helps to reduce the noise and outliers in a model. By decreasing the amount of regularization, increasing complexity and variation is introduced into the model for successful training of the model.
 #### *2. Increase the Duration of the training*
   Training the model for less time can also result in underfit model that is to try to train the model for more epochs. Ensuring that the loss is decreases gradually over the course of training. However, it is important to make sure that it should not lead to overtraining, and subsequently, overfitting. Finding the balance between the two will be key.
 #### *3. Feature selection*
   There will be a specific features in any model that can be used to determine a required outcome. If there are not enough predictive features present, then more features with greater importance or more features, should be introduced. This process can increase the complexity of the model, resulting in getting better training results.
## *Conclusion* 
 The get a good machine learning model with a good fit it should be between the underfitted and overfitted model, so that it can make accurate predictions.

 when we train our model for a particular time, the errors in the training data and testing data go down. But if we train the model for a long duration of time, then overfitting occurs which reduce the performance of model, as the model also learn the noise present in the dataset. The errors in the test dataset start increasing, so the point, just before the raising of errors, is the good point, and we can stop here for achieving a good model. 
 
### *By : Kusupati Deekshitha , Subham Nanda*
 
  
