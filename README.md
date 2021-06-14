# datascience Project portfolio
Exploring my data science portfolio

## [Project 1: Heart attack  prediction: Overview](https://github.com/NaveenKumarMaurya/HeartAttackPrediction#heartattackprediction)
This is a simple classification problem trained on python scikit-learn library.The classification model takes 
the independent variable eg. age,sex,cholestrol,blood pressure etc.,from heart attack data set to predict 
whether the person will get heart attack or not.

In this project chi-squre test has been used for to checkfeature importance of categorical variableand independent t-test is used to compare the mean of variables
grouped on the basis of output category and by finding correlation among numerical variables to get the 
importance of each variable in deciding output.Features renaming also done with help of google,you can find [here](https://archive.ics.uci.edu/ml/datasets/Heart+Disease/).

We also use Pipeline method to apply 10 classifiction algorithms(1.logistic Regression 2.Decisiontreeclassifier 
3.Randomforestclassifier 4.GaussianNB 5.KNN 6.Gboost Classifier 7.AdaboostClassifier8.SGDClassifier 9.SVC 10.MLP Classifier)
to get the best accuracy which i have got in KNN=82.4%.Then i also apply for loop to get the best random
state producing good accuracy and then we have got accuracy of 90% with choosing appropriate n_nieghour parameter which is n_neighour=6.
After plotting AUC-ROC curve we have got the value AUC=93% which is a good value for a model.

![](https://github.com/NaveenKumarMaurya/my-datascience-project/blob/main/heart-attack-silent%20(1).jpg)

## [Project 2: Car Price Prediction : Overview](https://github.com/NaveenKumarMaurya/datascience-project-portfolio/blob/main/car%20price%20prediction.ipynb)
This is a regression based ML plroject build  on python scikit-learn library.The aim of this project is to predict the price of the car on the basis of its given independent feature eg. model, engine feul type, engine HP, engine cylinder, transmission type, driven_Wheels, number of door, market category, vehicle size, vehicle style,
highway MPG, city mpg, Popularity.

We applied 11 ML regression algorithms to build our model, then we get best accuracy in Extra tree regressor which is 97%.
After analysing the features and correlation we found 'Engine HP' is the most important factor for the price of the car.

![](https://github.com/NaveenKumarMaurya/datascience-project-portfolio/blob/main/car%20images.jpg)![](https://github.com/NaveenKumarMaurya/datascience-project-portfolio/blob/main/car%20feature%20importance.png)

## [Project 3: Concrete Compressive Strength : Overview](https://github.com/NaveenKumarMaurya/datascience-project-portfolio/blob/main/concrete-data-eda-model-acc-97.ipynb)
This is a Regression problem trained on python scikit-learn libarary.The data is related to civil engineering/Architecture where compressive strengh of material being used is import factor to determone the stability, sustainibility of building/bridge/construction.The target aim of the model is to predict the compressive strength of concrete on the basis of independent  variable -cement,Blast Furnace Slag,Fly Ash,Water,Superplasticizer,Coarse Aggregate,Fine Aggregate.

we used 12 regression algorithms to build our ML model where we have got the best result with Extra Tree Regressor having accuracy of 97%,RSME=4.08 which is very good for a model.

![](https://github.com/NaveenKumarMaurya/datascience-project-portfolio/blob/main/concrete%20%20image.jpg)

## [Project 4: Credit card customer segmentation:](https://github.com/NaveenKumarMaurya/datascience-project-portfolio/blob/main/Credit%20card%20customer%20segmentation.ipynb)
this is a unsupervised machine learning problem where diifferent segment or group of customer has to be made using K-mean clusterring and hierarchical clustering algorithm .
After taht we have to identify our target customer who has more potential to give profit.
You can visit  the data by clicking on the image below.

[![](https://github.com/NaveenKumarMaurya/datascience-project-portfolio/blob/main/Credit-cards%20images.jpg) ](https://www.kaggle.com/aryashah2k/credit-card-customer-data)

## [Project 5: Covid-19 death analysis:](https://github.com/NaveenKumarMaurya/datascience-project-portfolio/blob/main/covid-19-death-eda-visualisation.ipynb)
This is visual analysis based project where we have the data set of 45 countries. Using plotlly chroropleth, seaborn, matplotlib we made different plot to analyse the count of 
daeth, rate of death due to covid-19 in different countries with respect to time.
We also found that Mexico has the highest count and rate of death.For more detail about the dataset please click on the image below

[![](https://github.com/NaveenKumarMaurya/datascience-project-portfolio/blob/main/covid-19%20death%20map.png)](https://www.kaggle.com/dhruvildave/covid19-deaths-dataset)

## [Project 6:Bank term deposit subscription:](https://github.com/NaveenKumarMaurya/datascience-project-portfolio/blob/main/Bank%20term%20deposit%20subscription.ipynb)
This is a imbalance classification based ML project where we are given a direct marketing campaign (phone call) dataset obtained from a bank. The data contains 3090 rows and each row refers to an individual customer record. The dataset contains 15 input attributes and one target or class label attribute (subscribed to term deposit).

We used 8 classification algorithms to build our model on sklearn. Smote technique has been used to tackle the imbalance of the data using [imblearn](https://pypi.org/project/imblearn/) package.
We got the best accuracy with Decision tree model accuracy =92% and f1-score =.93, after analysing the correlation and feature importance we got the'duration' is the most important factor to decide whether customer takes subscription or not.

![](https://github.com/NaveenKumarMaurya/datascience-project-portfolio/blob/main/bank-term-deposit.jpg)

## [Project 7: Bank Marketing:](https://github.com/NaveenKumarMaurya/datascience-project-portfolio/blob/main/Bank%20Marketing.ipynb)
Banking dataset that helps running a campaign. how do banks select target customers so that chances of selling their product maximize, this will help you understand these things.The targeted customer has to be predicted on the basis of some features eg.age,education,marital status,salary,housing etc. We have build a classification model to classify the targeted customer.
you can get the dataset from kaggle by clicking on below image

[![](https://github.com/NaveenKumarMaurya/datascience-project-portfolio/blob/main/bank%20marking%20image.jpg)](https://www.kaggle.com/dhirajnirne/bank-marketing)

## [Project 8: Telecom User Churn:](https://github.com/NaveenKumarMaurya/datascience-project-portfolio/blob/main/Telecom%20User%20Churn%20.ipynb)

predicting the churn, we can react in time and try to keep the client who wants to leave. Based on the data about the services that the client uses, we can make him a special offer, trying to change his decision to leave the operator. This will make the task of retention easier to implement than the task of attracting new users, about which we do not know anything yet.
We build the classification model to predict whether the customer will churn or nor.Here we have focused on recall rather than accuracy and we get recall upto 80%

