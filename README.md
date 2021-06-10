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

## [Project 2: Concrete Compressive Strength : Overview](https://github.com/NaveenKumarMaurya/datascience-project-portfolio/blob/main/concrete-data-eda-model-acc-97.ipynb)
This is a Regression problem trained on python scikit-learn libarary.The data is related to civil engineering/Architecture where compressive strengh of material being used is import factor to determone the stability, sustainibility of building/bridge/construction.The target aim of the model is to predict the compressive strength of concrete on the basis of independent  variable -cement,Blast Furnace Slag,Fly Ash,Water,Superplasticizer,Coarse Aggregate,Fine Aggregate.

we used 12 regression algorithms to build our ML model where we have got the best result with Extra Tree Regressor having accuracy of 97%,RSME=4.08 which is very good for a model.
![](https://github.com/NaveenKumarMaurya/datascience-project-portfolio/blob/main/concrete%20%20image.jpg)




