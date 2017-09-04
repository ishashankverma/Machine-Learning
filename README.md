# Intro to Machine Learning Project Submission

## Questions

__1. Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]__

The goal of this project is to identify the persons of interest (POIs) from the Enron corpus by using the financial records and email data of all the people to build a predictive model. This model can be used to find other suspects that were not found in the original investigation. A person of interest here refers to the people who commited fraud. The dataset consists of 146 people from the Enron corpus, 18 of which are POIs. Below are some important characteristics of the dataset:

Total no. of data points: 146

Total no. of POIs: 18

Total no. of non POIs: 128

Total no. of features: 21

No. of used features: 20

No. of NaN values in each feature: 

**Features**|**Number of NaNs**|
--- | --- |
salary|51|
to_messages|60|
deferral_payments|107|
total_payments|21|
loan_advances|142|
bonus|64|
email_address|35|
restricted_stock_deferred|128|
total_stock_value|20|
shared_receipt_with_poi|60|
long_term_incentive|80|
exercised_stock_options|44|
from_messages|60|
other|53|
from_poi_to_this_person|60|
from_this_person_to_poi|60|
poi|0|
deferred_income|97|
expenses|51|
restricted_stock|36|
director_fees|129|

Looking at the scatter plots, we can see that the dataset had 1 outlier "TOTAL", this appears to be due to some clerical error as it looks like the sum of all the other values. After checking the dataset, I found person named "THE TRAVEL AGENCY IN THE PARK". This looks like an agency instead of a person. Checking the dataset for NaNs, I found a person with all the records as NaN. I have removed these irregular values from the dataset.

__2. What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “intelligently select features”, “properly scale features”]__

I used the below final features in my POI identifier:

**Feature**|**Score**
---|---
exercised_stock_options|24.815079733218194
total_stock_value|24.182898678566879
bonus|20.792252047181535
salary|18.289684043404513
to_poi_ratio|16.409712548035792
total_wealth|15.369276864584535
deferred_income|11.458476579280369

I used SelectKBest for selecting the 7 best features to use in all the algorithms. 6 out of these 7 features are related to financial data and 1 feature is related to the email data. I also used min-max scaling to scale all the best features. I chose to use scaling because our email and financial data is varied. I used min-max scalers to rescale each feature to a common range. I created three new variables "to_poi_ratio", "from_poi_ratio" and "total_wealth". The feature "to_poi_ratio" that I created to depict the ratio of emails that a person sends to a POI, the feature "from_poi_ratio" depicts the ratio of emails that a person received from a POI and the feature "total_wealth" represents the sum of the 'salary', 'bonus', 'total_stock_value' & 'exercised_stock_options' features.

__3. What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]__

After trying Decision Tree and Naive Bayes, the naive bayes had the best precision and recall. So, I ended up using naive bayes as the final algorithm. Below is a table that shows the accuracy, precision and recall of the algorithms post tuning:

Algorithm|Accuracy|Precision|Recall
---|---|---|---|
Naive Bayes|0.851|0.415|0.345
Decision Tree|0.831|0.211|0.193

__4. What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric items: “discuss parameter tuning”, “tune the algorithm”]__

Parameter tuning of an algorithm refers to the adjustment and optimization of a particular algorithm to improve it's fit on the test set. If done well, parameter tuning results in the best performance from an algorithm. However, if done wrong, it may affect the accuracy, precision and recall and make them poor. For the decision tree classifier, I tuned the algorithm using GridSearchCV to get the best parameters. 

Below are the parameters of my tuned Decision Tree Classifier:

{min_samples_split=2, max_leaf_nodes=None, criterion='entropy', max_depth=None, min_samples_leaf=10}

__5. What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric items: “discuss validation”, “validation strategy”]__

Validation consists of a bunch of techniques to generalize our model with an independent/future dataset. A classic mistake, that can occur is overfitting a model. The model that has been overfit will perform well on the training set but score poorly on the test set. We can use cross validation techniques or reduce the number of features used in our dataset to avoid overfitting. I have used a low number of features to create my model and I trained the model using train test split with a 70:30 ratio. I also checked the precision and recall of my model.

__6. Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]__

I used accuracy, precision and recall as my evaluation metrics. 

The evaluation metrics for the test set and final project set is given below

Metric|Test Set Score|Final Set Score
---|---|---
Accuracy|0.851|0.85
Precision|0.414|0.466
Recall|0.345|0.348

Accuracy translates to the proximity/closeness of the measured value to the actual value. Accuracy of 0.851 means that the ratio of both the true predictions (true positives and true negatives) to the total predictions is 0.85. Precision is an algorithm's ability to classify actual true positives from the total true positives predicted. A precision of 0.466 means that out of 1000 people identified as POIs, 466 people are actually POIs. Recall measures an algorithm's ability to classify actual true positives from the actual true positives. A recall of 0.348 means that out of 1000 people who are actually POIs, 348 are classified correctly as POIs.


## References

https://docs.python.org/2/library/pprint.html

http://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection

http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html#sklearn.feature_selection.SelectKBest

http://scikit-learn.org/stable/modules/tree.html#tree-classification
