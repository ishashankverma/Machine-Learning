#!/usr/bin/python

import sys
import pickle
import pprint
import matplotlib.pyplot as plt
sys.path.append("../tools/")

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
from numpy import mean
from sklearn.tree import DecisionTreeClassifier

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
poi_label = ['poi']
email_features_list = [
    'from_messages',
    'from_poi_to_this_person',
    'from_this_person_to_poi',
    'shared_receipt_with_poi',
    'to_messages',
    ]
financial_features_list = [
    'bonus',
    'deferral_payments',
    'deferred_income',
    'director_fees',
    'exercised_stock_options',
    'expenses',
    'loan_advances',
    'long_term_incentive',
    'other',
    'restricted_stock',
    'restricted_stock_deferred',
    'salary',
    'total_payments',
    'total_stock_value',
]

features_list = poi_label + email_features_list + financial_features_list

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Print the total no. of data points
print("Total no. of data points: %d" %len(data_dict))
# Allocation across classes (POI/non-POI)
poi = 0
for person in data_dict:
    if data_dict[person]['poi'] == True: poi += 1
print("Total no. of POIs: %d" % poi)
print("Total no. of non POIs: %d" % (len(data_dict) - poi))
       
# No. of features used
all_features = data_dict[data_dict.keys()[0]].keys()
#print("There are %d features for each person in the dataset, and %d features \
#are used" %(len(all_features), len(features_list)))

print("Total no. of features: %d" %(len(all_features)))
print("No. of used features: %d" %(len(features_list)))

# Finding and printing no. of missing values
missing_values = {}
for feature in all_features:
    missing_values[feature] = 0
for person in data_dict:
    for feature in data_dict[person]:
        if data_dict[person][feature] == "NaN":
            missing_values[feature] += 1
print("No. of NaN values in each feature: \n")
for feature in missing_values:
    print("%s: %d" %(feature, missing_values[feature]))

### Task 2: Remove outliers
# Function to plot the outliers
def plot_outliers(data_set, feature_x, feature_y):
    """
    This function takes input as a dict and 2 feature names as strings.
    It draws a 2d plot of the 2 features. 
    POIs are marked in red while all the other values are marked in blue.
    """
    
    data = featureFormat(data_set, [feature_x, feature_y, 'poi'])
    for point in data:
        x = point[0]
        y = point[1]
        poi = point[2]
        if poi:
            color = 'red'
        else:
            color = 'blue'
        plt.scatter(x, y, color=color)
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.show()

# Visualize data to identify outliers
print(plot_outliers(data_dict, 'total_payments', 'total_stock_value'))
print(plot_outliers(data_dict, 'from_poi_to_this_person', 'from_this_person_to_poi'))
print(plot_outliers(data_dict, 'salary', 'bonus'))
print(plot_outliers(data_dict, 'total_payments', 'other'))

# Printing the expected outliers having the highest values
expected_outliers = []
for person in data_dict:
    if data_dict[person]['total_payments'] != "NaN":
        expected_outliers.append((person, data_dict[person]['total_payments']))
print("Outliers:")
print(sorted(expected_outliers, key = lambda x: x[1], reverse=True)[0:4])

# Printing the data dictionary
pp = pprint.PrettyPrinter()

#pp.pprint(data_dict)

# Function to remove outliers
def remove_outlier(dict_object, keys):
    """
    Takes in a dictionary and some keys as input.
    Removes the keys & their values from the dictionary.    
    """
    for key in keys:
        dict_object.pop(key, 0)


outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E']
remove_outlier(data_dict, outliers)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

# New features for fraction of emails from and to the POIs
for elem in my_dataset:
    person = my_dataset[elem]
    email_from_poi = person['from_poi_to_this_person']
    to_email = person['to_messages']
    if email_from_poi != 'NaN' and to_email != 'NaN':
        person['from_poi_ratio'] = email_from_poi/float(to_email)
    else:
        person['from_poi_ratio'] = 0
    email_to_poi = person['from_this_person_to_poi']
    from_email = person['from_messages']
    if email_from_poi != 'NaN' and to_email != 'NaN':
        person['to_poi_ratio'] = email_to_poi/float(from_email)
    else:
        person['to_poi_ratio'] = 0

# New feature for the combined total wealth of the person
for elem in my_dataset:
    person = my_dataset[elem]
    if(person['salary'] != 'NaN' and person['total_stock_value'] != 'NaN' and person['exercised_stock_options'] != 'NaN' and person['bonus'] != 'NaN'):
        person['total_wealth'] = sum([person[value] for value in ['salary', 'bonus', 'total_stock_value', 'exercised_stock_options']])
    else:
        person['total_wealth'] = 'NaN'

# Adding the new features to the features list
my_features_list = features_list + ['from_poi_ratio', 'to_poi_ratio', 'total_wealth']

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, my_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Best Feature Selection using SelectKBest
k = 7
kbest = SelectKBest(f_classif, k = 7)
kbest.fit_transform(features, labels)
scores = zip(my_features_list[1:],kbest.scores_)
sorted_scores = sorted(scores, key = lambda x: x[1], reverse=True)
print("Scores of all features:")
pp.pprint(sorted_scores)
kbest_features = dict(sorted_scores[:k])
print("Best features are:")
pp.pprint(kbest_features)
best_features_list = poi_label + kbest_features.keys()
#pp.pprint(best_features_list)

# Extract features and labels from dataset for the new features list
data = featureFormat(my_dataset, best_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Min Max Scaling
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

def eval_clf(grid_search, features, labels, parameters, iterations=100):
    accuracy, precision, recall = [], [], []
    for iteration in range(iterations):
        features_train, features_test, labels_train, labels_test = train_test_split(features, labels, random_state=iteration)
        grid_search.fit(features_train, labels_train)
        predictions = grid_search.predict(features_test)
        accuracy.append(accuracy_score(labels_test, predictions))
        precision.append(precision_score(labels_test, predictions))
        recall.append(recall_score(labels_test, predictions))
    print '\nAccuracy:', mean(accuracy)
    print 'Precision:', mean(precision)
    print 'Recall:', mean(recall)
    best_params = grid_search.best_estimator_.get_params()
    print '\nBest Parameters are: \n'
    for param_name in parameters.keys():
        print '%s=%r, ' % (param_name, best_params[param_name])

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
nb_clf = GaussianNB()
nb_param = {}
nb_grid_search = GridSearchCV(nb_clf, nb_param)
print '\nGaussianNB:'
eval_clf(nb_grid_search, features, labels, nb_param)

##GaussianNB:
##Accuracy: 0.851428571429
##Precision: 0.414515873016
##Recall: 0.344615079365

##DecisionTree:
##
##Accuracy: 0.830857142857
##Precision: 0.211265873016
##Recall: 0.193357142857
##
##Best Parameters are: 
##
##min_samples_split=2, 
##max_leaf_nodes=None, 
##criterion='entropy', 
##max_depth=None, 
##min_samples_leaf=10,


##
##tree_clf = DecisionTreeClassifier()
##tree_param = {'criterion': ['gini', 'entropy'],
##              'min_samples_split': [2, 5, 10, 20],
##              'max_depth': [None, 2, 5, 10],
##              'min_samples_leaf': [1, 5, 10],
##              'max_leaf_nodes': [None, 5, 10, 20]}
##tree_grid_search = GridSearchCV(tree_clf, tree_param)
##print '\nDecisionTree:'
##eval_clf(tree_grid_search, features, labels, tree_param)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn import naive_bayes
clf = GaussianNB()

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, best_features_list)
