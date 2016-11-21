
# coding: utf-8

# # Title: Supervising rain & snow or not

#     Group Members:
#         Bingyu Song    A20364641
#         Xin Liu        A20353208
#         Zhipeng liu    A20355209

#     Project Description:
#     This project aims to supervise whether one day would rain or snow, according to this day's weather conditon. The data we used is last ten years weather information of Chicago collect from ”https://www.wundrground.com/history/“. We use some methods from scrapy library.

# ###### 1.Load Data 
#     In the following part, we load the data that is already processed in the last phase.

# In[1]:

from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.model_selection import cross_val_score
import re


# In[2]:

data=[]
label=[]

f=open('collection','r')
content=f.read()
content=content.split('\n')
for i in range(0,len(content)-1):
	line=content[i].split(',')
	line = [float(j) for j in line]
	data.append(line[1:-1])
	label.append(line[-1])


# ###### 2. Scale data and print the data shape

# In[3]:

from sklearn.preprocessing import scale
data = scale(data,copy = 'False')
print("number of instance: ", len(data))
print("number of features: ", len(data[0]))


# ###### 3. Choose a performance measure
#     This is a classification tasks, and there are two results: rain or snow and not rain or snow. The weight of these two result is the same. Thus we choose accuracy as performance measure.

# ###### 4. Performance of baselines.
#     Since this is a classification tasks, we use performance of random prediction and performance of predicting the majority class all the time as the baselines.

# In[4]:

cout = 0;
for i in label:
	if(i == 0):
		cout = cout + 1
print("accuracy of predicting the majority class all the time: ",cout/2982)

import random
cout = 0;
for i in label:
	if(i == random.randint(0,1)):
		cout = cout + 1
print("accuracy of random prediction:", cout/2982)


# ###### 5. Model Selection    
#     We use two models, decision tree and logistic regression, to train the data. For each model we performed different parameter settings. Then we calculated the accuracy of each situation, and chose the model and parameter setting which has the highest accuracy.

#     Following show the performance of decision tree model.

# In[5]:

def entro(x,y):
	clf = tree.DecisionTreeClassifier(random_state = 0, criterion = "entropy",max_depth = x,min_weight_fraction_leaf = y) 
	clf = clf.fit(data,label)
	out = cross_val_score(clf,data,label,cv = 10,scoring = "accuracy")
	print("max_depth = ",x,"min weight frac = ",y,"accu : ",out.mean())
def gini(x,y):
	clf = tree.DecisionTreeClassifier(random_state = 0, criterion = "gini",max_depth = x,min_weight_fraction_leaf = y) 
	clf = clf.fit(data,label)
	out = cross_val_score(clf,data,label,cv = 10,scoring = "accuracy")
	print("max_depth = ",x,"min weight frac = ",y,"accu : ",out.mean())

print("1 entropy:")
clf = tree.DecisionTreeClassifier(random_state = 0, criterion = "entropy") 
clf = clf.fit(data,label)
out = cross_val_score(clf,data,label,cv = 10,scoring = "accuracy")
print(out.mean())
list1 = [5,10,20]
list2 = [0.1,0.15,0.2,0.3]
for i in list1:
	for j in list2:
		entro(i,j)
print("2 gini:")
clf = tree.DecisionTreeClassifier(random_state = 0, criterion = "entropy") 
clf = clf.fit(data,label)
out = cross_val_score(clf,data,label,cv = 10,scoring = "accuracy")
print(out.mean())
for i in list1:
	for j in list2:
		gini(i,j)


#     Following shows the performace of logistic regression.
#     
#     (Because in LogisticRegression when the penalty is L1, the solver only can be liblinear, we dericatly set it as 'liblinear')
#     
#     The mainly features are: CDT,Max TemperatureF,Mean TemperatureF,Min TemperatureF,Max Humidity, Mean Humidity, Min Humidity, Mean Sea Level PressureIn, Mean Wind SpeedMPH, CloudCover
#     The lable is: events
#     We think the mainly positive features should be: Humidity and CloudCover
# 

# In[6]:

def l2(x,y):
	clf = LogisticRegression(penalty = 'l2',random_state = 0, solver = x, fit_intercept = y)
	clf = clf.fit(data,label)
	out = cross_val_score(clf,data,label,cv = 10,scoring = "accuracy")
	print("solver = ",x,"  fit intercept = ",y,"   accu = ",out.mean())
def l1(x):
	clf = LogisticRegression(penalty = 'l1',random_state = 0, solver = "liblinear", fit_intercept = x)
	clf = clf.fit(data,label)
	out = cross_val_score(clf,data,label,cv = 10,scoring = "accuracy")
	print("fit intercept = ",x,"   accu = ",out.mean())

list1 = ["newton-cg", "lbfgs", "liblinear", "sag"]
list2 = [0,1]
print("l2 default:")
clf = LogisticRegression(penalty = 'l2',random_state = 0)
clf = clf.fit(data,label)
out = cross_val_score(clf,data,label,cv = 10,scoring = "accuracy")
print(out.mean())
print ("other combine:")
for i in list1:
	for j in list2:
		l2(i,j)

list3 = [0,1]
print("l1 default:")
clf = LogisticRegression(penalty = 'l1',random_state = 0)
clf = clf.fit(data,label)
out = cross_val_score(clf,data,label,cv = 10,scoring = "accuracy")
print(out.mean())
print ("other combine:")
for i in list3:
	l1(i)


#     According to the largest accuracy of decision tree model is smaller than the accurancy of logistoc regression,so we finally choose the logistic regression as the trainning model. And the settings is: random state = 0, use L2 penalty. The accuracy of choosen model is 0.786384268716.

# ###### 6. Important features
# Since we choose the logistic regression as the model, we should print the top features and their weights. The result is as following. 

# In[7]:

clf = LogisticRegression(penalty = 'l2',random_state = 0)
clf = clf.fit(data,label)
print("weight of every feature in model with best accu: ",clf.coef_[0])


#     According to our datasets, the top positive feature is cloud cover, the top negative features is min humidity.
#     It seems unreasonable for this negative feature, because the humidity should be positive infect. It because for humidity has min, max and avg, if we add those weight together, it's also positive influence, so we decided filter some redundant features.
# 
#     So finally, we decided use these features :
#             Mean TemperatureF,Mean Humidity, Mean Sea Level PressureIn, Mean Wind SpeedMPH, CloudCover
# 
#     So we need to repeat all process above with new features, as follow:
# 

# #### 1.lode data 

# In[8]:

import re

data=[]
label=[]
list1 = [2,5,7,8,9]
f=open('collection','r')
content=f.read()
content=content.split('\n')
for i in range(0,len(content)-1):
	line=content[i].split(',')
	line = [float(j) for j in line]
	tem = []
	for m in list1:
		tem.append(line[m])
	data.append(tem)
	label.append(line[-1])


# #### 2.Scale data and print the data shape

# In[9]:

from sklearn.preprocessing import scale
data = scale(data,copy = 'False')
print("number of instance: ", len(data))
print("number of features: ", len(data[0]))


# #### 3.baseline

# In[10]:

cout = 0;
for i in label:
	if(i == 0):
		cout = cout + 1
print("accuracy of predicting the majority class all the time: ",cout/2982)

import random
cout = 0;
for i in label:
	if(i == random.randint(0,1)):
		cout = cout + 1
print("accuracy of random prediction:", cout/2982)


# #### 4.decision tree

# In[11]:

from sklearn import tree
from sklearn.model_selection import cross_val_score


def entro(x,y):
	clf = tree.DecisionTreeClassifier(random_state = 0, criterion = "entropy",max_depth = x,min_weight_fraction_leaf = y) 
	clf = clf.fit(data,label)
	out = cross_val_score(clf,data,label,cv = 10,scoring = "accuracy")
	print("max_depth = ",x,"min weight frac = ",y,"accu : ",out.mean())
def gini(x,y):
	clf = tree.DecisionTreeClassifier(random_state = 0, criterion = "gini",max_depth = x,min_weight_fraction_leaf = y) 
	clf = clf.fit(data,label)
	out = cross_val_score(clf,data,label,cv = 10,scoring = "accuracy")
	print("max_depth = ",x,"min weight frac = ",y,"accu : ",out.mean())

print("1 entropy:")
clf = tree.DecisionTreeClassifier(random_state = 0, criterion = "entropy") 
clf = clf.fit(data,label)
out = cross_val_score(clf,data,label,cv = 10,scoring = "accuracy")
print(out.mean())
list1 = [5,10,20]
list2 = [0.1,0.15,0.2,0.3]
for i in list1:
	for j in list2:
		entro(i,j)

print("2 gini:")
clf = tree.DecisionTreeClassifier(random_state = 0, criterion = "entropy") 
clf = clf.fit(data,label)
out = cross_val_score(clf,data,label,cv = 10,scoring = "accuracy")
print(out.mean())
for i in list1:
	for j in list2:
		gini(i,j)


# #### 5.logistic regression

# In[12]:

from sklearn.preprocessing import scale
data = scale(data,copy = 'False')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def l2(x,y):
	clf = LogisticRegression(penalty = 'l2',random_state = 0, solver = x, fit_intercept = y)
	clf = clf.fit(data,label)
	out = cross_val_score(clf,data,label,cv = 10,scoring = "accuracy")
	print("solver = ",x,"  fit intercept = ",y,"   accu = ",out.mean())
def l1(x):
	clf = LogisticRegression(penalty = 'l1',random_state = 0, solver = "liblinear", fit_intercept = x)
	clf = clf.fit(data,label)
	out = cross_val_score(clf,data,label,cv = 10,scoring = "accuracy")
	print("fit intercept = ",x,"   accu = ",out.mean())

list1 = ["newton-cg", "lbfgs", "liblinear", "sag"]
list2 = [0,1]
print("l2 default:")
clf = LogisticRegression(penalty = 'l2',random_state = 0)
clf = clf.fit(data,label)
out = cross_val_score(clf,data,label,cv = 10,scoring = "accuracy")
print(out.mean())
print ("other combine:")
for i in list1:
	for j in list2:
		l2(i,j)

list3 = [0,1]
print("l1 default:")
clf = LogisticRegression(penalty = 'l1',random_state = 0)
clf = clf.fit(data,label)
out = cross_val_score(clf,data,label,cv = 10,scoring = "accuracy")
print(out.mean())
print ("other combine:")
for i in list3:
	l1(i)

clf = LogisticRegression(penalty = 'l2',random_state = 0)
clf = clf.fit(data,label)
print("weight of every feature in model with best accu: ",clf.coef_[0])


# #### 6. report

#     The largest accu in logistic regression function still bigger than accu in decision tree, so we also choose using logistic regression
#     so according to the new process, the top positive feature is cloudCover rate, the top negative features is  Mean Sea Level PressureIn
# 
#     The best accuracy is [-0.07818463  0.46048479 -0.31994331  0.25220156  1.38587607]
# 
#     This conclusion make sence.

# In[ ]:



