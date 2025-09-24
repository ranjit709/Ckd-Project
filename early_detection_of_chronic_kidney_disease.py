

#importing the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score

#read and load data csv files
data = pd.read_csv('kidney_disease (1).csv')

#Check the 10 samples of train data
data.head(10)

#Check the last 5 samples of train data
data.tail(5)

#Data preprocessing and cleaning
data.info()

NewCols={"bp":"blood_pressure","sg":"specific_gravity", "al":"albumin","su":"sugar","rbc":"red_blood_cells","pc":"pus_cell",
         "pcc":"pus_cell_clumps","ba":"bacteria","bgr":"blood_glucose_random","bu":"blood_urea","sc":"serum_creatinine",
         "sod":"sodium","pot":"potassium","hemo":"haemoglobin","pcv":"packed_cell_volume","wc":"white_blood_cell_count",
          "rc":"red_blood_cell_count","htn":"hypertension","dm":"diabetes_mellitus","cad":"coronary_artery_disease",
          "appet":"appetite","pe":"pedal_edema","ane":"anemia"}

# Change columns of CKD data to new columns
data.rename(columns=NewCols, inplace=True)

#checking the distribution of numerical data  such as count , mean , max , min  and standard deviation.
data.describe()

#check numbers of rows(samples) and columns(features)
data.shape

#check count of values for each features
data.count()

#Check total missing values in each feature
data.isnull().sum()

"""#Visualizations"""

#visualization of null values in features
plt.subplots(figsize=(10, 7))
((data.isnull().sum())).sort_values(ascending=False).plot(kind='bar')

# Drop id column 
data.drop(["id"],axis=1,inplace=True)

data[['red_blood_cells','pus_cell']] = data[['red_blood_cells','pus_cell']].replace(to_replace={'abnormal':1,'normal':0})
data[['pus_cell_clumps','bacteria']] = data[['pus_cell_clumps','bacteria']].replace(to_replace={'present':1,'notpresent':0})
data[['hypertension','diabetes_mellitus','coronary_artery_disease','pedal_edema','anemia']] = data[['hypertension','diabetes_mellitus','coronary_artery_disease','pedal_edema','anemia']].replace(to_replace={'yes':1,'no':0})
data[['appetite']] = data[['appetite']].replace(to_replace={'good':1,'poor':0,'no':np.nan})
data['coronary_artery_disease'] = data['coronary_artery_disease'].replace(to_replace='\tno',value=0)
data['diabetes_mellitus'] = data['diabetes_mellitus'].replace(to_replace={'\tno':0,'\tyes':1,' yes':1, '':np.nan})
data['classification'] = data['classification'].replace(to_replace={'ckd':1.0,'ckd\t':1.0,'notckd':0.0,'no':0.0})

data['pedal_edema'] = data['pedal_edema'].replace(to_replace='good',value=0) 
data['appetite'] = data['appetite'].replace(to_replace='no',value=0)
data['coronary_artery_disease']=data['coronary_artery_disease'].replace('yes',1)

"""#Missing values

"""

data['age']=data['age'].fillna(np.mean(data['age']))
data['blood_pressure']=data['blood_pressure'].fillna(np.mean(data['blood_pressure']))
data['albumin']=data['albumin'].fillna(np.mean(data['albumin']))

data['specific_gravity']=data['specific_gravity'].fillna(np.mean(data['specific_gravity']))
data['sugar']=data['sugar'].fillna(np.mean(data['sugar']))
data['blood_glucose_random']=data['blood_glucose_random'].fillna(np.mean(data['blood_glucose_random']))
data['blood_urea']=data['blood_urea'].fillna(np.mean(data['blood_urea']))
data['serum_creatinine']=data['serum_creatinine'].fillna(np.mean(data['serum_creatinine']))
data['haemoglobinhaemoglobin']=data['haemoglobin'].fillna(np.mean(data['haemoglobin']))
data['potassium']=data['potassium'].fillna(np.mean(data['potassium']))
data['sodium']=data['sodium'].fillna(np.mean(data['sodium']))

data = data.replace("\t?", np.nan)
data = data.replace(" ?", np.nan)
data = data.fillna(method='ffill')
data = data.fillna(method='backfill')

#Checking null values after imputation
data.isnull().sum()

"""#Outliers"""

#check outliers for specific gravity
fig, ax = plt.subplots()
ax.scatter(x = data['specific_gravity'], y = data['classification'])
plt.ylabel('specific_gravity', fontsize=13)
plt.xlabel('classfication', fontsize=13)
plt.show()

#check outliers for sugar
fig, ax = plt.subplots()
ax.scatter(x = data['sugar'], y = data['classification'])
plt.ylabel('sugar', fontsize=13)
plt.xlabel('classfication', fontsize=13)
plt.show()

#Blood_pressure
fig, ax = plt.subplots()
ax.scatter(x = data['blood_pressure'], y = data['classification'])
plt.ylabel('blood_pressure', fontsize=13)
plt.xlabel('classification', fontsize=13)
plt.show()

"""#Visualization"""

#Observing numerical and categorical features
numericalFeatures = data.select_dtypes(include=np.number)
categoricalFeatures = data.select_dtypes(include='object')

#Numerical features
numericalFeatures

"""#Correalation

"""

#Correalation of numerical columns
datacorrnumerical=numericalFeatures.corr()
sns.pairplot(numericalFeatures)

plt.subplots(figsize=(15, 15))
sns.heatmap(datacorrnumerical,annot=True)

plt.scatter(data['classification'],data['age'])
plt.xlabel('classification',fontsize=10)
plt.ylabel('age',fontsize=10)

plt.scatter(data['classification'],data['blood_pressure'])
plt.xlabel('classification',fontsize=10)
plt.ylabel('blood_pressure',fontsize=10)

plt.scatter(data['classification'],data['albumin'])
plt.xlabel('classification',fontsize=10)
plt.ylabel('albumin',fontsize=10)

sns.boxplot(x='hypertension', y='specific_gravity', data=data, palette='viridis')

sns.boxplot(x='hypertension', y='albumin', data=data, palette='viridis')

X = data.iloc[:, :-1]
y = data.iloc[:, 24]

X=X.drop('classification', axis=1)

X=pd.DataFrame(X)

"""#Feaute selection"""

from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
selector = RFE(estimator=model, n_features_to_select=14)

selector.fit(X, y)

selector.get_support(indices=True)

Features=X.columns

selected_features_idx = selector.get_support(indices=True)
selected_features_idx

selected_featuresDT = Features[selected_features_idx]
selected_featuresDT

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

rfc = RandomForestClassifier(random_state=0, criterion='gini')

selector = SelectFromModel(estimator=rfc)

selector.fit(X, y)

x=X[selected_featuresDT]

x.head()

"""#Splitting data

"""

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
x_train

#Decision Tree classifier
from sklearn.tree import DecisionTreeClassifier
modeldt = DecisionTreeClassifier()

modeldt.fit(x_train, y_train)

y_preddt = modeldt.predict(x_test)
y_preddt

CMDT=confusion_matrix(y_test,y_preddt)
CMDT

print(classification_report(y_test, y_preddt))

sns.set(font_scale=1.1)
sns.heatmap(CMDT, annot=True,fmt="g")
ax= plt.subplot()
plt.title("CM_CKD with DT")
#plt.tight_layout()

ax.xaxis.set_ticklabels(['NotCKD', 'CKD'])
ax.yaxis.set_ticklabels(['NotCKD', 'CKD'])
plt.ylabel(' True Label')
plt.xlabel(' Predicted Label ')
# plt.show()

#Performance metrics of Decision Tree
accuracy = accuracy_score(y_test,y_preddt)
print('Accuracy: %f' % accuracy)
accuracy = balanced_accuracy_score(y_test,y_preddt)
print('Balanced_Accuracy: %f' % accuracy)
precision = precision_score(y_test,y_preddt)
print('Precision: %f' % precision)
recall = recall_score(y_test,y_preddt)
print('Recall: %f' % recall)
f1 = f1_score(y_test,y_preddt)
print('F1 score: %f' % f1)
x_train

#Prediction
k=np.array([[1.017408,1,0,0,99,11.7,48,5000,2.5,0,1,1,1,0]])
predict_dt=modeldt.predict(k)
predict_dt=int(predict_dt.item())
# sclr=np.squeeze(predict_dt)
classes=np.array(['Normal','Kidney disease detected'])
predict_dt
classes[predict_dt]

#Model saving
import pickle

with open('Dt.pickle','wb') as f:
  pickle.dump(modeldt,f)

#Random forest classifier
from sklearn.ensemble import RandomForestClassifier
modelRF = RandomForestClassifier()

modelRF.fit(x_train, y_train)

y_predrf = modelRF.predict(x_test)
y_predrf

print(classification_report(y_test, y_predrf))

CMRF=confusion_matrix(y_test, y_predrf)
CMRF

sns.set(font_scale=1.1)
sns.heatmap(CMRF, annot=True,fmt="g")
ax= plt.subplot()
plt.title("CM_CKD with RF")
#plt.tight_layout()

ax.xaxis.set_ticklabels(['NotCKD', 'CKD'])
ax.yaxis.set_ticklabels(['NotCKD', 'CKD'])
plt.ylabel(' True Label')
plt.xlabel(' Predicted Label ')
# plt.show()

#Prediction
k=np.array([[1.017408,1,0,0,99,11.7,48,5000,2.5,0,1,1,1,0]])
predict_dt=modelRF.predict(k)
predict_dt=int(predict_dt.item())
# sclr=np.squeeze(predict_dt)
classes=np.array(['Normal','Kidney disease detected'])
predict_dt
classes[predict_dt]

#Model saving
import pickle

with open('RF.pickle','wb') as f:
  pickle.dump(modelRF,f)

###########  RF #############
accuracy = accuracy_score(y_test,y_predrf)
print('Accuracy: %f' % accuracy)
accuracy = balanced_accuracy_score(y_test,y_predrf)
print('Balanced_Accuracy: %f' % accuracy)
precision = precision_score(y_test,y_predrf)
print('Precision: %f' % precision)
recall = recall_score(y_test,y_predrf)
print('Recall: %f' % recall)
f1 = f1_score(y_test,y_predrf)
print('F1 score: %f' % f1)

#Support vector machine
from sklearn.svm import SVC
modelsvc = SVC(C=0.05)
modelsvc.fit(x_train, y_train)
y_predsvc = modelsvc.predict(x_test)

print(classification_report(y_test, y_predsvc))

CMsvm=confusion_matrix(y_test, y_predsvc)
CMsvm

sns.set(font_scale=1.1)
sns.heatmap(CMsvm, annot=True,fmt="g")
ax= plt.subplot()
plt.title("CM_CKD with SVM")
#plt.tight_layout()

ax.xaxis.set_ticklabels(['NotCKD', 'CKD'])
ax.yaxis.set_ticklabels(['NotCKD', 'CKD'])

plt.ylabel(' True Label')
plt.xlabel(' Predicted Label ')
# plt.show()


#Prediction
k=np.array([[1.017408,1,0,0,99,11.7,48,5000,2.5,0,1,1,1,0]])
predict_dt=modelsvc.predict(k)
predict_dt=int(predict_dt.item())
# sclr=np.squeeze(predict_dt)
classes=np.array(['Normal','Kidney disease detected'])
predict_dt
classes[predict_dt]

#Model saving
import pickle

with open('svc.pickle','wb') as f:
  pickle.dump(modelsvc,f)

############## SVM ##########
accuracy = accuracy_score(y_test,y_predsvc)
print('Accuracy: %f' % accuracy)
accuracy = balanced_accuracy_score(y_test,y_predsvc)
print('Balanced_Accuracy: %f' % accuracy)
precision = precision_score(y_test,y_predsvc)
print('Precision: %f' % precision)
recall = recall_score(y_test,y_predsvc)
print('Recall: %f' % recall)
f1 = f1_score(y_test,y_predsvc)
print('F1 score: %f' % f1)

#KNN 
from sklearn.neighbors import KNeighborsClassifier
modelknn = KNeighborsClassifier(n_neighbors=7)
modelknn.fit(x_train, y_train)
y_predknn = modelknn.predict(x_test)

print(classification_report(y_test, y_predknn))

CMknn=confusion_matrix(y_test, y_predknn)
CMknn

sns.set(font_scale=1.1)
sns.heatmap(CMknn, annot=True,fmt="g")
ax= plt.subplot()
plt.title("CM_CKD with KNN")
#plt.tight_layout()

ax.xaxis.set_ticklabels(['NotCKD', 'CKD'])
ax.yaxis.set_ticklabels(['NotCKD', 'CKD'])
plt.ylabel(' True Label')
plt.xlabel(' Predicted Label ')
# plt.show()


#Prediction
k=np.array([[1.017408,1,0,0,99,11.7,48,5000,2.5,0,1,1,1,0]])
predict_dt=modelknn.predict(k)
predict_dt=int(predict_dt.item())
# sclr=np.squeeze(predict_dt)
classes=np.array(['Normal','Kidney disease detected'])
predict_dt
classes[predict_dt]

#Model saving
import pickle

with open('knn.pickle','wb') as f:
  pickle.dump(modelknn,f)

############# KNN ##############
accuracy = accuracy_score(y_test,y_predknn)
print('Accuracy: %f' % accuracy)
accuracy = balanced_accuracy_score(y_test,y_predknn)
print('Balanced_Accuracy: %f' % accuracy)
precision = precision_score(y_test,y_predknn)
print('Precision: %f' % precision)
recall = recall_score(y_test,y_predknn)
print('Recall: %f' % recall)
f1 = f1_score(y_test,y_predknn)
print('F1 score: %f' % f1)

