import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

# Loading dataset
dataset = pd.read_csv("Pima_Indian_diabetes.csv")

# Converting all negative values into NONEs, since negative values are not valid in given data
dataset[dataset<0]=np.nan
# Converting zeros of specific features into NONEs
dataset['Insulin']=dataset['Insulin'].replace(0,np.nan)
dataset['Glucose']=dataset['Glucose'].replace(0,np.nan)
dataset['BMI']=dataset['BMI'].replace(0,np.nan)
dataset['DiabetesPedigreeFunction']=dataset['DiabetesPedigreeFunction'].replace(0,np.nan)
dataset['SkinThickness']=dataset['SkinThickness'].replace(0,np.nan)
dataset['BloodPressure']=dataset['BloodPressure'].replace(0,np.nan)
dataset['Age']=dataset['Age'].replace(0,np.nan)

# Removes the row with less than 6 non-null values
dataset.dropna(thresh=5,inplace = True)

# Function to add random error to the values used to fill the NONEs
def generate_vector(temp_min,temp_max,given,num_value):
    np.random.seed(1)
    min = given + temp_min
    max = given + temp_max
    return np.random.uniform(min,max,num_value)

# Filling NONEs of the feature 'Pregnancies'
pd.set_option('mode.chained_assignment', None)
var_preg = generate_vector(-1,1,dataset['Pregnancies'].median(),dataset['Pregnancies'].isnull().sum())
dataset['Pregnancies'].loc[dataset[dataset.Pregnancies.isna()].index] = var_preg
dataset['Pregnancies'].round

# Removing outliers from the feature 'Pregnancies'
dataset = dataset[dataset.Pregnancies<13]

# Grouping specific features with outcome
test_data = pd.DataFrame(index=range(0,768),columns=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])
test_data = dataset
test_data1 = pd.DataFrame(index=range(0,768),columns=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])
test_data1 = dataset
test_data2 = pd.DataFrame(index=range(0,768),columns=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])
test_data2 = dataset

test_data = test_data[test_data.Insulin!=0]
test_data1 = test_data1[test_data1.Glucose!=0]
test_data2 = test_data2[test_data2.DiabetesPedigreeFunction!=0]
i = test_data.groupby(['Outcome'])
g = test_data1.groupby(['Outcome'])
d = test_data2.groupby(['Outcome'])
ins=i['Insulin'].agg(pd.Series.mode)
glu1 = g['Glucose'].agg(pd.Series.mode)
glu2 = g['Glucose'].mean()
dia = d['DiabetesPedigreeFunction'].agg(pd.Series.mode)

# Filling NONEs of the features 'Glucose', 'Insulin', 'DiabetesPedigreeFunction'
temp_outcome = dataset['Outcome']
for j in temp_outcome:
    if j==0:
        dataset.Insulin = dataset.Insulin.fillna(ins[0])
        dataset.Glucose = dataset.Glucose.fillna(glu1[0])
        dataset.DiabetesPedigreeFunction = dataset.DiabetesPedigreeFunction.fillna(dia[0])
    elif j==1:
        dataset.Insulin = dataset.Insulin.fillna(ins[1])
        dataset.Glucose = dataset.Glucose.fillna(glu2[1])
        dataset.DiabetesPedigreeFunction = dataset.DiabetesPedigreeFunction.fillna(dia[1])


# Filling NONEs of the feature 'SkinThickness'
var_skin = generate_vector(-9,6,dataset['SkinThickness'].mode(),dataset['SkinThickness'].isnull().sum())
dataset['SkinThickness'].loc[dataset[dataset.SkinThickness.isna()].index]= var_skin

# Removing outliers from the feature 'SkinThickness'
dataset = dataset[dataset.SkinThickness<63]

# Filling NONEs of the feature 'Age'
dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean())

# Creating an instance for the function LinearRegression
lin_reg = LinearRegression()

# Filling NONEs of the feature 'BMI' using linear regression
data_with_null = dataset[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
data_without_null = data_with_null.dropna()
x = data_without_null.loc[:,['SkinThickness']]
y = data_without_null.iloc[:,5]
#fitting SkinThickness feature into linear regression to predict BMI
lin_reg.fit(x,y)
test_data = pd.DataFrame(index=range(0,768),columns=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])
test_data['BMI'] = pd.DataFrame(lin_reg.predict(data_with_null.loc[:,['SkinThickness']]))
dataset.BMI.fillna(test_data.BMI,inplace=True)

# Removing outliers from the feature 'BMI'
dataset = dataset[dataset.BMI>13]
dataset = dataset[dataset.BMI<50]

# Filling NONEs of the feature 'BloodPressure' using linear regression
data_with_null = dataset[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
data_without_null = data_with_null.dropna()
x = data_without_null.loc[:,['BMI','Age']]
y = data_without_null.iloc[:,2]
#fitting BMI and Age featuers into linear regression to predict BloodPressure
lin_reg.fit(x,y)
test_data = pd.DataFrame(index=range(0,768),columns=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])
test_data['BloodPressure'] = pd.DataFrame(lin_reg.predict(data_with_null.loc[:,['BMI','Age']]))
dataset.BloodPressure.fillna(test_data.BloodPressure,inplace=True)

# Removing outliers from the feature 'BloodPressure'
dataset = dataset[dataset.BloodPressure>40]
dataset = dataset[dataset.BloodPressure<105]

# Normalizing the Dataset
dataset = (dataset-dataset.min())/(dataset.max()-dataset.min())

# Dividing the dataset for feature extraction
x = dataset[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age',]]
y = dataset['Outcome']
# Creating an instance for the function logistic regression
log_reg = LogisticRegression(solver = 'lbfgs')

# Implementing PCA for feature extraction
# pca = PCA(n_components=3)
# x = pca.fit_transform(x)

# Splitting the data into training set and testing set
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 4)

#fitting testing data into logistic regression model
log_reg.fit(x_train,y_train)
pred = log_reg.predict(x_test)

# Calculating the accuracy of 'Outcome'
print(accuracy_score(y_test,pred))
