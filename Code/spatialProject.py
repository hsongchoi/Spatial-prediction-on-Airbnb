#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import itertools
import time

from scipy import stats
from scipy.stats import norm
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
from sklearn.metrics import r2_score


# In[2]:


data = pd.read_csv('input/AB_NYC_2019.csv',delimiter = ',')
data.head(5)
data.describe()


# In[3]:


data.columns
len(data)
len(data.columns)
data.shape #the number of rows and columns


# In[4]:


data.dtypes


# In[5]:


#looking to find out how many nulls are found in each column in dataset
data.isnull().sum()


# In[6]:


#The above distribution graph shows that there is a right-skewed distribution on price.
#log transformation will be used to make this feature less skewed.
#Since division by zero is a problem, log+1 transformation
plt.figure(figsize=(10,10))
sns.distplot(data['price'],fit=norm)
plt.title("Price Distribution Plot",size=15, weight='bold')


# In[7]:


data['price_log'] = np.log(data.price+1)


# In[8]:


plt.figure(figsize=(10,8))
sns.distplot(data['price_log'],fit=norm)
plt.title("Log-price Distribution Plot",size=15, weight='bold')

plt.figure(figsize=(6,6))
stats.probplot(data['price_log'],plot=plt)
plt.show()


# In[9]:


#dropping columns that are not significant
data1 = data.drop(['id','name','host_id','host_name', 'last_review','price'], axis=1) #axis : {0 or ‘index’, 1 or ‘columns’}, default 0
#examing the changesa
data1.head(5)
df = pd.DataFrame(data1)
df.head(5)
data2 = df.round(2)
data2.describe().round(2)


# In[10]:


df.room_type.value_counts()
df.neighbourhood_group.value_counts()
#df.neighbourhood.value_counts()
#pd.DataFrame(df.neighbourhood_group.unique())
df['neighbourhood'].groupby(df['neighbourhood_group']).describe()


# In[11]:


df.neighbourhood_group.unique()
len(set(data.neighbourhood_group)) #we use "set" to remove duplicates
df.room_type.unique()
len(set(data.room_type))
len(set(df.neighbourhood))


# In[ ]:





# In[12]:


pd.DataFrame(df.neighbourhood_group.unique())
df.room_type.value_counts()


# In[13]:


df.isnull().sum()
#Number of reviews features has some missing data.
#It will be replaced with zero.


# In[14]:


#replacing all NaN values in 'reviews_per_month' with 0
df.fillna({'reviews_per_month':0}, inplace=True)
df.reviews_per_month.isnull().sum()


# In[15]:


df['neighbourhood_group'] = df['neighbourhood_group'].astype("category").cat.codes
df['neighbourhood'] = df['neighbourhood'].astype("category").cat.codes
df['room_type'] = df['room_type'].astype("category").cat.codes
df.info()


# In[16]:


plt.figure(figsize=(15,10))
corr=df.corr(method='pearson')
sns.heatmap(corr, annot=True, square=True,cmap="BuPu",vmin=-0.7, vmax=0.7)
plt.title("Correlation Matrix", size =15, weight='bold')


# In[17]:


#multicollinearity will help to measure the relationship between explanatory variables in multiple regression
multicollinearity, V=np.linalg.eig(corr)
multicollinearity
#None of the eigenvalues is close to zero. no multicollinearity exists in the data.


# In[18]:


#Residual Plots
#detect outliers, non-linear data for regression models.
#the residual plots for each explanatory variables with the price.
df_x, df_y = df.iloc[:,:-1], df.iloc[:,-1]


# In[19]:


def fit_linear_reg(X,Y):
    #Fit linear regression model and return RSS and R squared values
    model_k = linear_model.LinearRegression(fit_intercept = True)
    model_k.fit(X,Y)
    RSS = mean_squared_error(Y,model_k.predict(X)) * len(Y)
    R_squared = model_k.score(X,Y)
    return RSS, R_squared


# In[20]:


export_csv = df.to_csv('input/df.csv', index = None, header=True)


# In[21]:


#Initialization variables
Y = df_y
X = df_x
k = 10

remaining_features = list(X.columns.values)
features = []
RSS_list, R_squared_list = [np.inf], [np.inf] #Due to 1 indexing of the loop...
features_list = dict()

for i in range(1,k+1):
    best_RSS = np.inf
    
    for combo in itertools.combinations(remaining_features,1):

            RSS = fit_linear_reg(X[list(combo) + features],Y)   #Store temp result 

            if RSS[0] < best_RSS:
                best_RSS = RSS[0]
                best_R_squared = RSS[1] 
                best_feature = combo[0]

    #Updating variables for next loop
    features.append(best_feature)
    remaining_features.remove(best_feature)
    
    #Saving values for plotting
    RSS_list.append(best_RSS)
    R_squared_list.append(best_R_squared)
    features_list[i] = features.copy()


# In[22]:


print('Forward stepwise subset selection')
print('Number of features |', 'Features |', 'RSS')
display([(i,features_list[i], round(RSS_list[i])) for i in range(1,5)])


# In[23]:


df1 = pd.concat([pd.DataFrame({'features':features_list}),pd.DataFrame({'RSS':RSS_list, 'R_squared': R_squared_list})], axis=1, join='inner')
df1['numb_features'] = df1.index


# In[24]:


#Initializing useful variables
m = len(Y)
p = 11
hat_sigma_squared = (1/(m - p -1)) * min(df1['RSS'])

#Computing
df1['C_p'] = (1/m) * (df1['RSS'] + 2 * df1['numb_features'] * hat_sigma_squared )
df1['AIC'] = (1/(m*hat_sigma_squared)) * (df1['RSS'] + 2 * df1['numb_features'] * hat_sigma_squared )
df1['BIC'] = (1/(m*hat_sigma_squared)) * (df1['RSS'] +  np.log(m) * df1['numb_features'] * hat_sigma_squared )
df1['R_squared_adj'] = 1 - ( (1 - df1['R_squared'])*(m-1)/(m-df1['numb_features'] -1))
df1


# In[25]:


df1['R_squared_adj'].idxmax()
df1['R_squared_adj'].max()


# In[26]:


variables = ['C_p', 'AIC','BIC','R_squared_adj']
fig = plt.figure(figsize = (18,6))

for i,v in enumerate(variables):
    ax = fig.add_subplot(1, 4, i+1)
    ax.plot(df1['numb_features'],df1[v], color = 'lightblue')
    ax.scatter(df1['numb_features'],df1[v], color = 'darkblue')
    if v == 'R_squared_adj':
        ax.plot(df1[v].idxmax(),df1[v].max(), marker = 'x', markersize = 20)
    else:
        ax.plot(df1[v].idxmin(),df1[v].min(), marker = 'x', markersize = 20)
    ax.set_xlabel('Number of predictors')
    ax.set_ylabel(v)

fig.suptitle('Subset selection using C_p, AIC, BIC, Adjusted R2', fontsize = 16)
plt.show()


# In[27]:


f, axes = plt.subplots(figsize=(10,7))
sns.residplot(df_x.iloc[:,0],df_y,lowess=True, scatter_kws={'alpha': 0.5}, 
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
f, axes = plt.subplots(figsize=(10,7))
sns.residplot(df_x.iloc[:,1],df_y,lowess=True, scatter_kws={'alpha': 0.5}, 
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})


# In[ ]:


f, axes = plt.subplots(figsize=(10,7))
sns.residplot(df_x.iloc[:,2],df_y,lowess=True, scatter_kws={'alpha': 0.5}, 
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})


# In[ ]:


f, axes = plt.subplots(figsize=(10,7))
sns.residplot(df_x.iloc[:,3],df_y,lowess=True, scatter_kws={'alpha': 0.5}, 
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})


# In[ ]:


f, axes = plt.subplots(figsize=(10,7))
sns.residplot(df_x.iloc[:,4],df_y,lowess=True, scatter_kws={'alpha': 0.5}, 
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})


# In[ ]:


f, axes = plt.subplots(figsize=(10,7))
sns.residplot(df_x.iloc[:,5],df_y,lowess=True, scatter_kws={'alpha': 0.5}, 
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})


# In[ ]:


f, axes = plt.subplots(figsize=(10,7))
sns.residplot(df_x.iloc[:,6],df_y,lowess=True, scatter_kws={'alpha': 0.5}, 
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})


# In[ ]:


f, axes = plt.subplots(figsize=(10,7))
sns.residplot(df_x.iloc[:,7],df_y,lowess=True, scatter_kws={'alpha': 0.5}, 
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})


# In[ ]:


f, axes = plt.subplots(figsize=(10,7))
sns.residplot(df_x.iloc[:,8],df_y,lowess=True, scatter_kws={'alpha': 0.5}, 
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})


# In[ ]:


f, axes = plt.subplots(figsize=(10,7))
sns.residplot(df_x.iloc[:,9],df_y,lowess=True, scatter_kws={'alpha': 0.5}, 
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})


# In[ ]:


df_x_transformed = StandardScaler().fit_transform(df_x)
x_train, x_test, y_train, y_test = train_test_split(df_x_transformed, df_y, test_size=0.3,random_state=42)


# In[ ]:


a = pd.DataFrame(df_x_transformed)
f, axes = plt.subplots(figsize=(10,7))
sns.residplot(a.iloc[:,3],df_y,lowess=True, scatter_kws={'alpha': 0.5}, 
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})


# In[ ]:


### Linear Regression ###

def linear_reg(input_x, input_y, cv=5):
    ## Defining parameters
    model_LR= LinearRegression()

    parameters = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}

    ## Building Grid Search algorithm with cross-validation and Mean Squared Error score.

    grid_search_LR = GridSearchCV(estimator=model_LR,  
                         param_grid=parameters,
                         scoring='neg_mean_squared_error',
                         cv=cv,
                         n_jobs=-1)

    ## finding the best parameters.

    grid_search_LR.fit(input_x, input_y)
    best_parameters_LR = grid_search_LR.best_params_  
    best_score_LR = grid_search_LR.best_score_ 
    print(best_parameters_LR)
    print(best_score_LR)


# linear_reg(nyc_model_x, nyc_model_y)


# In[ ]:


kfold_cv=KFold(n_splits=5, random_state=42, shuffle=False)
for train_index, test_index in kfold_cv.split(df_x_transformed,df_y):
    X_train, X_test = df_x_transformed[train_index], df_x_transformed[test_index]
    y_train, y_test = df_y[train_index],df_y[test_index]


# In[ ]:


Poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train = Poly.fit_transform(X_train)
X_test = Poly.fit_transform(X_test)


# In[ ]:


##Linear Regression
lr = LinearRegression(copy_X= True, fit_intercept = True, normalize = True)
lr.fit(X_train, y_train)
lr_pred= lr.predict(X_test)


# In[ ]:


print('MAE: %f'% mean_absolute_error(y_test, lr_pred)) #the difference between predictions and actual values
print('RMSE: %f'% np.sqrt(mean_squared_error(y_test, lr_pred))) 
print('R2 %f' % r2_score(y_test, lr_pred)) #the goodness of fit measure


# In[ ]:


#what hosts have the most listings on Airbnb platform and taking advantage of this service
#Interestingly, the first host has more than 300+ listings.
top_host = df.host_id.value_counts().head(10)
top_host


# In[ ]:


df.calculated_host_listings_count.max()


# In[ ]:


f, ax = plt.subplots(figsize=(8,6))
viz1 = top_host.plot(kind = 'bar')
viz1.set_title('Hosts with the most listings in NYC')
viz1.set_ylabel('Count of listings')
viz1.set_xlabel('Host IDs')


# In[ ]:


df['price'].groupby(df['neighbourhood_group']).describe()


# In[ ]:


#we make subgroup for each neighbourhood_group
sub_1 = df.loc[df['neighbourhood_group'] == 'Bronx']
sub_2 = df.loc[df['neighbourhood_group'] == 'Brooklyn']
sub_3 = df.loc[df['neighbourhood_group'] == 'Manhattan']
sub_4 = df.loc[df['neighbourhood_group'] == 'Queens']
sub_5 = df.loc[df['neighbourhood_group'] == 'Staten Island']


# In[ ]:


plt.figure()
#plt.xlim(0,500)
fig = sns.distplot(sub_2['price'],fit=norm);


# In[ ]:


res = stats.probplot(sub_1['price'], plot=plt)


# In[ ]:


#neighbourhood where airbnb concentrates on (top 10)
#pd.DataFrame
a = pd.DataFrame(df['neighbourhood'].value_counts().head(10).index.tolist())
#a = df.sort_values(by=['neighbourhood'], ascending=False).head(10)
neigh = []
neighh = []
#for i in range (0,10):
aa0 = a.iloc[0]
aa1 = a.iloc[1] 
aa2 = a.iloc[2]
aa3 = a.iloc[3]
neigh1 = df.loc[df['neighbourhood']== a.iloc[0][0]]
neigh2 = df.loc[df['neighbourhood']== aa1[0]]


# In[ ]:


#Finally I made it.
j = 0
neigh10 = []
for i in range (0,10):
    globals()["aa" + str(j)] = a.iloc[j]
    globals()["neigh" + str(j)] = df.loc[df['neighbourhood']== globals()["aa" + str(j)][0]]
    j += 1


# In[ ]:


neigh10 = pd.concat([neigh1, neigh2, neigh3, neigh4, neigh5, neigh6, neigh7, neigh8, neigh9])
neigh10.neighbourhood_group.value_counts()


# In[ ]:


import urllib
#initializing the figure size
plt.figure(figsize=(10,8))
#loading the png NYC image found on Google and saving to my local folder along with the project
i=urllib.request.urlopen('https://upload.wikimedia.org/wikipedia/commons/e/ec/Neighbourhoods_New_York_City_Map.PNG')
nyc_img=plt.imread(i)
#scaling the image based on the latitude and longitude max and mins for proper output
plt.imshow(nyc_img,zorder=0,extent=[-74.258, -73.7, 40.49,40.92])
ax=plt.gca()
#using scatterplot again
df.plot(kind='scatter', x='longitude', y='latitude', label='price', c='price', ax=ax, cmap=plt.get_cmap('jet'), colorbar=True, alpha=0.4, zorder=5)
plt.legend()
plt.show()


# In[ ]:


jupyter nbconvert-to script notebook_name.ipynb


# In[ ]:





# In[ ]:





# In[ ]:




