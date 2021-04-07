#!/usr/bin/env python
# coding: utf-8

# <a id='import_lib'></a>
# ## 1. Import Libraries

# **Let us import the required libraries and functions**

# In[1]:


# supress warnings 
from warnings import filterwarnings
filterwarnings('ignore')

# 'Os' module provides functions for interacting with the operating system 
import os

# 'Pandas' is used for data manipulation and analysis
import pandas as pd 

# 'Numpy' is used for mathematical operations on large, multi-dimensional arrays and matrices
import numpy as np

# 'Matplotlib' is a data visualization library for 2D and 3D plots, built on numpy
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# 'Seaborn' is based on matplotlib; used for plotting statistical graphics
import seaborn as sns

# 'Scikit-learn' (sklearn) emphasizes various regression, classification and clustering algorithms
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet

# 'Statsmodels' is used to build and analyze various statistical models
import statsmodels
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.tools.eval_measures import rmse
from statsmodels.compat import lzip
from statsmodels.graphics.gofplots import ProbPlot

# 'SciPy' is used to perform scientific computations
from scipy.stats import f_oneway
from scipy.stats import jarque_bera
from scipy import stats


# <a id='set_options'></a>
# ## 2. Set Options

# In[2]:


# display all columns of the dataframe
pd.options.display.max_columns = None

# display all rows of the dataframe
pd.options.display.max_rows = None

# return an output value upto 6 decimals
pd.options.display.float_format = '{:.6f}'.format


# <a id='Read_Data'></a>
# ## 3. Read Data

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="todo.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>Read and display data to get insights from the data<br> 
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# In[3]:


# read csv file using pandas
df_insurance = pd.read_csv("healthinsurance.csv")

# display the top 5 rows of the dataframe
df_insurance.head()


# <a id='data_preparation'></a>
# ## 4. Data Analysis and Preparation
# 

# <a id='Data_Understanding'></a>
# ### 4.1 Understand the Dataset

# <a id='Data_Shape'></a>
# ### 4.1.1 Data Dimension

# In[4]:


# .shape returns the dimension of the data
df_insurance.shape


# We see the dataframe has 13 columns and 15000 observations.

# <a id='Data_Types'></a>
# ### 4.1.2 Data Types

# **1. Check data types**

# In[5]:


# use .dtypes to view the data type of a variable
df_insurance.dtypes


# **2. Change the incorrect data types**

# In[6]:


# convert numeric variable 'smoker' to object (categorical) variable
df_insurance.smoker = df_insurance.smoker.astype('object')

# convert numeric variable 'diabetes' to object (categorical) variable
df_insurance.diabetes = df_insurance.diabetes.astype('object')

# convert 'regular_ex' variable diabetes to object (categorical) variable
df_insurance.regular_ex = df_insurance.regular_ex.astype('object')


# **3. Recheck the data types after the conversion**

# In[7]:


# recheck the data types using .dtypes
df_insurance.dtypes


# Note the data types are now as per the data definition. Now we can proceed with the analysis.

# <a id='Summary_Statistics'></a>
# ### 4.1.3 Summary Statistics

# **1. For numerical variables, we will use .describe()**

# In[8]:


# describe the numerical data
df_insurance.describe()


# **2. For categorical features, we will use .describe(include=object)**

# In[9]:


# describe the categorical data
df_insurance.describe(include = object)


# <a id='Missing_Values'></a>
# ### 4.1.4 Missing Values

# In[10]:


# obtain the total missing values for each variable
Total = df_insurance.isnull().sum().sort_values(ascending=False) 

# 'isnull().sum()' returns the number of missing values in each variable
Percent = (df_insurance.isnull().sum()*100/df_insurance.isnull().count()).sort_values(ascending=False)   

# concat the 'Total' and 'Percent' columns using 'concat' function
missing_data = pd.concat([Total, Percent], axis = 1, keys = ['Total', 'Percentage of Missing Values'])

# print the missing data
missing_data


# The missing values are present in the data for the `age` and `bmi` variables. There are 396 (2.6%) missing values for the variable `age` and 956 (6.4%) missing values for the variable `bmi`

# ### Visualize Missing Values using Heatmap

# In[11]:


# set the figure size
plt.figure(figsize=(15, 8))

# plot heatmap to check null values
sns.heatmap(df_insurance.isnull(), cbar=False)

# display the plot
plt.show()


# ### Deal with Missing Values

# In[12]:


# check the average age for male and female
df_insurance['age'].groupby(df_insurance['sex'], axis=0).mean()


# The average age for the male and female is nearly the same. We will fill in missing values with the mean age of the policyholder.

# In[13]:


# fill the missing values with the mean value of 'age' using 'fillna()'
df_insurance['age'].fillna(df_insurance['age'].mean(), inplace=True)


# Replace missing values by mean for the BMI.

# In[14]:


# fill the missing values with the mean value of 'bmi' using 'fillna()'
df_insurance['bmi'].fillna(df_insurance['bmi'].mean(), inplace=True)


# We have seen that the the minimum bloodpressure is 0, which is absurd. It implies that these are missing values. 
# Let us replace these missing values with the median value.

# In[15]:


# calculate the median of the bloodpressure using 'median()''
median_bloodpressure = df_insurance['bloodpressure'].median()

# replace zero values by median using 'replace()'
df_insurance['bloodpressure'] = df_insurance['bloodpressure'].replace(0,median_bloodpressure) 


# Recheck the summary statistics to confirm the missing value treatment for the variable 'bloodpressure'.

# In[16]:


# obtain the summary statistics of numeric variables using 'describe()'
df_insurance.describe()


# To confirm the data is valid, observe the minimum and maximum value of the variable `bloodpressure` is 40, which can be possible with patients suffering from low bloodpressure.

# View the missing value plot once again to see if the missing values have been imputed.

# In[17]:


# set the figure size
plt.figure(figsize=(15, 8))

# plot heatmap to check null values 
sns.heatmap(df_insurance.isnull(), cbar=False)

# display the plot
plt.show()


# Now, we obtain the dataset with no missing values.

# <a id='correlation'></a>
# ### 4.1.5 Correlation

# In[18]:


# select the numerical features in the dataset using 'select_dtypes()'
df_numeric_features = df_insurance.select_dtypes(include=np.number)

# print the names of the numeric variables 
df_numeric_features.columns


# The dataframe df_numeric_features has 6 numeric variables.

# In[19]:


# generate the correlation matrix
corr =  df_numeric_features.corr()

# print the correlation matrix
corr


# In[20]:


# set the figure size
plt.figure(figsize=(15, 8))

# plot the heat map
# corr: give the correlation matrix
# cmap: colour code used for plotting
# vmax: gives maximum range of values for the chart
# vmin: gives minimum range of values for the chart
# annot: prints the correlation values in the chart
# annot_kws: Sets the font size of the annotation
sns.heatmap(corr, cmap='YlGnBu', vmax=1.0, vmin=-1.0, annot = True, annot_kws={"size": 15}, )

# specify name of the plot using plt.title()
plt.title('Correlation between numeric features')

# display the plot
plt.show()


# <a id='categorical'></a>
# ### 4.1.6 Analyze Categorical Variables

# In[21]:


# display the summary statistics of categorical variables
df_insurance.describe(include=object)


# There are 6 categorical variables. From the output we see that the variable cities has most number of categories. There are 91 cities in the data, of which NewOrleans occurs highes number of times.
# 
# Let us visualize the variables. However, we shall exculde the variable `city` from it.

# In[22]:


# create a list of all categorical variables
df_categoric_features = df_insurance.select_dtypes(include='object').drop(['city'], axis=1)

# plot the count distribution for each categorical variable 
fig, ax = plt.subplots(3, 2, figsize=(25, 20))

# plot a count plot for all the categorical variables
for variable, subplot in zip(df_categoric_features, ax.flatten()):
    
    # plot the count plot using countplot()
    countplot = sns.countplot(y=df_insurance[variable], ax=subplot )
       
    # set the y-axis labels 
    countplot.set_ylabel(variable, fontsize = 30)

# avoid overlapping of the plots using tight_layout()    
plt.tight_layout()   

# display the plot
plt.show()


# Now consider the variable `city`.

# In[23]:


# set the figure size
plt.figure(figsize=(15, 15))

# plot the count plot using countplot()
countplot = sns.countplot(y=df_insurance['city'], orient="h")

# set the x-axis labels 
countplot.set_ylabel('City', fontsize = 30)

# display the plot
plt.show()


# <a id='categorical_numerical'></a>
# ### 4.1.7 Analyze Relationship Between Target and Categorical Variables

# In[24]:


# plot the boxplot for each categorical variable 
fig, ax = plt.subplots(3, 2, figsize=(25, 15))

# plot a boxplot for all the categorical variables 
for variable, subplot in zip(df_categoric_features, ax.flatten()):
    
    # x: variable on x-axis
    boxplt = sns.boxplot(x=variable, y='claim', data=df_insurance, ax=subplot)
    
    # set the x-axis labels 
    boxplt.set_xlabel(variable, fontsize = 30)

# avoid overlapping of the plots using tight_layout()    
plt.tight_layout()   

# display the plot
plt.show() 


# Since the variable `city` has 91 categories, we shall plot it separately.

# In[25]:


# set the figure size
plt.figure(figsize=(15, 8))

# plot the boxplot for categorical variable 'city'
ax = sns.boxplot(x=df_insurance["city"], y=df_insurance['claim'], data=df_insurance)

# set the x-axis labels 
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize = 10)

# display the plot
plt.show()


# <a id='Feature_Engineering'></a>
# ### 4.1.8 Feature Engineering

# In[26]:


# create a region column and combine the north-east cities
df_insurance['region'] = df_insurance['city'].replace(['NewYork', 'Boston', 'Phildelphia', 'Pittsburg', 'Buffalo',
                                                       'AtlanticCity','Portland', 'Cambridge', 'Hartford', 
                                                       'Springfield', 'Syracuse', 'Baltimore', 'York', 'Trenton',
                                                       'Warwick', 'WashingtonDC', 'Providence', 'Harrisburg',
                                                       'Newport', 'Stamford', 'Worcester'],
                                                      'North-East')


# In[27]:


# combine all the southern cities into the 'region' column
df_insurance['region'] = df_insurance['region'].replace(['Atlanta', 'Brimingham', 'Charleston', 'Charlotte',
                                                         'Louisville', 'Memphis', 'Nashville', 'NewOrleans',
                                                         'Raleigh', 'Houston', 'Georgia', 'Oklahoma', 'Orlando',
                                                         'Macon', 'Huntsville', 'Knoxville', 'Florence', 'Miami',
                                                         'Tampa', 'PanamaCity', 'Kingsport', 'Marshall'],
                                                         'Southern')


# In[28]:


# combine all the mid-west cities into the 'region' column
df_insurance['region'] = df_insurance['region'].replace(['Mandan', 'Waterloo', 'IowaCity', 'Columbia',
                                                         'Indianapolis', 'Cincinnati', 'Bloomington', 'Salina',
                                                         'KanasCity', 'Brookings', 'Minot', 'Chicago', 'Lincoln',
                                                         'FallsCity', 'GrandForks', 'Fargo', 'Cleveland', 
                                                         'Canton', 'Columbus', 'Rochester', 'Minneapolis', 
                                                         'JeffersonCity', 'Escabana','Youngstown'],
                                                         'Mid-West')


# In[29]:


# combine all the western cities into the 'region' column
df_insurance['region'] = df_insurance['region'].replace(['SantaRosa', 'Eureka', 'SanFrancisco', 'SanJose',
                                                         'LosAngeles', 'Oxnard', 'SanDeigo', 'Oceanside', 
                                                         'Carlsbad', 'Montrose', 'Prescott', 'Fresno', 'Reno',
                                                         'LasVegas', 'Tucson', 'SanLuis', 'Denver', 'Kingman',
                                                         'Bakersfield', 'Mexicali', 'SilverCity', 'Pheonix',
                                                         'SantaFe', 'Lovelock'],
                                                         'West')


# In[30]:


# check the unique values of the region using 'unique()'
df_insurance['region'].unique()


# In[31]:


df_insurance['region'].value_counts()


# In[32]:


# drop the 'city' variable from the dataset using drop()
df_insurance = df_insurance.drop(['city'], axis=1)


# Check whether the new variable added into the data frame or not.

# In[33]:


# display the top 5 rows of the dataframe
df_insurance.head()


# #### Analyze relationship between region and claim variable

# In[34]:


# set figure size
plt.figure(figsize=(15,8))

# boxplot of claim against region
ax = sns.boxplot(x="region", y="claim", data=df_insurance)

# rotate labels using set_ticklabels
ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=90)

# show the plot
plt.show()


# The plot shows that there is not much significant difference in the variance of the insurance claim across the regions.

# <a id='outliers'></a>
# ### 4.1.9 Discover Outliers

# **1. Plot boxplot for numerical data**

# In[35]:


# set the plot size
plt.rcParams['figure.figsize']=(15,8)

# create a boxplot for all numeric features
df_numeric_features.boxplot(column=['bmi', 'weight','no_of_dependents', 'bloodpressure', 'age'])

# to display the plot
plt.show()


# **2. Remove outliers by IQR method**

# In[36]:


# compute the first quartile using quantile(0.25)
Q1 = df_numeric_features.drop(['claim'], axis=1).quantile(0.25)

# compute the first quartile using quantile(0.75)
Q3 = df_numeric_features.drop(['claim'], axis=1).quantile(0.75)

# calculate of interquartile range 
IQR = Q3 - Q1

# print the IQR values for numeric variables
print(IQR)


# In[37]:


# filter out the outlier values
df_insurance = df_insurance[~((df_insurance < (Q1 - 1.5 * IQR)) | (df_insurance > (Q3 + 1.5 * IQR))).any(axis=1)]


# A simple way to know whether the outliers have been removed or not is to check the dimensions of the data. If the dimensions are reduced that implies outliers are removed

# In[38]:


# check the shape of data using shape
df_insurance.shape


# **4. Plot boxplot to recheck for outliers**

# In[39]:


# set figure size 
plt.rcParams['figure.figsize']=(15,8)

# recheck for outliers
df_insurance.boxplot(column=['bmi', 'weight','no_of_dependents', 'bloodpressure', 'age'])

# display only the plot
plt.show()


# Observing the range of the boxplot, we say that the outliers are removed from the original data. The new 'outliers' that you see are moderate outliers that lie within the min/max range before removing the actual outliers

# <a id='Recheck_Correlation'></a>
# ### 4.1.10 Recheck the Correlation

# In[40]:


# filter the numerical features in the dataset
df_numeric_features = df_insurance.select_dtypes(include=np.number)

# display the numeric features
df_numeric_features.columns


# In[41]:


# generate the correlation matrix 
corr =  df_numeric_features.corr()

# print the correlation matrix
corr


# In[42]:


# set the figure size
plt.figure(figsize=(15, 8))

# plot the heat map
sns.heatmap(corr, cmap='YlGnBu', vmax=1.0, vmin=-1.0, annot = True, annot_kws={"size": 15})

# specify name of the plot
plt.title('Correlation between numeric features')

# display the plot
plt.show()


# <a id='Data_Preparation'></a>
# ## 4.2 Prepare the Data

# <a id='Normality'></a>
# ### 4.2.1 Check for Normality

# **1. Plot a histogram and also perform the Jarque-Bera test**

# In[43]:


# check the distribution of target variable using hist()
df_insurance.claim.hist()

# display the plot
plt.show()


# Let us perform the Jarque-Bera test to check the normality of the target variable.

# The null and alternate hypothesis of Jarque-Bera test are as follows: <br>
#     
#     H0: The data is normally distributed
#     H1: The data is not normally distributed

# In[44]:


# normality test using jarque_bera()
stat, p = jarque_bera(df_insurance["claim"])

# to print the numeric outputs of the Jarque-Bera test upto 3 decimal places
print('Statistics=%.3f, p-value=%.3f' % (stat, p))

# display the conclusion
alpha = 0.05

# if the p-value is greater than alpha print we accept alpha 
if p > alpha:
    print('The data is normally distributed (fail to reject H0)')
else:
    print('The data is not normally distributed (reject H0)')


# **2. If the data is not normally distributed, use log transformation to get near normally distributed data**

# In[45]:


# log transformation for normality using np.log()
df_insurance['log_claim'] = np.log(df_insurance['claim'])

# display first 5 rows of the data
df_insurance.head()


# **3. Recheck for normality by plotting histogram and performing Jarque-Bera test**

# In[46]:


# recheck for normality 
df_insurance.log_claim.hist()

# display the plot
plt.show()


# Let us perform Jarque Bera test

# In[47]:


# recheck normality by Jarque-Bera test
statn, pv = jarque_bera(df_insurance['log_claim'])

# to print the numeric outputs of the Jarque-Bera test upto 3 decimal places
print('Statistics=%.3f, p-value=%.3f' % (stat, p))

# display the conclusion
alpha = 0.05

# if the p-value is greater than alpha print we accept alpha 
if p > alpha:
    print('The data is normally distributed (fail to reject H0)')
else:
    print('The data is not normally distributed (reject H0)')


# <a id='one_way_anova'></a>
# ### 4.2.2 One-Way Anova 

# #### One Way Anova for 'Sex' on 'Claim'

# In[48]:


# perform one way anova for sex on claim using f_oneway()
f_oneway(df_insurance['claim'][df_insurance['sex'] == 'male'], 
             df_insurance['claim'][df_insurance['sex'] == 'female'])


# The F-statistic = 68.99 and the p-value < 0.05, which indicates that there is a significant difference in the mean of the insurance claim across gender. We may consider building separate models for each gender. However, in this example we go ahead and build a single model for both genders.

# <a id='dummy'></a>
# ### 4.2.3 Dummy Encoding of Categorical Variables

# **1. Filter numerical and categorical variables**

# In[49]:


# filter the numerical features in the dataset using select_dtypes()
df_numeric_features = df_insurance.select_dtypes(include=np.number)

# display the numeric features
df_numeric_features.columns


# In[50]:


# filter the categorical features in the dataset using select_dtypes()
df_categoric_features = df_insurance.select_dtypes(include=[np.object])

# display categorical features
df_categoric_features.columns


# **2. Dummy encode the catergorical variables**

# In[51]:


# create data frame with only categorical variables that have been encoded

# for all categoric variables create dummy variables
for col in df_categoric_features.columns.values:
    
    # for a feature create dummy variables using get_dummies()
    dummy_encoded_variables = pd.get_dummies(df_categoric_features[col], prefix=col, drop_first=True)
    
    # concatenate the categoric features with dummy variables using concat()
    df_categoric_features = pd.concat([df_categoric_features, dummy_encoded_variables],axis=1)
    
    # drop the orginal categorical variable from the dataframe
    df_categoric_features.drop([col], axis=1, inplace=True)


# **3. Concatenate numerical and dummy encoded categorical variables**

# In[52]:


# concatenate the numerical and dummy encoded categorical variables using concat()
df_insurance_dummy = pd.concat([df_numeric_features, df_categoric_features], axis=1)

# display data with dummy variables
df_insurance_dummy.head()


# <a id='LinearRegression'></a>
# ## 5. Linear Regression (OLS)

# <a id='withLog'></a>
# ### 5.1 Multiple Linear Regression - Full Model - with Log Transformed Dependent Variable (OLS)

# **1. Split the data into training and test sets**

# In[53]:


# add the intercept column to the dataset
df_insurance_dummy = sm.add_constant(df_insurance_dummy)

# separate the independent and dependent variables
X = df_insurance_dummy.drop(['claim','log_claim'], axis=1)

# extract the target variable from the data set
y = df_insurance_dummy[['log_claim','claim']]

# split data into train subset and test subset for predictor and target variables
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# check the dimensions of the train & test subset for 

# print dimension of predictors train set
print("The shape of X_train is:",X_train.shape)

# print dimension of predictors test set
print("The shape of X_test is:",X_test.shape)

# print dimension of target train set
print("The shape of y_train is:",y_train.shape)

# print dimension of target test set
print("The shape of y_test is:",y_test.shape)


# **2. Build model using sm.OLS().fit()**

# In[54]:


# build a full model using OLS()
linreg_full_model_withlog = sm.OLS(y_train["log_claim"], X_train).fit()

# print the summary output
print(linreg_full_model_withlog.summary())


# **3. Predict the values using test set**

# In[55]:


# predict the 'log_claim' using predict()
linreg_full_model_withlog_predictions = linreg_full_model_withlog.predict(X_test)


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="todo.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b> Note that the predicted values are log transformed claim. In order to get claim values, we take the antilog of these predicted values by using the function np.exp()
# </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 
# 

# In[56]:


# take the exponential of predictions using np.exp()
predicted_claim = np.exp(linreg_full_model_withlog_predictions)

# extract the 'claim' values from the test data
actual_claim = y_test['claim']


# **4. Compute accuracy measures**
# 
# Now we calculate accuray measures like Root-mean-square-error (RMSE), R-squared and Adjusted R-squared.

# In[57]:


# calculate rmse using rmse()
linreg_full_model_withlog_rmse = rmse(actual_claim, predicted_claim)

# calculate R-squared using rsquared
linreg_full_model_withlog_rsquared = linreg_full_model_withlog.rsquared

# calculate Adjusted R-Squared using rsquared_adj
linreg_full_model_withlog_rsquared_adj = linreg_full_model_withlog.rsquared_adj 


# **5. Tabulate the results**

# In[58]:


# create dataframe 'score_card'
score_card = pd.DataFrame(columns=['Model_Name', 'R-Squared', 'Adj. R-Squared', 'RMSE'])

# print the score card
score_card


# In[59]:


# compile the required information
linreg_full_model_withlog_metrics = pd.Series({
                     'Model_Name': "Linreg full model with log of target variable",
                     'RMSE':linreg_full_model_withlog_rmse,
                     'R-Squared': linreg_full_model_withlog_rsquared,
                     'Adj. R-Squared': linreg_full_model_withlog_rsquared_adj     
                   })

# append our result table using append()
score_card = score_card.append(linreg_full_model_withlog_metrics, ignore_index=True)

# print the result table
score_card


# <a id='withoutLog'></a>
# ### 5.2 Multiple Linear Regression - Full Model - without Log Transformed Dependent Variable (OLS)

# **1. Build model using sm.OLS().fit()**

# In[60]:


# ordinary least squares regression

# build a full model using OLS()
linreg_full_model_withoutlog = sm.OLS(y_train['claim'], X_train).fit()

# print the summary output
print(linreg_full_model_withoutlog.summary())


# #### Calculate the p-values to know the insignificant variables

# In[61]:


# calculate the p-values for all the variables
linreg_full_model_withoutlog_pvalues = pd.DataFrame(linreg_full_model_withoutlog.pvalues, columns=["P-Value"])

# print the values
linreg_full_model_withoutlog_pvalues


# The above table shows the p-values for all the variables to decide the significant variables

# Let's create a list of insignificant variables

# In[62]:


# select insignificant variables
insignificant_variables = linreg_full_model_withoutlog_pvalues[
                                                        linreg_full_model_withoutlog_pvalues['P-Value']  > 0.05]

# get the position of a specified value
insigni_var = insignificant_variables.index

# convert the list of variables to 'list' type
insigni_var = insigni_var.to_list()

# get the list of insignificant variables
insigni_var


# **2. Predict the values using test set**

# In[63]:


# predict the claim using predict()
predicted_claim = linreg_full_model_withoutlog.predict(X_test)

# extract the 'claim' values from the test data
actual_claim = y_test['claim']


# **3. Compute model accuracy measures**
# 
# Now we calculate accuray measures like Root-mean-square-error (RMSE), R-squared and Adjusted R-squared.

# In[64]:


# calculate rmse using rmse()
linreg_full_model_withoutlog_rmse = rmse(actual_claim, predicted_claim)

# calculate R-squared using rsquared
linreg_full_model_withoutlog_rsquared = linreg_full_model_withoutlog.rsquared

# calculate Adjusted R-Squared using rsquared_adj
linreg_full_model_withoutlog_rsquared_adj = linreg_full_model_withoutlog.rsquared_adj 


# **4. Tabulate the results**

# In[65]:


# compile the required information
linreg_full_model_withoutlog_metrics = pd.Series({
                     'Model_Name': "Linreg full model without log of target variable",
                     'RMSE':linreg_full_model_withoutlog_rmse,
                     'R-Squared': linreg_full_model_withoutlog_rsquared,
                     'Adj. R-Squared': linreg_full_model_withoutlog_rsquared_adj     
                   })

# append our result table using append()
score_card = score_card.append(linreg_full_model_withoutlog_metrics, ignore_index=True)

# print the result table
score_card


# <a id='Finetuning'></a>
# ## 5.3. Fine Tune Linear Regression Model (OLS)

# <a id='RemovingInsignificantVariable'></a>
# ### 5.3.1 Linear Regression after Removing Insignificant Variable (OLS)
# 
# The null and alternate hypothesis of linear regression as follows: <br>
#     
#     H0: All beta coefficients are zero
#     H1: At least one beta coefficient is not zero

# **1. Consider the significant variables**

# In[66]:


# drop the insignificant variables
X_significant = df_insurance.drop(["sex","job_title","region","claim","log_claim"], axis=1)


# In[67]:


# filter the categorical features in the dataset using select_dtypes()
df_significant_categoric_features = X_significant.select_dtypes(include=[np.object])

# display categorical features
df_significant_categoric_features.columns


# **Dummy encode the catergorical variables**

# In[68]:


# create data frame with only categorical variables that have been encoded

# for all categoric variables create dummy variables
for col in df_significant_categoric_features.columns.values:
    
    # for a feature create dummy variables using get_dummies()
    dummy_encoded_variables = pd.get_dummies(df_significant_categoric_features[col], prefix=col, drop_first=True)
    
    # concatenate the categoric features with dummy variables using concat()
    df_significant_categoric_features = pd.concat([df_significant_categoric_features, dummy_encoded_variables],axis=1)
    
    # drop the orginal categorical variable from the dataframe
    df_significant_categoric_features.drop([col], axis=1, inplace=True)


# **Concatenate numerical and dummy encoded categorical variables**

# In[69]:


# concatenate the numerical and dummy encoded categorical variables using concat()
df_insurance_significant = pd.concat([df_numeric_features, df_significant_categoric_features], axis=1)

# display data with dummy variables
df_insurance_significant.head()


# **2. Split the data into training and test sets**

# In[70]:


# add the intercept column to the dataset
df_insurance_significant = sm.add_constant(df_insurance_significant)

# separate the independent and dependent variables
X = df_insurance_significant.drop(['claim','log_claim'], axis=1)

# extract the target variable from the data set
y = df_insurance_significant[['log_claim','claim']]

# split data into train subset and test subset for predictor and target variables
X_train_significant, X_test_significant, y_train, y_test = train_test_split(X, y, random_state=1)

# check the dimensions of the train & test subset for 

# print dimension of predictors train set
print("The shape of X_train is:",X_train.shape)

# print dimension of predictors test set
print("The shape of X_test is:",X_test.shape)

# print dimension of target train set
print("The shape of y_train is:",y_train.shape)

# print dimension of target test set
print("The shape of y_test is:",y_test.shape)


# **1. Build model using sm.OLS().fit()**

# In[71]:


# build a full model with significant variables using OLS()
linreg_model_with_significant_var = sm.OLS(y_train['claim'], X_train_significant).fit()

# to print the summary output
print(linreg_model_with_significant_var.summary())


# **2. Predict the values using test set**

# In[72]:


# predict the 'claim' using predict()
predicted_claim = linreg_model_with_significant_var.predict(X_test_significant)

# extract the 'claim' values from the test data
actual_claim = y_test['claim']


# **3. Compute model accuracy measures**
# 
# Now we calculate accuray measures like Root-mean-square-error (RMSE), R-squared and Adjusted R-squared.

# In[73]:


# calculate rmse using rmse()
linreg_model_with_significant_var_rmse = rmse(actual_claim, predicted_claim)

# calculate R-squared using rsquared
linreg_model_with_significant_var_rsquared = linreg_model_with_significant_var.rsquared

# calculate Adjusted R-Squared using rsquared_adj
linreg_model_with_significant_var_rsquared_adj = linreg_model_with_significant_var.rsquared_adj 


# **4. Tabulate the results**

# In[74]:


# compile the required information
linreg_model_with_significant_var_metrics = pd.Series({
                     'Model_Name': "Linreg full model with significant variables",
                     'RMSE': linreg_model_with_significant_var_rmse,
                     'R-Squared': linreg_model_with_significant_var_rsquared,
                     'Adj. R-Squared': linreg_model_with_significant_var_rsquared_adj     
                   })

# append our result table using append()
score_card = score_card.append(linreg_model_with_significant_var_metrics, ignore_index=True)

# print the result table
score_card


# <a id='Assumptions'></a>
# ### 5.3.2 Check the Assumptions of the Linear Regression
# 
# Now we perform test for checking presence of Autocorrelation and Heteroskedasticity.

# <a id='Autocorrelation'></a>
# ### 5.3.2.1 Detecting Autocorrelation
# 
# The null and alternate hypothesis of Durbin-Watson test is as follows: <br>
#         
#           H0: There is no autocorrelation in the residuals
#           H1: There is autocorrelation in the residuals

# <a id='Heteroskedasticity'></a>
# ### 5.3.2.2 Detecting Heteroskedasticity
# The null and alternate hypothesis of Breusch-Pagan test is as follows:<BR>
#     
#     H0: The residuals are homoskedastic
#     H1: The residuals are not homoskedastic

# In[75]:


# create vector of result parmeters
name = ['f-value','p-value']           

# perform Breusch-Pagan test using het_breushpagan()
test = sms.het_breuschpagan(linreg_model_with_significant_var.resid, linreg_model_with_significant_var.model.exog)

# print the output
lzip(name, test)           


# <a id='Linearity_of_Residuals'></a>
# ### 5.3.2.3 Linearity of Residuals

# In[76]:


# create subplots of scatter plots
fig, ax = plt.subplots(nrows = 4, ncols= 5, figsize=(25, 20))

# use for loop to create scatter plot for residuals and each independent variable (do not consider the intercept)
for variable, subplot in zip(X_train_significant.columns[1:], ax.flatten()):
    sns.scatterplot(X_train_significant[variable], linreg_model_with_significant_var.resid , ax=subplot)

# display the plot
plt.show()


# <a id='Normality_of_Residuals'></a>
# ### 5.3.2.4 Normality of Residuals

# In[77]:


# calculate fitted values
fitted_vals = linreg_model_with_significant_var.predict(X_test_significant)

# calculate residuals
resids = actual_claim - fitted_vals

# create subplots using subplots() such that there is one row having one plot
fig, ax = plt.subplots(1, 1, figsize=(15, 8))

# plot the probability plot to check the normality of the residuals
stats.probplot(resids, plot=plt)

# set the marker type using the set_marker() parameter
ax.get_lines()[0].set_marker('o')

# set the marker size using the set_markersize() parameter
ax.get_lines()[0].set_markersize(5.0)

# set the color of the trend line using set_markerfacecolor()
ax.get_lines()[0].set_markerfacecolor('r')

# set the trend line width
ax.get_lines()[1].set_linewidth(4.0)

# display the plot
plt.show()


# **The mean of the residuals always equals zero (assuming that your line is actually the line of “best fit”)** 

# In[78]:


# check the mean of the residual
linreg_model_with_significant_var.resid.mean()


# The mean of the residuals is very much closer to zero. Therefore, we can say that linearity is present.

# **Perform Jarque Bera test to check normality of the residuals**

# In[79]:


# normality test using 'jarque_bera'
stat, p = jarque_bera(resids)

# to print the numeric outputs of the Jarque-Bera test upto 3 decimal places
print('Statistics=%.3f, p-value=%.3f' % (stat, p))

# display the conclusion
alpha = 0.05

# if the p-value is greater than alpha print we accept alpha 
if p > alpha:
    print('The data is normally distributed (fail to reject H0)')
else:
    print('The data is not normally distributed (reject H0)')


# <a id='RemovingInsignificantVariable_scaleddata'></a>
# ### 5.3.3 Linear Regression after Removing Insignificant Variable (OLS) - Scaled Data

# **1. Perform standardization on train data**

# In[80]:


# scale the numeric variables
df_num_scaled = df_numeric_features.apply(lambda rec: (rec - rec.mean()) / rec.std())

# create a dataframe
df_insurance_scaled = pd.concat([df_num_scaled, df_categoric_features], axis = 1)

# add constant to the data
df_insurance_scaled =sm.add_constant(df_insurance_scaled)


# In[81]:


# on doing the predictions to compute RMSE the we need to unscale the predictions
mean_numeric_features = df_numeric_features.mean()

# store the standard deviation as 'std_numeric_features'
std_numeric_features = df_numeric_features.std()


# **2. Split the data into training and test sets**

# In[82]:


# separate the independent and dependent variables
X = df_insurance_scaled.drop(['claim','log_claim'], axis=1)

# extract the target variable from the train set
y = df_insurance_scaled['claim']

# split data into train subset and test subset for predictor and target variables
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X, y, random_state=1)

# check the dimensions of the train & test subset for 

# print dimension of predictors train set
print("The shape of X_train_scaled is:",X_train_scaled.shape)

# print dimension of predictors test set
print("The shape of X_test_scaled is:",X_test_scaled.shape)

# print dimension of target train set
print("The shape of y_train_scaled is:",y_train_scaled.shape)

# print dimension of target test set
print("The shape of y_test_scaled is:",y_test_scaled.shape)


# **3. Consider only the significant variables**

# In[83]:


# consider the significant variable in the training set
X_train_scaled_significant = X_train_scaled.drop(insigni_var, axis=1)

# consider the significant variable in the training set
X_test_scaled_significant = X_test_scaled.drop(insigni_var, axis=1)


# **4. Build model using sm.OLS().fit()**

# In[84]:


# ordinary least squares regression
linreg_model_with_significant_scaled_vars = sm.OLS(y_train_scaled,X_train_scaled_significant).fit()

# to print the summary output
print(linreg_model_with_significant_scaled_vars.summary())


# **5. Predict the values using test set**

# In[85]:


# predict the 'claim' using predict()
predicted_claim = linreg_model_with_significant_scaled_vars.predict(X_test_scaled_significant)

# unscale the data
y_pred_unscaled = (predicted_claim * std_numeric_features.claim) + mean_numeric_features.claim

# extract the 'claim' values from the test data
actual_claim = y_test['claim']


# **6. Compute model accuracy measures**
# 
# Now we calculate accuray measures like Root-mean-square-error (RMSE), R-squared and Adjusted R-squared.

# In[86]:


# calculate rmse using rmse()
linreg_model_with_significant_scaled_vars_rmse = rmse(actual_claim, y_pred_unscaled)

# calculate R-squared using rsquared
linreg_model_with_significant_scaled_vars_rsquared = linreg_model_with_significant_scaled_vars.rsquared

# calculate Adjusted R-Squared using rsquared_adj
linreg_model_with_significant_scaled_vars_rsquared_adj = linreg_model_with_significant_scaled_vars.rsquared_adj 


# **7. Tabulate the results**

# In[87]:


# compile the required information
linreg_model_with_significant_scaled_vars_metrics = pd.Series({
                     'Model_Name': "Linreg with scaled significant variables",
                     'RMSE': linreg_model_with_significant_scaled_vars_rmse,
                     'R-Squared': linreg_model_with_significant_scaled_vars_rsquared,
                     'Adj. R-Squared': linreg_model_with_significant_scaled_vars_rsquared_adj     
                   })

# append our result table using append()
score_card = score_card.append(linreg_model_with_significant_scaled_vars_metrics, ignore_index = True)

# print the result table
score_card


# <a id='Interaction'></a>
# ### 5.3.4 Linear Regression with Interaction (OLS)

# **1. Compute the interaction effect**

# In[88]:


# create a copy of the entire dataset to add the interaction effect using copy()
df_insurance_interaction = df_insurance_dummy.copy()

# add the interaction variable
df_insurance_interaction['bmi*smoker'] = df_insurance_interaction['bmi']*df_insurance_interaction['smoker_1'] 

# print the data with interaction
df_insurance_interaction.head()


# **2. Split the data into training and test sets**
# 
# Notice that there is a change in the data set. We have added a variable(bmi*smoker), we consider this data as our new data. We again spilt the data into a train and test set keeping the random_state as the same as before (please refer section 5.1 to confirm)

# In[89]:


# separate the independent and dependent variables
X = df_insurance_interaction.drop(['claim','log_claim'], axis=1)

# extract the target variable from the train set
y = df_insurance_interaction['claim']

# split data into train subset and test subset for predictor and target variables
X_train_interaction, X_test_interaction, y_train, y_test = train_test_split( X, y, random_state=1)

# check the dimensions of the train & test subset for 

# print dimension of predictors train set
print("The shape of X_train_interaction is:",X_train_interaction.shape)

# print dimension of predictors test set
print("The shape of X_test_interaction is:",X_test_interaction.shape)

# print dimension of target train set
print("The shape of y_train is:",y_train.shape)

# print dimension of target test set
print("The shape of y_test is:",y_test.shape)


# **3. Build model using sm.OLS().fit()**

# In[90]:


# building a full model with an interaction term using OLS()
linreg_with_interaction = sm.OLS(y_train, X_train_interaction).fit()

# print the summary output
print(linreg_with_interaction.summary())


# **4. Predict the values using test set**

# In[91]:


# predict the 'claim' using predict()
predicted_claim = linreg_with_interaction.predict(X_test_interaction)

# extract the 'claim' values from the test data
actual_claim = y_test


# **5. Compute model accuracy measures**
# 
# Now we calculate accuray measures like Root-mean-square-error (RMSE), R-squared and Adjusted R-squared.

# In[92]:


# calculate rmse using rmse()
linreg_with_interaction_rmse = rmse(actual_claim, predicted_claim)

# calculate R-squared using rsquared
linreg_with_interaction_rsquared = linreg_with_interaction.rsquared

# calculate Adjusted R-Squared using rsquared_adj
linreg_with_interaction_rsquared_adj = linreg_with_interaction.rsquared_adj 


# **6. Tabulate the results**

# In[93]:


# compile the required information
linreg_with_interaction_metrics = pd.Series({
                     'Model_Name': "linreg_with_interaction",
                     'RMSE': linreg_with_interaction_rmse,
                     'R-Squared': linreg_with_interaction_rsquared,
                     'Adj. R-Squared': linreg_with_interaction_rsquared_adj     
                   })

# append our result table using append()
# ignore_index=True: does not use the index labels
# python can only append a Series if ignore_index=True or if the Series has a name
score_card = score_card.append(linreg_with_interaction_metrics, ignore_index = True)

# print the result table
score_card


# <a id='regularization'></a>
# ## 6. Regularization (OLS)

# <a id='Ridge_Regression'></a>
# ### 6.1 Ridge Regression (OLS)

# **1. Split the data in the form of train and test sets**

# We shall use the scaled data. The train test split is alread conducted for the scaled data.

# **2. For different values of alpha create a tabular representation of parameter estimates and accuracy metrics**

# We first create a list of all the variable names and accuracy metrics whose values we want.

# In[94]:


# compile the column names of the output dataframe

# create list of variable names
col = list(X_train_scaled.columns)

# add 'Sum of square of Residuals' to the list
col.append('ssr')

# add 'R squared' to the list
col.append('R squared')

# add 'Adj. R squared' to the list
col.append('Adj. R squared')

# add 'Root mean Squared Error' to the list
col.append('RMSE')


# Fit a linear regression model by the OLS method.

# In[95]:


# build a OLS model using OLS()
ridge_regression = sm.OLS(y_train_scaled, X_train_scaled)

# the fitted ridge model
results_fu = ridge_regression.fit()


# In a for loop we pass different values of alpha. We tabulate all beta coeffients.

# In[96]:


# create an empty list
frames = []

# a 'for' loop for values of alpha in a given range
for n in np.arange(0.0001,10.1, 0.1).tolist():
    
    # fit a rigde regression to the linear model built using OLS
    results_fr = ridge_regression.fit_regularized(L1_wt=0, alpha=n, start_params=results_fu.params)
    
    # fit the model
    results_fr_fit = sm.regression.linear_model.OLSResults(model=ridge_regression, 
                                             params=results_fr.params, 
                                             normalized_cov_params=ridge_regression.normalized_cov_params)
    
    # compute the rmse
    results_fr_fit_predictions = results_fr_fit.predict(X_test_scaled)
    
    # obtain the rmse
    results_fr_fit_rmse = rmse(y_test, results_fr_fit_predictions)    
  
    # compile the necessary metrics used for comparing the models
    list_metric = [results_fr_fit.ssr, results_fr_fit.rsquared, results_fr_fit.rsquared_adj, results_fr_fit_rmse]
    
    # append the empty list 
    frames.append(np.append(results_fr.params, list_metric))
    
    # converting the list to a dataframe.
    df_params = pd.DataFrame(frames, columns= col)

# add column names to the dataframe  
df_params.index=np.arange(0.0001, 10.1, 0.1).tolist()

# add the first column name alpha to the data frame. 
df_params.index.name = 'alpha*'

# display the output
df_params


# Let us compare the results for low and high values of alpha

# In[97]:


# call the first two rows and last two rows
df_params.iloc[[0,1,-2,-1]]


# **3. Find the alpha for which RMSE is minimum**
# 
# Now, to know which model performs the best, we find the alpha value which has lowest root mean squared error.

# In[98]:


# find the alpha value for which RMSE is minimum
alpha = df_params.RMSE[df_params.RMSE == df_params.loc[:,'RMSE'].min()].index.tolist()

# print the conclusion
print('The required alpha is %.4f ' % (alpha[0]))


# Thus, we may say the model obtained by alpha = 0.0001 is performing the best since has the lowest root mean squared error.

# **In ridge regression, coefficients may tend to zero yet are never zero. This can be checked by the follwing code**

# In[99]:


# check for zeros in the above coefficients table
df_params.apply(lambda x: sum(x.values==0),axis=1)


# For all alpha values, the corresponding value is zero which impiles there are **no** coefficients with value zero.

# **4. Fit a ridge model by substituting the alpha value obtained in step 3**
# 
# We know that when aplha = 0.0001, the model performs better than other since it has the lowest RMSE and highest adjusted R- squared value. Let us now find its summary output.

# In[100]:


# build a ridge model for the desired alpha 
results_fr = ridge_regression.fit_regularized(L1_wt=0, alpha=0.0001, start_params=results_fu.params)

# fit the model 
ridge_regression_best = sm.regression.linear_model.OLSResults(model = ridge_regression, 
                                        params=results_fr.params, 
                                        normalized_cov_params=ridge_regression.normalized_cov_params)

# print the summary output 
print (ridge_regression_best.summary())


# **5. Predict the values using test set**

# In[101]:


# predict the scaled claim using predict()
predicted_claim = ridge_regression_best.predict(X_test_scaled)

# unscale the data
y_pred_unscaled = (predicted_claim * std_numeric_features.claim) + mean_numeric_features.claim

# extract the 'claim' values from the test data
actual_claim = y_test


# **6. Compute model accuracy measures**
# 
# Now we calculate accuray measures like Root-mean-square-error (RMSE), R-squared and Adjusted R-squared.

# In[102]:


# calculate rmse using rmse()
ridge_regression_best_rmse = rmse(actual_claim, y_pred_unscaled)

# calculate R-squared using rsquared
ridge_regression_best_rsquared = ridge_regression_best.rsquared

# calculate Adjusted R-Squared using rsquared_adj
ridge_regression_best_rsquared_adj = ridge_regression_best.rsquared_adj 


# **7. Tabulate the results**

# In[103]:


# compile the required information
ridge_regression_best_metrics = pd.Series({
                     'Model_Name': "Ridge Regression",
                     'RMSE': ridge_regression_best_rmse,
                     'R-Squared': ridge_regression_best_rsquared,
                     'Adj. R-Squared': ridge_regression_best_rsquared_adj     
                   })

# append our result table using append()
score_card = score_card.append(ridge_regression_best_metrics, ignore_index = True)

# print the result table
score_card


# <a id='Lasso_Regression'></a>
# ### 6.2 Lasso Regression (OLS)

# **1. Fit a regression model using OLS method**

# In[104]:


# we use the scaled data
lasso_regression = sm.OLS(y_train_scaled, X_train_scaled)

# the fitted lasso model
results_fu = lasso_regression.fit()


# **2. For different values of alpha create a tabular representation of parameter estimates and accuracy meterics**

# In[105]:


# create an empty list
frames = []

# a 'for' loop for values of alpha in a given range
for n in np.arange(0.0001, 0.02, 0.0001).tolist():
    
    # fit a lasso regression to the linear model built using OLS
    results_fr = lasso_regression.fit_regularized(L1_wt=1, alpha=n, start_params=results_fu.params)
     
    # fit the model 
    results_fr_fit = sm.regression.linear_model.OLSResults(model=lasso_regression, 
                                                params=results_fr.params, 
                                                normalized_cov_params=lasso_regression.normalized_cov_params)
    
    # calculate the rmse
    results_fr_fit_predictions = results_fr_fit.predict(X_test_scaled)
    
    # obtain the rmse
    results_fr_fit_rmse = rmse(y_test, results_fr_fit_predictions)    
  
    # compile the necessary metrics used for comparing the models
    list_metric = [results_fr_fit.ssr, results_fr_fit.rsquared, results_fr_fit.rsquared_adj, results_fr_fit_rmse]
    
    # append the empty list 
    frames.append(np.append(results_fr.params, list_metric))
    
    # convert the list to a dataframe
    df_params = pd.DataFrame(frames, columns= col)

# add column names to the dataframe    
df_params.index=np.arange(0.0001, 0.02, 0.0001).tolist()

# add the first column name alpha to the data frame
df_params.index.name = 'alpha*'

# display output
df_params


# Let us compare the results for low and high values of alpha

# In[106]:


# call the first two rows and last two rows
df_params.iloc[[0,1,-2,-1]]


# **3. Find the alpha for which RMSE is minimum**
# 
# Now to know which model performs the best, we find the alpha value which has lowest root mean squared error

# In[107]:


# find the alpha value for which RMSE is minimum
alpha = df_params.RMSE[df_params.RMSE == df_params.loc[:,'RMSE'].min()].index.tolist()

# print the conclusion
print('The required alpha is %.4f ' % (alpha[0]))


# We may say the model obtained by alpha = 0.0001 is performing the best.

# **In lasso regression, coefficients may be zero. This can be checked by the following code**

# In[108]:


# check for zeros in the above coefficients table
df_params.apply(lambda x: sum(x.values==0),axis=1)


# **4. Fit a lasso model by substituting the alpha value obtained in step 3**

# In[109]:


# fit the lasso regression model
results_fr = lasso_regression.fit_regularized(L1_wt=1, alpha=0.0001, start_params=results_fu.params)

# fit the lasso regression 
lasso_regression_best = sm.regression.linear_model.OLSResults(model=lasso_regression, 
                                              params=results_fr.params, 
                                              normalized_cov_params=lasso_regression.normalized_cov_params)

# print the summary output
print (lasso_regression_best.summary())


# **5. Predict the values using test set**

# In[110]:


# predict the 'claim' using predict()
predicted_claim = lasso_regression_best.predict(X_test_scaled)

# unscale the data
y_pred_unscaled = (predicted_claim * std_numeric_features.claim) + mean_numeric_features.claim

# extract the 'claim' values from the test data
actual_claim = y_test


# **6. Compute model accuracy measures**
# 
# Now we calculate accuray measures like Root-mean-square-error (RMSE), R-squared and Adjusted R-squared.

# In[111]:


# calculate rmse using rmse()
lasso_regression_best_rmse = rmse(actual_claim, y_pred_unscaled)

# calculate R-squared using rsquared
lasso_regression_best_rsquared = lasso_regression_best.rsquared

# calculate Adjusted R-Squared using rsquared_adj
lasso_regression_best_rsquared_adj = lasso_regression_best.rsquared_adj 


# **7. Tabulate the results**

# In[112]:


# compile the required information
lasso_regression_best_metrics = pd.Series({
                     'Model_Name': "Lasso Regression",
                     'RMSE': lasso_regression_best_rmse,
                     'R-Squared': lasso_regression_best_rsquared,
                     'Adj. R-Squared': lasso_regression_best_rsquared_adj     
                   })

# append our result table using append()
score_card = score_card.append(lasso_regression_best_metrics, ignore_index = True)

# print the result table
score_card


# <a id='Elastic_Net'></a>
# ### 6.3 Elastic Net Regression (OLS)

# **1. Fit a regression model using OLS method**

# In[113]:


# build a OLS model using OLS()
elastic_net_regression = sm.OLS(y_train_scaled, X_train_scaled)

# the fitted elastic net model
results_fu = elastic_net_regression.fit()


# **2. Use GridsearchCV to find the best penalty term**

# When the l1_ratio is set to 0 it is the same as ridge regression. When l1_ratio is set to 1 it is lasso. Elastic net is somewhere between 0 and 1 when setting the l1_ratio. Therefore, in our grid, we need to set several values of this argument. Below is the code.

# In[114]:


# normalize the data which is required for elastic net
elastic = ElasticNet(normalize=True)

# use gridsearchCV to find best penalty term 
search = GridSearchCV(estimator=elastic, 
                      param_grid={'l1_ratio':[.2,.4,.6,.8]},
                      scoring='neg_mean_squared_error', 
                      n_jobs=1, 
                      refit=True, 
                      cv=10)


# In[115]:


# fit the model to get best parameter
search.fit(X_train_scaled, y_train_scaled)

# get best parameter
search.best_params_


# The best l1_ratio is 0.2.

# **3. For different values of alpha create a tabular representation of parameter estimates and accuracy metrics**

# In[116]:


# create an empty list
frames = []

# a 'for' loop for values of alpha in a given range
for n in np.arange(0.0001, 1.5, 0.01).tolist():
    
    # fitting a elastic net regression to the elastic net model built using OLS 
    results_fr = elastic_net_regression.fit_regularized(method='elastic_net', 
                                                        L1_wt= 0.2, 
                                                        alpha=n, 
                                                        start_params=results_fu.params)
     
    # obtaining the parameters of the fitted model 
    results_fr_fit = sm.regression.linear_model.OLSResults(model=elastic_net_regression, 
                                    params=results_fr.params, 
                                    normalized_cov_params=elastic_net_regression.normalized_cov_params)
    
    # calculate rmse
    # call the test data
    results_fr_fit_predictions = results_fr_fit.predict(X_test_scaled)
    
    # obtaining the rmse
    results_fr_fit_rmse = rmse(y_test, results_fr_fit_predictions)    
  
    # compiling the necessary metrics used for comparing the models
    list_metric = [results_fr_fit.ssr, results_fr_fit.rsquared, results_fr_fit.rsquared_adj, results_fr_fit_rmse]
    
    # appending the empty list 
    frames.append(np.append(results_fr.params, list_metric))
    
    # converting the list to a dataframe
    df_params = pd.DataFrame(frames, columns= col)

# add column names to the dataframe    
df_params.index=np.arange(0.0001, 1.5, 0.01).tolist()

# add the first column name alpha to the data frame
df_params.index.name = 'alpha*'

# display output
df_params


# In[117]:


# to call the first two rows and last two rows
df_params.iloc[[0,1,-2,-1]]


# **4. Find the alpha for which RMSE is minimum**
# 
# Now to know which model performs the best, we find the alpha value which has lowest root mean squared error.

# In[118]:


# find the alpha value for which RMSE is minimum
# from the column EMSE of df_params obtain the minimum value of RMSE
# .index.tolist(): gets the corresponding index value, i.e. the alpha value
alpha = df_params.RMSE[df_params.RMSE == df_params.loc[:,'RMSE'].min()].index.tolist()

# print the conclusion
print('The required alpha is %.4f ' % (alpha[0]))


# We may say the model obtained by alpha = 0.0001 is performing the best.

# **In elastic net regression, coefficients may be zero. This can be checked by the following code.**

# In[119]:


# check for zeros in the above coefficients table
df_params.apply(lambda x: sum(x.values==0),axis=1)


# **5. Fit a elastic net model by substituting the alpha value obtained in step 4**

# In[120]:


# fit the elastic net regression model
results_fr = elastic_net_regression.fit_regularized(method='elastic_net', 
                                                    L1_wt=0.2, 
                                                    alpha= 0.0001, 
                                                    start_params=results_fu.params)

# fit elastic net regression
elastic_net_regression_best = sm.regression.linear_model.OLSResults(model=elastic_net_regression, 
                                              params=results_fr.params, 
                                              normalized_cov_params=elastic_net_regression.normalized_cov_params)

# print the summary output
print (elastic_net_regression_best.summary())


# **6. Predict the values using test set**

# In[121]:


# predict the 'claim' using predict()
predicted_claim = elastic_net_regression_best.predict(X_test_scaled)

# unscale the data
y_pred_unscaled = (predicted_claim * std_numeric_features.claim) + mean_numeric_features.claim

# extract the 'claim' values from the test data
actual_claim = y_test


# **7. Compute model accuracy measures**
# 
# Now we calculate accuray measures like Root-mean-square-error (RMSE), R-squared and Adjusted R-squared.

# In[122]:


# calculate rmse using rmse()
elastic_net_regression_best_rmse = rmse(actual_claim, y_pred_unscaled)

# calculate R-squared using rsquared
elastic_net_regression_best_rsquared = elastic_net_regression_best.rsquared

# calculate Adjusted R-Squared using rsquared_adj
elastic_net_regression_best_rsquared_adj = elastic_net_regression_best.rsquared_adj 


# **8. Tabulate the results**

# In[123]:


# compile the required information
elastic_net_regression_best_metrics = pd.Series({
                     'Model_Name': "Elastic net Regression",
                     'RMSE': elastic_net_regression_best_rmse,
                     'R-Squared': elastic_net_regression_best_rsquared,
                     'Adj. R-Squared': elastic_net_regression_best_rsquared_adj     
                   })

# append our result table using append()
score_card = score_card.append(elastic_net_regression_best_metrics, ignore_index = True)

# print the result table
score_card


# <a id='StochasticGradientDescent'></a>
# ## 7. Stochastic Gradient Descent - SGD

# <a id='LinearRegressionwithStochasticGradientDescent'></a>
# ### 7.1 Linear Regression with SGD

# **1. Create the train and test sets**
# 
# We use the orginal unscaled data.
# 
# The data is already split as train and training set.

# **2. Fit the linear regression using the SGD**

# In[124]:


# import SGDRegressor from sklearn to perform linear regression with stochastic gradient descent
from sklearn.linear_model import SGDRegressor

# build the model
linreg_with_SGD = SGDRegressor()

# we fit our model with train data
linreg_with_SGD = linreg_with_SGD.fit(X_train, y_train)


# **3. Predict the values using test set**

# In[125]:


# we use predict() to predict our values
linreg_with_SGD_predictions = linreg_with_SGD.predict(X_test)


# **4. Compute accuracy measures**
# 
# Now we calculate accuray measures Root-mean-square-error (RMSE), R-squared and Adjusted R-squared. 

# In[126]:


# calculate mse
linreg_SGD_mse = mean_squared_error(y_test, linreg_with_SGD_predictions)

# calculate rmse
linreg_SGD_rmse = np.sqrt(linreg_SGD_mse)

# calculate R-squared
linreg_SGD_r_squared = r2_score(y_test, linreg_with_SGD_predictions)

# calculate Adjusted R-squared
linreg_SGD_adjusted_r_squared = 1 - (1-linreg_SGD_r_squared)*(len(y_test)-1)/(len(y_test)- X_test.shape[1]-1)


# **5. Tabulate the results**

# In[127]:


# compile the required information
linreg_full_model_SGD = pd.Series({
                     'Model_Name': "Linear Regression SGD",
                     'RMSE': linreg_SGD_rmse ,
                     'R-Squared': linreg_SGD_r_squared,
                     'Adj. R-Squared': linreg_SGD_adjusted_r_squared   
                   })

# append our result table using append()
score_card = score_card.append(linreg_full_model_SGD, ignore_index = True)

# print the result table
score_card


# <a id='StochasticGradientDescentwithGridSearchCV'></a>
# ### 7.2 Linear Regression with SGD using GridSearchCV

# **1. Fit the linear regression using the SGD with GridSearchCV**

# In[128]:


# to supress the warnings 
from warnings import filterwarnings
filterwarnings('ignore')

# import gridsearchcv from sklearn to optimize best parameter
from sklearn.model_selection import GridSearchCV

# Grid search 
param_grid = { 'alpha': 10.0 ** -np.arange(1, 7), 
        
               'loss': ['squared_loss'], 
    
               'penalty': ['l2', 'l1', 'elasticnet'],
    
               'learning_rate': ['constant', 'optimal', 'invscaling']
}

# using sklearn’s GridSearchCV, we define our grid of parameters to search over and then run the grid search
clf = GridSearchCV(linreg_with_SGD, param_grid)

# fit the model on train data
clf.fit(X_train, y_train)


# Let us print the optimal parameters obtained by using GridSearchCV.

# In[129]:


# print the best parameters
print('Best alpha:', clf.best_estimator_.alpha)

# print best tolerance
print('Best tol:', clf.best_estimator_.tol) 

# print best starting rate (Use eta0 to specify the starting learning)
print('Best eta0:', clf.best_estimator_.eta0)

# print best learning rate
print('Best learning rate:', clf.best_estimator_.learning_rate) 


# We have obtained the optimal parameters. Now substituting these values in SGDRegressor() we build the model.

# In[130]:


# build the model using best parameters
linreg_SGD_using_best_parameter = SGDRegressor(alpha=1e-06,
                                               eta0=0.01, 
                                               learning_rate='optimal')

# fit the SGD model using best parameter
linreg_SGD_using_best_parameter.fit(X_train,y_train)


# **2. Predict the values using test set**

# In[131]:


# predict the values on test data using predict
linreg_SGD_using_best_parameter_predictions = linreg_SGD_using_best_parameter.predict(X_test)


# **3. Compute accuracy measures**
# 
# Now we calculate accuray measures Root-mean-square-error (RMSE), R-squared and Adjusted R-squared

# In[132]:


# calculate mse
linreg_SGD_using_best_parameter_mse = mean_squared_error(y_test, linreg_SGD_using_best_parameter_predictions)

# calculate rmse
linreg_SGD_using_best_parameter_rmse = np.sqrt(linreg_SGD_using_best_parameter_mse)

# calculate R-squared
linreg_SGD_using_best_parameter_r_squared = r2_score(y_test, linreg_SGD_using_best_parameter_predictions)

# calculate Adjusted R-squared
linreg_SGD_using_best_parameter_adjusted_r_squared = 1 - (1-linreg_SGD_using_best_parameter_r_squared)*(len(y_test)-1)/(len(y_test)- X_test.shape[1]-1)


# **4. Tabulate the results**

# In[133]:


# compile the required information
linreg_full_model_SGD = pd.Series({
                     'Model_Name': "Linear Regression SGD",
                     'RMSE': linreg_SGD_using_best_parameter_rmse ,
                     'R-Squared': linreg_SGD_using_best_parameter_r_squared,
                     'Adj. R-Squared': linreg_SGD_using_best_parameter_adjusted_r_squared   
                   })

# append our result table using append()
score_card = score_card.append(linreg_full_model_SGD, ignore_index = True)

# print the result table
score_card


# <a id='rmse_and_r-squared'></a>
# ## 8. Conclusion and Interpretation

# In[134]:


# view the result table
score_card


# From the table we exclude 'Linear Regression SGD' and 'Linear regression SGD using best parameters' models since its outputs are uninterpretable.

# In[135]:


# drop rows 8 and 9 from above table
score_card = score_card.drop(score_card.index[[8, 9]])
score_card


# **Let visualize graphically the above table**

# In[136]:


# plot the accuracy measure for all models
score_card.plot(secondary_y=['R-Squared','Adj. R-Squared'])

# display just the plot
plt.show()

