import pandas as pd
file_path =  'C:/Users/Sahana Kommalapati/Downloads/heart_2022_with_nans.csv'

# Read the CSV file into a pandas DataFrame
data = pd.read_csv(file_path)

# Display the first few rows of the DataFrame to verify that the data was read correctly
data.head()
data.tail()


# In[4]:


data.info()


# In[5]:


data.shape


# In[6]:


data.columns


# In[7]:


data.describe()


# # DATA CLEANING

# # 1. Formatting column names

# In[8]:


#clean columns in the dataset
data.columns = data.columns.str.replace('_',' ')
data.columns = data.columns.str.title()
data.columns


# # 2. Removing Duplicates

# In[9]:


# Check for duplicate rows
duplicates = data.duplicated().sum()

# Display the duplicate rows
print(duplicates)


# In[10]:


# Remove duplicate rows
data.drop_duplicates(inplace=True)

# Check the shape of the DataFrame after removing duplicates
print("Shape of DataFrame after removing duplicates:", data.shape)



# # 3. Removing Null Values

# In[11]:


# Check for null values in the dataset
null_values = data.isnull().sum()

# Display the count of null values for each column
print(null_values)


# In[12]:


# Select non-numeric columns
non_numeric_columns = data.select_dtypes(exclude=['number']).columns

# Remove rows with null values only for non-numeric columns
data_cleaned = data.dropna(subset=non_numeric_columns)

# Display the first few rows of the cleaned DataFrame
data_cleaned.head()


# In[13]:


data_cleaned.shape


# In[14]:


# Check for null values in the dataset
null_values = data_cleaned.isnull().sum()

# Display the count of null values for each column
print(null_values)


# # 4. Replacing Null values by taking mean

# In[15]:


# Select only the float columns
float_columns = data_cleaned.select_dtypes(include=['float64'])

# Fill null values with the mean of each column
data_cleaned.loc[:, float_columns.columns] = float_columns.fillna(float_columns.mean())



# In[16]:


# Check for null values in the dataset
null_values = data_cleaned.isnull().sum()

# Display the count of null values for each column
print(null_values)


# 
# # 5. Removing Outliers

# In[17]:


import matplotlib.pyplot as plt

# Select only numeric columns
numeric_data = data_cleaned.select_dtypes(include='number')

# Create boxplots for each numeric column
plt.figure(figsize=(10, 6))
numeric_data.boxplot()
plt.title('Boxplot of Numeric Columns')
plt.xticks(rotation=45)
plt.show()


# In[18]:


# Select only numeric columns
numeric_data = data_cleaned.select_dtypes(include='number')

# Calculate the IQR (Interquartile Range) for each column in the numeric DataFrame
Q1 = numeric_data.quantile(0.25)
Q3 = numeric_data.quantile(0.75)
IQR = Q3 - Q1

# Define the threshold for identifying outliers (e.g., 1.5 times the IQR)
threshold = 1.5

# Identify outliers for each column
outliers = (numeric_data < (Q1 - threshold * IQR)) | (numeric_data > (Q3 + threshold * IQR))

# Remove rows containing any outliers
data_filtered = data_cleaned[~outliers.any(axis=1)]

# Check the shape of the DataFrame after removing outliers
print("Shape of DataFrame after removing outliers:", data_filtered.shape)


# # 6. Dropping Unnecessary Columns

# In[19]:


# Define the list of columns to remove
columns_to_remove = ['Removed Teeth', 'Deaforhardofhearing', 'Blindorvisiondifficulty']

# Remove the specified columns from the dataset
filtered_columns = data_filtered.drop(columns=columns_to_remove)

# Display the first few rows of the filtered DataFrame
filtered_columns.head()


# In[20]:


# Print the column names in the DataFrame
filtered_columns


# # 7. Normalization

# In[21]:


from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Select only numeric columns for normalization and standardization
numeric_columns = filtered_columns.select_dtypes(include=['float64', 'int64']).columns

# Perform normalization
scaler = MinMaxScaler()
data_normalized = filtered_columns.copy()
data_normalized[numeric_columns] = scaler.fit_transform(data_normalized[numeric_columns])


# In[22]:


data_normalized.head()


# # 8. Standardization

# In[23]:


# Perform standardization
scaler = StandardScaler()
data_standardized = filtered_columns.copy()
data_standardized[numeric_columns] = scaler.fit_transform(data_standardized[numeric_columns])


# In[24]:


data_standardized.head(20)


# # 9. Feature Engineering

# In[25]:


data_standardized.columns


# In[26]:


print(data['Agecategory'].unique())


# In[27]:


# Define meaningful age group labels
age_group_labels = {
    'Age 18 to 24': '18-24',
    'Age 25 to 29': '25-29',
    'Age 30 to 34': '30-34',
    'Age 35 to 39': '35-39',
    'Age 40 to 44': '40-44',
    'Age 45 to 49': '45-49',
    'Age 50 to 54': '50-54',
    'Age 55 to 59': '55-59',
    'Age 60 to 64': '60-64',
    'Age 65 to 69': '65-69',
    'Age 70 to 74': '70-74',
    'Age 75 to 79': '75-79',
    'Age 80 or older': '80+'
}

# Map age categories to meaningful labels
data_standardized['AgeGroup'] = data_standardized['Agecategory'].map(age_group_labels)

# Drop original 'AgeCategory' column
data_standardized.drop('Agecategory', axis=1, inplace=True)


# In[28]:


data_standardized['AgeGroup']


# # 10. Strings to Binary Values for easy computation

# In[29]:


data_standardized['Haddiabetes'] = data_standardized['Haddiabetes'].apply(lambda x: 1 if x == 'Yes' else 0)


# In[30]:


data_standardized['Physical Activities'] = data_standardized['Physical Activities'].apply(lambda x: 1 if x == 'Yes' else 0)


# In[31]:


data_standardized['Had Heart Attack'] = data_standardized['Had Heart Attack'].apply(lambda x: 1 if x == 'Yes' else 0)


# In[32]:


data_standardized['Had Angina'] = data_standardized['Had Angina'].apply(lambda x: 1 if x == 'Yes' else 0)


# In[33]:


data_standardized['Had Stroke'] = data_standardized['Had Stroke'].apply(lambda x: 1 if x == 'Yes' else 0)


# In[34]:


data_standardized['Had Asthma'] = data_standardized['Had Asthma'].apply(lambda x: 1 if x == 'Yes' else 0)
data_standardized['Had Skin Cancer'] = data_standardized['Had Skin Cancer'].apply(lambda x: 1 if x == 'Yes' else 0)
data_standardized['Had Copd'] = data_standardized['Had Copd'].apply(lambda x: 1 if x == 'Yes' else 0)
data_standardized['Haddepressivedisorder'] = data_standardized['Haddepressivedisorder'].apply(lambda x: 1 if x == 'Yes' else 0)
data_standardized['Hadkidneydisease'] = data_standardized['Hadkidneydisease'].apply(lambda x: 1 if x == 'Yes' else 0)
data_standardized['Hadarthritis'] = data_standardized['Hadarthritis'].apply(lambda x: 1 if x == 'Yes' else 0)
data_standardized['Difficultyconcentrating'] = data_standardized['Difficultyconcentrating'].apply(lambda x: 1 if x == 'Yes' else 0)
data_standardized['Difficultywalking'] = data_standardized['Difficultywalking'].apply(lambda x: 1 if x == 'Yes' else 0)
data_standardized['Difficultydressingbathing'] = data_standardized['Difficultydressingbathing'].apply(lambda x: 1 if x == 'Yes' else 0)
data_standardized['Difficultyerrands'] = data_standardized['Difficultyerrands'].apply(lambda x: 1 if x == 'Yes' else 0)
data_standardized['Alcoholdrinkers'] = data_standardized['Alcoholdrinkers'].apply(lambda x: 1 if x == 'Yes' else 0)
data_standardized['Hivtesting'] = data_standardized['Hivtesting'].apply(lambda x: 1 if x == 'Yes' else 0)
data_standardized['Fluvaxlast12'] = data_standardized['Fluvaxlast12'].apply(lambda x: 1 if x == 'Yes' else 0)
data_standardized['Pneumovaxever'] = data_standardized['Pneumovaxever'].apply(lambda x: 1 if x == 'Yes' else 0)
data_standardized['Highrisklastyear'] = data_standardized['Highrisklastyear'].apply(lambda x: 1 if x == 'Yes' else 0)
data_standardized['Covidpos'] = data_standardized['Covidpos'].apply(lambda x: 1 if x == 'Yes' else 0)


# In[35]:


data_standardized.head()


# In[36]:


data_standardized.columns


# # EXPLORATORY DATA ANALYSIS

# In[37]:


import seaborn as sns
import matplotlib.pyplot as plt

numeric_data = data.select_dtypes(include=['float64', 'int64'])

# Calculate correlation matrix
correlation_matrix = numeric_data.corr()

# Plot correlation heatmap with custom colors
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='YlGnBu', fmt=".2f", linewidths=0.5)
plt.title('Correlation Plot')
plt.show()


# In[38]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Univariate Analysis
# Visualize distributions of numerical features
sns.histplot(data_standardized['AgeGroup'], bins=10, kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age Category')
plt.ylabel('Count')
plt.show()



# In[39]:


# Plotting a bar plot for heart attacks by age category
plt.figure(figsize=(10, 6))
sns.countplot(x='AgeGroup', hue='Had Heart Attack', data=data_standardized)
plt.title('Heart Attacks by Age Category')
plt.xlabel('Age Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Had Heart Attack', labels=['No', 'Yes'])
plt.show()


# In[40]:


# 6. Scatter Plots
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Weightinkilograms', y='Heightinmeters', data=data_standardized, hue='Had Heart Attack')
plt.title('Scatter Plot of Weight vs. Height')
plt.xlabel('Weight (kg)')
plt.ylabel('Height (m)')
plt.show()


# In[41]:


plt.figure(figsize=(8, 8))
data[data['Had Heart Attack'] == 'Yes']['Sex'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'salmon'])
plt.title('Distribution of Heart Attacks by Gender')
plt.ylabel('')
plt.show()


# In[42]:


# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have your data in a pandas DataFrame named 'data'

# Group data by 'Had Heart Attack'
grouped_data = data.groupby('Had Heart Attack')

# Extract mean values for each group
mean_values = grouped_data[['Physical Health Days', 'Mental Health Days']].mean()

# Create the bar plot
plt.figure(figsize=(10, 6))
mean_values.plot(kind='bar', color=['blue', 'green', 'orange'])
plt.xlabel('Health Factors')
plt.ylabel('Average Value')
plt.title('Average Health Values by Heart Attack Occurrence')
plt.xticks(rotation=0)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()


# In[43]:


import matplotlib.pyplot as plt
import pandas as pd

# Convert the dictionary into a DataFrame
data_df = pd.DataFrame.from_dict(data_standardized)

# Define the order of age groups
age_groups = ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54',
              '55-59', '60-64', '65-69', '70-74', '75-79', '80+']

# Plotting bar charts for each age group
plt.figure(figsize=(15, 10))

# Iterate over each age group
for i, age_group in enumerate(age_groups):
    plt.subplot(4, 4, i + 1)  # Create subplots
    data_df[data_df['AgeGroup'] == age_group]['Alcoholdrinkers'].value_counts().plot(kind='bar', color='skyblue')
    plt.title(f"Alcohol Drinkers Distribution - {age_group}")
    plt.xlabel('Alcohol Drinkers')
    plt.ylabel('Count')
    plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# In[44]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'data' is your DataFrame
# Replace 'data' with the name of your DataFrame if it's different

# Plot the bar plot
plt.figure(figsize=(10, 6))
sns.countplot(x='Had Heart Attack', hue='Pneumovaxever', data=data_standardized)
plt.title('Had Heart Attack vs PneumoVaxEver')
plt.xlabel('Had Heart Attack')
plt.ylabel('Count')
plt.show()



# In[45]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.countplot(x='AgeGroup', hue='Had Heart Attack', data=data_standardized)
plt.title('Heart Disease Cases by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.show()


# In[46]:


plt.figure(figsize=(10, 8))
data['State'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title('Proportion of Heart Disease Cases by State')
plt.ylabel('')
plt.show()


# In[47]:


plt.figure(figsize=(8, 6))
sns.barplot(x='Had Heart Attack', y='Bmi', data=data, palette='pastel')
plt.title('BMI Distribution by Heart Disease')
plt.xlabel('Had Heart Attack')
plt.ylabel('BMI')
plt.show()


# In[48]:


plt.figure(figsize=(10, 6))
sns.countplot(x='Raceethnicitycategory', hue='Had Heart Attack', data=data, palette='Set2')
plt.title('Distribution of Heart Attack by Race')
plt.xlabel('Race')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Had Heart Attack', loc='upper right')
plt.show()


# In[49]:


plt.figure(figsize=(10, 6))
sns.countplot(x='Smokerstatus', hue='Had Heart Attack', data=data)
plt.title('Heart Disease Cases by Smoking Status')
plt.xlabel('Smoking Status')
plt.ylabel('Count')
plt.show()


# In[50]:


import seaborn as sns
import matplotlib.pyplot as plt

# Plot KDE plot between sleep hours and had heart attack
plt.figure(figsize=(8, 6))
sns.kdeplot(data=data, x='Sleep Hours', hue='Had Heart Attack', fill=True, common_norm=False)
plt.title('KDE Plot of Sleep Hours by Had Heart Attack')
plt.xlabel('Sleep Hours')
plt.ylabel('Density')
plt.xlim(0, 15)
plt.legend(title='Had Heart Attack', loc='upper right')
plt.show()


# In[51]:


import matplotlib.pyplot as plt
import seaborn as sns

# Plotting a bar plot between 'Had Heart Attack' and 'Had Asthma'
plt.figure(figsize=(8, 6))
sns.countplot(x='Had Heart Attack', hue='Had Asthma', data=data)
plt.title('Had Heart Attack vs Had Asthma')
plt.xlabel('Had Heart Attack')
plt.ylabel('Count')
plt.show()


# In[52]:


import matplotlib.pyplot as plt
import seaborn as sns

# Plotting a bar plot for alcohol drinkers with heart attack
plt.figure(figsize=(8, 6))
sns.countplot(x='Alcoholdrinkers', hue='Had Heart Attack', data=data)
plt.title('Alcohol Drinkers with Heart Attack')
plt.xlabel('Alcohol Drinkers')
plt.ylabel('Count')
plt.show()


# # PHASE 2

# ### ANALYSIS

# #### Splitting Data to Test and Train

# In[53]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, roc_auc_score,auc
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Encoded categorical variables and standardize data
data_encoded = pd.get_dummies(data_standardized)

# Spliting data into features (X) and target variable (y)
X = data_encoded.drop(columns=['Had Heart Attack'])  # Features
y = data_encoded['Had Heart Attack']  # Target variable

# Spliting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ### Logistic Regression

# In[78]:


logreg = LogisticRegression(max_iter=1000)  # Training logistic regression model with increased max_iter

logreg.fit(X_train, y_train) # fitting model

y_pred = logreg.predict(X_test) # Predictions on test set


# In[89]:


#Model Evaluation

# Calculating accuracy
accuracy_lg = accuracy_score(y_test, y_pred)
print("Logistic Regression - Accuracy:", accuracy_lg)

# Calculating precision
precision_lg = precision_score(y_test, y_pred, average='macro')  # need to use 'micro' or 'weighted' for multiclass
print("Logistic Regression - Precision:", precision_lg)
#The `macro` average computes the average of precision scores across all classes with equal weighting.

# Calculating recall
recall_lg = recall_score(y_test, y_pred, average='macro')
print("Logistic Regression - Recall:", recall_lg)

# Calculating F1-score
f1_lg = f1_score(y_test, y_pred, average='macro')
print("Logistic Regression - F1-score:", f1_lg)


# In[82]:


cm = confusion_matrix(y_test, y_pred)# Creating confusion matrix

# Plotting confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Logistic Regression')
plt.show()

y_prob = logreg.predict_proba(X_test)[:, 1]# predicted probabilities for the positive class

fpr, tpr, thresholds = roc_curve(y_test, y_prob)# Calculating ROC curve

# Calculating AUC score
auc_score = roc_auc_score(y_test, y_prob)

# Plotting ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % auc_score)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Logistic Regression')
plt.legend(loc='lower right')
plt.show()

# Calculating precision-recall curve
precision, recall, _ = precision_recall_curve(y_test, y_prob)

# Calculating average precision score
average_precision = average_precision_score(y_test, y_prob)

# Plotting precision-recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve (AP = %0.2f)' % average_precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve for Logistic Regression')
plt.legend(loc='upper right')
plt.show()


# ### Naive Bayes

# In[85]:


nb = GaussianNB()
nb.fit(X_train, y_train) #fitting the model
nb_pred = nb.predict(X_test) # prediction of test data


# In[88]:


# Calculating accuracy
accuracy_nb = accuracy_score(y_test, nb_pred)
print("Naive Bayes - Accuracy:", accuracy_nb)

# Calculating precision
precision_nb = precision_score(y_test, nb_pred)
print("Naive Bayes - Precision:", precision_nb)

# Calculating recall
recall_nb = recall_score(y_test, nb_pred)
print("Naive Bayes - Recall:", recall_nb)

# Calculating F1-score
f1_nb = f1_score(y_test, nb_pred)
print("Naive Bayes - F1-score:", f1_nb)


# In[87]:


cm = confusion_matrix(y_test, nb_pred)# Creating confusion matrix

# Plotting confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Naive Bayes')
plt.show()

# ROC CURVE
y_prob_nb = nb.predict_proba(X_test)[:, 1]# Getting predicted probabilities for the positive class

# Calculating ROC curve
fpr_nb, tpr_nb, thresholds_nb = roc_curve(y_test, y_prob_nb)

# Calculating AUC score
auc_score_nb = roc_auc_score(y_test, y_prob_nb)

# Plotting ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_nb, tpr_nb, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % auc_score_nb)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Naive Bayes')
plt.legend(loc='lower right')
plt.show()

# Precision vs Recall Plot
plt.figure(figsize=(8, 6))
precision, recall, _ = precision_recall_curve(y_test, nb_pred)
plt.plot(recall, precision, color='blue')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Naive Bayes')
plt.show()


# ### KNN

# In[75]:


# Convert data to numpy arrays
X_train_np = X_train.to_numpy()
X_test_np = X_test.to_numpy()
y_train_np = y_train.to_numpy()
y_test_np = y_test.to_numpy()

knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean') # Initialize KNN classifier with different parameters

knn.fit(X_train_np, y_train_np)# Training the model with new parameters

knn_pred = knn.predict(X_test_np)# Predict on test data with new model

y_score = knn.predict_proba(X_test_np)[:, 1]# Get probability estimates for positive class for PR curve with new model


# In[98]:


# Evaluate the KNN model

# Calculating accuracy
accuracy_knn = accuracy_score(y_test_np, knn_pred)
print("KNN Model - Accuracy:", accuracy_knn)

# Calculating precision
precision_knn = precision_score(y_test_np, knn_pred)
print("KNN Model - Precision:", precision_knn)

# Calculating Recall
recall_knn = recall_score(y_test_np, knn_pred)
print("KNN Model - Recall:", recall_knn)

# Calculating F1-Score
f1_knn = f1_score(y_test_np, knn_pred)
print("KNN Model - F1-score:", f1_knn)


# In[91]:


cm = confusion_matrix(y_test_np, knn_pred)# Plot Confusion Matrix for the new model

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for KNN Model")
plt.show()

# Calculate ROC curve for KNN
fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_test_np, y_score)

# Calculate AUC score for KNN
auc_score_knn = roc_auc_score(y_test_np, y_score)

# Plot ROC curve for KNN
plt.figure(figsize=(8, 6))
plt.plot(fpr_knn, tpr_knn, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % auc_score_knn)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for KNN Model')
plt.legend(loc='lower right')
plt.show()

# Precision vs Recall Plot for KNN
precision_knn, recall_knn, _ = precision_recall_curve(y_test_np, y_score)
plt.figure(figsize=(8, 6))
plt.plot(recall_knn, precision_knn, color='blue')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for KNN Model')
plt.show()


# ### SVM

# In[92]:


# Initialize SVM classifier with 'sigmoid' kernel
svm_classifier_sigmoid = SVC(kernel='sigmoid', C=1.0, gamma='scale')

svm_classifier_sigmoid.fit(X_train, y_train)# Training the SVM model with sigmoid kernel

svm_pred_sigmoid = svm_classifier_sigmoid.predict(X_test)# Prediction on test data with SVM model using sigmoid kernel


# In[99]:


# Evaluate the SVM model with sigmoid kernel

#Calculate accuracy
accuracy_svm_sigmoid = accuracy_score(y_test, svm_pred_sigmoid)
print("SVM Model (Sigmoid Kernel) - Accuracy:", accuracy_svm_sigmoid)

#calculate precision
precision_svm_sigmoid = precision_score(y_test, svm_pred_sigmoid)
print("SVM Model (Sigmoid Kernel) - Precision:", precision_svm_sigmoid)

# calculate recall
recall_svm_sigmoid = recall_score(y_test, svm_pred_sigmoid)
print("SVM Model (Sigmoid Kernel) - Recall:", recall_svm_sigmoid)

# calculate f1 Score
f1_svm_sigmoid = f1_score(y_test, svm_pred_sigmoid)
print("SVM Model (Sigmoid Kernel) - F1-score:", f1_svm_sigmoid)


# In[94]:


# Predictions are stored in svm_pred_sigmoid, y_score_svm_sigmoid is the score for positive class
y_score_svm_sigmoid = svm_classifier_sigmoid.decision_function(X_test)

# Plot Confusion Matrix for SVM model with sigmoid kernel
conf_matrix_sigmoid = confusion_matrix(y_test, svm_pred_sigmoid)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_sigmoid, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for SVM Model (Sigmoid Kernel)")
plt.show()

# Plot ROC Curve for SVM model with sigmoid kernel
fpr_sigmoid, tpr_sigmoid, _ = roc_curve(y_test, y_score_svm_sigmoid)
roc_auc_sigmoid = roc_auc_score(y_test, y_score_svm_sigmoid)
plt.figure(figsize=(8, 6))
plt.plot(fpr_sigmoid, tpr_sigmoid, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_sigmoid)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for SVM Model (Sigmoid Kernel)')
plt.legend(loc="lower right")
plt.show()

# Plot Precision-Recall Curve for SVM model with sigmoid kernel
precision_sigmoid, recall_sigmoid, _ = precision_recall_curve(y_test, y_score_svm_sigmoid)
average_precision_sigmoid = average_precision_score(y_test, y_score_svm_sigmoid)
plt.figure(figsize=(8, 6))
plt.step(recall_sigmoid, precision_sigmoid, color='b', alpha=0.2, where='post')
plt.fill_between(recall_sigmoid, precision_sigmoid, step='post', alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve: AP={0:0.2f} for SVM Model (Sigmoid Kernel)'.format(average_precision_sigmoid))
plt.show()


# ### Decision Tree

# In[100]:


# Creating a Decision Tree classifier
model_dt = DecisionTreeClassifier()

model_dt.fit(X_train, y_train)# Training the model

y_pred_dt = model_dt.predict(X_test)# Making predictions


# In[101]:


# Evaluating the model

# Calculating accuracy
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Decision Tree - Accuracy:", accuracy_dt)

# Calculating precision
# For binary classification, you can specify the averaging method as 'binary'.
precision_dt = precision_score(y_test, y_pred_dt, average='binary')
print("Decision Tree - Precision:", precision_dt)

#calculating recall
recall_dt = recall_score(y_test, y_pred_dt, average='binary')
print("Decision Tree - Recall:", recall_dt)

#calculating F1-Score
f1_dt = f1_score(y_test, y_pred_dt, average='binary')
print("Decision Tree - F1-score:", f1_dt)


# In[102]:


# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix for Decision Tree')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Calculate the false positive rate (fpr) and true positive rate (tpr)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# Calculate the area under the ROC curve (AUC)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Compute precision-recall pairs
precision, recall, _ = precision_recall_curve(y_test, model_dt.predict_proba(X_test)[:, 1])

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.')
plt.title('Precision-Recall Curve for Decision Tree')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()


# ### Random Forest

# In[103]:


# Creating a Random Forest classifier
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)

model_rf.fit(X_train, y_train)# Training the model

y_pred_rf = model_rf.predict(X_test)# Making predictions


# In[105]:


# Evaluating the model

# Calculate accuarcy
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest - Accuracy:", accuracy_rf)

# Calculate precision
precision_rf = precision_score(y_test, y_pred_rf)
print("Random Forest - Precision:", precision_rf)

# Calculate recall
recall_rf = recall_score(y_test, y_pred_rf)
print("Random Forest - Recall:", recall_rf)

# Calculate F1-score
f1_rf = f1_score(y_test, y_pred_rf)
print("Random Forest - F1-score:", f1_rf)


# In[106]:


# Plot confusion matrix
plt.figure(figsize=(10, 5))
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix for Random Forest")
plt.colorbar()
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.xticks([0, 1], ['No Heart Attack', 'Had Heart Attack'])
plt.yticks([0, 1], ['No Heart Attack', 'Had Heart Attack'])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='black')
plt.show()

# Get predicted probabilities for the positive class
y_pred_proba = model_rf.predict_proba(X_test)[:, 1]

# Compute ROC curve and ROC area
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Random Forest')
plt.legend(loc="lower right")
plt.show()

# Compute precision-recall pairs
precision, recall, _ = precision_recall_curve(y_test, model_rf.predict_proba(X_test)[:, 1])

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.')
plt.title('Precision-Recall Curve for Random Forest')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()


# ### Ridge Classifier

# In[107]:


from sklearn.linear_model import RidgeClassifier

ridge_classifier = RidgeClassifier()# Initialize the Ridge Classifier

ridge_classifier.fit(X_train, y_train)# Train the model

y_pred_ridge = ridge_classifier.predict(X_test)# Make predictions on the test set


# In[110]:


# Evaluate the model

#Calculating accuracy
accuracy_ridge = accuracy_score(y_test, y_pred_ridge)

#calculating precision
precision_ridge = precision_score(y_test, y_pred_ridge, average='binary')

#calculating recall
recall_ridge = recall_score(y_test, y_pred_ridge, average='binary')

#calculating F1-Score
f1_ridge = f1_score(y_test, y_pred_ridge, average='binary')

print("Ridge Classifier - Accuracy:", accuracy_ridge)
print("Ridge Classifier - Precision:", precision_ridge)
print("Ridge Classifier - Recall:", recall_ridge)
print("Ridge Classifier - F1-score:", f1_ridge)


# In[111]:


# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix for Ridge Classifier')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Calculate ROC curve
y_prob = ridge_classifier.decision_function(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Calculate AUC score
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Ridge Classifier')
plt.legend(loc='lower right')
plt.show()

# Calculate precision-recall curve
precision, recall, _ = precision_recall_curve(y_test, y_prob)

# Calculate average precision score
average_precision = average_precision_score(y_test, y_prob)

# Plot precision-recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve (AP = %0.2f)' % average_precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve for Ridge Classifier')
plt.legend(loc='upper right')
plt.show()


# In[113]:


algorithms = ['Logistic Regression', 'Naive Bayes', 'KNN', 'SVM', 'Decision Tree', 'Random Forest', 'Ridge Classifier']
accuracies = [95.77, 71.28, 95.30, 92.17, 93.24, 95.69, 95.70]
precisions = [75.74, 11.48, 33.15, 12.00, 25.25, 57.14, 63.16]
recalls = [59.67, 82.40, 6.70, 12.32, 27.37, 7.72, 5.45]
f1_scores = [63.63, 20.15, 11.15, 12.16, 26.27, 13.61, 10.04]

x = np.arange(len(algorithms))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots(figsize=(14, 8))

# Plotting data
rects1 = ax.bar(x - 3*width/2, accuracies, width, label='Accuracy')
rects2 = ax.bar(x - width/2, precisions, width, label='Precision')
rects3 = ax.bar(x + width/2, recalls, width, label='Recall')
rects4 = ax.bar(x + 3*width/2, f1_scores, width, label='F1-Score')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Algorithms')
ax.set_ylabel('Scores')
ax.set_title('Comparison of ML Algorithms for Heart Disease Prediction')
ax.set_xticks(x)
ax.set_xticklabels(algorithms, rotation=45)
ax.legend()

# Autolabel function to display the label on top of the bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
fig.tight_layout()
plt.show()


# In[ ]:




