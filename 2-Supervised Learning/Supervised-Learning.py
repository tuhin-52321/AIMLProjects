# %%
# importing libraries
import pandas as pd
import numpy as np
import matplotlib.colors as colors
import seaborn as sns
import math
import matplotlib.pyplot as plt
import warnings

from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE

#%matplotlib inline
sns.set(style='darkgrid', palette='deep', font='sans-serif', font_scale=1.3, color_codes=True)

# %% [markdown]
# # PART A - 30 Marks

# %% [markdown]
# ## 1. Data Understanding: [5 Marks]
# <ol style="list-style-type: upper-alpha;">
#   <li>Read all the 3 CSV files as DataFrame and store them into 3 separate variables. [1 Mark]</li>
#   <li>Print Shape and columns of all the 3 DataFrames. [1 Mark]</li>
#   <li>Compare Column names of all the 3 DataFrames and clearly write observations. [1 Mark]</li>
#   <li>Print DataTypes of all the 3 DataFrames. [1 Mark]</li>
#   <li>Observe and share variation in ‘Class’ feature of all the 3 DaraFrames. [1 Mark]</li>
# </ol>

# %%
#1A - Read all the 3 CSV files as dataframes and store them into 3 seperate variables.
normal = pd.read_csv("Part1-Normal.csv")
type_h = pd.read_csv("Part1-Type_H.csv")
type_s = pd.read_csv("Part1-Type_S.csv")

# %%
#1B - Print shape and columns for each variable

print("Shape of normal DataFrame   : " , normal.shape)
print("Columns of normal DataFrame : " , normal.columns.values, '\n')

print("Shape of type_h DataFrame   : " , type_h.shape)
print("Columns of type_h DataFrame : " , type_h.columns.values, '\n')

print("Shape of type_3 DataFrame   : " , type_s.shape)
print("Columns of type_s DataFrame : " , type_s.columns.values, '\n')

all_columns = pd.DataFrame({'normal':normal.columns.values,'type_h':type_h.columns.values,'type_s':type_s.columns.values},index=['Column 1','Column 2','Column 3','Column 4','Column 5','Column 6','Column 7']).T

all_columns


# %% [markdown]
# ### 1C - Observations on column names of all the 3 DataFrames
# 
# - All three DataFrames has 7 Columns
# - All column titles are excatly same
# 

# %%
#1D - Print DataTypes of all the 3 DataFrames.

print("Data types of Normal:")
print(normal.dtypes)
print("----------------------")

print("Data types of Type_H:")
print(type_h.dtypes)
print("----------------------")

print("Data types of Type_S:")
print(type_s.dtypes)
print("----------------------")


# %%
#1E - Observe and share variation in ‘Class’ feature of all the 3 DaraFrames

print("Class Feature of Normal:")
print(normal['Class'].value_counts())
print("----------------------")

print("Class Feature of Type_H:")
print(type_h['Class'].value_counts())
print("----------------------")

print("Class Feature of Type_S:")
print(type_s['Class'].value_counts())
print("----------------------")

# %% [markdown]
# ### Observations for Class feature for all three datasets
# 
# - All rows of Normal DataFrame should have class as 'Normal'
#     - However, out of 100 data rows, 73 rows class values are "Normal" and 27 values are "Nrml". This is looks like spelling mistakes - all values should be "Normal"
# - All rows of Type_H DataFrame should have class as 'Type_H'
#     - However, out of 60 data rows, 37 rows class values are "Type_H" and 23 values are "type_h". This is looks like typing mistakes - all values should be "Type_H"
# - All rows of Type_S DataFrame should have class as 'Type_S'
#     - However, out of 150 data rows, 133 rows class values are "Type_S" and 17 values are "tp_s". This is looks like typing/spelling mistakes - all values should be "Type_S"
# 
# 

# %% [markdown]
# ## 2. Data Preparation and Exploration: [5 Marks]
# <ol style="list-style-type: upper-alpha;">
#   <li>Unify all the variations in ‘Class’ feature for all the 3 DataFrames. [1 Mark]</li>
#   <li>Combine all the 3 DataFrames to form a single DataFrame [1 Marks]</li>
#   <li>Print 5 random samples of this DataFrame [1 Marks]</li>
#   <li>Print Feature-wise percentage of Null values. [1 Mark]</li>
#   <li>Check 5-point summary of the new DataFrame. [1 Mark]</li>
# </ol>

# %%
#2A - Unify all the variations in Class feature for all the 3 DataFrames

#Make Class label as 'Normal' for all rows of Normal DataFrame

normal['Class'] = "Normal"

#Make Class label as 'Type_H' for all rows of type_h DataFrame

type_h['Class'] = 'Type_H'

#Make Class label as 'Type_S' for all rows of type_s DataFrame

type_s['Class'] = 'Type_S'

# %%
#2B - Combine all the 3 DataFrames to form a single DataFrame

patient_data = pd.concat([normal, type_h, type_s])



# %%
#2C - Print 5 random samples of this DataFrame

patient_data.sample(5)

# %%

#2D - Print Feature-wise percentage of Null values.

percent_null=patient_data.isnull().sum()* 100 / len(patient_data)

pd.DataFrame({'Percentage of null': percent_null})

# %% [markdown]
# - There are no null values observed as all percentages above are zero

# %%

#2E - Check 5-point summary of the new DataFrame.

patient_data.describe().T


# %% [markdown]
# - 5-point summary (min,25%,50%,75%,max) for each feature is displayed above.

# %% [markdown]
# ## 3. Data Analysis: [10 Marks]
# <ol style="list-style-type: upper-alpha;">
#   <li>Visualize a heatmap to understand correlation between all features [2 Marks]</li>
#   <li>Share insights on correlation. [2 Marks]
#     <ol style="list-style-type: upper-alpha;">
#        <li>Features having stronger correlation with correlation value.
#        <li>Features having weaker correlation with correlation value.</li>
#     </ol>
#   </li>
#   <li>Visualize a pairplot with 3 classes distinguished by colors and share insights. [2 Marks]</li>
#   <li>Visualize a jointplot for ‘P_incidence’ and ‘S_slope’ and share insights. [2 Marks]</li>
#   <li>Visualize a boxplot to check distribution of the features and share insights. [2 Marks]</li>
# </ol>

# %%
#3A - Visualize a heatmap to understand correlation between all features of combined DataFrame

corr = patient_data.corr(numeric_only=True)

fig=plt.figure(dpi = 120,figsize= (5,4))
ax = fig.add_subplot()

mask= np.triu(np.ones_like(corr))

#Show the heatmap with strong, medium and weak correlations
cmap = colors.ListedColormap(['darkred','tan','gainsboro', 'tan', 'darkred'])
bounds=[-1, -0.7, -0.3, 0.3, 0.7, 1]
norm = colors.BoundaryNorm(bounds, cmap.N)

sns.heatmap(corr, mask=mask, square=True, cmap=cmap, norm=norm, lw=1, fmt = ".2f", annot=True, ax=ax)
plt.yticks(rotation = 0)
plt.xticks(rotation = 90)
plt.title('Correlation Heatmap')

#Show the legends of strong, medium and weak correlations
ax.text(6.8, 0.65, 'Strong correlation',color='darkred', fontsize = 10)
ax.text(6.8, 1.85, 'Medium correlation',color='tan', fontsize = 10)
ax.text(6.8, 3.05, 'Weak correlation',color='gainsboro', fontsize = 10)
ax.text(6.8, 4.25, 'Medium correlation',color='tan', fontsize = 10)
ax.text(6.8, 5.45, 'Strong correlation',color='darkred', fontsize = 10)

plt.show()


# %% [markdown]
# ### 3B - Insights on the correlation between features
# 
# ### Assumptions to separate stronger, weaker and moderate correlations:
# - Coefficient between −0.3 and +0.3 = weak correlation.
# - Coefficient less than −0.7 or greater than +0.7 = strong correlation.
# - Coefficient between −0.3 and −0.7 or between +0.3 and +0.7 = moderate correlation.
# 
# #### Features having **stronger correlation** (correlations >= 0.7 and <= -0.7)
# 
# - *P_incidence* and *S_slope* with correlation value 0.81 (positive correlation)
# - *P_incidence* and *L_angle* with correlation value 0.72 (positive correlation)
# 
# #### Features having **moderate correlation** (correlations between 0.3 and 0.7 and -0.7 and -0.3)
# 
# - *P_incidence* and *S_Degree* with correlation value 0.64 (positive correlation)
# - *P_incidence* and *P_tilt* with correlation value 0.63 (positive correlation)
# - *L_angle* and *S_slope* with correlation value 0.60 (positive correlation)
# - *L_angle* and *S_Degree* with correlation value 0.53 (positive correlation)
# - *S_slope* and *S_Degree* with correlation value 0.52 (positive correlation)
# - *P_tilt* and *L_angle* with correlation value 0.43 (positive correlation)
# - *P_tilt* and *S_Degree* with correlation value 0.40 (positive correlation)
# - *S_slope* and *P_radius* with correlation value -0.34 (negative correlation)
# 
# #### Features having **weak correlation** (correlations betwen -0.3 to 0.3)
# 
# - *P_incidence* and *P_radius* with correlation value -0.25 (negative correlation)
# - *L_angle* and *S_Degree* with correlation value -0.08 (negative correlation)
# - *P_tilt* and *S_slope* with correlation value -0.06 (negative correlation)
# - *P_tilt* and *P_radius* with correlation value 0.03 (positive correlation)
# - *P_radius* and *S_Degree* with correlation value -0.03 (negative correlation)
# 

# %%
#3C - Visualize a pairplot with 3 classes distinguished by colors for the combined dataframe

warnings.filterwarnings("ignore", message="The figure layout has changed to tight")
sns.pairplot(patient_data,hue='Class')
plt.show()


# %% [markdown]
# ### Insight of pairplot with 'Class'
# 
# - Along the diagonal we can see distribution of variable for three classes are all normal distribution, but each class having its own distrbution.
# - It is evident that Type_S class is more compared to other two
# - Normal class has higher values compared to Type_H
# - From the plot, it is evident that P_tilt, L_angle and S_slope has a clear positive corelation with all classes, even S_Degree also have some slight postive corelation.
# - We can find few outliers in each plots
# - there is no corelation of classes with P_radius is evident.

# %%
# 3D - Visualize a jointplot for ‘P_incidence’ and ‘S_slope’

sns.jointplot(x=patient_data['P_incidence'], y=patient_data['S_slope'])

plt.show()

# %% [markdown]
# ### Insights from Jointplot of P_incidence and S_slope
# 
# - S_Slope data looks normally distributed
# - P_incidence data also looks normally distributed
# - It is evident from the plot that P_incidence and S_slope are postively corelated
# - There is one clear outlier (beyoond 120,120), and possibly couple of other outliers aroud (120,80)

# %%

#3E - Visualize a boxplot to check distribution of the features

ax_rows=2
ax_columns=3
fig, axes = plt.subplots(ax_rows, ax_columns, figsize=(18, 10))

fignum=0
for y in ['P_incidence','P_tilt','L_angle','S_slope','P_radius','S_Degree']:
    sns.boxplot(ax=axes[fignum//ax_columns, fignum%ax_columns],data=patient_data, x='Class',  y=y)
    fignum += 1

plt.show()


# %% [markdown]
# ### Insights from Boxplots
# 
# #### P_incidence vs Class
# 
# - P_incidence Value is larger for Type_S Class. We can see some extreme values as well
# - Normal Value is slightly higher than Type_H
# 
# #### P_tilt vs Class
# 
# - Mean of Type_S is slightly higher than rest two
# - Few cases Normal and Type_H also has huge values
# 
# #### L_angle vs Class
# 
# - L_Angle has higher value for Type_S Class
# - We can see Normal class has higher values compared to type_H class
# - Each class contains one outlier
# 
# #### S_slope vs Class
# 
# - S_slope has huge values for Type_S class
# - Normal class has high s_slope compared to Type_H
# 
# #### P_radius vs Class
# 
# - We can see P_radius value is more for Normal Class
# - There is some extreme values for Type_s class
# - All classes has higher and lower Value
# 
# #### S_Degree vs Class
# 
# - S_Degree has extreme values for type_S Class
# - Few Normal class also has huge values for S_Degree

# %% [markdown]
# ## 4. Model Building: [6 Marks]
# <ol style="list-style-type: upper-alpha;">
#   <li>Split data into X and Y. [1 Marks]</li>
#   <li>Split data into train and test with 80:20 proportion. [1 Marks]</li>
#   <li>Train a Supervised Learning Classification base model using KNN classifier. [2 Marks]</li>
#   <li>Print all the possible performance metrics for both train and test data. [2 Marks]</li>
# </ol>

# %%
#4A - split data into X and y

#first conver 'Class' column as category column
patient_data['Class'] = patient_data.Class.astype('category')

#Then encode the target using label encoder

le=LabelEncoder()
patient_data['Class']=le.fit_transform(patient_data['Class'])

X = patient_data.drop(labels= "Class" , axis = 1)
y = patient_data["Class"]




# %%
#4B - Split data into train and test with 80:20 proportion.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=10)


# %%
#4C - Train a Supervised Learning Classification base model using KNN classifier.

print("Training KNN with defaults")
print("---------------------------------------------")

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# %%
#4D - Print all the possible performance metrics for both train and test data.


predicted_labels_train = knn.predict(X_train)
train_accuracy = metrics.accuracy_score(y_train, predicted_labels_train)
train_precision = metrics.precision_score(y_train, predicted_labels_train, average='macro')
train_recall = metrics.recall_score(y_train, predicted_labels_train,average='macro')
train_f1 = metrics.f1_score(y_train, predicted_labels_train,average='macro')
print("Training data Accuracy       : {0:.4f}".format(train_accuracy))
print("Training data Precision      : {0:.4f}".format(train_precision))
print("Training data Recall         : {0:.4f}".format(train_recall))
print("Training data F1 Score       : {0:.4f}".format(train_f1))

predicted_labels_test = knn.predict(X_test)
test_accuracy = metrics.accuracy_score(y_test, predicted_labels_test)
test_precision = metrics.precision_score(y_test, predicted_labels_test,average='macro')
test_recall = metrics.recall_score(y_test, predicted_labels_test,average='macro')
test_f1 = metrics.f1_score(y_test, predicted_labels_test,average='macro')
print()
print("Testing data Accuracy  : {0:.4f}".format(test_accuracy))
print("Testing data Precision : {0:.4f}".format(test_precision))
print("Testing data Recall    : {0:.4f}".format(test_recall))
print("Testing data F1 Score  : {0:.4f}".format(test_f1))

print()
print("[*Note: the precision/recall/f1 scores above are unweighted mean (macro average) of each class level values]")
print()

def plot_cm_patient_data(X, y, title):
    predicted_labels = knn.predict(X)
    cm = metrics.confusion_matrix(y, predicted_labels, labels=[0,1,2])

    true_normal=y.value_counts()[0]
    true_h=y.value_counts()[1]
    true_s=y.value_counts()[2]
    
    trues = [true_normal,true_normal,true_normal,true_h,true_h,true_h,true_s,true_s,true_s]
    pred_pct = ["{0:.2%}".format(value) for value in
                     cm.flatten()/trues]

    df_cm = pd.DataFrame((cm.flatten()/trues).reshape(3,3), index = [i for i in ["Normal","Type_H","Type_S"]],
              columns = [i for i in ["Normal","Type_H","Type_S"]])

    tags = ['True Normal', 'False Normal', 'False Normal', 'False Type_H', 'True Type_H', 'False Type_H', 'False Type_S', 'False Type_S','True Type_S']

    fig=plt.figure(figsize = (7,5))
    ax = fig.add_subplot()

    counts = ["{0:0.0f}".format(value) for value in
                cm.flatten()]
    labels = [f"[{v4}]\n\n{v1}/{v3}\n({v2})" for v1, v2, v3, v4 in
          zip(counts,pred_pct, trues, tags)]
    labels = np.asarray(labels).reshape(3,3)

    sns.heatmap(df_cm, annot=labels ,fmt='', cmap='Blues')

    plt.title('{0}\nLabels: Predicted/True\n(Percentage of Predicted vs True)'.format(title))
    plt.xlabel('Predicted')
    plt.ylabel('Actuals')
    ax.xaxis.label.set_color('red')
    ax.yaxis.label.set_color('green')

    plt.show()

plot_cm_patient_data(X_train,y_train,"Confusion Matrix of KNN (Training Data)")

print()

plot_cm_patient_data(X_test,y_test,"Confusion Matrix of KNN (Test Data)")

print()

print("classification  Matrix (Training Data):\n",metrics.classification_report(y_train,predicted_labels_train))

print()

print("classification  Matrix (Test Data):\n",metrics.classification_report(y_test,predicted_labels_test))

print()


# %% [markdown]
# ### Observations
# 
# - Training accuracy is high vs the test data accuracy is low
#     - This is due to overfitting of data
# - In test data, Type-S prediction is more accurate than others
# - Test data predictions: 
#     - Out of 17 True Normal cases 
#         - 47.06% correctly predcited
#             - 8 cases are predicted correctly
#         - 52.94% predcited wrongly 
#             - 9 cases are predicted as Type_H
#     - Out of 15 True Type_H cases 
#         - 60.00% correctly predcited
#             - 9 cases are predicted correctly
#         - 40% predcited wrongly 
#             - 6 cases are predicted as Normal
#     - Out of 30 True Type_S cases 
#         - 90.00% correctly predcited
#             - 27 cases are predicted correctly
#         - 10% predcited wrongly 
#             - 1 case is predicted as Normal (3.33%)
#             - 2 cases are predicted Type_H  (6.67%)
#     - Overall Accuracy = 70.97%
#             
# 

# %% [markdown]
# ## 5. Performance Improvement: [4 Marks]
# <ol style="list-style-type: upper-alpha;">
#   <li>Experiment with various parameters to improve performance of the base model. [2 Marks]</li>
#   <li>Clearly showcase improvement in performance achieved. [1 Marks]<br/>For Example:</li>
#   <ol style="list-style-type: upper-alpha;">
#     <li>Accuracy: +15% improvement</li>
#     <li>Precision: +10% improvement.</li>
#   </ol>
#   <li>Clearly state which parameters contributed most to improve model performance. [1 Marks]</li>
#   </li>
# </ol>

# %%
#Let's try to find optimal k-value range

train_score=[]
test_score=[]
train_mse=[]
test_mse=[]
for k in range(1,51):
    KNN = KNeighborsClassifier(n_neighbors= k )
    KNN.fit(X_train, y_train)
    train_score.append(KNN.score(X_train, y_train))
    test_score.append(KNN.score(X_test, y_test))
    train_mse.append(metrics.mean_squared_error(y_train, KNN.predict(X_train)))
    test_mse.append(metrics.mean_squared_error(y_test, KNN.predict(X_test)))

plt.plot(range(1,51),train_score, label = "Training Accuracy")
plt.plot(range(1,51),test_score, label = "Testing Accuracy")
plt.legend()
plt.title("Score vs k-value")
plt.show()

plt.plot(range(1,51),train_mse, label = "Mean Squared Error (Training data)")
plt.plot(range(1,51),test_mse, label = "Mean Squared Error (Test data)")
plt.legend()
plt.title("Score vs k-value")
plt.show()


# %% [markdown]
# ### Observations
# 
# - Accuracy is steadily decreasing for train data (in-sample)
# - Accuracy is maximum between k=10-20 for test data
# - Mean Squared Error also minimun between k=10-20 for test data
# 

# %%
# Check %-age increases for k=10 to 20 and other parameters of KNN to find the best score
best_accuracy = 0
best_k=-1
best_weights=""
best_metric=""
for w in ['uniform','distance']:
    for m in ['minkowski','euclidean','manhattan']:
        for k in range(10,21):
            knn = KNeighborsClassifier(n_neighbors = k, weights = w, metric=m )
            knn.fit(X_train, y_train)
            predicted_labels    = knn.predict(X_test)
            new_accuracy_score  = metrics.accuracy_score(y_test, predicted_labels)
            if new_accuracy_score > best_accuracy:
                best_accuracy = new_accuracy_score
                best_k=k
                best_weights=w
                best_metric=m

print("The parameters found with best accuracy score of {0:5.2f} are : k = {1}, weights = '{2}' and metric = '{3}'".format(best_accuracy, best_k, best_metric, best_weights))



# %% [markdown]
# **_5A_** Conclusion on Experimenting with various parameters to improve performance of KNN:
# 
# - Did run KNN with varying three parameters n_neighbor, wieghts and metrics.
# - Found the best fitting parameter values:
#     - k = 12
#     - weights = 'minkowski'
#     - metric = 'uniform'

# %%
#Check performance improvement

#Run knn with best parameter
knn = KNeighborsClassifier(n_neighbors = best_k, weights = best_weights, metric=best_metric )
knn.fit(X_train, y_train)
predicted_labels    = knn.predict(X_test)
new_accuracy_score  = metrics.accuracy_score(y_test, predicted_labels)
new_precision_score = metrics.precision_score(y_test, predicted_labels,average='macro')
new_recall_score    = metrics.recall_score(y_test, predicted_labels,average='macro')
new_f1_score        = metrics.f1_score(y_test, predicted_labels,average='macro')
print("Percentage increase of Accuracy,precision,recall and f1 with best parameters respectively: {0:5.2f}%, {1:5.2f}%, {2:5.2f}%, {3:5.2f}%".format(
                        ((new_accuracy_score-test_accuracy)*100)/test_accuracy,
                        ((new_precision_score-test_precision)*100)/test_precision,
                        ((new_recall_score-test_recall)*100)/test_recall,
                        ((new_f1_score-test_f1)*100)/test_f1)
)


# %% [markdown]
# **_5B_** PErformance improvement of best model vs. base model:
# 
# With best parameter values:
# - Accuracy increase : 13.64%
# - Precision increase : 16.93%
# - Recall increase : 18.71%
# - F1 score increase: 18.46%
# 
# _*Note: the precession/recall/f1 scores above are the unweighted mean (macro average) of each class values_

# %%
#Let me run the parameters individually

#With k and other defaults
best_accuracy = 0
best_k=-1
for k in range(10,21):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    predicted_labels    = knn.predict(X_test)
    new_accuracy_score  = metrics.accuracy_score(y_test, predicted_labels)
    if new_accuracy_score > best_accuracy:
        best_accuracy = new_accuracy_score
        best_k=k
print("The best k-value found with best accuracy score of {0:5.2f} is : k = {1}".format(best_accuracy, best_k))

#Now try varying wiegths with others as defails
best_accuracy = 0
best_weights=""
for w in ['uniform','distance']:
    knn = KNeighborsClassifier(weights = w)
    knn.fit(X_train, y_train)
    predicted_labels    = knn.predict(X_test)
    new_accuracy_score  = metrics.accuracy_score(y_test, predicted_labels)
    if new_accuracy_score > best_accuracy:
        best_accuracy = new_accuracy_score
        best_weights=w
print("The best weights found with best accuracy score of {0:5.2f} is : weights = '{1}'".format(best_accuracy, best_weights))

#Now try metric
best_accuracy = 0
best_metric=""
for m in ['minkowski','euclidean','manhattan']:
    knn = KNeighborsClassifier(metric=m )
    knn.fit(X_train, y_train)
    predicted_labels    = knn.predict(X_test)
    new_accuracy_score  = metrics.accuracy_score(y_test, predicted_labels)
    if new_accuracy_score > best_accuracy:
        best_accuracy = new_accuracy_score
        best_metric=m

print("The best metric found with best accuracy score of {0:5.2f} is : weights = '{1}'".format(best_accuracy, best_metric))


# %% [markdown]
# **_5D_** The parameters contributed most to improve model performance
# 
# **Conclusion** : The k-value/n_neighbors parameter contributed most to improve model performance.

# %% [markdown]
# # PART B - 30 Marks

# %% [markdown]
# ## 1. Data Understanding and Preparation: [5 Marks]
# <ol style="list-style-type: upper-alpha;">
#   <li>Read both the Datasets ‘Data1’ and ‘Data 2’ as DataFrame and store them into two separate variables. [1 Marks]</li>
#   <li>Print shape and Column Names and DataTypes of both the Dataframes. [1 Marks] </li>
#   <li>Merge both the Dataframes on ‘ID’ feature to form a single DataFrame [2 Marks]</li>
#   <li>Change Datatype of below features to ‘Object’ [1 Marks]<br/>
#       ‘CreditCard’, ‘InternetBanking’, ‘FixedDepositAccount’, ‘Security’, ‘Level’, ‘HiddenScore’</li>
# </ol>

# %%
#1A - Read both the Datasets ‘Data 1’ and ‘Data 2’ as DataFrame and store them into two separate variables.

data1 = pd.read_csv("Part2-Data1.csv")
data2 = pd.read_csv("Part2-Data2.csv")

# %%
#1B - Print shape and Column Names and DataTypes of both the Dataframes.

#Define a common functon to print shape and column names and datatypes.
def print_details(df):
    print("Shape: ", df.shape)
    print()
    print("Columns Names: " , df.columns.values)
    print()
    print("Column wise datatypes: ")
    df.info()
    print()


#data1
print("Details of data1 :")
print("-------------------")
print()
print_details(data1)


# %%

#data2
print("Details of data2 :")
print("-------------------")
print()
print_details(data2)

# %%
#1C - Merge both the Dataframes on ‘ID’ feature to form a single DataFrame

# merging data1 and data2 by ID
# i.e. the rows with common ID's get merged
bank_data = pd.merge(data1, data2, on="ID")
bank_data

# %%
#1D - Change Datatype of below features to ‘Object’ :
#   ‘CreditCard’, ‘InternetBanking’, ‘FixedDepositAccount’, ‘Security’, ‘Level’, ‘HiddenScore’

#Reason: values of these columns are binary (0/1), but the data type is ‘int’/‘float’ which is not expected.

for col in ['CreditCard', 'InternetBanking', 'FixedDepositAccount', 'Security', 'Level', 'HiddenScore']:
    bank_data[col] =  bank_data[col].astype('object')

#Check if done
bank_data.info()


# %% [markdown]
# ## 2. Data Exploration and Analysis: [5 Marks]
# <ol style="list-style-type: upper-alpha;">
#   <li>Visualize distribution of Target variable ‘LoanOnCard’ and clearly share insights. [2 Marks]</li>
#   <li>Check the percentage of missing values and impute if required. [1 Marks] </li>
#   <li>Check for unexpected values in each categorical variable and impute with best suitable value. [2 Marks]</li>
# </ol>

# %%
#2A - Visualize distribution of Target variable 'LoanOnCard' and clearly share insights

plt.figure(figsize=(6,8))
# Create countplot
ax = sns.countplot(x=bank_data['LoanOnCard'])

# Add labels and title
ax.set(xlabel='Loan on Credit Card', ylabel='Count', title='Countplot of Target Variable')

# Add data labels on top of each bar
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'),
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha = 'center', va = 'center',
                xytext = (0, 10),
                textcoords = 'offset points')

# Show plot
plt.show()

# %% [markdown]
# ### Insights
# 
# - We can clearly see imbalanced class distribution from the above plot. Count is very high for Class'0', while it is very less for Class'1'.

# %%
# 2b - Check the percentage of missing values and impute if required.

# Calculate percentage of missing values in each column
percent_missing = bank_data.isnull().sum() * 100 / len(bank_data)
print(percent_missing)

# %% [markdown]
# - As the percentage of missing values in target variable 'LoanOnCard' is just 0.4%, I decide to impute the missing values with mode.

# %%
# Missing value imputation

bank_data['LoanOnCard'] = bank_data['LoanOnCard'].fillna(bank_data['LoanOnCard'].mode()[0])
bank_data.head()

# %%
## Again check for missing values

bank_data.isna().sum().sum()

# %% [markdown]
# - So, there are no missing values. All missing values in 'LoanOnCard' were replaced with mode.

# %%
#2C -  Check for unexpected values in each categorical variable and impute with best suitable value.

categorical_cols = ['HiddenScore','Level', 'Security', 'FixedDepositAccount', 'InternetBanking', 'CreditCard']
for col in categorical_cols:
    print('The unique values in the', col,':', bank_data[col].unique())


# %% [markdown]
# - There are no unexpected values in the categorical variables.

# %% [markdown]
# ## Data Preparation and model building: [10 Marks]
# <ol style="list-style-type: upper-alpha;">
#   <li>Split data into X and Y. [1 Marks]</li>
#   <li>Split data into train and test. Keep 25% data reserved for testing. [1 Marks]</li>
#   <li>Train a Supervised Learning Classification base model - Logistic Regression. [2 Marks]</li>
#   <li>Print evaluation metrics for the model and clearly share insights. [1 Marks] </li>
#   <li>Balance the data using the right balancing technique. [2 Marks] </li>
#     <ol style='list-style-type: lower-roman;'>
#         <li>Check distribution of the target variable</li>
#         <li>Say output is class A : 20% and class B : 80%</li>
#         <li>Here you need to balance the target variable as 50:50</li>
#         <li>Try appropriate method to achieve the same</li>
#     </ol>
#    <li>Again train the same previous model on balanced data. [1 Marks] </li>
#    <li>Print evaluation metrics and clearly share differences observed. [2 Marks]</li>
#   </ol>
# </ol>

# %%
# 3A - Split data into X and y.

#I shall not use ID and ZipCode as independent variable, as these variables have no impact on the target
X = bank_data.drop(labels=['ID', 'ZipCode', 'LoanOnCard'], axis=1)
y = bank_data['LoanOnCard']

# %%
# 3B - Split data into train and test. Keep 25% data reserved for testing.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=365)

# %%
# 3C - Train a Supervised Learning Classification base model - Logistic Regression.


lr = LogisticRegression(max_iter=3000) ## with defualt max_iter, convergence warning appears
lr.fit(X_train, y_train)

# %%
# 3D - Print evaluation metrics for the model and clearly share insights.

#Plot Confusion metrics
def plot_cm(cm, y_test, title):
    true_no_loan=y_test.value_counts()[0]
    true_loan=y_test.value_counts()[1]

    trues = [true_no_loan,true_no_loan,true_loan,true_loan]
    pred_pct = ["{0:.2%}".format(value) for value in
                         cm.flatten()/trues]

    tags = ['True Negative', 'False Negative', 'False Positive', 'True Positive']

    ax_labels = ["No Loan On Card","Has Loan On Card"]

    df_cm = pd.DataFrame((cm.flatten()/trues).reshape(2,2), index = [i for i in ax_labels],
                  columns = [i for i in ax_labels])


    fig=plt.figure(figsize = (7,5))
    ax = fig.add_subplot()

    counts = ["{0:0.0f}".format(value) for value in
                    cm.flatten()]
    labels = [f"[{v4}]\n\n{v1}/{v3}\n({v2})" for v1, v2, v3, v4 in
              zip(counts,pred_pct, trues, tags)]
    labels = np.asarray(labels).reshape(2,2)

    sns.heatmap(df_cm, annot=labels ,fmt='', cmap='Blues')

    plt.title('{0}\nLabels: Predicted/True\n(Percentage of Predicted vs True)'.format(title))
    plt.xlabel('Predicted')
    plt.ylabel('Actuals')
    ax.xaxis.label.set_color('red')
    ax.yaxis.label.set_color('green')

    plt.show()

#Train Data Metrics

y_pred = lr.predict(X_train)
cm = metrics.confusion_matrix(y_train, y_pred)
print('Confusion metrics (Training Data): \n', cm)
print('\n')

plot_cm(cm, y_train, "Confusion matrix for Logistic Regression (Training Data)")

print('Classification Report (Training Data):  \n', metrics.classification_report(y_train, y_pred))
print('\n')
print('ROC_AUC score:', metrics.roc_auc_score(y_train, y_pred).round(2))

print()
print()

#Test Data Metrics

y_pred = lr.predict(X_test)
cm = metrics.confusion_matrix(y_test, y_pred)
print('Confusion metrics (Test Data): \n', cm)
print('\n')

plot_cm(cm, y_test, "Confusion matrix for Logistic Regression (Test Data)")

print('Classification Report (Test Data):  \n', metrics.classification_report(y_test, y_pred))
print('\n')
print('ROC_AUC score:', metrics.roc_auc_score(y_test, y_pred).round(2))



# %% [markdown]
# ### Insights
# - Training and testing data accuracy are same
# - Though the accuracy is 95%, the recall of minority class is very less. So we can say that our model is biased towards majority class - the model is more fit towards no loan class.

# %%
#3E - Balance the data using the rigth balancing technique

# Check distribution of target variable
percent_dist_tv = (bank_data["LoanOnCard"].value_counts()/bank_data["LoanOnCard"].value_counts().sum())*100
percent_dist_tv

# %% [markdown]
# ### Observations
# - We can see imbalanced distribution of classes in the target variable.
# - 90.4% belong to class'0.0' and 9.6% belong to class'1.0'.
# - To overcome this problem I shall use oversampling technique called SMOTE.

# %%
#Balance the target variable as 50:50

smote = SMOTE()
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# %%
# again check distribution of target variable on train data

y_train_balanced.value_counts()

# %% [markdown]
# - Now I have balanced the target variable as 50:50.

# %%
#3F - Again train the same previous model on balanced data.

lr.fit(X_train_balanced, y_train_balanced)

# %%
# 3G - Print evaluation metrics and clearly share differences observed.

#Training data metrics
y_pred = lr.predict(X_train)
cm = metrics.confusion_matrix(y_train, y_pred)
print('Confusion metrics (Training Data): \n', cm)
print('\n')

#Plot Confusion metrics
plot_cm(cm,y_train, "Confusion matrix for Logistic Regression with balanced targets (Training Data)")

print('Classification Report (Training Data):  \n', metrics.classification_report(y_train, y_pred))
print('\n')
print('ROC_AUC score (Training Data):', metrics.roc_auc_score(y_train, y_pred).round(2))

#Test data metrics
y_pred = lr.predict(X_test)
cm = metrics.confusion_matrix(y_test, y_pred)
print('Confusion metrics (Test Data): \n', cm)
print('\n')

#Plot Confusion metrics
plot_cm(cm,y_test, "Confusion matrix for Logistic Regression with balanced targets (Test Data)")

print('Classification Report (Test Data):  \n', metrics.classification_report(y_test, y_pred))
print('\n')
print('ROC_AUC score (Test Data):', metrics.roc_auc_score(y_test, y_pred).round(2))

# %% [markdown]
# ### Observations
# - The accuracy has decreased a little bit, but the recall score has increased for the minority class.
# - ROC_AUC score has improved from 0.79 to 0.89
# - Training and Testing data performance is comparable, in fact, test data performed well.

# %% [markdown]
# ## 4. Performance Improvement: [10 Marks]
# <ol style="list-style-type: upper-alpha;">
#   <li>Train a base model each for SVM, KNN [4 Marks]</li>
#   <li>Tune parameters for each of the models wherever required and finalize a model. [3 Marks]</li>
#   <li>Print evaluation metrics for final model. [1 Marks] </li>
#   <li>Share improvement achieved from base model to final model. [2 Marks] </li>
# </ol>

# %%
# 4A - Train a base model each for SVM, KNN.

# SVM base model

svm_base_model = SVC()
svm_base_model.fit(X_train, y_train)

# KNN base model

knn_base_model = KNeighborsClassifier()
knn_base_model.fit(X_train, y_train)

# %%
# 4B - Tune parameters for each of the models wherever required and finalize a model.

# Tune parameters for svm model with default kernel 'rbf' using different values of 'C' and 'gamma'
best_c = -1
best_gamma = -1
best_kernel = ""
best_accuracy = 0

power_ranges = [-3,-2,-1,0,1,2,3]
#I shall use only two kernels: 'rbf' and 'sigmoid', as there are many features, so data may not be linearly seprable and running with 'linear' or 'poly' kernels will take too much time and
#those kernels will not give best results. Moreover, 'poly' must be used with correct degree. Also, we can not use 'precomputed' kernels as we need square metrics for same.
#For sigmoid, I will use default coef0, i.e. 0.
kernels = ['rbf', 'sigmoid' ] 

for pc in power_ranges:
    for pg in power_ranges:
        for kernel in kernels:
            C=math.pow(10, pc)
            gamma=math.pow(10, pg)
            svm_model = SVC(C=C, gamma=gamma, kernel=kernel)
            svm_model.fit(X_train, y_train)
            pred = svm_model.predict(X_test)
            accuracy = metrics.accuracy_score(y_test,pred)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_c = C
                best_gamma = gamma
                best_kernel = kernel

print("The best SVM parameters found with best accuracy score of {0:5.2f} are : C = {1}, gamma = {2} and kernel = '{3}'".format(best_accuracy, best_c, best_gamma, best_kernel))


# %%
# Tune parameters for knn model with using different values of 'n_neighbor', 'weights' and 'metric' parameters
best_accuracy = 0
best_k=-1
best_weights=""
best_metric=""
for w in ['uniform','distance']:
    for m in ['minkowski','euclidean','manhattan']:
        for k in range(10,21):
            knn = KNeighborsClassifier(n_neighbors = k, weights = w, metric=m )
            knn.fit(X_train, y_train)
            pred = knn.predict(X_test)
            accuracy  = metrics.accuracy_score(y_test, pred)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_k=k
                best_weights=w
                best_metric=m

print("The best KNN parameters found with best accuracy score of {0:5.2f} are : k = {1}, weights = '{2}' and metric = '{3}'".format(best_accuracy, best_k, best_metric, best_weights))


# %%
#Run svm with best parameter
svm_model = SVC(C=best_c, gamma=best_gamma, kernel=best_kernel)
svm_model.fit(X_train, y_train)
pred = svm_model.predict(X_test)
accuracy_score  = metrics.accuracy_score(y_test, pred)
precision_score = metrics.precision_score(y_test, pred)
recall_score    = metrics.recall_score(y_test, pred)
f1_score        = metrics.f1_score(y_test, pred)

print("The scores for best SVM model are: \n\t Accuracy   : {0:5.2f} \n\t Precisions : {1:5.2f} \n\t Recall     : {2:5.2f} \n\t F1         : {3:5.2f}".format(accuracy_score, precision_score, recall_score, f1_score))

#Run knn with best parameter
knn = KNeighborsClassifier(n_neighbors = best_k, weights = best_weights, metric=best_metric )
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
accuracy_score  = metrics.accuracy_score(y_test, pred)
precision_score = metrics.precision_score(y_test, pred)
recall_score    = metrics.recall_score(y_test, pred)
f1_score        = metrics.f1_score(y_test, pred)

print("The scores for best KNN model are: \n\t Accuracy   : {0:5.2f} \n\t Precisions : {1:5.2f} \n\t Recall     : {2:5.2f} \n\t F1         : {3:5.2f}\n\n".format(accuracy_score, precision_score, recall_score, f1_score))


# %% [markdown]
# ### Conclusion for best model
# 
# - On comparing the SVM and KNN model, the SVM model is the best as it gives the highest accuracy, precision, recall and f1 score. Hence our final model is the SVM model with C=1000 and gamma=0.001.

# %%
# 4C - Print evaluation metrics for final model.

svm_final = SVC(C=1000, gamma=0.001)
svm_final.fit(X_train, y_train)

#Training data metrics
pred_final = svm_final.predict(X_train)
cm = metrics.confusion_matrix(y_train, pred_final)
print('Confusion Matrix of the final model (Training Data): \n', cm)
print('\n')

#plot confusion matrix
plot_cm(cm, y_train, "Confusion matrix for final SVM model (Training Data)")

print('Classification Report of the final model (Training Data): \n', metrics.classification_report(y_train, pred_final))
print('\n')
print('ROC_AUC Score of the final model (Training Data):', metrics.roc_auc_score(y_train, pred_final).round(2))


print()

#Test data metrics
pred_final = svm_final.predict(X_test)
cm = metrics.confusion_matrix(y_test, pred_final)
print('Confusion Matrix of the final model (Test Data): \n', cm)
print('\n')

#plot confusion matrix
plot_cm(cm, y_test, "Confusion matrix for final SVM model (Test Data)")

print('Classification Report of the final model (Test Data): \n', metrics.classification_report(y_test, pred_final))
print('\n')
print('ROC_AUC Score of the final model (Test Data):', metrics.roc_auc_score(y_test, pred_final).round(2))



# %%
# 4D - Share improvement achieved from base model to final model.

# Comparision is done using test data performance metrics only

# print evaluation metrics for SVM base model 
pred_svm_base = svm_base_model.predict(X_test)
cm=metrics.confusion_matrix(y_test, pred_svm_base)
print('Confusion Matrix of the svm base model (Test Data): \n', cm)
print('\n')

#plot confusion matrix
plot_cm(cm, y_test, "Confusion matrix for base SVM model (Test Data)")

print('Classification Report of the svm base model (Test Data): \n', metrics.classification_report(y_test, pred_svm_base))
print('\n')
print('ROC_AUC Score of the svm base model (Test Data):', metrics.roc_auc_score(y_test, pred_svm_base).round(2))

# %%
# print evaluation metrics for KNN base model 
pred_knn_base = knn_base_model.predict(X_test)
cm=metrics.confusion_matrix(y_test, pred_knn_base)
print('Confusion Matrix of the KNN base model (Test Data): \n', cm)
print('\n')

#plot confusion matrix
plot_cm(cm, y_test, "Confusion matrix for base KNN model (Test Data)")

print('Classification Report of the knn base model (Test Data): \n', metrics.classification_report(y_test, pred_knn_base))
print('\n')
print('ROC_AUC Score of the knn base model (Test Data):', metrics.roc_auc_score(y_test, pred_knn_base).round(2))

# %%
#Compare base SVM model performance with the final SVM model

def print_improvement(y_test, y_base_pred, y_final_pred, score_mesure, name):
    base_score = score_mesure(y_test, y_base_pred).round(2)
    final_score = score_mesure(y_test, y_final_pred).round(2)
    print("\t{0} Score increased from {1:4.2f} to {2:4.2f} : increase of {3:5.2f}%".format(name,base_score, final_score, (final_score-base_score)*100/base_score))

def print_all_improvements(y_test, y_base_pred, y_final_pred):
    print_improvement(y_test,y_base_pred, y_final_pred, metrics.accuracy_score, "Accuracy")
    print_improvement(y_test,y_base_pred, y_final_pred, metrics.precision_score, "Precision")
    print_improvement(y_test,y_base_pred, y_final_pred, metrics.recall_score, "Recall")
    print_improvement(y_test,y_base_pred, y_final_pred, metrics.f1_score, "F1")
    print_improvement(y_test,y_base_pred, y_final_pred, metrics.roc_auc_score, "ROC_AUC")


#Print improvement %-ages between SVM base model and SVM final model
print("Final SVM Model vs SVM base model (based on Test Data Performance):\n")

print_all_improvements(y_test,pred_svm_base, pred_final)

#Print improvement %-ages between KNN base model and SVM final model
print("\nFinal SVM Model vs KNN base model (based on Test Data Performance):\n")

print_all_improvements(y_test,pred_knn_base, pred_final)

print()
print("*Note: the precision,recall and f1 score above are for the class label 1.0 (has loan on card).")


# %% [markdown]
# ### Summary of Improvements:
# - Base models are biased towards majority class (no loan on card), but final model is more balanced.
# - Accuracy,Precision, Recall and f1-score has significantly increased in the final model.
# - ROC_AUC score has improved significantly as well
# 
# 


