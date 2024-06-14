import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer


ds = pd.read_csv("C:/Users/User/Downloads/Machine Learning/water_potability.csv")
ds

dh=ds.head()
print(dh)

dc=ds.describe()
print(dc)

dw=ds.isna().sum()
print(dw)

# Impute missing values using mean
imp_mean = SimpleImputer(strategy="mean")
ds_im = imp_mean.fit_transform(ds)

# Convert the NumPy array to a Pandas DataFrame
df_im = pd.DataFrame(ds_im, columns=ds.columns)

# Display the DataFrame without truncation
pd.set_option('display.max_rows', None)  # To display all rows
pd.set_option('display.max_columns', None)  # To display all columns
print(df_im)

target_count = df_im['Potability'].value_counts()

# Print the counts of each class
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')

# Plot the target count
target_count.plot(kind='bar', title='Count (Potability)')
plt.xlabel('Potability')
plt.ylabel('Count')
plt.show()

from sklearn.utils import resample

# Separate majority and minority classes
majority_class = df_im[df_im['Potability'] == 0]
minority_class = df_im[df_im['Potability'] == 1]

# Upsample the minority class
minority_upsampled = resample(minority_class, 
                              replace=True,     # sample with replacement
                              n_samples=len(majority_class),    # to match majority class
                              random_state=42)  # reproducible results

# Combine the upsampled minority class with the majority class
balanced_df = pd.concat([majority_class, minority_upsampled])

# Display new class counts
print(balanced_df['Potability'].value_counts())

# Plot the balanced target count
balanced_df['Potability'].value_counts().plot(kind='bar', title='Count (Potability) After Oversampling')
plt.xlabel('Potability')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10,5))
sns.heatmap(balanced_df.corr())
plt.show()



# Split the data into features and target
x = balanced_df.drop('Potability', axis=1)
y = balanced_df['Potability']


#feature scaling Decision Tree
from sklearn.preprocessing import StandardScaler
sc_xdt=StandardScaler()
xdt_train=sc_x.fit_transform(x)

#fitting Decision Tree classification to train set
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy', random_state=42)
classifier.fit(xdt_train,y)

# Get feature importances
importances = classifier.feature_importances_
feature_importances = pd.DataFrame(importances, index=x.columns, columns=['Importance']).sort_values(by='Importance', ascending=False)
print(feature_importances)

# Visualize feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances.index, y=feature_importances['Importance'])
plt.title('Feature Importances from Decision Tree')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.show()

from sklearn.feature_selection import RFE

# Initialize RFE with the decision tree classifier
rfe = RFE(estimator=DecisionTreeClassifier(random_state=42), n_features_to_select=5, step=1)

# Fit RFE
rfe.fit(xdt_train,y)

# Get the selected feature names
selected_features = x.columns[rfe.support_]
print("Selected features by RFE:", selected_features)

# Use only selected features
X_selected = xdt_train[:, rfe.support_]

# Visualize selected features
plt.figure(figsize=(10, 6))
sns.barplot(x=selected_features, y=rfe.estimator_.feature_importances_)
plt.title('Selected Feature Importances from Decision Tree')
plt.xlabel('Selected Features')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.show()

#splitting dataset to train and test  to 80% and 20% for Decision Tree
from sklearn.model_selection import train_test_split
xdt2_train, xdt_test, ydt_train, ydt_test = train_test_split(X_selected,y,test_size=0.20, random_state=42)

#Decision Tree predict target variable
classifier.fit(xdt2_train, ydt_train)
ydt_pred=classifier.predict(xdt_test)
ydt_test,ydt_pred

#Confusion matrix for Decision Tree
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ydt_test,ydt_pred)
cm

#Evaluation for Decision Tree
from sklearn.metrics import classification_report, accuracy_score

print("Accuracy:", accuracy_score(ydt_test, ydt_pred))
print("Classification Report:\n", classification_report(ydt_test, ydt_pred))

# Visualize the confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns

conf_matrix = confusion_matrix(ydt_test, ydt_pred)


plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Potable', 'Potable'], yticklabels=['Not Potable', 'Potable'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
