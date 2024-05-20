
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
import pickle
import pandas as pd

NasaDataSet=pd.read_csv("Asteroid_Data.csv")
print(NasaDataSet)

import numpy as np
df = pd.DataFrame(NasaDataSet)
columns1 = ['Est_Diameter_Min', 'Est_Diameter_Max', 'Relative_Velocity', 'Absolute_Magnitude','Orbit_ID']
def remove_outliers(df, columns1):
    no_outliers = df.copy()
    for col in columns1:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR
        no_outliers = no_outliers[(no_outliers[col] > lower_limit) & (no_outliers[col] < upper_limit)]
    return no_outliers

columns_to_filter = ['Est_Diameter_Min', 'Est_Diameter_Max', 'Relative_Velocity', 'Absolute_Magnitude','Orbit_ID']
no_outliers = remove_outliers(NasaDataSet, columns_to_filter)

print("Original DataSet Shape:", NasaDataSet.shape)
print("DataSet Shape after Removing Outliers:", no_outliers.shape)

#Before diving into model building, our initial step is to separate the features into X and the target variable into y.
X = no_outliers.iloc[:, :-1].values #meaning that all the feature except the last one because it's the target.
y = no_outliers.iloc[:, 8].values #meaning that the last feature since Python indexing starts from 0.

from sklearn import preprocessing
s_scaler = preprocessing.StandardScaler()
X_s = s_scaler.fit_transform(X)

X_train5, X_test5, y_train5, y_test5 = train_test_split(X_s, y, test_size=0.3, random_state=42, stratify=y)
AdaBoost_New = AdaBoostClassifier()
params = {'n_estimators': [50, 100, 150], 'learning_rate': [0.1, 0.5, 1]}

from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(AdaBoost_New, params, cv=5)
grid.fit(X_train5, y_train5)

AdaBoostNeww = grid.best_estimator_
AdaBoostNeww.fit(X_train5, y_train5)
y_pred5 = AdaBoostNeww.predict(X_test5)

import numpy as np

# Assuming 'data' is your dataset with 'Miss_Distance' feature
# Extract the 'Miss_Distance' feature from the dataset
miss_distance = no_outliers['Miss_Distance']

# Calculate mean and standard deviation
mean_distance = np.mean(miss_distance)
std_distance = np.std(miss_distance)

# Standardize the Miss_Distance feature
standardized_distance = (miss_distance - mean_distance) / std_distance
print("Standardized Miss_Distance:")
print(standardized_distance)


pickle.dump(AdaBoostNeww,open('ada_model.pkl','wb'))