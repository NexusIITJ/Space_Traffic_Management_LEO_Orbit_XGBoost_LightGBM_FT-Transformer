import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import xgboost 


# Load Dataset 
df = pd.read_excel("Train_2000.xlsx")
print(df.info())


df['HighRisk'] = ((df['cdmPc'] > 1e-6) & (df['cdmMissDistance'] < 2000)).astype(int)

from sklearn.preprocessing import LabelEncoder # for ecnoding Categorical Features 

categorical_cols = ['SAT1_CDM_TYPE', 'SAT2_CDM_TYPE',
                    'rso1_objectType', 'rso2_objectType',
                    'org1_displayName', 'org2_displayName']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))


# Build Feature Matrix

features = [
    'cdmMissDistance', 'cdmPc',
    'SAT1_CDM_TYPE', 'SAT2_CDM_TYPE',
    'rso1_objectType', 'rso2_objectType',
    'org1_displayName', 'org2_displayName',
    'condition_24H<tca<72H', 'condition_Pc>1e-6',
    'condition_missDistance<2000m', 'condition_Radial<100m',
    'condition_InTrack<500m', 'condition_CrossTrack<500m',
    'condition_sat2posUnc>1km', 'condition_sat2Obs<25'
]

X = df[features]
y = df['HighRisk']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
