import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv('E:\Github\TechnocolabsInternshipProjects\Project 2\deploy_resource\cleaned_data.csv')
x = df[['PAY_1', 'PAY_AMT1', 'LIMIT_BAL', 'BILL_AMT1']].values
y = df['default payment next month'].values
rf = RandomForestClassifier(n_estimators=150, max_depth=8)

rf.fit(x, y)

pickle.dump(rf, open('model.pkl','wb'))





