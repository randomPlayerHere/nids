# %%
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, RobustScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn import metrics

# %%

train_path = "data/KDDTrain+.csv"
test_path  = "data/KDDTest+.csv"

columns = (['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot'
,'num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations'
,'num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate'
,'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count'
,'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate'
,'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','outcome','level'])

# %%
df = pd.read_csv(train_path)
df.columns = columns
df.head()

# %%
df.describe()

# %%
df['outcome'] = df['outcome'].map(lambda a : 'normal' if a=='normal' else 'attack')

# %%
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.pie(df['protocol_type'].value_counts(), labels = df['protocol_type'].value_counts().index, autopct='%1.0f%%')
plt.subplot(1,2,2)
plt.pie(df['outcome'].value_counts(), labels=df['outcome'].value_counts().index, autopct='%1.0f%%')
plt.show()

# %%
cat_cols = ['protocol_type', 'service', 'flag']
num_cols = [col for col in columns if col not in cat_cols + ['outcome', 'level', 'is_host_login', 'land', 'logged_in', 'is_guest_login']]
binary_cols = ['is_host_login', 'land', 'logged_in', 'is_guest_login']

# %%
X = df.drop(['outcome', 'level'], axis=1)
y = df['outcome'].map(lambda x: 0 if x == 'normal' else 1).astype('int')
y_reg = df['level'].values

# %%
preprocessor = ColumnTransformer(
    transformers=[
        ('num', RobustScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols),
        ('binary', 'passthrough', binary_cols)
    ])

# pipeline_full = Pipeline([
#     ('preprocessor', preprocessor),
#     ('classifier', KNeighborsClassifier(n_neighbors=20))
# ])

pipeline_pca = Pipeline([
    ('preprocessor', preprocessor),
    ('pca', PCA(n_components=20)),
    ('classifier', KNeighborsClassifier(n_neighbors=20))
])

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
kernal_evals = dict()
def evaluate_classification(model, name, X_train, X_test, y_train, y_test):
    train_accuracy = metrics.accuracy_score(y_train, model.predict(X_train))
    test_accuracy = metrics.accuracy_score(y_test, model.predict(X_test))
    
    train_precision = metrics.precision_score(y_train, model.predict(X_train))
    test_precision = metrics.precision_score(y_test, model.predict(X_test))
    
    train_recall = metrics.recall_score(y_train, model.predict(X_train))
    test_recall = metrics.recall_score(y_test, model.predict(X_test))
    
    kernal_evals[str(name)] = [train_accuracy, test_accuracy, train_precision, test_precision, train_recall, test_recall]
    print("Training Accuracy " + str(name) + " {}  Test Accuracy ".format(train_accuracy*100) + str(name) + " {}".format(test_accuracy*100))
    print("Training Precesion " + str(name) + " {}  Test Precesion ".format(train_precision*100) + str(name) + " {}".format(test_precision*100))
    print("Training Recall " + str(name) + " {}  Test Recall ".format(train_recall*100) + str(name) + " {}".format(test_recall*100))
    
    actual = y_test
    predicted = model.predict(X_test)
    confusion_matrix = metrics.confusion_matrix(actual, predicted)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['normal', 'attack'])

    fig, ax = plt.subplots(figsize=(10,10))
    ax.grid(False)
    cm_display.plot(ax=ax)

# # %%
# pipeline_full.fit(X_train, y_train)
# evaluate_classification(pipeline_full, "KNeighborsClassifier_Pipeline", X_train, X_test, y_train, y_test)

# %%
pipeline_pca.fit(X_train, y_train)
evaluate_classification(pipeline_pca, "KNeighborsClassifier_PCA_Pipeline", X_train, X_test, y_train, y_test)

# %%
joblib.dump(pipeline_pca, 'models/knn_pipeline.pkl')
# %%
