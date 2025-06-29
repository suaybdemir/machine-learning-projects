import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataset_root = 'datasets/nsl-kdd'
train_file = os.path.join(dataset_root, 'KDDTrain+.txt')
test_file = os.path.join(dataset_root, 'KDDTest+.txt')
header_names = ['duration', 'protocol_type', 'service', 'flag',
                'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
                'urgent', 'hot', 'num_failed_logins', 'logged_in',
                'num_compromised', 'root_shell', 'su_attempted',
                'num_root', 'num_file_creations', 'num_shells',
                'num_access_files', 'num_outbound_cmds',
                'is_host_login', 'is_guest_login', 'count',
                'srv_count', 'serror_rate', 'srv_serror_rate',
                'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
                'diff_srv_rate', 'srv_diff_host_rate',
                'dst_host_count', 'dst_host_srv_count',
                'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
                'dst_host_serror_rate', 'dst_host_srv_serror_rate',
                'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
                'attack_type', 'success_pred']
col_names = np.array(header_names)

nominal_index = [1, 2, 3]
binary_index = [6, 11, 13, 14, 20, 21]
numeric_index = list(set(range(41)).difference(nominal_index).difference(binary_index))

nominal_cols = col_names[nominal_index].tolist()
binary_cols = col_names[binary_index].tolist()
numeric_cols = col_names[numeric_index].tolist()

category = defaultdict(list)

attack_mapping = {}

for attack in ['spy', 'perl', 'loadmodule', 'rootkit', 'buffer_overflow']:
    attack_mapping[attack] = 'u2r'

for attack in ['phf', 'multihop', 'ftp_write', 'imap', 'warezmaster', 'guess_passwd', 'warezclient']:
    attack_mapping[attack] = 'r2l'

for attack in ['nmap', 'portsweep', 'ipsweep', 'satan']:
    attack_mapping[attack] = 'probe'

for attack in ['land', 'pod', 'teardrop', 'back', 'smurf', 'neptune']:
    attack_mapping[attack] = 'dos'

attack_mapping['normal'] = 'benign'

train_df = pd.read_csv(train_file, names=header_names)
train_df['attack_category'] = train_df['attack_type'].map(lambda x: attack_mapping.get(x, 'unknown'))
train_df.drop(['success_pred'], axis=1, inplace=True)

test_df = pd.read_csv(test_file, names=header_names)
test_df['attack_category'] = test_df['attack_type'].map(lambda x: attack_mapping.get(x, 'unknown'))
test_df.drop(['success_pred'], axis=1, inplace=True)

train_attack_types = train_df['attack_type'].value_counts()
train_attack_cats = train_df['attack_category'].value_counts()
test_attack_types = test_df['attack_type'].value_counts()
test_attack_cats = test_df['attack_category'].value_counts()

train_attack_types.plot(kind='barh', figsize=(20, 10), fontsize=14)
plt.title("Train Attack Types", fontsize=18)
plt.xlabel("Count", fontsize=14)
plt.ylabel("Attack Type", fontsize=14)
plt.show()

train_attack_cats.plot(kind='barh', figsize=(20, 10), fontsize=14)
plt.title("Train Attack Categories", fontsize=18)
plt.xlabel("Count", fontsize=14)
plt.ylabel("Attack Category", fontsize=14)
plt.show()

train_Y = train_df['attack_category']
train_x_raw = train_df.drop(['attack_category', 'attack_type'], axis=1)
test_Y = test_df['attack_category']
test_x_raw = test_df.drop(['attack_category', 'attack_type'], axis=1)

combined_df_raw = pd.concat([train_x_raw, test_x_raw])
combined_df = pd.get_dummies(combined_df_raw, columns=nominal_cols, drop_first=True)

train_x = combined_df[:len(train_x_raw)]
test_x = combined_df[len(train_x_raw):]

dummy_variables = list(set(train_x) - set(combined_df_raw))

from sklearn.preprocessing import StandardScaler

durations = train_x['duration'].values.reshape(-1, 1)
standard_scaler = StandardScaler().fit(durations)
scaled_durations = standard_scaler.transform(durations)

print(f"Scaled durations {pd.Series(scaled_durations.flatten()).describe()}")

from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler().fit(durations)
min_max_scaled_durations = min_max_scaler.transform(durations)
print(f"Min max scaled durations {pd.Series(min_max_scaled_durations.flatten()).describe()}")

from sklearn.preprocessing import RobustScaler

min_max_scaler = RobustScaler().fit(durations)
robust_scaled_durations = min_max_scaler.transform(durations)
print(f"Robust scaled {pd.Series(robust_scaled_durations.flatten()).describe()}")

standard_scaler = StandardScaler().fit(train_x[numeric_cols])

train_x[numeric_cols] = train_x[numeric_cols].astype(float)
test_x[numeric_cols] = test_x[numeric_cols].astype(float)


train_x.loc[:, numeric_cols] = standard_scaler.transform(train_x[numeric_cols])
test_x.loc[:, numeric_cols] = standard_scaler.transform(test_x[numeric_cols])


from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix , zero_one_loss

sm = SMOTE(random_state=42)
X_resampled,y_resampled = sm.fit_resample(train_x,train_Y)

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state=42)
clf.fit(X_resampled, y_resampled)
pred_y = clf.predict(test_x)
results = confusion_matrix(test_Y,pred_y)
error = zero_one_loss(test_Y,pred_y)

import seaborn as sns
from sklearn.metrics import confusion_matrix

labels = train_Y.unique()
cm = confusion_matrix(test_Y,pred_y,labels=labels)

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()