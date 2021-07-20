import pandas as pd
import numpy as np
import tensorflow as tf
import autokeras as ak
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers

x = pd.read_csv('train-values.csv').set_index('id')
y = pd.read_csv('train-labels.csv')
x_test = pd.read_csv('test-values.csv').set_index('id')
test_ids = list(x_test.index)
df_train = pd.merge(x, y, on='id').set_index('id')

aux_df = pd.DataFrame(index=x.columns, data={'uniques': x.nunique(),
                                             'type': x.dtypes,
                                             'nulls': x.isnull().sum(),
                                             'nulls%': x.isnull().sum() / len(x) * 100
                                             })
cat_vars = aux_df[aux_df['type'] == 'object'].index
num_vars = aux_df[aux_df['type'] != 'object'].index

x[cat_vars] = x[cat_vars].fillna('none')
x[num_vars] = x[num_vars].fillna(0)

encoder = LabelEncoder()
y = encoder.fit_transform(y.status_group)
print(encoder.classes_)

# scaler = StandardScaler()
# x[num_vars] = scaler.fit_transform(x[num_vars])
# print(x)

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.3)

# train_set = tf.data.Dataset.from_tensor_slices(
#     (tf.cast(x_train.values, tf.float32),
#      tf.cast(y_train, tf.int32))
# )
# valid_set = tf.data.Dataset.from_tensor_slices(
#     (tf.cast(x_valid.values, tf.float32),
#      tf.cast(y_valid, tf.int32))
# )

# for features_tensor, target_tensor in train_set:
#     print(f'features:{features_tensor} target:{target_tensor}')

# input_node = ak.StructuredDataInput()
# output_node = ak.StructuredDataBlock(categorical_encoding=True)(input_node)
# output_node = ak.ClassificationHead(multi_label=True)(output_node)

# clf = ak.AutoModel(inputs=input_node, outputs=output_node, overwrite=True, max_trials=3)

clf = ak.StructuredDataClassifier(overwrite=False, multi_label=True, max_trials=20)

clf.fit(x_train, y_train, epochs=50)
print(clf.evaluate(x_valid, y_valid))
predict = np.hstack(clf.predict(x_test)).astype(np.int64)

predict_df = pd.DataFrame({'id': test_ids,
                           'status_group': encoder.inverse_transform(predict)}).set_index('id')
print(predict_df['status_group'].value_counts())
predict_df.to_csv('submission_ak.csv')

# TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
# TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"
#
# train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
# test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)
#
# # Initialize the structured data classifier.
# clf = ak.StructuredDataClassifier(
#     overwrite=True, max_trials=3
# )  # It tries 3 different models.
# # Feed the structured data classifier with training data.
# clf.fit(
#     # The path to the train.csv file.
#     train_file_path,
#     # The name of the label column.
#     "survived",
#     epochs=10,
# )
# # Predict with the best model.
# predicted_y = clf.predict(test_file_path)
# # Evaluate the best model with testing data.
# print(clf.evaluate(test_file_path, "survived"))