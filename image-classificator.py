import os
import re

import csv
import tensorflow as tf
import tensorflow.python.platform
from PIL import Image
from pickle import Unpickler
from tensorflow.python.platform import gfile
import numpy as np
import pandas as pandas
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC, LinearSVC
import matplotlib.pyplot as plt
import pickle
import tarfile

model_dir = "imagenet"
images_dir = "test/"
list_images =  [images_dir + fileName for fileName in sorted(os.listdir(images_dir), key=lambda fn : int(fn.split('.')[0])) if re.search("^.+(\.png|\.PNG)$", fileName)]

list_image_ids = []
print('so far so good')


def create_graph():
    with gfile.FastGFile(os.path.join(
            model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name="")


def find_image_label(csv_file_location, id):
    with open(csv_file_location) as inf:
        reader = csv.reader(inf)

        for row in reader:
            if row[0] == id:
                # print('Found: {}'.format(row))
                return row[1]


def extract_features(list_images):
    nb_features = 2048  # 2048
    features = np.empty((len(list_images), nb_features))
    labels = []

    create_graph()

    with tf.Session() as sess:

        next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')

        for index, image in enumerate(list_images):
            if (index == 100):
                print('Processing ' + image + ' index: ' + str(index))
            if not gfile.Exists(image):
                tf.logging.fatal('file does not exist ' + image)

            #image_array = np.array(image)# convert the png to a numpy array cause InceptionV3 does not work with png files
            image_data = Image.open(image)  # gfile.FastGFile(image, 'rb').read()

            image_data_array = np.array(image_data)[:, :, 0:3]
            prediction = sess.run(next_to_last_tensor, {'DecodeJpeg:0': image_data_array})
            features[index, :] = np.squeeze(prediction)

            image_name = image.split('/')[1]
            image_name_without_extension = image_name.split('.')[0]
            list_image_ids.append(image_name_without_extension)

            print(image_name_without_extension)
            # image_label = find_image_label('trainLabels.csv', image_name_without_extension)
            #
            # labels.append(image_label) # use the labels from the labels file
    return features


features = extract_features(list_images)

# save the results to the hdd
pickle.dump(features, open('test-features-dump', 'wb'))
# pickle.dump(labels, open('test-dump-small', 'wb'))

print('!!!!!!!!!FINISHED GENERATING FEATURES AND LABELS !!!!!!!!!!!!!')

# # Load the features and labels
features = pickle.load(open('features', 'rb'))
labels = pickle.load(open('labels', 'rb'))
#
X_train = features
y_train = labels

X_test = pickle.load(open('test-features-dump', 'rb'))  # test data extracted features
#
# # classify them
clf = LinearSVC(C=1.0, loss='squared_hinge', penalty='l2', multi_class='ovr')
clf.fit(X_train, y_train)

# X_test find train features

y_predict = clf.predict(X_test)

def plot_confusion_matrix(y_true, y_pred):
    cm_array = confusion_matrix(y_true, y_pred)
    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    plt.imshow(cm_array[:-1, :-1], interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix", fontsize=16)
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_label('Number of images', rotation=270, labelpad=30, fontsize=12)
    xtick_marks = np.arange(len(true_labels))
    ytick_marks = np.arange(len(pred_labels))
    plt.xticks(xtick_marks, true_labels, rotation=90)
    plt.yticks(ytick_marks, pred_labels)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12
    fig_size[1] = 12
    plt.rcParams["figure.figsize"] = fig_size


# y_test expected labels

# print("Accuracy: {0:0.1f}%".format(accuracy_score(y_test,y_predict)*100))


# export to .csv file

pandas.DataFrame({'id': list_image_ids, 'label': y_predict}).to_csv('kaggle-submission.csv', index=False)
