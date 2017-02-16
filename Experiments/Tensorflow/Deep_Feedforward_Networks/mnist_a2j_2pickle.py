"""
Download notMNIST and generate a pickle file
Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
"""
# On command line: python3 mnist_a2j_2pickle.py
# Prerequisite: tensorflow (see tensorflow.org)

from __future__ import print_function

import numpy as np
import pickle
import os
import sys
import tarfile
import random
import matplotlib.image as img

from os.path import join
from six.moves.urllib.request import urlretrieve

url = 'http://yaroslavvb.com/upload/notMNIST/'

def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        print('Downloading ', filename, " ...")
        filename, _ = urlretrieve(url + filename, filename)
        statinfo = os.stat(filename)
        if statinfo.st_size == expected_bytes:
            print('Found and verified', filename)
        else:
            raise Exception('Failed to verify' +
                            filename + '. Can you get to it with a browser?')
    else:
        print('Found and verified', filename)
    return filename

def extract(filename):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    data_folders = []
    if os.path.exists(root):
        data_folders = [os.path.join(root, d)
                        for d in sorted(os.listdir(root)) if d != '.DS_Store']
    if len(data_folders) == num_classes:
        print("Using previously extracted files...")
        print(data_folders)
        return data_folders
    tar = tarfile.open(filename)
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    sys.stdout.flush()
    tar.extractall()
    tar.close()
    data_folders = [os.path.join(root, d)
                    for d in sorted(os.listdir(root)) if d != '.DS_Store']
    if len(data_folders) != num_classes:
        raise Exception(
            'Expected %d folders, one per class. Found %d instead.' % (
                num_classes, len(data_folders)))
    print(data_folders)
    return data_folders

def getfiles_fromlist(dirs):
    files = []
    for dir in dirs:
        files.extend([os.path.join(dir,f) for f in sorted(os.listdir(dir)) if f != '.DS_Store'])
    return files

def readfile(path):
    try:
        data = img.imread(path)
        return data
    except:
        print("Error reading: ", path)
        return np.array([])

def read_image_files(files):
    imagelabels = []
    imagedata = []
    for file in files:
        parent_dir = os.path.dirname(file)
        label =  (np.arange(num_classes) == (ord(parent_dir[-1])-ord('A')) ).astype(np.float32)
        data = readfile(file)
        if (data.size > 0):
            imagelabels.append(label)
            imagedata.append(data)
    return np.array(imagedata),np.array(imagelabels)

train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)

num_classes = 10

train_folders = extract(train_filename)
test_folders = extract(test_filename)

train_files = np.array(getfiles_fromlist(train_folders))
test_files = np.array(getfiles_fromlist(test_folders))
random.shuffle(train_files)

all_dataset, all_labels = read_image_files(train_files)
test_dataset, test_labels = read_image_files(test_files)
image_size = all_dataset.shape[2]

all_dataset = all_dataset.reshape((-1,image_size*image_size)).astype(np.float32)
test_dataset = test_dataset.reshape((-1,image_size*image_size)).astype(np.float32)

data = { "test_labels" : test_labels, "all_labels" : all_labels, "test_dataset" : test_dataset,
         "all_dataset" : all_dataset }

pickle_file = open( "mnist_a2j.pickle", "wb" )
pickle.dump( data, pickle_file )
pickle_file.close()