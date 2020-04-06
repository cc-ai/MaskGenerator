import os
import numpy as np
import glob
import random
import json

def get_train_test_dict(root, train_r):
    """
    Make json files of images and masks paths.
    Format is {0: {x: path_to_image, m: path_to_mask}, 1: ...}
    train_r: ratio of train test 
    """
    filenames = [os.path.basename(elem) for elem in glob.glob(os.path.join(root, "Normal/*"))]
    random.shuffle(filenames)
    
    train_index = int(train_r*len(filenames))
    train = filenames[:train_index]
    test = filenames[train_index:]

    train_dict = []
    for i in range(len(train)):
        train_dict.append( {'x': os.path.join(root, "Normal",train[i]),  'm': os.path.join(root, "Mask",train[i])})
    test_dict = []
    for i in range(len(test)):
        test_dict.append( {'x': os.path.join(root, "Normal",test[i]),  'm': os.path.join(root, "Mask",test[i])})
    return(train_dict, test_dict)

def make_filenames_json(train_dict, test_dict, root, postfix = ""):
    with open(os.path.join(root, "train" + postfix+ ".json"), "w") as write_file:
        json.dump(train_dict, write_file, indent=4)
    with open(os.path.join(root, "test" + postfix+ ".json"), "w") as write_file:
        json.dump(test_dict, write_file, indent=4)