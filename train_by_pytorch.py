# -*- coding: utf-8 -*-
import os
import sys
import itertools
import math
import logging
import json
import re
import random
from collections import OrderedDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon


# Root directory of the project
ROOT_DIR = os.path.abspath("/home/g1g/Desktop/Mask_RCNN/balloon_project")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_balloon.h5")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
import balloon
#from mrcnn import model

# %%

config = balloon.BalloonConfig()
BALLOON_DIR = os.path.join(ROOT_DIR, "splice_2/")

#dataset = balloon.BalloonDataset()
#dataset.load_balloon(BALLOON_DIR, "train")
#
## Must call before using the dataset
#dataset.prepare()
# %% train class

#def train(model):
#    """Train the model."""
#    # Training dataset.
#    dataset_train = balloon.BalloonDataset()
#    dataset_train.load_balloon(BALLOON_DIR, "train")
#    dataset_train.prepare()
#
#    # Validation dataset
#    dataset_val = balloon.BalloonDataset()
#    dataset_val.load_balloon(BALLOON_DIR, "val")
#    dataset_val.prepare()
#
#    # *** This training schedule is an example. Update to your needs ***
#    # Since we're using a very small dataset, and starting from
#    # COCO trained weights, we don't need to train too long. Also,
#    # no need to train all layers, just the heads should do it.
#    print("Training network heads")
#    model.train(dataset_train, dataset_val,
#                learning_rate=config.LEARNING_RATE,
#                epochs=30,
#                layers='heads')
    


# %% A cell
#print("Image Count: {}".format(len(dataset.image_ids)))
#print("Class Count: {}".format(dataset.num_classes))
#for i, info in enumerate(dataset.class_info):
#    print("{:3}. {:50}".format(i, info['name']))


# %% A cell
#image_ids = np.random.choice(dataset.image_ids, 4)
#for image_id in image_ids:
#    image = dataset.load_image(image_id)
#    mask, class_ids = dataset.load_mask(image_id)
#    visualize.display_top_masks(image, mask, class_ids, dataset.class_names)
    


# %% A cell
#image_id = random.choice(dataset.image_ids)
#image = dataset.load_image(image_id)
#mask, class_ids = dataset.load_mask(image_id)
## Compute Bounding box
#bbox = utils.extract_bboxes(mask)
#
## Display image and additional stats
#print("image_id ", image_id, dataset.image_reference(image_id))
#log("image", image)
#log("mask", mask)
#log("class_ids", class_ids)
#log("bbox", bbox)
## Display image and instances
#visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)

# %% training
# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)

dataset_train = balloon.BalloonDataset()
dataset_train.load_balloon(BALLOON_DIR, "train")
dataset_train.prepare()

# Validation dataset
dataset_val = balloon.BalloonDataset()
dataset_val.load_balloon(BALLOON_DIR, "val")
dataset_val.prepare()

model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=50, 
            layers='heads')
