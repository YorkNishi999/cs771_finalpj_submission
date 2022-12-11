import json 
from PIL import Image
from transformers import ViltConfig, ViltProcessor, ViltForQuestionAnswering
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

DATA_DIR = './AdDataset/' # path to the parent dir for the dataset
IMAGE_DIR = DATA_DIR + 'images/'
ANNOTATION_DIR = DATA_DIR + 'annotations_images/image/'
SENTIMENTS_JSON = ANNOTATION_DIR + 'Sentiments.json'
TOPICS_JSON = ANNOTATION_DIR + 'Topics.json'
ACTIONS_JSON = ANNOTATION_DIR + 'QA_Action.json'
REASONS_JSON = ANNOTATION_DIR + 'QA_Reason.json'
SENTIMENTS_LIST = ANNOTATION_DIR + 'Sentiments_List2.txt' # path to the Sentiments_List2.txt file
TOPICS_LIST = ANNOTATION_DIR + 'Topics_List2.txt' # path to the Topics_List2.txt file

DICT_JSON = {'SENTIMENTS': SENTIMENTS_JSON, 'TOPICS': TOPICS_JSON, 'ACTIONS': ACTIONS_JSON, 'REASONS': REASONS_JSON}
DICT_LIST = {'SENTIMENTS': SENTIMENTS_LIST, 'TOPICS': TOPICS_LIST}

# config for Transformer
config = ViltConfig.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# processor for Transformer
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
