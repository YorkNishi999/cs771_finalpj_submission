import json 
from PIL import Image
from transformers import ViltConfig, ViltProcessor, ViltForQuestionAnswering
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os


acc_train = [0.01590958316,0.02091086302,0.07383779314,0.1655344872,0.2613464075,0.3696024573,0.4927245161,0.6077342627,0.6580817926,0.6670210881]
acc_val	= [0.01470848839,0.03632819422,0.1059720007,0.1759702286,0.2160198476,0.233740918,0.2387028177,0.2335637072,0.2309055467,0.2220450115]
loss = [747580.8746,50212.92917,46036.48589,39431.21508,33097.28982,27520.45986,22428.75571,17913.8652,14504.2213,12476.44728]

# draw result
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

x = np.linspace(0, 10, 10)

c1,c2,c3 = "blue","green","red"      # 各プロットの色
l1,l2,l3 = "Training acc","Validation accu","Training Loss"   # 各ラベル

ax1.plot(x, acc_train, color=c1, label=l1)
ax1.plot(x, acc_val, color=c2, label=l2)
ax2.plot(x, loss, color=c3, label=l3)

ax1.legend(loc = 'upper right') 
ax2.legend(loc = 'upper right') 

fig.tight_layout()
fig.savefig('./' + 'action' + '_epoch_' + str(10) + '.png')