import json 
from PIL import Image
from transformers import ViltConfig, ViltProcessor, ViltForQuestionAnswering
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import *
from commons import *
import math
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os


# Setting

ANALYSIS = 'ACTIONS'

assert ANALYSIS in ['ACTIONS', 'REASONS']

print("Training by " + ANALYSIS)

# Create Questions
js = open(DICT_JSON[ANALYSIS])
print("Loading " + DICT_JSON[ANALYSIS])
row_q = json.load(js)

questions = []

for i in row_q.keys():
    dic = {}
    dic['image_id'] = i
    if ANALYSIS == 'ACTIONS':
        dic['question'] = 'What should I do?'
    elif ANALYSIS == 'REASONS':
        dic['question'] = 'Why should I do?'
    questions.append(dic)

# Load Answers annotations
f = open(DICT_JSON[ANALYSIS])

annotation_name_list = []
data_annotations = json.load(f)
annotations = []
text = [] # Corpus set 
# Get the Corpus set 
for i in data_annotations.keys():
    for j in range(len(data_annotations[i])):
        # Each element is a sentense, with surrounding space removed, ending period removed
        # All in lower cases
        cur = data_annotations[i][j]
        cur = cur.strip()
        cur = cur.strip(string.punctuation)
        cur = cur.lower()
        text.append(cur)

tv = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None)
tv_fit = tv.fit_transform(text)

print("Start to calc tfidf score")
filename = './models/tfidf_' + ANALYSIS + '.txt'
if os.path.exists(filename):
    print('Load tfidf score from file')
    annotations = joblib.load(filename)
else:
    for i in tqdm(data_annotations.keys()):
        di = {}
        di['image_id'] = i
        count = 0
        answers = data_annotations[i]
        ansl = []
        # Find the sentence with max avg tfidf_score
        for j in range(len(answers)):
            cur = answers[j]
            cur = cur.strip()
            cur = cur.strip(string.punctuation)
            cur = cur.lower()
            countOfWords = len(cur.split())
            #invalid sentence with zero word, skip
            if(countOfWords <= 0):
                continue
            cur = [cur]
            cur_fit = tv.transform(cur)
            max_ind = cur_fit.toarray().argmax()
            dic_a = {}
            dic_a['answer'] =  tv.get_feature_names_out()[max_ind]
            #print(dic_a['answer'])
            dic_a['answer_id'] = count
            count = count + 1
            ansl.append(dic_a)
        di['answers'] = ansl
        annotations.append(di)
    joblib.dump(annotations, filename, compress=3)

count = 0
config.label2id.clear()
config.id2label.clear()
print("Start to create annotations")
for annotation in tqdm(annotations):
    answers = annotation['answers']
    answer_count = {}
    for answer in answers:
        answer_ = answer["answer"]
        answer_count[answer_] = answer_count.get(answer_, 0) + 1
    labels = []
    scores = []
    
    for answer in answer_count:
        if answer not in list(config.label2id.keys()):
            config.label2id[answer] = count 
            config.id2label[count] = answer
            count += 1
        labels.append(config.label2id[answer])
        score = get_score(answer_count[answer])
        scores.append(score)
    annotation['labels'] = labels
    annotation['scores'] = scores


# Create Dataset for VQA
dataset = VQADatasetForAd(questions=questions,
                     annotations=annotations,
                     processor=processor)

print("done")

# Instantiate the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-mlm",
                                                 num_labels=len(config.id2label),
                                                 id2label=config.id2label,
                                                 label2id=config.label2id)
model.to(device)


# dataset 
N_TRAIN_RATIO = 0.9
BATCH_SIZE = 10
n_train = int(len(dataset) * N_TRAIN_RATIO)
n_val   = int(len(dataset) - n_train)

print("=====================================")
print("n_val", n_val)

train, val = torch.utils.data.random_split(dataset, [n_train, n_val])
train_dataloader = DataLoader(train, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)


# Training
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

training_loss_list = []
training_acc_list = []
val_acc_list = []

NUM_EPOCH = 10

for epoch in range(NUM_EPOCH):  # loop over the dataset multiple times # 10 is best after that overfitting
    print("Start Training")
    
    model.train()
    print(f"Epoch: {epoch}")
    acc_per_epoch = 0
    accumulate_loss = 0

    for batch in tqdm(train_dataloader):
        # get the inputs; 
        batch = {k:v.to(device) for k,v in batch.items()}

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        accumulate_loss += loss.item()

        # calc accuracy
        model.eval()
        with torch.no_grad():
            pred = outputs.logits.argmax(dim=1)
            true_label_idx=batch['labels'].argmax(dim=1)
            acc_per_epoch += sum(pred == true_label_idx).item()
    
    # validation accuracy
    print("Training Loss: ", accumulate_loss)
    model.eval()
    num_data = len(val)
    acc_val = 0
    print("Start Validation")
    for batch in tqdm(val_dataloader):
        batch = {k:v.to(device) for k,v in batch.items()}
        # forward pass
        outputs = model(**batch)
        pred = outputs.logits.argmax(dim=1)
        true_label_idx=batch['labels'].argmax(dim=1)
        acc_val += sum(pred == true_label_idx).item()

    print("Training accuracy: {}".format(acc_per_epoch/n_train))
    print("Validation accuracy: {}".format(acc_val/n_val))
    training_loss_list.append(accumulate_loss)
    training_acc_list.append(acc_per_epoch/n_train)
    val_acc_list.append(acc_val/n_val)
    
    if epoch % 5 == 0 or epoch == NUM_EPOCH - 1:


        # save model
        model.save_pretrained('./models/' + ANALYSIS + '_epoch_' + str(NUM_EPOCH))

        # CSV
        with open("./output_data" + ANALYSIS +'_epoch_' + str(NUM_EPOCH) + ".txt", "a") as output:
            output.write(str(training_acc_list))
            output.write(str(val_acc_list))
            output.write(str(training_loss_list))

# draw result
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

x = np.linspace(0, NUM_EPOCH, NUM_EPOCH)

c1,c2,c3 = "blue","green","red"      # 各プロットの色
l1,l2,l3 = "Training acc","Validation accu","Training Loss"   # 各ラベル

ax1.plot(x, training_acc_list, color=c1, label=l1)
ax1.plot(x, val_acc_list, color=c2, label=l2)
ax2.plot(x, training_loss_list, color=c3, label=l3)

ax1.legend(loc = 'upper right') 
ax2.legend(loc = 'upper right') 

fig.tight_layout()
fig.savefig('./' + ANALYSIS + '_epoch_' + str(NUM_EPOCH) + '.png')