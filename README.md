Note that the environment and code is implemented in Ubuntu 20.04, and there are some dependency on Linux OS. So if you use this repo, use the same OS environment.

# Environment / Data download
1. Run the command
```
git clone 
cd /path/to/repo
conda env create -f environment.yml
```

2. Data download
```
sudo sh download.sh
```

3. Move .txt file to data dir
```
mv Sentiments_List2.txt ./AdDataset/annotations_images/image
mv Topics_List2.txt ./AdDataset/annotations_images/image
```

# Usage
1. Use the following commands
```
conda activate bert01
```

2. If you want to train 'Sentiments' or 'Topics', use `training_sentiments_topics.py`. If you want to train 'Reasons' or 'Actions', use `training_reasons_actions.py`.

```
python training_sentiments_topics.py
```

 - Change the variables of `ANALYSIS` to change the object of your analysis between sentiments and topics, and between reasons and actions in each file, respectively.

