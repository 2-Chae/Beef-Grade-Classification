import torch
import torch.nn as nn
import random
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import csv
from sklearn.utils import shuffle
import sklearn
from sklearn.metrics import confusion_matrix, cohen_kappa_score



USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print("train with", device)

seed_number = 0
random.seed(seed_number)
torch.manual_seed(seed_number)

if device == 'cuda':
    torch.cuda.manual_seed_all(seed_number)

lr = 0.00006  # slow learner
print(lr)
epochs = 15 
batch_size = 100
test_batch_size = 1
drop_prob = 0.5
num_classes = 5


# load train data
with open('./data/train_data_final.pkl', 'rb') as f:
    train_data = pickle.load(f)

with open('./data/test_data_final.pkl', 'rb') as f:
    test_data = pickle.load(f)

selected_models = [0,1,2]
train_data["model_num"] = len(selected_models)
test_data["model_num"] = len(selected_models)

# model
embedding_dim1 = 128 
embedding_dim2 = 128
linear1 = nn.Linear(train_data["model_num"]*num_classes, embedding_dim1)
linear2 = nn.Linear(embedding_dim1, embedding_dim2)
linear3 = nn.Linear(embedding_dim2, num_classes)
dropout = nn.Dropout(p=drop_prob)
gelu = nn.GELU()
layer_norm1 = nn.LayerNorm(embedding_dim1)
layer_norm2 = nn.LayerNorm(embedding_dim2)

nn.init.xavier_uniform_(linear1.weight)
nn.init.xavier_uniform_(linear2.weight)
nn.init.xavier_uniform_(linear3.weight)

model = nn.Sequential(linear1, layer_norm1, gelu, dropout, linear2, layer_norm2, gelu, dropout, linear3).cuda()

# cost function and optimizer
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                      momentum=0.9, weight_decay=5e-4)

concatenated_features = []
cnt = 0
for i in selected_models:
    if cnt == 0:
        train = np.array(train_data["features"][i])
    else:
        temp = np.array(train_data["features"][i])
        train = np.concatenate((train, temp), axis=1)

    cnt += 1

train_label = np.array(train_data["label"])

train = torch.tensor(train, dtype=torch.float32).cuda()
train_label = torch.tensor(train_label).cuda()

cnt = 0
for i in selected_models:
    if cnt == 0:
        test = np.array(test_data["features"][i])
    else:
        temp = np.array(test_data["features"][i])
        test = np.concatenate((test, temp), axis=1)

    cnt += 1

test_label = np.array(test_data["label"])
test_label = torch.tensor(test_label).cuda()

test = torch.tensor(test, dtype=torch.float32).cuda()
test_batch_size = test.shape[0]

def evaluate():
    model.eval()
    gt = []
    pred = []
    for epoch in range(1):
        for i in range(0, test.shape[0], test_batch_size):
            input = test[i:i+test_batch_size]
            target = test_label[i:i+test_batch_size]
            
            output = model(input)
            _, predicted = output.max(1)

            gt.extend(target.cpu())
            pred.extend(predicted.cpu().tolist())

    
    score = cohen_kappa_score(pred, gt, weights='quadratic', labels=None, sample_weight=None)
    print("cohen kappa score: %f" % score)

    sample_weight = sklearn.utils.class_weight.compute_sample_weight(class_weight="balanced", y=gt, indices=None)
    score = cohen_kappa_score(pred, gt, weights='quadratic', labels=None, sample_weight=sample_weight)
    print("cohen kappa score with sample weight: %f" % score)



# start training
total_batch = int(train.shape[0] / batch_size)
print(total_batch)


for epoch in range(epochs):
    print('epoch:', epoch)
    loss_val = 0.
    cnt = 0.
    model.train()
    for i in range(0, train.shape[0], batch_size):
        input = train[i:i+batch_size]
        target = train_label[i:i+batch_size]

        output = model(input)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val += loss.item()
        cnt += 1

    evaluate()
