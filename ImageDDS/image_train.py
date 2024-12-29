import random
import torch.nn.functional as F
import torch.nn as nn
from image_model import ImageDDS
from utils_test import *
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, recall_score, \
    balanced_accuracy_score
from sklearn import metrics
from creat_data_img import creat_data
import pandas as pd
import datetime
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau

result_name = 'Image'

modeling = ImageDDS


parser = argparse.ArgumentParser(description='PyTorch Implementation of DDs')
# image fusing graph
parser.add_argument('--use_image_fusion', action='store_true', default=False, help='是否使用图像进行融合')
parser.add_argument('--lambda_fusion_graph', type=float, default=0, help='graph 融合的比例')
parser.add_argument('--lambda_fusion_image', type=float, default=0, help='image 融合的比例')
# contrastive learning
parser.add_argument('--use_cl', action='store_true', default=False, help='是否使用对比学习')
parser.add_argument('--lambda_cl', type=float, default=0.1, help='')
parser.add_argument('--temperature', type=float, default=0.1, help='')
parser.add_argument('--base_temperature', type=float, default=0.075, help='')
parser.add_argument('--TRAIN_BATCH_SIZE', type=int, default=128, help='')
parser.add_argument('--TEST_BATCH_SIZE', type=int, default=128, help='')
# log dir
# parser.add_argument('--log_dir', default='./experiments/', help='path to log')
args = parser.parse_args()
print('args:',args)


# if not os.path.exists(args.log_dir):
#     os.makedirs(args.log_dir)

# training function at each epoch
def train(model, device, drug1_loader_train, drug2_loader_train, optimizer, epoch, use_image_fusion=False, use_cl=False, lambda_cl = 0.1):
    print("===============")
    print('Training on {} samples...'.format(len(drug1_loader_train.dataset)))
    model.train()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()

    zipped = zip(drug1_loader_train, drug2_loader_train)
    enumerate_data = enumerate(zipped)
    for batch_idx, data in enumerate_data:  
        data1 = data[0]
        data2 = data[1]
        data1.image = data1.image.reshape(-1, 3, 224, 224)
        data2.image = data2.image.reshape(-1, 3, 224, 224)
        data1 = data1.to(device)
        data2 = data2.to(device)
        y = data[0].y.view(-1, 1).float().to(device)
        y = y.squeeze(1)
        optimizer.zero_grad()
        output, data_dict = model(data1, data2)
        loss = loss_fn(output, y)
        total_loss = loss
        cl_loss = torch.zeros(1).to(device)
        if use_cl:
            cl_loss = model.cal_cl_loss(data_dict)
            total_loss += cl_loss * lambda_cl

        # print('loss', loss)
        total_loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data1.x),
                                                                           len(drug1_loader_train.dataset),
                                                                           100. * batch_idx / len(drug1_loader_train),
                                                                           loss.item()))
        ys = output.to('cpu').data.numpy()
        predicted_labels = list(map(lambda x: int(x>0.5), ys))
        predicted_scores = list(map(lambda x: x, ys))
        total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
        total_prelabels = torch.cat((total_prelabels, torch.Tensor(predicted_labels)), 0)
        total_labels = torch.cat((total_labels, data1.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_prelabels.numpy().flatten(),total_loss


def predicting(model, device, drug1_loader_test, drug2_loader_test, use_image_fusion=False):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(drug1_loader_test.dataset)))
    with torch.no_grad():
        for data in zip(drug1_loader_test, drug2_loader_test):
            data1 = data[0]
            data2 = data[1]
            data1.image = data1.image.reshape(-1, 3, 224, 224)
            data2.image = data2.image.reshape(-1, 3, 224, 224)
            data1 = data1.to(device)
            data2 = data2.to(device)
            output, data_dict = model(data1, data2)
            ys = output.to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: int(x>0.5), ys))
            predicted_scores = list(map(lambda x: x, ys))
            total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
            total_prelabels = torch.cat((total_prelabels, torch.Tensor(predicted_labels)), 0)
            total_labels = torch.cat((total_labels, data1.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_prelabels.numpy().flatten()


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(len(dataset) * ratio)
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


# CPU or GPU

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    print('The code uses CPU!!!')



TRAIN_BATCH_SIZE = args.TRAIN_BATCH_SIZE
TEST_BATCH_SIZE = args.TEST_BATCH_SIZE
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 100

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

cellfile = 'data/cell_features_954.csv'
drug_smiles_file = 'data/smiles.csv'  
datafile = 'data/new_labels_0_10.csv' 
dataset = 'new_labels_0_10'


print('开始处理源文件....')
drug1, drug2, cell, label, smile_graph, cell_features, smile_imageidx = creat_data(dataset, cellfile)
print('从源文件提取特征成功！')

print('载入数据...')
drug1_data = TestbedDataset(root='data', dataset=dataset + '_drug1', img_root=f"data/processed/{dataset}/images", xd=drug1, xt=cell, y=label, smile_graph=smile_graph,
                            xt_featrue=cell_features, smile_imageidx=smile_imageidx)
drug2_data = TestbedDataset(root='data', dataset=dataset + '_drug2', img_root=f"data/processed/{dataset}/images", xd=drug2, xt=cell, y=label, smile_graph=smile_graph,
                            xt_featrue=cell_features, smile_imageidx=smile_imageidx)


print('载入数据完成！')

lenth = len(drug1_data)
pot = int(lenth / 5)
print('lenth', lenth)
print('pot', pot)

random_num = random.sample(range(0, lenth), lenth)  

folder_path = './result/' + result_name
if not os.path.exists(folder_path):
    os.makedirs(folder_path)


for i in range(5):

    test_num = random_num[pot * i:pot * (i + 1)]  
    train_num = random_num[:pot * i] + random_num[pot * (i + 1):] 

    drug1_data_train = drug1_data[train_num]  
    drug1_data_test = drug1_data[test_num]  
    drug1_loader_train = DataLoader(drug1_data_train, batch_size=TRAIN_BATCH_SIZE,
                                    shuffle=None) 
    drug1_loader_test = DataLoader(drug1_data_test, batch_size=TRAIN_BATCH_SIZE,
                                   shuffle=None)  


    drug2_data_test = drug2_data[test_num]
    drug2_data_train = drug2_data[train_num]
    drug2_loader_train = DataLoader(drug2_data_train, batch_size=TRAIN_BATCH_SIZE, shuffle=None)
    drug2_loader_test = DataLoader(drug2_data_test, batch_size=TRAIN_BATCH_SIZE, shuffle=None)

    
    model = modeling(use_image_fusion=args.use_image_fusion, use_cl=args.use_cl, temperature=args.temperature,
                     base_temperature=args.base_temperature, lambda_fusion_graph=args.lambda_fusion_graph,
                     lambda_fusion_image=args.lambda_fusion_image, batch_size=TRAIN_BATCH_SIZE, device=device).to(device)
    # torch.save(model.state_dict(), 'save_model/'+ result_name + '_'+ str(i)+'_epoch=0.pt')
    #loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.955, patience=7, verbose=True)
    now = datetime.datetime.now()
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")

    file_AUCs = folder_path + '/'+ result_name + '_' + str(i) + '--AUCs--' + dataset + '_'  + '.txt'
    AUCs = ('Epoch\tAUC_dev\tPR_AUC\tACC\tBACC\tPREC\tTPR\tKAPPA\tRECALL')
    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')

    best_auc = 0
    for epoch in range(NUM_EPOCHS):
        train_T, train_S, train_Y ,total_loss= train(model, device, drug1_loader_train, drug2_loader_train, optimizer, epoch + 1,
                                          use_image_fusion=args.use_image_fusion, use_cl=args.use_cl, lambda_cl=args.lambda_cl)
        T, S, Y = predicting(model, device, drug1_loader_test, drug2_loader_test, use_image_fusion=args.use_image_fusion)
        # T is correct label
        # S is predict score
        # Y is predict label

        # compute preformence
        AUC = roc_auc_score(T, S)
        precision, recall, threshold = metrics.precision_recall_curve(T, S)
        PR_AUC = metrics.auc(recall, precision)
        BACC = balanced_accuracy_score(T, Y)
        tn, fp, fn, tp = confusion_matrix(T, Y).ravel()
        TPR = tp / (tp + fn)
        PREC = precision_score(T, Y)
        ACC = accuracy_score(T, Y)
        KAPPA = cohen_kappa_score(T, Y)
        recall = recall_score(T, Y)

        train_AUC = roc_auc_score(train_T, train_S)
        train_precision, train_recall, train_threshold = metrics.precision_recall_curve(T, S)
        train_PR_AUC = metrics.auc(train_recall, train_precision)
        train_ACC = accuracy_score(train_T, train_Y)
        
        print("Train: AUC={}, PR_AUC={}, ACC={}".format(train_AUC, train_PR_AUC, train_ACC))
        print("Test: AUC={}, PR_AUC={}, ACC={}".format(AUC, PR_AUC, ACC))

        scheduler.step(AUC)

        # save data

        if best_auc < AUC:
            best_auc = AUC

            AUCs = [epoch, AUC, PR_AUC, ACC, BACC, PREC, TPR, KAPPA, recall]
            save_AUCs(AUCs, file_AUCs)
        
        print('best_auc', best_auc)  
    save_AUCs("best_auc:" + str(best_auc), file_AUCs)
    # torch.save(model.state_dict(), 'save_model/'+ result_name + '_'+ str(i)+'_epoch=400.pt')
