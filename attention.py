import torch.nn as nn
import torch
from torch.nn import functional as F
from utils import *
from layers2 import GAT1
# from layers3 import Method
import time
from sklearn import metrics
import warnings

warnings.filterwarnings('ignore')

# 定义超参数
early_stop = 50
BATCH_SIZE = 512
EPOCH = 200
Device = "cuda:0" if torch.cuda.is_available() else 'cpu'
hidden_dim = 96
dropout = 0.5  # 一般为0.5, 0.2, mr为0.2
leaky_alpha = 0.2
n_head = 8
learning_rate = 0.001  # 1e-2, 5e-2, 1e-3, 5e-3, 1e-4, 5e-4
weight_decay = 0  # 1e-2, 1e-3, #1e-4, 1e-5
dataset = 'sensitive'
train_adj, train_embed, train_y, val_adj, val_embed, val_y, test_adj, test_embed, test_y = load_data(dataset)


# 加载训练集，验证集，测试集
print("load training set")
train_adj, train_mask = preprocess_adj1(train_adj)
train_feature = preprocess_features(train_embed)
print("load validation set")
val_adj, val_mask = preprocess_adj1(val_adj)
val_feature = preprocess_features(val_embed)
print("load test set")
test_adj, test_mask = preprocess_adj1(test_adj)
test_feature = preprocess_features(test_embed)

# 构造torch的数据加载器
train_dataset = TensorDataset(train_feature, train_adj, train_y, train_mask)
train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=BATCH_SIZE)

val_dataset = TensorDataset(val_feature, val_adj, val_y, val_mask)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

test_dataset = TensorDataset(test_feature, test_adj, test_y, test_mask)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# 定义参数
feature_dim = train_feature.size()[2]
n_class = len(set(train_y.tolist()))
test_class = len(set(test_y.tolist()))
# model = Method(feature_dim, hidden_dim, hidden_dim, node_list[0], node_list[1], node_list[2],  n_class)
model = GAT1(feature_dim, hidden_dim, n_class, n_head, dropout, leaky_alpha)
# model = GAT4(feature_dim, hidden_dim, n_class, n_head,  dropout, leaky_alpha)
# model = GAT1(feature_dim, hidden_dim, n_class)
# model = GatDiffPooling(feature_dim, hidden_dim, n_class, dropout, leaky_alpha)
model.to(Device)
print(model)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def train_eval(cate, loader, model, optimizer, loss_func, Device):
    model.train() if cate=='train' else model.eval()
    run_loss = 0.0
    preds, labels = [], []
    for batch_idx, data in enumerate(loader, 0):
        if cate == "train":
            optimizer.zero_grad()
        features, adj, label, mask = data[0].float(), data[1].float(), data[2].long(), data[3].float()
        features, adj, label, mask = features.to(Device), adj.to(Device), label.to(Device), mask.to(Device)
        output = model(features, adj, mask)

        loss = loss_func(output, label)
        # loss = model.loss(output, label, mask, adj)
        _, predict = torch.max(output.data, dim=1)
        preds.append(predict)
        labels.append(label.data)

        if cate == "train":
            loss.backward()
            optimizer.step()
            # print("iters={:4d}".format(batch_idx),
            #       "training_loss={:.4f}".format(loss.data))
        run_loss += loss.data


    preds = torch.cat(preds).tolist()
    labels = torch.cat(labels).tolist()
    loss = run_loss / (batch_idx+1)
    acc = metrics.accuracy_score(labels, preds) * 100
    return loss, acc, preds, labels


print("Training start!")
best_acc = 0.
best_loss = 10000
patience = 0
for epoch in range(EPOCH):
    start_time = time.time()
    train_loss, train_acc, _, _ = train_eval("train", train_dataloader, model, optimizer, criterion, Device)
    end_time = time.time()
    val_loss, val_acc, _, _ = train_eval("eval", val_dataloader, model, optimizer, criterion, Device)
    patience += 1
    if val_acc >= best_acc:
        test_loss, test_acc, test_preds, test_labels = train_eval("test", test_dataloader, model, optimizer, criterion, Device)
        best_acc = val_acc
        # torch.save(model, './model/' + str(dataset) + '_best_model.pkl')
        best_epoch = epoch + 1
        best_test_loss = test_loss
        best_test_acc = test_acc
        best_test_preds = test_preds
        best_test_labels = test_labels
        patience = 0
        print("Epoch={:04d}".format(epoch+1),
              "Train_loss={:.4f}".format(train_loss),
              "Train_acc={:.4f}".format(train_acc),
              "Val_loss={:.4f}".format(val_loss),
              "Val_acc={:.4f}".format(val_acc),
              "Test_loss={:.4f}".format(test_loss),
              "Test_acc={:.4f}".format(test_acc),
              "time={:.4f}".format(end_time-start_time)
             )
    else:
        print("Epoch={:04d}".format(epoch + 1),
              "Train_loss={:.4f}".format(train_loss),
              "Train_acc={:.4f}".format(train_acc),
              "Val_loss={:.4f}".format(val_loss),
              "Val_acc={:.4f}".format(val_acc),
              "time={:.4f}".format(end_time - start_time))
    if patience == early_stop:
        break
print("Best Epoch={:04d}".format(best_epoch), "Test_loss={:.04f}".format(best_test_loss),
      "Test_acc={:.4f}".format(best_test_acc))
print("Test Precision, Recall and F1-Score...")
print(metrics.classification_report(best_test_labels, best_test_preds, digits=4))
print("Macro average Test Precision, Recall and F1-Score...")
print(metrics.precision_recall_fscore_support(best_test_labels, best_test_preds, average='macro'))
print("Micro average Test Precision, Recall and F1-Score...")
print(metrics.precision_recall_fscore_support(best_test_labels, best_test_preds, average='micro'))

