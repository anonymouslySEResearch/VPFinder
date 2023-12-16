import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, BertModel
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
best_f1_path = "model_best/SVDAP_multi_f1.txt"
best_model_path = "model_best/SVDAP_multi_model.pth"
# 对按照CWE顶层关系分配样本后的数据集

# Load data
df = pd.read_csv("data/dataset_project_relation_layer.csv", low_memory=False)

# Split data into train and test sets
df_train, df_test = np.split(df, [int(.9 * len(df))])

# Load pre-trained tokenizers
tokenizer1 = AutoTokenizer.from_pretrained("./mycodebert")
tokenizer2 = BertTokenizer.from_pretrained("./mybert")
labels = {'positive': 1, 'negative': 0}

class MultiClassDataset(Dataset):
    def __init__(self, dataframe=df_train):
        self.message = [tokenizer2(message, padding='max_length', max_length=256, truncation=True, return_tensors="pt") for message in dataframe['message']]
        self.add_patch = [tokenizer1.encode_plus(str(patch), padding='max_length', max_length=256, truncation=True, return_tensors="pt") for patch in dataframe['add_patch']]
        self.del_patch = [tokenizer1.encode_plus(str(patch), padding='max_length', max_length=256, truncation=True, return_tensors="pt") for patch in dataframe['del_patch']]
        self.labels = [int(cwe_id) - 1 for cwe_id in dataframe['CWE_ID']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_labels(self):
        return self.labels

    def get_batch_labels(self, idx):
        print(self.labels[idx])
        return np.array(self.labels[idx])

    def get_batch_message(self, idx):
        return self.message[idx]

    def get_batch_add(self, idx):
        return self.add_patch[idx]

    def get_batch_del(self, idx):
        return self.del_patch[idx]

    def __getitem__(self, idx):
        batch_message = self.get_batch_message(idx)
        batch_add = self.get_batch_add(idx)
        batch_del = self.get_batch_del(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_message, batch_add, batch_del, batch_y

# Model
class MultiClassifier(nn.Module):
    def __init__(self):
        super(MultiClassifier, self).__init__()
        self.bert1 = AutoModelForSequenceClassification.from_pretrained("mycodebert")
        self.bert2 = BertModel.from_pretrained("mybert")
        self.dropout = 0.3
        self.hidden = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(16, 12),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        self.att = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)

    def forward(self, message_id, add_id, del_id, message_mask, add_mask, del_mask):
        message = self.bert2(input_ids=message_id, attention_mask=message_mask, return_dict=False)[0]   # [20, 256, 768]
        add_patch = self.bert1.roberta(input_ids=add_id, attention_mask=add_mask, return_dict=False)[0]   # [20, 256, 768]
        del_patch = self.bert1.roberta(input_ids=del_id, attention_mask=del_mask, return_dict=False)[0]   # [20, 256, 768]

        patch = torch.cat((add_patch, del_patch), dim=1)   # [20, 512, 768]

        message = message.repeat(1, 2, 1)
        att_output = self.att(message, patch, patch)   # [20, 512, 768]

        res_output = message + att_output[0]   # [20, 512, 768]

        avg = torch.mean(res_output, dim=1)

        output = self.hidden(avg)

        return output

# Train Multi-Class Classification Model
def get_model_output(model, val_data):
    val_dataset = MultiClassDataset(dataframe=val_data)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=20)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    best_f1_score = 0.0
    print("Now loading the best multi-class model parameters...")
    try:
        with open(best_f1_path, 'r') as f:
            best_f1_score = float(f.read())
            f.close()
        model.load_state_dict(torch.load(best_model_path))
        print("Read best model successfully. Starting training...")
    except:
        print("Best model not found!")

    model.to(device)
    criterion = nn.CrossEntropyLoss()

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    model.eval()
    all_preds = []
    all_labels = []
    total_loss_val = 0
    total_acc_val = 0

    with torch.no_grad():
        for messages, adds, dels, labels in tqdm(val_dataloader):
            meesage_ids = messages['input_ids'].squeeze(1).to(device)
            add_ids = adds['input_ids'].squeeze(1).to(device)
            del_ids = dels['input_ids'].squeeze(1).to(device)
            message_mask = messages['attention_mask'].squeeze(1).to(device)
            add_mask = adds['attention_mask'].squeeze(1).to(device)
            del_mask = dels['attention_mask'].squeeze(1).to(device)
            labels = labels.long().to(device)

            outputs = model(meesage_ids, add_ids, del_ids, message_mask, add_mask, del_mask)
            preds = torch.argmax(outputs, dim=1)

            loss = criterion(outputs, labels)

            total_loss_val += loss.item()
            total_acc_val += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    for pred, label in zip(all_preds, all_labels):
        print(pred, label)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(
        f"Val Loss: {total_loss_val / len(val_dataset):.4f} | Val Accuracy: {total_acc_val / len(val_dataset):.4f}")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}")
    print()


# Train the Multi-Class Classification Model
model = MultiClassifier()
filtered_df_test = df_test[df_test['flag'] == 'positive']
get_model_output(model, df_test)