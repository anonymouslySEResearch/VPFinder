import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, BertModel
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
best_f1_path = "model_best/VPFinder-S_binary_f1.txt"
best_model_path = "model_best/VPFinder-S_binary_model.pth"

# Load data
df = pd.read_csv("data/dataset_project_relation_layer.csv", low_memory=False)

# Split data into train and test sets
df_train, df_test = np.split(df, [int(.9 * len(df))])

# Load pre-trained tokenizers
tokenizer1 = AutoTokenizer.from_pretrained("./mycodebert")
tokenizer2 = BertTokenizer.from_pretrained("./mybert")
labels = {'positive': 1, 'negative': 0}

class MyDataset(Dataset):
    def __init__(self, dataframe=df_train):
        self.messages = [tokenizer2(message, padding='max_length', max_length=512, truncation=True, return_tensors="pt")
                         for message in dataframe['message']]
        self.patch_del = [tokenizer1.encode_plus(str(patch), padding='max_length', max_length=256, truncation=True,
                                                 return_tensors="pt") for patch in dataframe['del_patch']]
        self.patch_add = [tokenizer1.encode_plus(str(patch), padding='max_length', max_length=256, truncation=True,
                                                 return_tensors="pt") for patch in dataframe['add_patch']]
        self.labels = [labels[label] for label in dataframe['flag']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_labels(self):
        return self.labels

    def get_batch_labels(self, idx):
        print(self.labels[idx])
        return np.array(self.labels[idx])

    def get_batch_del(self, idx):
        return self.patch_del[idx]

    def get_batch_add(self, idx):
        return self.patch_add[idx]

    def get_batch_messages(self, idx):
        return self.messages[idx]

    def __getitem__(self, idx):
        batch_del = self.get_batch_del(idx)
        batch_add = self.get_batch_add(idx)
        batch_y = self.get_batch_labels(idx)
        batch_messages = self.get_batch_messages(idx)
        return batch_del, batch_add, batch_messages, batch_y

# Model
class VulClassifier(nn.Module):
    def __init__(self):
        super(VulClassifier, self).__init__()
        self.bert1 = AutoModelForSequenceClassification.from_pretrained("mycodebert")
        self.bert2 = BertModel.from_pretrained("mybert")
        self.dropout = 0.3
        self.hidden = nn.Sequential(
            nn.Linear(768 * 3, 1024),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(8, 2),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )


    def forward(self, del_id, add_id, message_id, del_mask, add_mask, message_mask):
        pooled_output1 = self.bert1.roberta(input_ids=del_id, attention_mask=del_mask, return_dict=False)[0]
        output_del = pooled_output1[:, 0, :]
        pooled_output2 = self.bert1.roberta(input_ids=add_id, attention_mask=add_mask, return_dict=False)[0]
        output_add = pooled_output2[:, 0, :]
        pooled_output3 = self.bert2(input_ids=message_id, attention_mask=message_mask, return_dict=False)[0]
        output_message = pooled_output3[:, 0, :]
        output1 = torch.cat((output_del, output_add, output_message), dim=1)
        # need new compare method
        output = self.hidden(output1)
        return output

# Train
def train(model, val_data):
    val_dataset = MyDataset(dataframe=val_data)

    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=20)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Now using {device}")

    best_f1_score = 0.0
    print("Now loading the best model parameters...")
    try:
        with open(best_f1_path, 'r') as f:
            best_f1_score = float(f.read())
            f.close()
        model.load_state_dict(torch.load(best_model_path))
        print("Read best model successfully. Starting training...")
    except:
        print("Best model not found. Starting training from scratch...")

    model.to(device)
    criterion = nn.CrossEntropyLoss()

    if use_cuda:
        print("with cuda...")
        model = model.cuda()
        criterion = criterion.cuda()

    model.eval()
    all_preds = []
    all_labels = []
    total_loss_val = 0
    total_acc_val = 0

    with torch.no_grad():
        for delete, add, messages, labels in tqdm(val_dataloader):
            del_ids = delete['input_ids'].squeeze(1).to(device)
            add_ids = add['input_ids'].squeeze(1).to(device)
            message_ids = messages['input_ids'].squeeze(1).to(device)
            del_mask = delete['attention_mask'].squeeze(1).to(device)
            add_mask = add['attention_mask'].squeeze(1).to(device)
            message_mask = messages['attention_mask'].squeeze(1).to(device)
            labels = labels.long().to(device)

            outputs = model(del_ids, add_ids, message_ids, del_mask, add_mask, message_mask)
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

    with open("out/ablation.txt", 'a') as f:
        f.write("IRPredicter-S:\n")
        f.write(
            f"Val Loss: {total_loss_val / len(val_dataset):.4f} | Val Accuracy: {total_acc_val / len(val_dataset):.4f}\n")
        f.write(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}\n\n")


model = VulClassifier()
train(model, df_test)