import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, BertModel
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
best_f1_path = "model_best/VPFinder-ID_binary_f1.txt"
best_model_path = "model_best/VPFinder-ID_binary_model.pth"

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
        self.texts = [tokenizer2(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt") for
                      text in dataframe['text']]
        self.comments = [
            tokenizer2(str(comment), padding='max_length', max_length=512, truncation=True, return_tensors="pt") for
            comment in dataframe['comment']]
        self.codes = [
            tokenizer1.encode_plus(code, padding='max_length', max_length=512, truncation=True, return_tensors="pt") for
            code in dataframe['code']]
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

    def get_batch_texts(self, idx):
        # Fetch a batch of texts
        return self.texts[idx]

    def get_batch_comments(self, idx):
        # Fetch a batch of comments
        return self.comments[idx]

    def get_batch_codes(self, idx):
        # Fetch a batch of codes
        return self.codes[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_comments = self.get_batch_comments(idx)
        batch_codes = self.get_batch_codes(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_comments, batch_codes, batch_y

# Model
class VulClassifier(nn.Module):
    def __init__(self):
        super(VulClassifier, self).__init__()
        self.bert1 = AutoModelForSequenceClassification.from_pretrained("mycodebert")
        self.bert2 = BertModel.from_pretrained("mybert")
        self.dropout = 0.3
        self.hidden = nn.Sequential(
            nn.Linear(768 * 3, 768),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(768, 256),
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

    def forward(self, text_id, comment_id, code_id, text_mask, comment_mask, code_mask):
        text_output = self.bert2(input_ids=text_id, attention_mask=text_mask, return_dict=False)[0]
        comment_output = self.bert2(input_ids=comment_id, attention_mask=comment_mask, return_dict=False)[0]
        code_output = self.bert1.roberta(input_ids=code_id, attention_mask=code_mask, return_dict=False)[0]
        text_input = text_output[:, 0, :]
        comment_input = comment_output[:, 0, :]
        code_input = code_output[:, 0, :]
        input = torch.cat((text_input, comment_input, code_input), dim=1)
        # need new compare method
        output = self.hidden(input)
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
        for texts, comments, codes, labels in tqdm(val_dataloader):
            text_ids = texts['input_ids'].squeeze(1).to(device)
            comment_ids = comments['input_ids'].squeeze(1).to(device)
            code_ids = codes['input_ids'].squeeze(1).to(device)
            text_mask = texts['attention_mask'].squeeze(1).to(device)
            comment_mask = comments['attention_mask'].squeeze(1).to(device)
            code_mask = codes['attention_mask'].squeeze(1).to(device)
            labels = labels.long().to(device)

            outputs = model(text_ids, comment_ids, code_ids, text_mask, comment_mask, code_mask)
            preds = torch.argmax(outputs, dim=1)

            loss = criterion(outputs, labels)

            total_loss_val += loss.item()
            total_acc_val += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    for pred, label in zip(all_preds, all_labels):
        print(pred, label)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')

    print(
        f"Val Loss: {total_loss_val / len(val_dataset):.4f} | Val Accuracy: {total_acc_val / len(val_dataset):.4f}")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}")
    print()

    with open("out/ablation.txt", 'a') as f:
        f.write("IRPredicter-ID:\n")
        f.write(
            f"Val Loss: {total_loss_val / len(val_dataset):.4f} | Val Accuracy: {total_acc_val / len(val_dataset):.4f}\n")
        f.write(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}\n\n")


model = VulClassifier()
train(model, df_test)