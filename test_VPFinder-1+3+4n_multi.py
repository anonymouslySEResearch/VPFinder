import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, BertModel
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset
from torch.optim import AdamW
from tqdm import tqdm
from torchsampler import ImbalancedDatasetSampler
from sklearn.metrics import precision_score, recall_score, f1_score
best_f1_path = "model_best/VPFinder-1+3+4n_multi_f1.txt"
best_model_path = "model_best/VPFinder-1+3+4n_multi_model.pth"

# Load data
df = pd.read_csv("data/dataset_project_relation_layer.csv", low_memory=False)

# Split data into train and test sets
df_train, df_test = np.split(df, [int(.9 * len(df))])

# Load pre-trained tokenizers
tokenizer1 = AutoTokenizer.from_pretrained("./mycodebert")
tokenizer2 = BertTokenizer.from_pretrained("./mybert")
labels = {'positive': 1, 'negative': 0}

with open("data/cwe_1_8_classes.json", 'r') as f:
    content = f.read()
vul_label_info_first = json.loads(content)
for key, value in vul_label_info_first.items():
    vul_label_info_first[key] = np.array(value)

with open("data/cwe_2_4_classes.json", 'r') as f:
    content = f.read()
vul_label_info_second = json.loads(content)
for key, value in vul_label_info_second.items():
    vul_label_info_second[key] = np.array(value)

class MultiClassDataset(Dataset):
    def __init__(self, dataframe):
        self.texts = [tokenizer2(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt") for text in dataframe['text']]
        self.comments = [tokenizer2(str(comment), padding='max_length', max_length=512, truncation=True, return_tensors="pt") for comment in dataframe['comment']]
        self.messages = [tokenizer2(message, padding='max_length', max_length=512, truncation=True, return_tensors="pt") for message in dataframe['message']]
        self.patch_del = [tokenizer1.encode_plus(str(patch), padding='max_length', max_length=256, truncation=True, return_tensors="pt") for patch in dataframe['del_patch']]
        self.patch_add = [tokenizer1.encode_plus(str(patch), padding='max_length', max_length=256, truncation=True, return_tensors="pt") for patch in dataframe['add_patch']]
        self.codes = [tokenizer1.encode_plus(code, padding='max_length', max_length=512, truncation=True, return_tensors="pt") for code in dataframe['code']]
        self.labels = [int(cwe_id) - 1 for cwe_id in dataframe['CWE_ID']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_labels(self):
        return self.labels

    def get_batch_labels(self, idx):
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def get_batch_comments(self, idx):
        # Fetch a batch of comments
        return self.comments[idx]

    def get_batch_del(self, idx):
        return self.patch_del[idx]

    def get_batch_add(self, idx):
        return self.patch_add[idx]

    def get_batch_codes(self, idx):
        # Fetch a batch of codes
        return self.codes[idx]

    def get_batch_messages(self, idx):
        return self.messages[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_comments = self.get_batch_comments(idx)
        batch_codes = self.get_batch_codes(idx)
        batch_del = self.get_batch_del(idx)
        batch_add = self.get_batch_add(idx)
        batch_y = self.get_batch_labels(idx)
        batch_messages = self.get_batch_messages(idx)
        return batch_texts, batch_comments, batch_codes, batch_del, batch_add, batch_messages, batch_y

# Multi-class Classification Model
class MultiClassifier(nn.Module):
    def __init__(self, num_classes_first_level, num_classes_second_level, hidden_dim=64):
        super(MultiClassifier, self).__init__()
        self.num_classes_first_level = num_classes_first_level
        self.num_classes_second_level = num_classes_second_level
        self.bert1 = AutoModelForSequenceClassification.from_pretrained("mycodebert")
        self.bert2 = BertModel.from_pretrained("mybert")
        dropout_rate = 0.3
        self.patch_del_hidden = nn.Linear(768, 384)
        self.patch_add_hidden = nn.Linear(768, 384)
        self.hidden = nn.Sequential(
            nn.Linear(768 * 2, 768),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.label_info = nn.Linear(768, hidden_dim)
        self.label_info_second = nn.Linear(768, hidden_dim)
        self.att1 = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)
        self.att2 = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)
        self.att3 = nn.MultiheadAttention(embed_dim=768 * 2, num_heads=8, batch_first=True)

    def forward(self, input_id, comment_id, code_id, del_id, add_id, message_id, mask, comment_mask, code_mask, del_mask, add_mask, message_mask, vul_label_first_layer, vul_label_second_layer):
        text_output = self.bert2(input_ids=input_id, attention_mask=mask, return_dict=False)[0]
        comment_output = self.bert2(input_ids=comment_id, attention_mask=comment_mask, return_dict=False)[0]
        code_output = self.bert1.roberta(input_ids=code_id, attention_mask=code_mask, return_dict=False)[0]
        del_output = self.bert1.roberta(input_ids=del_id, attention_mask=del_mask, return_dict=False)[0]
        add_output = self.bert1.roberta(input_ids=add_id, attention_mask=add_mask, return_dict=False)[0]
        message_output = self.bert2(input_ids=message_id, attention_mask=message_mask, return_dict=False)[0]


        text_output = text_output[:, 1:, :]
        comment_output = comment_output[:, 1:, :]

        code_output = code_output[:, 0, :]
        code_output = code_output.unsqueeze(1)
        code_output = code_output.repeat(1, 511, 1)

        att_out1 = self.att1(code_output, text_output, text_output)
        att_out2 = self.att2(code_output, comment_output, comment_output)
        description_output = torch.cat((att_out1[0], att_out2[0]), dim=2)

        del_output = del_output[:, 0, :]
        del_output = self.patch_del_hidden(del_output)
        add_output = add_output[:, 0, :]
        add_output = self.patch_add_hidden(add_output)
        message_output = message_output[:, 0, :]
        solution_output = torch.cat((message_output, del_output, add_output), dim=1)
        solution_output = solution_output.unsqueeze(1)
        solution_output = solution_output.repeat(1, 511, 1)

        att_out3 = self.att3(solution_output, description_output, description_output)
        hidden_input = torch.mean(att_out3[0], dim=1)

        hidden_output = self.hidden(hidden_input)

        mapped_label_info_first = self.label_info(vul_label_first_layer)

        label_info_first = mapped_label_info_first.squeeze(dim=1)

        mapped_label_info_second = self.label_info_second(vul_label_second_layer)

        first_layer_output = torch.mm(hidden_output, label_info_first.t())

        label_info_second = mapped_label_info_second.squeeze(dim=1)

        second_layer_output = torch.mm(hidden_output, label_info_second.t())

        output = torch.cat((first_layer_output, second_layer_output), dim=1)

        return output


# Train Multi-Class Classification Model
def train_multi_model(model, val_data):
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
        print("Read best multi-class model successfully. Starting training...")
    except:
        print("Best multi-class model not found. Starting training from scratch...")

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
        for inputs, comments, codes, delete, add, messages, labels in tqdm(val_dataloader):
            input_ids = inputs['input_ids'].squeeze(1).to(device)
            comment_ids = comments['input_ids'].squeeze(1).to(device)
            code_ids = codes['input_ids'].squeeze(1).to(device)
            del_ids = delete['input_ids'].squeeze(1).to(device)
            add_ids = add['input_ids'].squeeze(1).to(device)
            message_ids = messages['input_ids'].squeeze(1).to(device)
            attention_mask = inputs['attention_mask'].squeeze(1).to(device)
            comment_mask = comments['attention_mask'].squeeze(1).to(device)
            code_mask = codes['attention_mask'].squeeze(1).to(device)
            del_mask = delete['attention_mask'].squeeze(1).to(device)
            add_mask = add['attention_mask'].squeeze(1).to(device)
            message_mask = messages['attention_mask'].squeeze(1).to(device)
            labels = labels.long().to(device)
            vul_label_first_layer = torch.tensor([vul_label_info_first[label] for label in vul_label_info_first], dtype=torch.float32)
            vul_label_first_layer = vul_label_first_layer.to(device)
            vul_label_second_layer = torch.tensor([vul_label_info_second[label] for label in vul_label_info_second], dtype=torch.float32)
            vul_label_second_layer = vul_label_second_layer.to(device)

            outputs = model(input_ids, comment_ids, code_ids, del_ids, add_ids, message_ids, attention_mask, comment_mask, code_mask, del_mask, add_mask, message_mask, vul_label_first_layer, vul_label_second_layer)
            preds = torch.argmax(outputs, dim=1)

            loss = criterion(outputs, labels)

            total_loss_val += loss.item()
            total_acc_val += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(
        f"Val Loss: {total_loss_val / len(val_dataset):.4f} | Val Accuracy: {total_acc_val / len(val_dataset):.4f}")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}")
    print()


# Train the Multi-Class Classification Model
NUM_CLASSES_MULTI = 8
multi_model = MultiClassifier(NUM_CLASSES_MULTI, 4)
filtered_df_test = df_test[df_test['flag'] == 'positive']
train_multi_model(multi_model, filtered_df_test)