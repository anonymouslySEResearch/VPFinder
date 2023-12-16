import json
import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def joint_cwe_info():
    data = pd.read_csv("data/1000.csv", header=0, index_col=None)
    cwe = {}
    for index, row in data.iterrows():
        cwe[str(index)] = str(row[0]) + '. ' + str(row[3]) + ' ' + str(row[4]) + ' ' + str(row[13])

    with open("data/joint_cwe.json", 'w', encoding='utf-8') as f:
        json.dump(cwe, f, indent=4, ensure_ascii=False)

# 对每个CWE信息进行embedding
def emb_cwe_info():
    with open("data/joint_cwe.json", 'r', encoding='utf-8') as f:
        content = f.read()
    cwe = json.loads(content)
    embedded_dict = {}

    for key, value in cwe.items():
        inputs = tokenizer(value, return_tensors='pt', padding='max_length', truncation=True, max_length=512)

        with torch.no_grad():
            outputs = model(**inputs)

        embedding = outputs.last_hidden_state[:, 0, :].numpy()

        embedded_dict[key] = embedding.tolist()

    with open("data/embedded_cwe.json", 'w', encoding='utf-8') as f:
        json.dump(embedded_dict, f, indent=4, ensure_ascii=False)

# 从下而上更新向量
def emb_from_bottom():
    with open("data/embedded_cwe.json", 'r') as f:
        content = f.read()
    cwe = json.loads(content)
    for key, value in cwe.items():
        cwe[key] = np.array(value)
    with open("data/sorted_cwe.json", 'r') as f:
        content = f.read()
    relation = json.loads(content)
    cwe_new = {}
    for parent_key, child_keys in relation.items():
        if len(child_keys) == 0:
            cwe_new[parent_key] = cwe[parent_key]
        elif parent_key == '1000':
            cwe_new[parent_key] = np.mean([cwe[child_key] for child_key in child_keys], axis=0)
        else:
            parent_vector = cwe[parent_key]

            updated_parent_vector = parent_vector + np.mean([cwe[child_key] for child_key in child_keys], axis=0)
            cwe_new[parent_key] = updated_parent_vector

    cwe_list = {}
    for key, value in cwe_new.items():
        cwe_list[key] = value.tolist()

    cwe_list["0"] = [[-i for i in emb] for emb in cwe_list["1000"]]

    with open("data/embedded_from_bottom_cwe.json", 'w') as f:
        json.dump(cwe_list, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    joint_cwe_info()
    emb_cwe_info()
    emb_from_bottom()