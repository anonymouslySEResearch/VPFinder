import json
import numpy as np

def emb_class():
    with open("data/embedded_from_bottom_cwe.json") as f:
        content = f.read()
    cwe_emb = json.loads(content)
    for key, value in cwe_emb.items():
        cwe_emb[key] = np.array(value)

    cwe1 = cwe_emb['664']
    cwe2 = cwe_emb['707']
    cwe3 = cwe_emb['710']
    cwe4 = cwe_emb['682']
    cwe5 = cwe_emb['691']
    cwe6_emb = {}
    for i in ['693', '435', '697', '703', '417']:
        cwe6_emb[i] = cwe_emb[i]
    cwe6 = np.mean([cwe6_emb[key] for key in cwe6_emb.keys()], axis=0)
    cwe7 = cwe_emb['284']
    cwe8 = cwe_emb['1000']

    cwe_8_classes = {}
    cwe_8_classes['0'] = cwe1
    cwe_8_classes['1'] = cwe2
    cwe_8_classes['2'] = cwe3
    cwe_8_classes['3'] = cwe4
    cwe_8_classes['4'] = cwe5
    cwe_8_classes['5'] = cwe6
    cwe_8_classes['6'] = cwe7
    cwe_8_classes['7'] = cwe8
    for key, value in cwe_8_classes.items():
        cwe_8_classes[key] = value.tolist()

    with open("data/cwe_8_classes.json", 'w') as f:
        json.dump(cwe_8_classes, f, indent=4, ensure_ascii=False)

def emb_class_layer():
    # 读取向量
    with open("data/embedded_from_bottom_cwe.json") as f:
        content = f.read()
    cwe_emb = json.loads(content)
    for key, value in cwe_emb.items():
        cwe_emb[key] = np.array(value)

    # 第一层
    cwe1 = cwe_emb['664']
    cwe2 = cwe_emb['707']
    cwe3 = cwe_emb['710']
    cwe4 = cwe_emb['682']
    cwe5 = cwe_emb['691']
    cwe6_emb = {}
    for i in ['693', '435', '697', '703']:
        cwe6_emb[i] = cwe_emb[i]
    cwe6 = np.mean([cwe6_emb[key] for key in cwe6_emb.keys()], axis=0)
    cwe7 = cwe_emb['284']
    cwe8 = cwe_emb['1000']

    cwe_1_8_classes = {}
    cwe_1_8_classes['0'] = cwe1
    cwe_1_8_classes['1'] = cwe2
    cwe_1_8_classes['2'] = cwe3
    cwe_1_8_classes['3'] = cwe4
    cwe_1_8_classes['4'] = cwe5
    cwe_1_8_classes['5'] = cwe6
    cwe_1_8_classes['6'] = cwe7
    cwe_1_8_classes['7'] = cwe8
    for key, value in cwe_1_8_classes.items():
        cwe_1_8_classes[key] = value.tolist()

    with open("data/cwe_1_8_classes.json", 'w') as f:
        json.dump(cwe_1_8_classes, f, indent=4, ensure_ascii=False)

    # 第二层 CWE-664
    cwe9 = cwe_emb['118']
    cwe10 = cwe_emb['404']
    cwe11 = cwe_emb['668']
    cwe12_emb = {}
    for i in ['913', '706', '704', '669', '666', '665']:
        cwe12_emb[i] = cwe_emb[i]
    cwe12 = np.mean([cwe12_emb[key] for key in cwe12_emb.keys()], axis=0)

    cwe_2_4_classes = {}
    cwe_2_4_classes['0'] = cwe9
    cwe_2_4_classes['1'] = cwe10
    cwe_2_4_classes['2'] = cwe11
    cwe_2_4_classes['3'] = cwe12
    for key, value in cwe_2_4_classes.items():
        cwe_2_4_classes[key] = value.tolist()

    with open("data/cwe_2_4_classes.json", 'w') as f:
        json.dump(cwe_2_4_classes, f, indent=4, ensure_ascii=False)

def old_emb():
    with open("data/embedded_cwe.json", 'r') as f:
        content = f.read()
    cwe_emb = json.loads(content)
    for key, value in cwe_emb.items():
        cwe_emb[key] = np.array(value)

    # 第一层
    cwe1 = cwe_emb['664']
    cwe2 = cwe_emb['707']
    cwe3 = cwe_emb['710']
    cwe4 = cwe_emb['682']
    cwe5 = cwe_emb['691']
    cwe6_emb = {}
    for i in ['693', '435', '697', '703']:
        cwe6_emb[i] = cwe_emb[i]
    cwe6 = np.mean([cwe6_emb[key] for key in cwe6_emb.keys()], axis=0)
    cwe7 = cwe_emb['284']


    with open("data/embedded_from_bottom_cwe.json") as f:
        content = f.read()
    cwe_new = json.loads(content)
    for key, value in cwe_new.items():
        cwe_new[key] = np.array(value)
    cwe8 = cwe_new['1000']

    cwe_1_8_classes = {}
    cwe_1_8_classes['0'] = cwe1
    cwe_1_8_classes['1'] = cwe2
    cwe_1_8_classes['2'] = cwe3
    cwe_1_8_classes['3'] = cwe4
    cwe_1_8_classes['4'] = cwe5
    cwe_1_8_classes['5'] = cwe6
    cwe_1_8_classes['6'] = cwe7
    cwe_1_8_classes['7'] = cwe8
    for key, value in cwe_1_8_classes.items():
        cwe_1_8_classes[key] = value.tolist()

    with open("data/cwe_1_8_classes_old.json", 'w') as f:
        json.dump(cwe_1_8_classes, f, indent=4, ensure_ascii=False)

    # 第二层 CWE-664
    cwe9 = cwe_emb['118']
    cwe10 = cwe_emb['404']
    cwe11 = cwe_emb['668']
    cwe12_emb = {}
    for i in ['913', '706', '704', '669', '666', '665']:
        cwe12_emb[i] = cwe_emb[i]
    cwe12 = np.mean([cwe12_emb[key] for key in cwe12_emb.keys()], axis=0)

    cwe_2_4_classes = {}
    cwe_2_4_classes['0'] = cwe9
    cwe_2_4_classes['1'] = cwe10
    cwe_2_4_classes['2'] = cwe11
    cwe_2_4_classes['3'] = cwe12
    for key, value in cwe_2_4_classes.items():
        cwe_2_4_classes[key] = value.tolist()

    with open("data/cwe_2_4_classes_old.json", 'w') as f:
        json.dump(cwe_2_4_classes, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    emb_class()
    emb_class_layer()
    old_emb()