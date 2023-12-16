import json
import pandas as pd

def get_cwe_relationship():
    data = pd.read_csv("data/1000.csv")
    cwe = {}
    leave_number = 0
    for index, row in data.iterrows():
        if index in [284, 435, 664, 682, 691, 693, 697, 703, 707, 710]:
            cwe[index] = '1000'
            continue
        relation = str(row[5])
        if relation:
            leave_number += 1
            if index not in [682, 690, 692, 1299]:
                relation = relation[:58]
                relation = relation.replace('::NATURE:', '')
                relation = relation.replace(':VIEW ID:1000:ORDINAL:Primary::', '')
                relation = relation.replace('ChildOf:CWE ID:', '')
                relation = relation.replace(':VIEW ID:1000ChildOf:C', '')
                relation = relation.replace(':VIEW ID:1000:ORDINAL:Primary:', '')
                relation = relation.replace('StartsWith:CWE ID:', '')
                relation = relation.replace(':VIEW ID:709:CHAIN ID:680::', '')
                relation = relation.replace('N', '')
                relation = relation.replace('W', '')
                cwe[index] = relation
            elif index == 1299:
                relation = relation.replace('::NATURE:PeerOf:CWE ID:1191:VIEW ID:1194:ORDINAL:Primary::NATURE:ChildOf:CWE ID:', '')
                relation = relation.replace(':VIEW ID:1000:ORDINAL:Primary::NATURE:ChildOf:CWE ID:288:VIEW ID:1000::', '')
                cwe[index] = relation
            elif index == 682:
                continue
            elif index == 690:
                relation = relation.replace('::NATURE:StartsWith:CWE ID:252:VIEW ID:709:CHAIN ID:690::NATURE:ChildOf:CWE ID:', '')
                relation = relation.replace(':VIEW ID:1000:ORDINAL:Primary::', '')
                cwe[index] = relation
            elif index == 692:
                relation = relation.replace('::NATURE:StartsWith:CWE ID:184:VIEW ID:709:CHAIN ID:692::NATURE:ChildOf:CWE ID:', '')
                relation = relation.replace(':VIEW ID:1000:ORDINAL:Primary::', '')
                cwe[index] = relation

    # lack = ['125', '119', '118', '664', '1000']
    cwe['118'] = '664'
    cwe['119'] = '118'
    cwe['125'] = '119'
    with open("data/1000.json", 'w') as f:
        f.write(json.dumps(cwe, indent=4, ensure_ascii=False))

def build_tree():
    with open("data/1000.json", 'r') as f:
        content = f.read()
    cwe = json.loads(content)
    nodes = {}

    for key, value in cwe.items():
        nodes[key] = value

    traverse = []
    for i in range(1500):
        traverse.append([])
    for node in nodes.keys():
        traverse[int(nodes[node])].append(node)

    roots = [284, 435, 664, 682, 691, 693, 697, 703, 707, 710]
    tree6 = {}
    for root1 in roots:
        children1 = tree_parser(traverse, root1)
        tree5 = {}
        for root2 in children1:
            children2 = tree_parser(traverse, root2)
            tree4 = {}
            for root3 in children2:
                children3 = tree_parser(traverse, root3)
                tree3 = {}
                for root4 in children3:
                    children4 = tree_parser(traverse, root4)
                    tree2 = {}
                    for root5 in children4:
                        children5 = tree_parser(traverse, root5)
                        tree1 = {}
                        for root6 in children5:
                            children6 = tree_parser(traverse, root6)
                            tree1[root6] = children6
                        tree2[root5] = tree1
                    tree3[root4] = tree2
                tree4[root3] = tree3
            tree5[root2] = tree4
        tree6[root1] = tree5
    tree7 = {}
    tree7["1000"] = tree6
    with open("data/my_CWE_tree.json", 'w') as f:
        json.dump(tree7, f, indent=4, ensure_ascii=False)
    with open("data/go_CWE_tree.json", 'w') as f:
        json.dump(tree7, f, ensure_ascii=False)

def tree_parser(traverse, root):
    children = traverse[int(root)]
    return children

def extract_children_recursive(node):
    children_dict = {}

    for key, value in node.items():
        if isinstance(value, dict) and value:
            children_dict[key] = list(value.keys())
            children_dict.update(extract_children_recursive(value))
        elif isinstance(value, list) and value:
            children_dict[key] = []
            for item in value:
                if isinstance(item, dict):
                    children_dict[key].append(list(item.keys()))
                    children_dict.update(extract_children_recursive(item))
        else:
            children_dict[key] = []

    return children_dict

def reverse_dict(input):
    reversed_dict = {}
    reversed_keys = list(input.keys())[::-1]
    for key in reversed_keys:
        value = input[key]
        reversed_dict[key] = value
    with open("data/sorted_cwe.json", 'w') as f:
        json.dump(reversed_dict, f, indent=4, ensure_ascii=False)
    return reversed_dict

if __name__ == "__main__":
    get_cwe_relationship()
    build_tree()
    with open("data/my_CWE_tree.json", 'r') as f:
        content = f.read()
    cwe = json.loads(content)
    children_dict = extract_children_recursive(cwe)
    final = reverse_dict(children_dict)
    print(children_dict)
    print(final)