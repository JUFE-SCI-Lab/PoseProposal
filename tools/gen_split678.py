import json
import copy
import numpy as np

if __name__ == '__main__':
    anchor_split_idx = 5
    with open(f'data/mp100/annotations/mp100_split{anchor_split_idx}_train.json', 'r') as f:
        train_json = json.load(f)
    with open(f'data/mp100/annotations/mp100_split{anchor_split_idx}_test.json', 'r') as f:
        test_json = json.load(f)

    traintest = {}
    for k in ['images', 'annotations', 'categories']:
        traintest[k] = train_json[k] + test_json[k]
    traintest['info'] = train_json['info']
    with open(f'data/mp100/annotations/mp100_split{anchor_split_idx}_traintest.json', 'w') as f:
        json.dump(traintest, f)

    ori_train_cids = [d['id'] for d in train_json['categories']]
    ori_test_cids = [d['id'] for d in test_json['categories']]

    perm_train_cids = list(np.random.permutation(ori_train_cids))

    for i in range(3):
        new_split_idx = anchor_split_idx + 1 + i
        del_num = 10 + i * 10

        base_cids = ori_train_cids[:-del_num]
        novel_cids = ori_test_cids + ori_train_cids[-del_num:]
        print(f'split{new_split_idx}: {len(base_cids)}:{len(novel_cids)}')
        print(base_cids)
        print(novel_cids)
        assert sorted(base_cids + novel_cids) == sorted(ori_train_cids + ori_test_cids)
        i
