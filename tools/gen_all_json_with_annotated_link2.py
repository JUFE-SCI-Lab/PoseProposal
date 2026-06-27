import json
import numpy as np

# split-1
test_cids = [10, 77, 24, 47, 33, 29, 70, 68, 81, 73, 2, 14, 30, 53, 42, 3, 39, 84, 78, 60]
val_cids = [48, 22, 12, 91, 96, 6, 35, 95, 66, 92]
train_cids = [37, 36, 56, 89, 65, 69, 74, 19, 9, 23, 79, 13, 93, 82, 99, 94, 34, 4, 61, 57, 51, 52, 7, 85, 21, 90, 76,
              38, 5, 63, 59, 50, 88, 100, 31, 32, 26, 41, 71, 15, 87, 54, 43, 27, 58, 55, 16, 20, 28, 83, 44, 46, 86,
              98, 17, 49, 45, 40, 8, 64, 80, 62, 11, 18, 1, 97, 75, 67, 25, 72]

all_super_categories = {'Giraffidae', 'hand', 'Castoridae', 'Ursidae', 'Mephitidae', 'Elephantidae', 'Cercopithecidae',
                        'animal', 'Muridae', 'bird', 'clothes', 'Felidae', 'furniture', 'Sciuridae', 'Rhinocerotidae',
                        'Cricetidae', 'Leporidae', 'Hippopotamidae', 'Canidae', 'Mustelidae', 'Bovidae', 'Suidae',
                        'Hominidae', 'vehicle', 'Equidae', 'person', 'Procyonidae', 'Cervidae', 'animal_face'}

if __name__ == '__main__':
    # img_root = '/home/chenjunjie/workspace/datasets/PoseDataset/mp100/mp100_images_cc/'
    with open(f'data/mp100/annotations/mp100_all_link.json', 'r') as f:
        ori_json = json.load(f)

    new_json = {}
    for k in ['images', 'annotations', 'categories', 'info']:
        new_json[k] = ori_json[k]

    for i in range(100):
        cdict = new_json['categories'][i]
        cid = cdict['id']

        ori_links = ori_json['categories'][i]['skeleton']
        if cid in [84]:
            new_links = [i for i in ori_links if i != [9, 2] and 10 not in i] + \
                        [[2, 7], [7, 8], [3, 8]]
        elif cid in [39]:
            new_links = ori_links + [[9, 12]]
        else:
            new_links = ori_links
        new_json['categories'][i]['skeleton'] = new_links
        i

    with open(f'data/mp100/annotations/mp100_all_link_0319.json', 'w') as f:
        json.dump(new_json, f)
