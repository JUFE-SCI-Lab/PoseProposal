import json

if __name__ == '__main__':
    for split_id in [1, 2, 3, 4, 5]:
        # coco_train = COCO(f'data/mp100/annotations/mp100_split{split_id}_train.json')
        # coco_val = COCO(f'data/mp100/annotations/mp100_split{split_id}_val.json')
        # coco_test = COCO(f'data/mp100/annotations/mp100_split{split_id}_test.json')
        # img_root = '/home/chenjunjie/workspace/datasets/PoseDataset/mp100/mp100_images_cc/'
        with open(f'data/mp100/annotations/mp100_split{split_id}_val.json', 'r') as f:
            val_json = json.load(f)
        with open(f'data/mp100/annotations/mp100_split{split_id}_test.json', 'r') as f:
            test_json = json.load(f)

        valtest_json = {}
        for k in ['images', 'annotations', 'categories']:
            valtest_json[k] = val_json[k] + test_json[k]
        valtest_json['info'] = test_json['info']

        cross_class_qs_infos = []
        for query_class in valtest_json['categories']:

            info = {}
            info.update(query_class)
            info['support_class_ids'] = []

            for support_class in valtest_json['categories']:
                if query_class['id'] == support_class['id']:
                    continue
                qstr = '-'.join(query_class['keypoints'])
                sstr = '-'.join(support_class['keypoints'])
                if qstr == sstr:
                    if '1' in query_class['keypoints'] and \
                            query_class['supercategory'] != support_class['supercategory']:
                        print(f'Skip!')
                        print(query_class)
                        print(support_class)
                        continue
                    info['support_class_ids'].append(support_class['id'])
            cross_class_qs_infos.append(info)

        valtest_json['info']['cross_class_qs_infos'] = cross_class_qs_infos
        with open(f'data/mp100/annotations/mp100_split{split_id}_valtest.json', 'w') as f:
            json.dump(valtest_json, f)
        split_id
