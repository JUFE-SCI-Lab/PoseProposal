from xtcocotools.coco import COCO
import os

if __name__ == '__main__':
    hit_count = 0
    miss_count = {}

    for i in [1, 2, 3, 4, 5]:
        coco_train = COCO(f'data/mp100/annotations/mp100_split{i}_train.json')
        coco_val = COCO(f'data/mp100/annotations/mp100_split{i}_val.json')
        coco_test = COCO(f'data/mp100/annotations/mp100_split{i}_test.json')

        # img_root = 'data/mp100/images/'
        # img_root = 'data/mp100_split1/'
        # img_root = '/home/chenjunjie/workspace/datasets/PoseDataset/raw_complete_mp100/mp100/'
        img_root = '/home/chenjunjie/workspace/datasets/PoseDataset/mp100/mp100_images_cc/'

        for coco in [coco_train, coco_test, coco_val]:
            for iid, info in coco.imgs.items():
                img_path = img_root + info['file_name']
                if os.path.exists(img_path):
                    hit_count += 1
                else:
                    name = os.path.dirname(info['file_name'])
                    if name in miss_count:
                        if img_path not in miss_count[name]:
                            miss_count[name].append(img_path)
                    else:
                        miss_count[name] = [img_path]
            iid
        i

    r = {}
    for k, v in miss_count.items():
        r[k] = len(set(v))
    print(r)
    d = 1
