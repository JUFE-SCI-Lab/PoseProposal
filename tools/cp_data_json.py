from xtcocotools.coco import COCO
import os
import shutil

if __name__ == '__main__':
    for i in [1, 2, 3, 4, 5]:
        coco_train = COCO(f'data/mp100/annotations/mp100_split{i}_train.json')
        coco_val = COCO(f'data/mp100/annotations/mp100_split{i}_val.json')
        coco_test = COCO(f'data/mp100/annotations/mp100_split{i}_test.json')
        # img_root = 'data/mp100/images/'
        img_root = '/home/chenjunjie/workspace/datasets/PoseDataset/raw_complete_mp100/mp100/'
        target_root = '/home/chenjunjie/workspace/datasets/PoseDataset/mp100_images_cc'
        hit_count = 0
        miss_count = {}

        for coco in [coco_train, coco_test, coco_val]:
            for iid, info in coco.imgs.items():
                img_path = img_root + info['file_name']
                if os.path.exists(img_path):
                    img_folder = os.path.dirname(info['file_name'])
                    img_name = os.path.basename(info['file_name'])
                    folder_path = f'{target_root}/{img_folder}'
                    os.makedirs(folder_path, exist_ok=True)
                    shutil.copy(img_path, f'{folder_path}/{img_name}')
                    hit_count += 1
                    if hit_count % 100 == 0:
                        print(hit_count)
                else:
                    name = os.path.dirname(info['file_name'])
                    if name in miss_count:
                        miss_count[name].append(img_path)
                    else:
                        miss_count[name] = [img_path]
            iid
        i

    r = {}
    for k, v in miss_count.items():
        r[k] = len(set(v))

    d = 1
