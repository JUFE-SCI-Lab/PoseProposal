import json

all_super_categories = {'Giraffidae', 'hand', 'Castoridae', 'Ursidae', 'Mephitidae', 'Elephantidae', 'Cercopithecidae',
                        'animal', 'Muridae', 'bird', 'clothes', 'Felidae', 'furniture', 'Sciuridae', 'Rhinocerotidae',
                        'Cricetidae', 'Leporidae', 'Hippopotamidae', 'Canidae', 'Mustelidae', 'Bovidae', 'Suidae',
                        'Hominidae', 'vehicle', 'Equidae', 'person', 'Procyonidae', 'Cervidae', 'animal_face'}

if __name__ == '__main__':
    super_class_dict = {
        'HumanBody': [72],
        'HumanFace': [18, 40],
        'vehicle': [51, 38, 32],
        'furniture': [98, 49, 45, 39, 84],
    }

    with open(f'data/mp100/annotations/mp100_all_link_0319.json', 'r') as f:
        all_json = json.load(f)

    for cinfo in all_json['categories']:
        if cinfo['supercategory'] in ['furniture', 'vehicle']:
            pass

        if cinfo['supercategory'] in ['Hominidae', 'person']:
            continue

        # [a for a in all_json['annotations'] if a['category_id'] == 35]

        print(cinfo)
    d = 1
