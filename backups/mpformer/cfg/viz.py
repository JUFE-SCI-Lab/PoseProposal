from mpformer.cfg.s1 import *

data['samples_per_gpu'] = 2
data['workers_per_gpu'] = 0

data['viz'] = dict(type='VizPoseDataset',
                   # ann_file=f'{data_root}/annotations/mp100_split1_valtest.json',
                   ann_file=f'{data_root}/annotations/mp100_split1_test.json',
                   img_prefix=f'{data_root}/mp100_images_cc',
                   data_cfg=data_cfg,
                   # valid_class_ids=[84, 73, 29, 70, 53, 14, ],
                   valid_class_ids=None,
                   max_kpt_num=channel_cfg['max_kpt_num'],
                   num_shots=1,
                   num_queries=1,
                   num_episodes=1,
                   pck_threshold_list=[0.05, 0.10, 0.15, 0.2],
                   pipeline=test_pipeline)
'''
good_test_viz_cid=[84,2,73,42,29,70,53,14,]
good_test_viz_cid=(84, 'bed'),(2, 'horse_body'),(73, 'lion_body'),(42, 'sheep_body'),(29, 'rabbit_body'),
(70, 'skunk_body'),(53, 'Tern'), (14, 'Woodpecker')
'''

cates = [(37, 'camel_face'), (36, 'quokka_face'), (56, 'onager_face'), (89, 'proboscismonkey_face'),
         (65, 'pademelon_face'), (69, 'olivebaboon_face'), (74, 'germanshepherddog_face'), (19, 'arcticwolf_face'),
         (9, 'panther_body'), (23, 'fallowdeer_face'), (79, 'grizzlybear_face'), (13, 'californiansealion_face'),
         (93, 'gerbil_face'), (82, 'capebuffalo_face'), (99, 'capybara_face'), (94, 'blackbuck_face'),
         (34, 'greyseal_face'), (4, 'fennecfox_face'), (61, 'gibbons_face'), (57, 'bonobo_face'), (51, 'suv'),
         (52, 'leopard_body'), (7, 'chipmunk_face'), (85, 'cheetah_body'), (21, 'bobcat_body'), (90, 'raccoon_body'),
         (76, 'polar_bear_body'), (38, 'bus'), (5, 'panda_body'), (63, 'ferret_face'), (59, 'Grebe'),
         (50, 'otter_body'), (88, 'rat_body'), (100, 'wolf_body'), (31, 'rhino_body'), (32, 'car'),
         (26, 'elephant_body'), (41, 'Kingfisher'), (71, 'giraffe_body'), (15, 'pig_body'), (87, 'cow_body'),
         (54, 'hippo_body'), (43, 'spider_monkey_body'), (27, 'deer_body'), (58, 'antelope_body'), (55, 'zebra_body'),
         (16, 'Wren'), (20, 'cat_body'), (28, 'Gull'), (83, 'short_sleeved_outwear'), (44, 'locust'), (46, 'Sparrow'),
         (86, 'Warbler'), (98, 'table'), (17, 'sling'), (49, 'sofa'), (45, 'chair'), (40, 'face'), (8, 'sling_dress'),
         (64, 'long_sleeved_dress'), (80, 'hand'), (62, 'vest'), (11, 'vest_dress'), (18, 'face'), (1, 'skirt'),
         (97, 'long_sleeved_shirt'), (75, 'shorts'), (67, 'trousers'), (25, 'short_sleeved_shirt'), (72, 'person'),
         (48, 'goldenretriever_face'), (22, 'guanaco_face'), (12, 'przewalskihorse_face'), (91, 'beaver_body'),
         (96, 'gentoopenguin_face'), (6, 'hamster_body'), (35, 'gorilla_body'), (95, 'weasel_body'), (66, 'fly'),
         (92, 'macaque'), (10, 'klipspringer_face'), (77, 'commonwarthog_face'), (24, 'dassie_face'),
         (47, 'alpaca_face'), (33, 'squirrel_body'), (29, 'rabbit_body'), (70, 'skunk_body'), (68, 'fox_body'),
         (81, 'bighornsheep_face'), (73, 'lion_body'), (2, 'horse_body'), (14, 'Woodpecker'), (30, 'bison_body'),
         (53, 'Tern'), (42, 'sheep_body'), (3, 'dog_body'), (39, 'swivelchair'), (84, 'bed'),
         (78, 'long_sleeved_outwear'), (60, 'short_sleeved_dress')]
