# from .cape_head import *  # noqa
# from .cape_detector import *  # noqa
# from .cape_transformer import *  # noqa
from .cfg.namefunc import set_outdir_by_exp_name
from .meta_pose import *
from .cpt_modules.predict_head import PredictHead
# from ckit.cpiplines import ColorAug
from .cpt_modules.dataset_train import MyTrainSet
from .cpt_modules.dataset_test import MyTestSet
from .cpt_modules.dataset_test_W import MyTestSet_W