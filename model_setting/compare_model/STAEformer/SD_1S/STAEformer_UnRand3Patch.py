import os
import sys
import json

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
import torch
from easydict import EasyDict

from basic_ts.losses import masked_mae, masked_rmse
from basic_ts.utils import load_adj

from main.main_data.forecast_dataset import ForecastDataset
from main.main_runner.forecast_runner import CompareForecastRunner

from compare_model.BTS_STAEformer import STAEformerSeq2Seq

CFG = EasyDict()

# ================= general ================= #
CFG.DESCRIPTION = "STAEformer on SD configuration"
CFG.RUNNER = CompareForecastRunner
CFG.DATASET_CLS = ForecastDataset
CFG.DATASET_NAME = "SD_1S"
CFG.DATASET_TYPE = "Traffic flow"
CFG.DATASET_DESCRIBE = EasyDict(json.load(open(os.path.join("dataset_describe", CFG.DATASET_NAME), "r")))

CFG.DATASET_INPUT_LEN = CFG.DATASET_DESCRIBE.get("src_len")
CFG.DATASET_OUTPUT_LEN = CFG.DATASET_DESCRIBE.get("trg_len")
CFG.GPU_NUM = 1
CFG.NULL_VAL = 0.0

CFG.DATASET_ARGS = {
    "seq_len": CFG.DATASET_DESCRIBE.get("pretrain_src_len")
}
CFG.SMETA_PATH = os.path.join(CFG.DATASET_DESCRIBE.get("folder"), CFG.DATASET_DESCRIBE.get("meta_file"))

CFG.GRAPH_PATH = os.path.join(CFG.DATASET_DESCRIBE.get("folder"), "adj_mx.pkl")

adj_mx, _ = load_adj(CFG.GRAPH_PATH, "doubletransition")
adj_mx = torch.tensor(adj_mx[0], dtype=torch.float32)

# ================= environment ================= #
CFG.ENV = EasyDict()
CFG.ENV.SEED = 1
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "STAEformer_UnRand3Patch"
CFG.MODEL.ARCH = STAEformerSeq2Seq

CFG.MODEL.MAIN_FEATURES = [0]

CFG.MODEL.TMETA_FEATURES = [1, 2, 3, 4] # tod, dow, dom, doy
CFG.MODEL.SMETA_FEATURES = None

CFG.MODEL.TARGET_FEATURES = [0]
CFG.MODEL.INPUT_DIM = len(CFG.MODEL.MAIN_FEATURES)
CFG.MODEL.OUTPUT_DIM = len(CFG.MODEL.TARGET_FEATURES)

CFG.FORECAST_MASK_METHOD = "unrand"
CFG.RANDOM_SEED = 2024

CFG.TrainSample = False
CFG.TestSample = True

CFG.MODEL.PARAM = {
    "in_steps": CFG.DATASET_INPUT_LEN,
    "out_steps": CFG.DATASET_OUTPUT_LEN,
    "steps_per_day": 16*4,
    "num_nodes" : CFG.DATASET_DESCRIBE.get("num_nodes"),
    "input_dim": CFG.MODEL.INPUT_DIM,
    "output_dim": CFG.MODEL.OUTPUT_DIM,
    "tod_embedding_dim": 0,
    "dow_embedding_dim": 0,
    "feed_forward_dim": 32,
    "num_heads": 4,
    "num_layers": 2,
    "use_mixed_proj": True, 

    "num_tokens": None, 
    "patch_length": 2,

    "forecast_mask_args": {
        "rand_mask_ration": 0.8,
        "unrand_mask_ration":0.3,
        "min_unrand_token_length":1,
        "max_unrand_token_length":4
    }
}

CFG.MODEL.PARAM["num_tokens"] = CFG.DATASET_DESCRIBE.get("src_len") // CFG.MODEL.PARAM["patch_length"]
# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.002,
    "weight_decay": 0,
    "eps": 1.0e-8
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [50],
    "gamma": 0.5
}

# ================= train ================= #
CFG.TRAIN.CLIP_GRAD_PARAM = {
    "max_norm": 3.0
}
CFG.TRAIN.NUM_EPOCHS = 100
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "result/compare_model/STAEformer", CFG.DATASET_NAME,
    "_".join([CFG.MODEL.NAME, str(CFG.TRAIN.NUM_EPOCHS)])
)

# train data
CFG.TRAIN.DATA = EasyDict()
# read data
CFG.TRAIN.DATA.DIR = CFG.DATASET_DESCRIBE.get("folder")
# dataloader args, optional
CFG.TRAIN.DATA.NUM_WORKERS = 8
CFG.TRAIN.DATA.BATCH_SIZE = 8
CFG.TRAIN.DATA.PREFETCH = False
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.PIN_MEMORY = True

# ================= validate ================= #
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 2
# validating data
CFG.VAL.DATA = EasyDict()
# read data
CFG.VAL.DATA.DIR = CFG.TRAIN.DATA.DIR
# dataloader args, optional
CFG.VAL.DATA.BATCH_SIZE = 32
CFG.VAL.DATA.PREFETCH = False
CFG.VAL.DATA.SHUFFLE = False
CFG.VAL.DATA.NUM_WORKERS = 8
CFG.VAL.DATA.PIN_MEMORY = True

# ================= test ================= #
CFG.TEST = EasyDict()

CFG.TEST.SAVE_RESULT_NPZ = False

CFG.TEST.INTERVAL = 10
# test data
CFG.TEST.DATA = EasyDict()
# read data
CFG.TEST.DATA.DIR = CFG.TRAIN.DATA.DIR
# dataloader args, optional
CFG.TEST.DATA.BATCH_SIZE = 16
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 8
CFG.TEST.DATA.PIN_MEMORY = True

# ================= eval ================= #
CFG.EVAL = EasyDict()
CFG.EVAL.HORIZONS = range(1, CFG.DATASET_OUTPUT_LEN+1)

CFG.TEST.SAVE_RESULT_NPZ = True