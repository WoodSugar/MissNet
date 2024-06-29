# -*- coding: utf-8 -*-
""" 
@Time   : 2023/10/11
 
@Author : Shen Fang
"""

import os
import sys
import json

from easydict import EasyDict

from basic_ts.losses import masked_rmse, masked_mse, masked_mae ,l2_loss

from main.main_data.forecast_dataset import ForecastDataset

from main.main_runner.forecast_runner import MetaForecastRunner

from main.main_arch.backbone import PreTrainEncoderDecoder
from main.main_arch.predictor import HeaderWithBackBone
from main.main_arch.header import UniversalHeader, UniversalSTPHeader, Header, LRHeader, MLPHeader
from main.main_arch.module import TransformerLayers

from model_utils import MLP 

import torch

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

CFG = EasyDict()
# ================= general ================= #
CFG.DESCRIPTION = "Finetune(SD) configuration"
CFG.RUNNER = MetaForecastRunner
CFG.DATASET_CLS = ForecastDataset
CFG.DATASET_NAME = "SD_1S"
CFG.DATASET_TYPE = "Traffic flow"
CFG.DATASET_DESCRIBE = EasyDict(json.load(open(os.path.join("dataset_describe", CFG.DATASET_NAME), "r")))

CFG.DATASET_INPUT_LEN = CFG.DATASET_DESCRIBE.get("pretrain_src_len")
CFG.DATASET_OUTPUT_LEN = CFG.DATASET_DESCRIBE.get("trg_len")
CFG.GPU_NUM = 1
CFG.NULL_VAL = 0.0

CFG.DATASET_ARGS = {
    "seq_len": CFG.DATASET_DESCRIBE.get("pretrain_src_len")
}

# ================= environment ================= #
CFG.ENV = EasyDict()
CFG.ENV.SEED = 0
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "MLP_ShortRand8"
CFG.MODEL.ARCH = HeaderWithBackBone
CFG.MODEL.FORWARD_FEATURES = [0]
CFG.MODEL.TARGET_FEATURES = [0]
CFG.MODEL.INPUT_DIM = len(CFG.MODEL.FORWARD_FEATURES)
CFG.MODEL.OUTPUT_DIM = len(CFG.MODEL.TARGET_FEATURES)

# TODO, 这里需要区分输入数据的不同语义，这是唯一指示数据特征维度的信息，需要牢记。

CFG.MODEL.MAIN_FEATURES = [0]
CFG.MODEL.TMETA_FEATURES = [1,2,3,4]
# CFG.MODEL.SMETA_FEATURES = list(range(5, 755+5))

CFG.GRAPH_PATH = os.path.join(CFG.DATASET_DESCRIBE.get("folder"), "adj_mx.pkl")
CFG.BackBone_GRAPH = None

CFG.SMETA_PATH = os.path.join(CFG.DATASET_DESCRIBE.get("folder"), CFG.DATASET_DESCRIBE.get("meta_file"))

CFG.TrainSample = False
CFG.TestSample = True
CFG.FORECAST_MASK_METHOD = "rand"
CFG.RANDOM_SEED = 2024

from model_setting.pretrain_model.SD_1S.SD_1S import CFG as BackBone

CFG.MODEL.PARAM = {
    "backbone_path": "result/PreTrain/SD_1S/PreTrain_100/1f2481f0a32829617badf35a2e483e31/PreTrain_best_val_MAE.pt", 
    "backbone_model": PreTrainEncoderDecoder, 
    "backbone_args": BackBone.MODEL.PARAM,

    "e2d_model": MLP, 
    "e2d_args": {
        "channels":(64, 64), 
        "act_type": "linear", 
        "norm_type": None, 
        "bias":True,
    },
    
    "predictor_model": MLPHeader, 
    "predictor_args": {
        "t_meta_emb": [4, 4], 
        "t_meta_emb_act": "linear", 
        "t_meta_emb_norm":"layer",

        "s_meta_dense_emb":[2, 4], 
        "s_meta_sparse_emb":[753, 64], 
        "s_meta_full_emd":[68, 32],
        "s_meta_embed_act": "linear", 
        "s_meta_emb_norm":"layer",
                 
        "data_in_c": 64, 
        "data_hid_c": 64, 
        "data_out_c": CFG.MODEL.INPUT_DIM, 
        "num_tokens": None, 
        "patch_length": BackBone.MODEL.PARAM["encoder_args"]["patch_length"],
        "trg_len": CFG.DATASET_OUTPUT_LEN,
        "patch2seq_method":"MLP", 
        "out_method":"Transformer",
        "num_nodes": CFG.DATASET_DESCRIBE.get("num_nodes"),
        "s_conv_k": [3], 
        "s_conv_d": [1]
    },

    "aux_compute_args": {
        "source_in_dim": BackBone.MODEL.INPUT_DIM,
        "target_in_dim": CFG.MODEL.INPUT_DIM,
    }
}

# CFG.Reverse = False

CFG.MODEL.PARAM["backbone_args"]["run_mode"] = "forecast"

# finetune stage from out source
CFG.MODEL.PARAM["backbone_args"]["encoder_args"]["num_tokens"] =  CFG.DATASET_DESCRIBE.get("pretrain_src_len") // BackBone.MODEL.PARAM["encoder_args"]["patch_length"]
CFG.MODEL.PARAM["backbone_args"]["encoder_args"]["forecast_mask_args"] = {
    "rand_mask_ration": 0.8,
    "unrand_mask_ration":0.5,
    "min_unrand_token_length":2,
    "max_unrand_token_length":4
}


CFG.MODEL.PARAM["predictor_args"]["num_tokens"] =  CFG.DATASET_DESCRIBE.get("pretrain_src_len") // BackBone.MODEL.PARAM["encoder_args"]["patch_length"]

# pretrain stage
# CFG.MODEL.PARAM["decoder_args"]["out_dim"] =  CFG.MODEL.PARAM["encoder_args"]["patch_length"] * CFG.MODEL.INPUT_DIM


# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = masked_mae

CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 2e-3,
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
    "max_norm": 5.0
}
CFG.TRAIN.NUM_EPOCHS = 100
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "result/FineTune", CFG.DATASET_NAME,
    "_".join([CFG.MODEL.NAME, str(CFG.TRAIN.NUM_EPOCHS)])
)
# train data
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.DIR = CFG.DATASET_DESCRIBE.get("folder")
CFG.TRAIN.DATA.NUM_WORKERS = 8
CFG.TRAIN.DATA.BATCH_SIZE = 32
CFG.TRAIN.DATA.PREFETCH = False

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
CFG.VAL.DATA.PIN_MEMORY = False

# ================= test ================= #
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 10
# test data
CFG.TEST.DATA = EasyDict()
# read data
CFG.TEST.DATA.DIR = CFG.TRAIN.DATA.DIR
# dataloader args, optional
CFG.TEST.DATA.BATCH_SIZE = 32
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 8
CFG.TEST.DATA.PIN_MEMORY = False

CFG.TEST.SAVE_RESULT_NPZ = True
# # ================= eval ================= #
CFG.EVAL = EasyDict()
CFG.EVAL.HORIZONS = range(1, CFG.DATASET_OUTPUT_LEN+1)