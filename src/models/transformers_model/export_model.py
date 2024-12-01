from dataclasses import dataclass

import torch

from dir_configs import add_rootpath
from models.transformers_model.data_module import GLUEDataModule
from models.transformers_model.model import GLUETransformer
import pytorch_lightning as pl


@dataclass
class ExportConfig:
    checkpoint_path: str = add_rootpath(
       'data/trainings/tiny_v2_losswt_10_1/lightning_logs/version_1/checkpoints/epoch=9-step=1050.ckpt'
    )
    model_name: str = 'hf-internal-testing/tiny-albert'
    export_model_path: str = 'data/model_repository/tiny_albert_spam_detector_352k_wt_10_1_v1.onnx'

config = ExportConfig()

pl.seed_everything(42)
dm = GLUEDataModule(model_name_or_path=config.model_name, task_name="cola")
dm.prepare_data()
dm.setup("fit")

# model = GLUETransformer(
#     model_name_or_path="albert-base-v2",
#     num_labels=dm.num_labels,
#     eval_splits=dm.eval_splits,
#     task_name=dm.task_name,
# )
model = GLUETransformer.load_from_checkpoint(checkpoint_path=config.checkpoint_path, strict=False, map_location='cpu')


for data in dm.test_dataloader():
    ids, mask = data['input_ids'], data['attention_mask']
    break
onnx_model_path = add_rootpath(config.export_model_path)
opset_version = 15
torch.onnx.export(
    model.model,
    (ids, mask),
    onnx_model_path,
    opset_version=opset_version,
    input_names=["input_ids", "attention_mask"],
    output_names=["output"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_len"},
        "attention_mask": {0: "batch_size", 1: "sequence_len"},
        "output": {0: "batch_size"},
    }
)