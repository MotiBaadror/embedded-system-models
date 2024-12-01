import torch

from dir_configs import add_rootpath
from models.transformers_model.data_module import GLUEDataModule
from models.transformers_model.model import GLUETransformer
import pytorch_lightning as pl

pl.seed_everything(42)
checkpoint_path = add_rootpath('data/trainings/lightning_logs/lightning_logs/version_1/checkpoints/epoch=2-step=291.ckpt')
dm = GLUEDataModule(model_name_or_path="albert-base-v2", task_name="cola")
dm.prepare_data()
dm.setup("fit")

# model = GLUETransformer(
#     model_name_or_path="albert-base-v2",
#     num_labels=dm.num_labels,
#     eval_splits=dm.eval_splits,
#     task_name=dm.task_name,
# )
model = GLUETransformer.load_from_checkpoint(checkpoint_path=checkpoint_path, strict=False, map_location='cpu')

trainer = pl.Trainer(
    max_epochs=1,
    accelerator="auto",
    devices=1,
)
# trainer.fit(model, datamodule=dm)
# trainer.validate(model, dm)
for data in dm.test_dataloader():
    ids, mask = data['input_ids'], data['attention_mask']
    break
onnx_model_path = add_rootpath( 'data/model_repository/spam_detector_v1.onnx')
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