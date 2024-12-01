import torch

from models.transformers_model.data_module import GLUEDataModule
from models.transformers_model.model import GLUETransformer
import pytorch_lightning as pl

pl.seed_everything(42)

dm = GLUEDataModule(model_name_or_path="albert-base-v2", task_name="cola")
dm.prepare_data()
dm.setup("fit")

model = GLUETransformer(
    model_name_or_path="albert-base-v2",
    num_labels=dm.num_labels,
    eval_splits=dm.eval_splits,
    task_name=dm.task_name,
)

trainer = pl.Trainer(
    max_epochs=1,
    accelerator="auto",
    devices=1,
)
# trainer.fit(model, datamodule=dm)
# trainer.validate(model, dm)
for data in dm.test_dataloader():
    ids, mask = data['input_ids'], data['attention_mask']
    # input = {'input_ids':data['input_ids'],'attention_mask':data['attention_mask'], 'token_type_ids':data['token_type_ids']}
    # model.to_onnx(
    #     'my_model.onnx',
    #     input_sample=input,
    #     input_names=["input_ids", "attention_mask", "token_type_ids"],
    # )
    # print(data)
    break
# model.to_onnx('my_model.onnx', input_sample=(("ids,mask)))
# print(ids.shape, mask.shape)
onnx_model_path = 'my_model.onnx'
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