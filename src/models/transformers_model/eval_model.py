import os
from dataclasses import dataclass
from torchmetrics import Accuracy, ConfusionMatrix

from dir_configs import add_rootpath
from models.transformers_model.data_module import GLUEDataModule
from models.transformers_model.entypoint_tf_model import MyTrainingConfig
from models.transformers_model.model import GLUETransformer

@dataclass
class MyEvalConfig:
    version: int
    checkpoint_name: str
    exp_name: str
    mismatch_count: int = 0
    base_dir: str = add_rootpath('data/trainings')
    checkpoint_path: str = None
    out_dir: str = None

    def __post_init__(self):
        self.checkpoint_path = self.base_dir+f'/{self.exp_name}/lightning_logs/version_{self.version}/checkpoints/{self.checkpoint_name}'
        self.out_dir = self.base_dir+f'/{self.exp_name}/lightning_logs/version_{self.version}'


train_config = MyTrainingConfig(epochs=10)

config = MyEvalConfig(
    version=1,
    checkpoint_name='epoch=9-step=1050.ckpt',
    exp_name='tiny_v2_losswt_10_1',
)

# pl.seed_everything(42)

dm = GLUEDataModule(model_name_or_path=train_config.model, task_name="cola")
dm.prepare_data()
dm.setup("fit")



exp_name = config.exp_name
model = GLUETransformer.load_from_checkpoint(checkpoint_path=config.checkpoint_path, strict=False, map_location='cpu')
model.eval()
accuracy = Accuracy(task='binary')
conf_mat = ConfusionMatrix(task='binary')

def eval_model(accuracy= accuracy, conf_mat=conf_mat, model=model, dm=dm):

    for data in dm.test_dataloader():
        ids, mask = data['input_ids'], data['attention_mask']
        input_values = dict(
            input_ids=ids,
            attention_mask=mask
        )
        # labels = data['labels']
        out = model(**data)
        cls = model.get_cls(out[1])
        if cls != data['labels']:
            print(dm.tokenizer.decode(data['input_ids'][0]))
        accuracy.update(cls, data['labels'])
        conf_mat.update(cls,data['labels'])

    print('acc  ', accuracy.compute())
    accuracy.reset()
    print('confusion matrix  ', conf_mat.compute())
    conf_mat.reset()
