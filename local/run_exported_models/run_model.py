from dir_configs import add_rootpath
from models.transformers_model.inference_lightning_model import OnnxRunner


runner =OnnxRunner(
    model_path=add_rootpath('data/model_repository/my_model_bs_1.onnx')
)
example_batch = {'text':['i am doing great']}
out = runner.run_inference(example_batch=example_batch)
print(out)