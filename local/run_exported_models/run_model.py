from dir_configs import add_rootpath
from models.transformers_model.inference_exported_model import OnnxRunner

runner =OnnxRunner(
    model_path=add_rootpath('data/model_repository/spam_detector_v1.onnx'),
    tokenizer_model='albert-base-v2'#'hf-internal-testing/tiny-albert'
)
example_batch = {'text':['i am doing great']}
out = runner.run_inference(example_batch=example_batch)
print(out)