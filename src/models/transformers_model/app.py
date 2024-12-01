import logging

from flask import Flask, request, jsonify

from dir_configs import add_rootpath
try:
    from models.transformers_model.inference_exported_model import OnnxRunner
except:
    from inference_exported_model import OnnxRunner

app = Flask(__name__)

tokenizer_model = 'hf-internal-testing/tiny-albert'#'albert-base-v2'#'hf-internal-testing/tiny-albert'
exported_model = 'tiny_albert_spam_detector_352k_wt_10_1_v1.onnx'
runner = OnnxRunner(
    add_rootpath(f'data/model_repository/{exported_model}'),
    tokenizer_model=tokenizer_model
)

@app.route('/predict', methods=["POST"])
def main_app():
    example_batch = request.json.get('example_batch')
    logging.info(f'{example_batch}')
    out = runner.run_inference(example_batch=example_batch)
    logging.info('runner finished with output ',out)
    return jsonify(result = f"{out}")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)