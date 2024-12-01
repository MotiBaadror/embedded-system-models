import logging

from flask import Flask, request, jsonify

from dir_configs import add_rootpath
try:
    from models.transformers_model.inference_exported_model import OnnxRunner
except:
    from inference_exported_model import OnnxRunner

app = Flask(__name__)

runner = OnnxRunner(add_rootpath('data/model_repository/my_model_bs_1.onnx'))

@app.route('/predict', methods=["POST"])
def main_app():
    example_batch = request.json.get('example_batch')
    logging.info(f'{example_batch}')
    out = runner.run_inference(example_batch=example_batch)
    logging.info('runner finished with output ',out)
    return jsonify(result = f"{out}")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)