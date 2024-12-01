import numpy as np
import torch
from onnxruntime import SessionOptions, InferenceSession
from transformers import AutoModelForSequenceClassification


def create_inference_session(
        model_path: str,
        intra_op_num_threads: int = 8,
        provider: str = 'CPUExecutionProvider'
) -> InferenceSession:
    """
    Create onnx runtime InferenceSession.

    model_path : str
        onnx model file.

    intra_op_num_threads : int
        Remember to tune this parameter.

    provider : str
        get_all_providers function can list all available providers.
        e.g. CUDAExecutionProvider
    """

    options = SessionOptions()
    options.intra_op_num_threads = intra_op_num_threads

    # load the model as a onnx graph
    session = InferenceSession(model_path, options, providers=[provider])
    session.disable_fallback()
    return session


input_ids = [[101,
              2031,
              2057,
              9471,
              2000,
              8439,
              2060,
              9490,
              2015,
              2040,
              2106,
              3576,
              2021,
              6827,
              4395,
              1029,
              102,
              2024,
              2045,
              2060,
              2796,
              9490,
              2015,
              4237,
              2013,
              2019,
              4014,
              13970,
              19661,
              2040,
              2031,
              2288,
              1996,
              4495,
              2000,
              3710,
              2004,
              11274,
              1997,
              2037,
              2110,
              1005,
              1055,
              4533,
              8924,
              1029,
              102]]

attention_mask = [[1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1,
  1]]

input_feed = {
    "input_ids": np.array(input_ids),
    "attention_mask": np.array(attention_mask)
}






onnx_model_path = "../../text_classification.onnx"

model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')
model.eval()
input_ids_tensor = torch.LongTensor(input_ids).to(model.device)
attention_mask_tensor = torch.LongTensor(attention_mask).to(model.device)
with torch.no_grad():
    torch_output = model(input_ids_tensor, attention_mask_tensor).logits.detach().cpu().numpy()

session = create_inference_session(onnx_model_path, provider="CUDAExecutionProvider")
onnx_output = session.run(output_names=["output"], input_feed=input_feed)[0]

if np.allclose(torch_output, onnx_output, atol=1e-5):
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

print("input_ids: ", input_ids)
print("onnx output: ", onnx_output)
print('torch output ', torch_output)
