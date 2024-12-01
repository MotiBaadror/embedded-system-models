from onnxruntime import SessionOptions, InferenceSession
from transformers import AutoModelForSequenceClassification

from models.transformers_model.data_module import GLUEDataModule


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







onnx_model_path = "../models/transformers_model/my_model.onnx"

model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')
model.eval()
# input_ids_tensor = torch.LongTensor(input_ids).to(model.device)
# attention_mask_tensor = torch.LongTensor(attention_mask).to(model.device)
# with torch.no_grad():
#     torch_output = model(input_ids_tensor, attention_mask_tensor).logits.detach().cpu().numpy()

dm = GLUEDataModule(model_name_or_path="albert-base-v2", task_name="cola")
dm.prepare_data()
dm.setup("fit")

for data in dm.train_dataloader():
    ids, mask = data['input_ids'], data['attention_mask']

input_feed = {
    "input_ids": ids.cpu().numpy(),
    "attention_mask": mask.cpu().numpy()
}


session = create_inference_session(onnx_model_path, provider="CUDAExecutionProvider")
onnx_output = session.run(output_names=["output"], input_feed=input_feed)[0]

# if np.allclose(torch_output, onnx_output, atol=1e-5):
#     print("Exported model has been tested with ONNXRuntime, and the result looks good!")

# print("input_ids: ", input_ids)
print("onnx output: ", onnx_output)
# print('torch output ', torch_output)
