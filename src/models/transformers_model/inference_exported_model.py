import numpy as np
from onnxruntime import SessionOptions, InferenceSession
from transformers import AutoTokenizer


class OnnxRunner():
    def __init__(self, model_path='my_model_bs_1.onnx', tokenizer_model="albert-base-v2"):
        self.model_path = model_path

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, use_fast=True)
        self.session = self.create_inference_session()
    def create_inference_session(self,
            # model_path: str,
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
        session = InferenceSession(self.model_path, options, providers=[provider])
        session.disable_fallback()
        return session

    def preprocess_batch(self,example_batch, max_seq_length=128):
        texts_or_text_pairs = example_batch['text']
        features = self.tokenizer.batch_encode_plus(
                    texts_or_text_pairs, max_length=max_seq_length, pad_to_max_length=True, truncation=True
        )
        return features


    def postprocess(self,logits):
        # Apply softmax to convert logits to probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # Stability trick
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


        # Class labels
        classes = ["true", "false"]

        # Find the index of the maximum probability
        predicted_index = np.argmax(logits)

        # Retrieve the corresponding label and probability
        predicted_label = classes[predicted_index]
        predicted_probability = probabilities[0][predicted_index]
        return {'isScam':predicted_label,'confidence':float(predicted_probability)}






    def run_inference(self,example_batch):
        processed_input  = self.preprocess_batch(example_batch)

        # processed_input['input_ids'], processed_input['attention_mask']
        input_feed = {
            "input_ids": np.array(processed_input['input_ids']),
            "attention_mask":np.array(processed_input['attention_mask'])
        }
        onnx_output = self.session.run(output_names=["output"], input_feed=input_feed)[0]

    # if np.allclose(torch_output, onnx_output, atol=1e-5):
    #     print("Exported model has been tested with ONNXRuntime, and the result looks good!")

        print("onnx output: ", onnx_output)
        onnx_output = self.postprocess(onnx_output)
        return onnx_output
