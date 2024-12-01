from models.transformers_model.preprocess import preprocess_batch

def test_preprocess():
    example = {'text':['Hey get this done in 1 day']}
    out = preprocess_batch(example_batch=example)
    assert isinstance(out['input_ids'], list)