from transformers import AutoConfig, AutoModelForSequenceClassification

model_name_or_path = 'distilbert-base-uncased'
num_labels = 2

config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)

model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=config)
print(model)
# input_ids_tensor
# attention_mask_tensor
print(config)