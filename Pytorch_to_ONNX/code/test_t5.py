from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForSeq2SeqLM

model_name = "t5-small"
save_directory = "tmp/onnx/"
model = ORTModelForSeq2SeqLM.from_pretrained(save_directory, export=False)
tokenizer = AutoTokenizer.from_pretrained(model_name)

onnx_translation = pipeline("translation_en_to_fr", model=model, tokenizer=tokenizer)
text = "He never went out without a book under his arm, and he often came back with two."
result = onnx_translation(text)
print(result)
# [{'translation_text': "Il n'est jamais sorti sans un livre sous son bras, et il est souvent revenu avec deux."}]
