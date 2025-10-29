from pprint import pprint

from transformers import AutoTokenizer

input_text = """BERT base model (uncased)
Pretrained model on English language using a masked language modeling (MLM) objective. It was introduced in this paper and first released in this repository. This model is uncased: it does not make a difference between english and English.

Disclaimer: The team releasing BERT did not write a model card for this model so this model card has been written by the Hugging Face team."""

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
out = tok(input_text, return_tensors="pt")
pprint(input_text)
pprint(out)
