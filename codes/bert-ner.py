from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

nlp = pipeline("ner", model=model, tokenizer=tokenizer)
example = "Finland officially the Republic of Finland is a Nordic country in Northern Europe. the city Helsinki is one of the largest cities in Finland."

ner_results = nlp(example)
print(ner_results)