import spacy


NER = spacy.load("en_core_web_sm")

raw_text="Finland officially the Republic of Finland is a Nordic country in Northern Europe"

text1= NER(raw_text)

for word in text1.ents:
    print(word.text,word.label_)
