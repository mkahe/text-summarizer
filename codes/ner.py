import spacy
from spacy import displacy

NER = spacy.load("en_core_web_sm")

raw_text="Finland is a vast country. The capital is Helsinki and Oulu is one of the northern cities. The Indian Space Research Organisation or is the national space agency of India, headquartered in Bengaluru. It operates under Department of Space which is directly overseen by the Prime Minister of India while Chairman of ISRO acts as executive of DOS as well."

text1= NER(raw_text)

for word in text1.ents:
    print(word.text,word.label_)

print(spacy.explain("ORG"))

print(spacy.explain("GPE"))

displacy.render(text1,style="ent",jupyter=False)