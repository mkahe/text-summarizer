import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity


named_entity_rec = spacy.load("en_core_web_sm")

class Sentence:
    def __init__(self, text, tag):
        self.text = text
        # tag: heading, subheading, paragraph
        # polish the text to remove special characters
        # remove the withespaces and commas
        self.text = re.sub('[^a-zA-Z0-9]', ' ', self.text)
        self.text = re.sub(' +', ' ', self.text)
        self.tag = tag
        self.POS = 0
        self.weight = 0
        self.named_entities = []
        self.add_named_entity()


    def add_named_entity(self):
        ner_text = named_entity_rec(self.text)
        for word in ner_text.ents:
            self.named_entities.append(word.text)

        if self.named_entities == []:
            self.named_entities = ""

        else:
            self.named_entities = " ".join(self.named_entities)
        

    
# read a text file
def read_file(file_name):
    with open(file_name, 'r') as f:
        text = f.read()
    return text


# split text into sentences
def split_text(text):
    # split text into sentences
    sentences = sent_tokenize(text)
    return sentences


# define create_sentence function
def create_sentence(text, tag):
    text = text.strip()
    # split text into sentences
    sentences = text.split('.')
    # create a list of Sentence objects
    sentence_objects = []
    for sentence in sentences:
        if sentence:
            sentence_objects.append(Sentence(sentence, tag))
    return sentence_objects

# define sent_tokenize function
# this function will use html tags to split text into sentences
# and the sentences are separated by period.
def sent_tokenize(text):
    # split text into sentences
    sentences = []
    # split text into paragraphs based on html tags
    # regex to find text between <h1> and </h1> including whitespace
    # or regex to find text between <p> and <p> including whitespace
    # pattern = re.compile(r'(<h1>(.*?)</h1>)|(<p>.*?</p>)', re.DOTALL)
    pattern = re.compile(r'<h1>(.*?)<\/h1>|<h2>(.*?)</h2>|<p>(.*?)<\/p>', re.DOTALL)
    # pattern = re.compile(r'<h1>(.*?)<\/h1>')
    # find all matches
    matches = pattern.finditer(text)
    # loop through matches
    for match in matches:   
        # get the text between html tags
        # if the match is a heading
        if match.group(1):
            # get the text between <h1> and </h1>
            text = match.group(1)
            # set tag to heading
            tag = 'heading'
            # create a list of Sentence objects
            sentence_objects = create_sentence(text, tag)
            # add the list of Sentence objects to sentences
            sentences.extend(sentence_objects)
        # match h2
        elif match.group(2):
            text = match.group(2)
            tag = 'heading 2'
            sentence_objects = create_sentence(text, tag)
            sentences.extend(sentence_objects)


        # if the match is a paragraph
        else:
            # get the text between <p> and </p>
            text = match.group(3)
            # set tag to paragraph
            tag = 'paragraph'
            # create a list of Sentence objects
            sentence_objects = create_sentence(text, tag)
            # add the list of Sentence objects to sentences
            sentences.extend(sentence_objects)

    # print all the sentences
    # for sentence in sentences:
    #     print("structure: %s, Text: %s" %( sentence.tag, sentence.text.strip()))

    return sentences


def lemmatizer(sentences):
    # create lemmatize object
    lemmmatizer = WordNetLemmatizer()
    # get sentences
    for i in range(len(sentences)):
        words = word_tokenize(sentences[i])
        # List comprehension
        words = [lemmmatizer.lemmatize(
            word.lower()) for word in words if word not in set(stopwords.words('english'))]
        sentences[i] = ' '.join(words)
    return sentences

# the return list order is the same as the input list
def get_score_based_on_tfidf(sentence):
    cv = TfidfVectorizer()
    x = cv.fit_transform(sentence)
    rlist = []
    # add the index 0 of each sentence 
    # to the POS attribute of the Sentence object
    for i in range(len(x.toarray())):
        rlist.append(np.sum(x.toarray()[i]))
    
    return rlist


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

# sort the sentences based on the weight decending
# then select the top 20 sentences
# add the first sentence to the summary
# calculate the similarity of the first sentence with the rest of the sentences
def get_summary(sentences):
    summary = []
    # sort the sentences based on the weight decending

    #Load AutoModel from huggingface model repository
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
    model = AutoModel.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")

    sentences.sort(key=lambda x: x.weight, reverse=True)

    summary.append((sentences[0], 0))

    # calculate the similarity of the first sentence with the rest of the sentences
    scores_s1 = [1]
    for i in range(1, len(sentences[:20])):
        sens = [sentences[0].text, sentences[i].text]
        encoded_input = tokenizer(sens, padding=True, truncation=True, max_length=128, return_tensors='pt')
        #Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        #Perform pooling. In this case, mean pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embedding_n= sentence_embeddings.numpy()
        score = cosine_similarity([sentence_embedding_n[0]],sentence_embedding_n[1:])
        scores_s1.append(score[0][0])

    scores_s1[0] = 1000
    for i in range(1, 10):
        # find the minimum score and its index
        min_score = min(scores_s1)
        min_index = scores_s1.index(min_score)
        # add the sentence with the minimum score to the summary
        summary.append((sentences[min_index], min_index))
        # recalulate the similarity of the new sentence with the sentences from index 0 to min_index
        for j in range(0, 20):
            sens = [sentences[min_index].text, sentences[j].text]
            # print(sens, scores_s1[min_index], scores_s1[j])
            encoded_input = tokenizer(sens, padding=True, truncation=True, max_length=128, return_tensors='pt')
            #Compute token embeddings
            with torch.no_grad():
                model_output = model(**encoded_input)
            
            #Perform pooling. In this case, mean pooling
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            sentence_embedding_n= sentence_embeddings.numpy()
            score = cosine_similarity([sentence_embedding_n[0]],sentence_embedding_n[1:])
            scores_s1[j] += score[0][0]
        scores_s1[min_index] = 1000

    for sen in sentences[:20]:
        print(sen.text)

    for i, j in summary:
        print("sentence: ", j, "TEXT: ", i.text)




    

if __name__ == "__main__":
    # read a text file
    text = read_file('dataset/finland.txt')
    # split text into sentences
    sentences = split_text(text)
    print(sentences[1].named_entities)
    
    text_list = [i.text for i in sentences]
    text_list = lemmatizer(text_list)
    sentence_tfidf = get_score_based_on_tfidf(text_list)

    ne_list = [i.named_entities for i in sentences]
    ne_list = lemmatizer(ne_list)
    ne_score_tfidf = get_score_based_on_tfidf(ne_list)
    
    for i in range(len(sentences)):
        sentences[i].weight = sentence_tfidf[i] + 2 * ne_score_tfidf[i]
    
    get_summary(sentences)

    
