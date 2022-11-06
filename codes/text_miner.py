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
from rouge import Rouge


named_entity_rec = spacy.load("en_core_web_sm")

class Sentence:
    def __init__(self, text, tag, order):
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
        self.order = order
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


# split text into sentences``
def split_html_text(text):
    # split text into sentences
    sentences = sent_tokenize(text)
    return sentences


def split_text(text):
    pattern = re.compile(r'.*?\.', re.DOTALL)
    matches = pattern.finditer(text)
    sentences = []
    i = 0
    for match in matches:
        print(match , match.group())
        text = match.group(0)
        sentence_object = Sentence(text, None, i)
        sentences.append(sentence_object)
        i += 1
    return sentences


# define create_sentence function
def create_sentence(text, tag):
    text = text.strip()
    # split text into sentences
    sentences = text.split('.')
    # create a list of Sentence objects
    sentence_objects = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        if sentence:
            sentence_objects.append(Sentence(sentence, tag, i))
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


def remove_stopwords(sentence):
    words = word_tokenize(sentence)
    words = [word for word in words if word not in set(stopwords.words('english'))]
    sentence = ' '.join(words)
    return sentence


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

    summary_sentence_count = 10
    scores_s1[0] = 1000
    for i in range(1, summary_sentence_count):
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

    # for sen in sentences[:20]:
    #     print(sen.text)

    summary.sort(key=lambda x: x[0].order)

    # for i, j in summary:
    #     print("sentence: ", j, "TEXT: ", i.text)

    return summary


def get_rouge_score(candidate, reference):
    rouge = Rouge()
    scores = rouge.get_scores(candidate, reference)
    return scores

def main():
    # read a text file
    text = read_file('dataset/1/candidate.txt')
    reference = read_file('dataset/1/reference.txt')
    # split text into sentences
    sentences = split_html_text(text)
    print(sentences[1].named_entities)
    
    text_list = [i.text for i in sentences]
    text_list = lemmatizer(text_list)
    sentence_tfidf = get_score_based_on_tfidf(text_list)

    ne_list = [i.named_entities for i in sentences]
    ne_list = lemmatizer(ne_list)
    ne_score_tfidf = get_score_based_on_tfidf(ne_list)
    
    for i in range(len(sentences)):
        sentences[i].weight = sentence_tfidf[i] + 2 * ne_score_tfidf[i]
    
    summary = get_summary(sentences)

    # sen1 = """ The area of Lapland was split between two counties of the Swedish Realm from 1634 to 1809. The northern and western
    # areas were part of Västerbotten County, while the southern areas were part of Ostrobothnia
    # County. The northern and western areas were transferred in 1809 to Oulu County, which
    # became Oulu Province. Under the royalist constitution of Finland during the first half of 1918, Lapland was to
    # become a Grand Principality and part of the inheritance of the proposed king of Finland. Lapland Province was
    # separated from Oulu Province in 1938."""

    # sen1_sw = remove_stopwords(sen1)

    # sen2 = "The area of the Lapland region is 100 367 km which consists of 92 667 km of dry land 6 316 km fresh water and 1 383 km of sea water. The very first snowflakes fall to the ground in late August or early September over the higher peaks. After Finland made a separate peace with the Soviet Union in 1944 the Soviet Union demanded that Finland expel the German army from its soil "
    # sen2_sw = remove_stopwords(sen2)
    # print(get_rouge_score(sen1_sw, sen2_sw))

    # for i, j in summary:
    #     print("sentence: ", j, "TEXT: ", i.text)

    summary_as_text = '. '.join([i[0].text for i in summary])
    print(summary_as_text)
    print(get_rouge_score(summary_as_text, reference))

    print("***************")

    luhun1 = """
    From the outset of the COVID-19 pandemic, news coverage and policymaking have prominently featured concerns that government-mandated restrictions on economic activity and personal mobility might increase domestic violence (DV).1 This attention to DV is well-motivated because of its high social and economic costs (Garcia- Moreno & Watts, 2011) and because stress, economic disruption, and social isolation are established predictors of DV (Berg & Tertilt, 2012; Bright et al., 2020). Despite the initial set of papers yielding mixed results, the claim that shutdowns increase DV incidence has been presented as an established fact in media coverage and in political debates about pandemic restrictions (e.g., Biggs, 2020). This paper is motivated by the observation that the empirical studies finding increases in DV in US cities examine DV service calls as their exclusive (Leslie & Wilson, 2020; McCrary & Sanga, 2021; Nix & Richards, 2021) or primary (Hsu & Henke, 2021) outcome measure. Papers that examine DV crime rates are more likely to find decreases in DV, particularly when they account for seasonal variation using data from prior years (Abrams, 2021; Ashby, 2020a; Bullinger et al., 2021; Miller et al., 2020).2 However, because studies of the different police outcomes have differed in their geographic coverage, it is unclear if the divergence in estimates comes from systematic differences between the two types of police data or from geographic variation in the impact of shutdowns. We address this important question by studying the 18 large, urban US police departments, serving over 14 million people, for which we were able to obtain incident-level data on both DV calls for service and DV assault crimes. We find a decrease in DV assaults but an increase in DV calls during shutdowns. We also estimate models that account for the finding in the prior literature of an increase in DV calls during the period of voluntarily lower mobility that followed the nationwide emergency declaration but pre-ceded mandated shutdowns (e.g., McCrary & Sanga, 2021). When we estimate models that also control for the pre-shutdown emergency period, we find both DV assault crimes and DV calls are lower during shutdowns, relative to the immediately preceding period. We also find no evidence that intimate partner homicides or reports of intimate partner violence in the National Crime Victimization Survey increased during shutdown months; suicides, which have been linked to DV (Stevenson & Wolfers, 2006), were lower. These results fail to provide empirical support for claims that DV increased because of pandemic shutdowns, and instead suggest that violence may have decreased.
    """
    print("Luhun:")
    print(get_rouge_score(luhun1, reference))

    lsa1 = """
    From the outset of the COVID-19 pandemic, news coverage and policymaking have prominently featured concerns that government-mandated restrictions on economic activity and personal mobility might increase domestic violence (DV).1 This attention to DV is well-motivated because of its high social and economic costs (Garcia- Moreno & Watts, 2011) and because stress, economic disruption, and social isolation are established predictors of DV (Berg & Tertilt, 2012; Bright et al., 2020). Federal stimulus payments enacted in response to the pandemic also significantly lowered poverty rates, which may have reduced DV (Wheaton et al., 2021; Erten et al., 2022). As a result of these opposing factors, the effects of shutdowns on overall DV levels were theoretically ambiguous and likely to vary across populations. Determining the overall impact of shutdowns on DV requires careful empirical analysis, but results need to be produced and disseminated rapidly to contribute to ongoing debates about pandemic policy (Single Gonzalez et al., 2020). Because of this urgency, researchers from a variety of disciplines relied on readily available administrative data to assess DV incidence. Despite the initial set of papers yielding mixed results, the claim that shutdowns increase DV incidence has been presented as an established fact in media coverage and in political debates about pandemic restrictions (e.g., Biggs, 2020). This paper is motivated by the observation that the empirical studies finding increases in DV in US cities examine DV service calls as their exclusive (Leslie & Wilson, 2020; McCrary & Sanga, 2021; Nix & Richards, 2021) or primary (Hsu & Henke, 2021) outcome measure. Papers that examine DV crime rates are more likely to find decreases in DV, particularly when they account for seasonal variation using data from prior years (Abrams, 2021; Ashby, 2020a; Bullinger et al., 2021; Miller et al., 2020).2 However, because studies of the different police outcomes have differed in their geographic coverage, it is unclear if the divergence in estimates comes from systematic differences between the two types of police data or from geographic variation in the impact of shutdowns. We also estimate models that account for the finding in the prior literature of an increase in DV calls during the period of voluntarily lower mobility that followed the nationwide emergency declaration but pre-ceded mandated shutdowns (e.g., McCrary & Sanga, 2021). These results fail to provide empirical support for claims that DV increased because of pandemic shutdowns, and instead suggest that violence may have decreased.
    """
    print("LSA:")
    print(get_rouge_score(lsa1, reference))

    edmunson1 = """
    From the outset of the COVID-19 pandemic, news coverage and policymaking have prominently featured concerns that government-mandated restrictions on economic activity and personal mobility might increase domestic violence (DV).1 This attention to DV is well-motivated because of its high social and economic costs (Garcia- Moreno & Watts, 2011) and because stress, economic disruption, and social isolation are established predictors of DV (Berg & Tertilt, 2012; Bright et al., 2020). Nevertheless, shutdowns were unprecedented, and they could reduce DV in some households by lowering exposure to DV triggers such as infidelity and alcohol consumption outside the home (Nemeth et al., 2012), limiting contact between non-cohabiting and former couples (Ivandic et al., 2020), and even strengthening some relationships (Sachser et al., 2021). Furthermore, increased public and private funding to support DV victims and survivors, together with in- creased media attention devoted to DV, around the time shutdowns were imposed (Bright et al., 2020) could have reduced repeated violence and escalation. Federal stimulus payments enacted in response to the pandemic also significantly lowered poverty rates, which may have reduced DV (Wheaton et al., 2021; Erten et al., 2022). As a result of these opposing factors, the effects of shutdowns on overall DV levels were theoretically ambiguous and likely to vary across populations. Determining the overall impact of shutdowns on DV requires careful empirical analysis, but results need to be produced and disseminated rapidly to contribute to ongoing debates about pandemic policy (Single Gonzalez et al., 2020). Despite the initial set of papers yielding mixed results, the claim that shutdowns increase DV incidence has been presented as an established fact in media coverage and in political debates about pandemic restrictions (e.g., Biggs, 2020). This paper is motivated by the observation that the empirical studies finding increases in DV in US cities examine DV service calls as their exclusive (Leslie & Wilson, 2020; McCrary & Sanga, 2021; Nix & Richards, 2021) or primary (Hsu & Henke, 2021) outcome measure. We address this important question by studying the 18 large, urban US police departments, serving over 14 million people, for which we were able to obtain incident-level data on both DV calls for service and DV assault crimes. These results fail to provide empirical support for claims that DV increased because of pandemic shutdowns, and instead suggest that violence may have decreased.
    """
    print("Edmunson:")
    print(get_rouge_score(edmunson1, reference))

    textrank1 = """
    From the outset of the COVID-19 pandemic, news coverage and policymaking have prominently featured concerns that government-mandated restrictions on economic activity and personal mobility might increase domestic violence (DV).1 This attention to DV is well-motivated because of its high social and economic costs (Garcia- Moreno & Watts, 2011) and because stress, economic disruption, and social isolation are established predictors of DV (Berg & Tertilt, 2012; Bright et al., 2020). Nevertheless, shutdowns were unprecedented, and they could reduce DV in some households by lowering exposure to DV triggers such as infidelity and alcohol consumption outside the home (Nemeth et al., 2012), limiting contact between non-cohabiting and former couples (Ivandic et al., 2020), and even strengthening some relationships (Sachser et al., 2021). Furthermore, increased public and private funding to support DV victims and survivors, together with in- creased media attention devoted to DV, around the time shutdowns were imposed (Bright et al., 2020) could have reduced repeated violence and escalation. This paper is motivated by the observation that the empirical studies finding increases in DV in US cities examine DV service calls as their exclusive (Leslie & Wilson, 2020; McCrary & Sanga, 2021; Nix & Richards, 2021) or primary (Hsu & Henke, 2021) outcome measure. Papers that examine DV crime rates are more likely to find decreases in DV, particularly when they account for seasonal variation using data from prior years (Abrams, 2021; Ashby, 2020a; Bullinger et al., 2021; Miller et al., 2020).2 However, because studies of the different police outcomes have differed in their geographic coverage, it is unclear if the divergence in estimates comes from systematic differences between the two types of police data or from geographic variation in the impact of shutdowns. We address this important question by studying the 18 large, urban US police departments, serving over 14 million people, for which we were able to obtain incident-level data on both DV calls for service and DV assault crimes. We find a decrease in DV assaults but an increase in DV calls during shutdowns. We also estimate models that account for the finding in the prior literature of an increase in DV calls during the period of voluntarily lower mobility that followed the nationwide emergency declaration but pre-ceded mandated shutdowns (e.g., McCrary & Sanga, 2021). When we estimate models that also control for the pre-shutdown emergency period, we find both DV assault crimes and DV calls are lower during shutdowns, relative to the immediately preceding period. We also find no evidence that intimate partner homicides or reports of intimate partner violence in the National Crime Victimization Survey increased during shutdown months; suicides, which have been linked to DV (Stevenson & Wolfers, 2006), were lower.
    """
    print("Textrank:")
    print(get_rouge_score(textrank1, reference))

    lexrank1 = """
    From the outset of the COVID-19 pandemic, news coverage and policymaking have prominently featured concerns that government-mandated restrictions on economic activity and personal mobility might increase domestic violence (DV).1 This attention to DV is well-motivated because of its high social and economic costs (Garcia- Moreno & Watts, 2011) and because stress, economic disruption, and social isolation are established predictors of DV (Berg & Tertilt, 2012; Bright et al., 2020). Nevertheless, shutdowns were unprecedented, and they could reduce DV in some households by lowering exposure to DV triggers such as infidelity and alcohol consumption outside the home (Nemeth et al., 2012), limiting contact between non-cohabiting and former couples (Ivandic et al., 2020), and even strengthening some relationships (Sachser et al., 2021). Federal stimulus payments enacted in response to the pandemic also significantly lowered poverty rates, which may have reduced DV (Wheaton et al., 2021; Erten et al., 2022). As a result of these opposing factors, the effects of shutdowns on overall DV levels were theoretically ambiguous and likely to vary across populations. Because of this urgency, researchers from a variety of disciplines relied on readily available administrative data to assess DV incidence. Papers that examine DV crime rates are more likely to find decreases in DV, particularly when they account for seasonal variation using data from prior years (Abrams, 2021; Ashby, 2020a; Bullinger et al., 2021; Miller et al., 2020).2 However, because studies of the different police outcomes have differed in their geographic coverage, it is unclear if the divergence in estimates comes from systematic differences between the two types of police data or from geographic variation in the impact of shutdowns. We address this important question by studying the 18 large, urban US police departments, serving over 14 million people, for which we were able to obtain incident-level data on both DV calls for service and DV assault crimes. We find a decrease in DV assaults but an increase in DV calls during shutdowns. We also estimate models that account for the finding in the prior literature of an increase in DV calls during the period of voluntarily lower mobility that followed the nationwide emergency declaration but pre-ceded mandated shutdowns (e.g., McCrary & Sanga, 2021). These results fail to provide empirical support for claims that DV increased because of pandemic shutdowns, and instead suggest that violence may have decreased.
    """
    print("LexRank:")
    print(get_rouge_score(lexrank1, reference))

def init():
    text_list = ["this is a test", "test is a test", "this is a test"]
    get_score_based_on_tfidf(text_list)

    

if __name__ == "__main__":
    # main()
    import sys
    # read input arguments
    command = sys.argv[1]
    if command == "--init":
        init()
    elif command == "--main":
        main()
    else:
        print("Invalid command")