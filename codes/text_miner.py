import re
class Sentence:
    def __init__(self, text, tag):
        self.text = text
        # tag: heading, subheading, paragraph
        self.tag = tag
        self.POS = 0
    
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


# define sent_tokenize function
# this function will use html tags to split text into sentences
# and the sentences are separated by period.
def sent_tokenize(text):
    # split text into sentences
    sentences = []
    # split text into paragraphs based on html tags
    # regext to find text between <h1> and </h1>
    pattern = re.compile(r'<h1>(.*?)<\/h1>')
    # find all matches
    matches = pattern.finditer(text)
    
    # loop through matches
    for match in matches:        
        # get the text between <h1> and </h1>
        paragraph = match.group(1)
        # split paragraph into sentences
        paragraph_sentences = paragraph.split('.')
        # loop through sentences
        for sentence in paragraph_sentences:
            # create a sentence object
            sentence_obj = Sentence(sentence, 'heading')
            # append sentence object to sentences
            sentences.append(sentence_obj)
    
    # print all the sentences
    for sentence in sentences:
        print(sentence.text)

    return sentences
    

if __name__ == "__main__":
    # read a text file
    text = read_file('dataset/finland.html')
    # split text into sentences
    sentences = split_text(text)
