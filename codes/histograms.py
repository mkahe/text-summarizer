import pandas as pd
# natural language processing: n-gram ranking
import re
import unicodedata
import nltk
from nltk.corpus import stopwords
# add appropriate words that will be ignored in the analysis
ADDITIONAL_STOPWORDS = ['oulu']

import matplotlib.pyplot as plt



def basic_clean(text):
  """
  A simple function to clean up the data. All the words that
  are not designated as a stop word is then lemmatized after
  encoding and basic regex parsing are performed.
  """
  wnl = nltk.stem.WordNetLemmatizer()
  stopwords = nltk.corpus.stopwords.words('english') + ADDITIONAL_STOPWORDS
  text = (unicodedata.normalize('NFKD', text)
    .encode('ascii', 'ignore')
    .decode('utf-8', 'ignore')
    .lower())
  words = re.sub(r'[^\w\s]', '', text).split()
  return [wnl.lemmatize(word) for word in words if word not in stopwords]


df = ["Oulu, a Finnish city and the provincial capital of North Ostrobothnia."
    , "It is located at the mouth of the Oulu River of the Perä Sea on the coast in the province of North Ostrobothnia."
    , "Oulu is founded in 1605, Oulu is the oldest city in Northern Finland and in terms of population the fifth largest city and the fourth largest urban area."
    , """The population of Oulu was 209,934 inhabitants June 30, 2022 and the areas 3,817.69 square
    kilometers, of which 2,972, 45 is land, 80.32 is inland water and the remaining 764.92 square kilometers is sea."""
    , "Oulu is Finland's largest coastal municipality by area."]
words = basic_clean(' '.join(df))
print(words)

unigrams_series = (pd.Series(nltk.ngrams(words, 1)).value_counts())[:20]
bigrams_series = (pd.Series(nltk.ngrams(words, 2)).value_counts())[:20]

unigrams_series.sort_values().plot.barh(color='blue', width=.9, figsize=(12, 8))
plt.title('Most Frequently Occuring Uigrams')
plt.ylabel('Unigram')
plt.xlabel('# of Occurances')
plt.show()