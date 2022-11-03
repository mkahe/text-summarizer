from rake_nltk import Rake

# Uses stopwords for english from NLTK, and all puntuation characters by
# default
r = Rake()

# mytext = '''
# Black-on-black ware is a 20th- and 21st-century pottery tradition developed by the Puebloan Native American ceramic artists in Northern New Mexico. Traditional reduction-fired blackware has been made for centuries by pueblo artists. Black-on-black ware of the past century is produced with a smooth surface, with the designs applied through selective burnishing or the application of refractory slip. Another style involves carving or incising designs and selectively polishing the raised areas. For generations several families from Kha'po Owingeh and P'ohwhóge Owingeh pueblos have been making black-on-black ware with the techniques passed down from matriarch potters. Artists from other pueblos have also produced black-on-black ware. Several contemporary artists have created works honoring the pottery of their ancestors.
# '''

mytext = """The area of Lapland was split between two counties of the Swedish Realm from 1634 to 1809. The northern and western
    areas were part of Västerbotten County, while the southern areas were part of Ostrobothnia
    County. The northern and western areas were transferred in 1809 to Oulu County, which
    became Oulu Province. Under the royalist constitution of Finland during the first half of 1918, Lapland was to
    become a Grand Principality and part of the inheritance of the proposed king of Finland. Lapland Province was
    separated from Oulu Province in 1938."""

# Extraction given the text.
r.extract_keywords_from_text(mytext)

# Extraction given the list of strings where each string is a sentence.
#r.extract_keywords_from_sentences(<list of sentences>)

# To get keyword phrases ranked highest to lowest.
print(r.get_ranked_phrases())

# To get keyword phrases ranked highest to lowest with scores.
print(r.get_ranked_phrases_with_scores())