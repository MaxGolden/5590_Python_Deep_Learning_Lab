from bs4 import BeautifulSoup
import urllib.request
from nltk.stem import WordNetLemmatizer
import nltk
import sklearn
from sklearn.datasets import fetch_20newsgroups

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk import wordpunct_tokenize, pos_tag, ne_chunk
import nltk.chunk
from nltk.util import ngrams
import string
import re
import nltk
from nltk.corpus import stopwords
import string



def remove_punct(text):
    words = text.split()
    return " ".join([w.rstrip(string.punctuation) for w in words])


stop_words = set(stopwords.words('english'))
file = open("nlp_input.txt").read()
stokens = nltk.sent_tokenize(file)
#removes punctuation
file = remove_punct(file)
files =""

#removes ariticles and stop words
for w in file.split():
    if w not in stop_words and w not in string.punctuation:
        files = files+(w)+" "
file = files

#tokenizes words

wtokens = nltk.word_tokenize(file)

#lemmatizes each word
#******* word tokenization and lemmatization*************
lemmatizer = WordNetLemmatizer()

wtokensClean = list()
for t in wtokens:
    if t not in string.punctuation:
        if t not in stop_words:
            wtokensClean.append( lemmatizer.lemmatize(t.lower()))

wtokens = wtokensClean

print("********shows Trigram of word tokens*************")
trigrams = []
trigrams = list(ngrams(wtokens, 3))
print(trigrams)

print("********extract top ten trigrams*************")


#creates a list of sets
triSets = []
for tri in trigrams:
    my_set = set()
    my_set.add(tri[0])
    my_set.add(tri[1])
    my_set.add(tri[2])
    triSets.append(my_set)

#print (triSets)

#prints the repeated trigrams represented as sets
print("list of all trigram sets that repeate")
repeating_trigrams = list()
mapCountTrigrams = {}
for n1 in range(0,len(triSets)):
    for n2 in range(n1+1,len(triSets)):
        if triSets[n1]==triSets[n2] and n1 != n2:
            if triSets[n1] not in repeating_trigrams:
                repeating_trigrams.append(triSets[n1])


#prints the list of repeating trigrams
# no sets of trigrams repeat more than once, therefor
# all trigrams are equal
print (repeating_trigrams)
#since they all contain at most 2 I just took the first 10
repeating_trigrams_top_10 = repeating_trigrams[:10]
print ("top ten:\n",repeating_trigrams_top_10)

result = ""
for s in stokens:
    swtokens = nltk.word_tokenize(s)
    swtokensClean = []
    for swt in swtokens:
        swtokensClean.append(lemmatizer.lemmatize(swt.lower()))
    for tri in repeating_trigrams_top_10:
       triList = list(tri)
       if triList[0] in swtokensClean:
            if triList[1] in swtokensClean:
                if triList[2] in swtokensClean:
                    result += s
                    break

print ("******************result:*******************")
print (result)





#
#
#
# print("********shows POS of word tokens*************")
# print(nltk.pos_tag(wtokens), "\n")
#
#
#
#
# print("********shows Stem of word tokens*************")
# n =0
# for t in wtokens:
#     n+=1
#     pStemmer = PorterStemmer()
#     print(pStemmer.stem(t))
#     lStemmer = LancasterStemmer()
#     print(lStemmer.stem(t))
#     sStemmer = SnowballStemmer('english')
#     print(sStemmer.stem(t))
#     if n > 20:
#         break
#
#
# print("********shows Lemmatization of word tokens*************")
#
# n =0
# lemmatizer = WordNetLemmatizer()
# for t in wtokens:
#     n+=1
#
#     print(t, ">>>>>" , lemmatizer.lemmatize(t))
#     if n > 100:
#         break
#
#
#
#
# print("********shows NER of word tokens*************")
# n = 0
# for s in stokens:
#     n+=1
#     print(ne_chunk(pos_tag(wordpunct_tokenize(s))), "\n")
#     if n > 6:
#         break
#
#
#

