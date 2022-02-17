import json
import pickle
import makedb as makedb
from pathlib import Path
import numpy as np
import pandas as pd
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Clean Data
def preProcess(s):
    ps = PorterStemmer()
    s = word_tokenize(s)
    stopwords_set = set(stopwords.words())
    stop_dict = {s: 1 for s in stopwords_set}
    s = [w for w in s if w not in stop_dict]
    s = [ps.stem(w) for w in s]
    s = ' '.join(s)
    s = s.translate(str.maketrans('', '', string.punctuation + u'\xa0'))
    s = s.translate(str.maketrans(string.whitespace, ' ' * len(string.whitespace), ''))
    return s
data = []

if not Path('database.json').exists():
    makedb.make_json('lyrics-data.csv', 'database.json')

#First time search, dump on pickle, else load pickle
if not Path('cleaned.pickle').exists():
    document = pd.read_json('database.json')
    s = document.agg(lambda x: list(x))['data']
    for index, subset in s.iteritems():
        data.append((subset['artist'], subset['song'], preProcess(subset['lyric'])))
    with open('cleaned.pickle', 'wb') as fin:
        pickle.dump(data, fin)
else:
    with open('cleaned.pickle', 'rb') as fin:
        data = pickle.load(fin)

data = pd.DataFrame(data, columns=['Artist','Song Name','Lyrics'])
print(data)

#First time create vocabulary list (tf) else load pickle
if not Path('vectorizer.pickle').exists():
    vectorizer = CountVectorizer(ngram_range=(1, 3))
    X = vectorizer.fit_transform(data['Lyrics'])
    with open('vectorizer.pickle', 'wb') as fin:
        pickle.dump(vectorizer, fin)
    print(X.shape)
else:
    pass

#First time create vocabulary list (tf-idf) else load pickle
if not Path('tfidfvectorizer.pickle').exists():
    tfidfvectorizer = TfidfVectorizer(ngram_range=(1, 3))
    X = tfidfvectorizer.fit_transform(data['Lyrics'])
    with open('tfidfvectorizer.pickle', 'wb') as fin:
        pickle.dump(tfidfvectorizer, fin)
    print(X.shape)
else:
    pass

def exactly_match_artist(input):
    match = []
    input = input.lower()
    input = input.replace(" ","-")
    for i,index in enumerate(data['Artist']):
        if index == input:
            match.append(data['Song Name'][i])
    return sorted(match)

def exactly_match_song_name(input):
    match = []
    input = input.lower()
    input = input.replace(" ","-")
    input = input.replace("'", "")
    for i,index in enumerate(data['Song Name']):
        index = index.lower()
        index = index.replace(" ","-")
        index = index.replace("'","")
        if index == input:
            match.append((data['Artist'][i],data['Lyrics'][i]))
    return match

#tf only
def tf_score(input):
    if exactly_match_artist(input):
        jsonresult = []
        match = exactly_match_artist(input)
        for item in match:
            jsonresult.append({"Song Name" : item})
        return json.dumps(jsonresult, indent=4)
    elif exactly_match_song_name(input):
        jsonresult = []
        match = exactly_match_song_name(input)
        for i,index in enumerate(match):
            jsonresult.append({"Artist": index[0], "Lyrics": index[1]})
        return json.dumps(jsonresult, indent=4)
    else:
        with open('vectorizer.pickle', 'rb') as fin:
            vectorizer = pickle.load(fin)
        X = vectorizer.transform(data['Lyrics'])
        X.data = np.log10(X.data + 1)
        query = vectorizer.transform([preProcess(input)])
        result = cosine_similarity(X,query).reshape((-1,))
        jsonresult = []
        for i,index in enumerate(result.argsort()[-10:][::-1]):
            jsonresult.append({'rank': i+1, 'artist': data['Artist'][index], 'song': data['Song Name'][index], 'lyric': input})
            print(str(i+1), data['Song Name'][index],"--",data['Artist'][index],"--",result[index])
        with open('top10tf.json', 'w', encoding='utf-8') as jsonf:
            jsonf.write(json.dumps(jsonresult, indent=4))
        return json.dumps(jsonresult, indent=4)

def tfidf_score(input):
    if exactly_match_artist(input):
        jsonresult = []
        match = exactly_match_artist(input)
        for item in match:
            jsonresult.append({"Song Name": item})
        return json.dumps(jsonresult, indent=4)
    elif exactly_match_song_name(input):
        jsonresult = []
        match = exactly_match_song_name(input)
        for i, index in enumerate(match):
            jsonresult.append({"Artist": index[0], "Lyrics": index[1]})
        return json.dumps(jsonresult, indent=4)
    else:
        with open('tfidfvectorizer.pickle', 'rb') as fin:
            tfidfvectorizer = pickle.load(fin)
        X = tfidfvectorizer.transform(data['Lyrics'])
        query = tfidfvectorizer.transform([preProcess(input)])
        result = cosine_similarity(X, query).reshape((-1,))
        jsonresult = []
        for i, index in enumerate(result.argsort()[-10:][::-1]):
            jsonresult.append({'rank': i + 1, 'artist': data['Artist'][index], 'song': data['Song Name'][index], 'lyric': input})
            print(str(i + 1), data['Song Name'][index], "--", data['Artist'][index], "--", result[index])
        with open('top10tfidf.json', 'w', encoding='utf-8') as jsonf:
            jsonf.write(json.dumps(jsonresult, indent=4))
        return json.dumps(jsonresult, indent=4)

def bm25_score(input):
    if exactly_match_artist(input):
        jsonresult = []
        match = exactly_match_artist(input)
        for item in match:
            jsonresult.append({"Song Name": item})
        return json.dumps(jsonresult, indent=4)
    elif exactly_match_song_name(input):
        jsonresult = []
        match = exactly_match_song_name(input)
        for i, index in enumerate(match):
            jsonresult.append({"Artist": index[0], "Lyrics": index[1]})
        return json.dumps(jsonresult, indent=4)
    else:
        desc = []
        input = preProcess(input)
        cleaned_description = pickle.load(open('cleaned.pickle', 'rb'))
        for i,index in enumerate(cleaned_description):
            desc.append(index[2])
        bm25 = BM25()
        bm25.fit(desc)
        Y = bm25.transform(input, desc)
        jsonresult = []
        for i, index in enumerate(Y.argsort()[-10:][::-1]):
            jsonresult.append({'rank': i + 1, 'artist': data['Artist'][index], 'song': data['Song Name'][index], 'lyric': input})
            print(str(i + 1), data['Song Name'][index], "--", data['Artist'][index], "--", Y[index])
        with open('top10bm25.json', 'w', encoding='utf-8') as jsonf:
            jsonf.write(json.dumps(jsonresult, indent=4))
        return json.dumps(jsonresult, indent=4)

class BM25(object):
    def __init__(self, b=0.75, k1=1.6):
        # self.vectorizer = pickle.load(open('tfidfvectorizer.pickle', 'rb'))
        self.vectorizer = TfidfVectorizer(ngram_range=(1,3),norm=None, smooth_idf=False)
        self.b = b
        self.k1 = k1
    def fit(self, X):
        """ Fit IDF to documents X """
        self.vectorizer.fit(X)
        y = super(TfidfVectorizer, self.vectorizer).transform(X)
        self.avdl = y.sum(1).mean()
    def transform(self, q, X):
        """ Calculate BM25 between query q and documents X """
        b, k1, avdl = self.b, self.k1, self.avdl
        # apply CountVectorizer
        X = super(TfidfVectorizer, self.vectorizer).transform(X)
        len_X = X.sum(1).A1
        q, = super(TfidfVectorizer, self.vectorizer).transform([q])
        assert sparse.isspmatrix_csr(q)
        # convert to csc for better column slicing
        X = X.tocsc()[:, q.indices]
        denom = X + (k1 * (1 - b + b * len_X / avdl))[:, None]
        # idf(t) = log [ n / df(t) ] + 1 in sklearn, so it need to be coneverted
        # to idf(t) = log [ n / df(t) ] with minus 1
        idf = self.vectorizer._tfidf.idf_[None, q.indices] - 1.
        numer = X.multiply(np.broadcast_to(idf, X.shape)) * (k1 + 1)
        return (numer / denom).sum(1).A1


