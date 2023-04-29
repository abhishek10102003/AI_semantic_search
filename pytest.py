import nltk
import spacy
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from enchant.checker import SpellChecker
from gensim.models import LdaModel
from gensim.corpora.dictionary import Dictionary
import re
import polyglot
from polyglot.detect import Detector
from polyglot.text import Text
from pyxdameraulevenshtein import damerau_levenshtein_distance

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Define paths to documents and stop words file
DOCUMENTS_PATH = "documents.txt"
STOPWORDS_PATH = "stopwords.txt"

# Read documents from file
with open(DOCUMENTS_PATH, "r") as f:
    documents = f.readlines()

# Read stop words from file
with open(STOPWORDS_PATH, "r") as f:
    stop_words = set([line.strip() for line in f.readlines()])

# Tokenize and preprocess documents
lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer("english")
preprocessed_docs = []
for doc in documents:
    # Tokenize words
    words = word_tokenize(doc)
    # Remove stop words and lemmatize or stem remaining words
    words = [word.lower() for word in words if not word in stop_words]
    words = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words]
    words = [stemmer.stem(word) for word in words]
    # Join words back into a single string
    preprocessed_docs.append(' '.join(words))

# Build TF-IDF vectorizer and calculate document similarity matrix
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_docs)
similarity_matrix = cosine_similarity(tfidf_matrix)

# Load spaCy model for NER
nlp = spacy.load("en_core_web_sm")

# Build dictionary and corpus for LDA topic modeling
dictionary = Dictionary([doc.split() for doc in preprocessed_docs])
corpus = [dictionary.doc2bow(doc.split()) for doc in preprocessed_docs]

# Train LDA model and get topic probabilities for each document
lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=5)
topic_probabilities = [lda_model.get_document_topics(corpus[i]) for i in range(len(corpus))]

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

# Function to perform semantic search
def semantic_search(query, documents, similarity_matrix, vectorizer, include_score=True, language='en', max_distance=1):
    # Detect query language and translate if necessary
    if language != 'en':
        text = Text(query, hint_language_code=language)
        query = text.translate('en').string.lower()
    # Preprocess query
    words = word_tokenize(query)
    # Check spelling and correct if necessary
    chkr = SpellChecker("en_US")
    for word in words:
        chkr.set_text(word)
        if not chkr.check():
            for suggestion in chkr.suggest():
                if suggestion in wordnet.words():
                    query = query.replace(word, suggestion)
                    break
    # Remove stop words and lemmatize
    words = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words if not word in stop_words]
    # Expand query using synonyms
    expanded_query = []
