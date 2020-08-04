from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

SYMBOLS = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n,—'"
STOP_WORDS = stopwords.words('english')
LEMMATIZER = WordNetLemmatizer()
STEMMER = SnowballStemmer("english")


def remove_symbols(text):
    for symbol in SYMBOLS:
        text = text.replace(symbol, ' ')
    return text


def remove_stop_words(vocabulary):
    valid_words = [x for x in vocabulary if x not in STOP_WORDS and x.isalpha()]
    return valid_words


def stem_and_lemmatize(vocabulary):
    stemmed_lemmatized_words = [STEMMER.stem(LEMMATIZER.lemmatize(x)) for x in vocabulary]
    return stemmed_lemmatized_words


def clean_text(text):
    text = text.lower()
    text = remove_symbols(text)
    vocabulary = word_tokenize(text)
    vocabulary = remove_stop_words(vocabulary)
    stemmed_lemmatized_words = stem_and_lemmatize(vocabulary)
    cleaned_text = " ".join(stemmed_lemmatized_words)
    return cleaned_text, vocabulary, stemmed_lemmatized_words
