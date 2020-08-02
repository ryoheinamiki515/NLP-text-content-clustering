import os
import wikipedia
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

SUMMARY_PATH = "./data/wiki_data/summary/"
CONTENT_PATH = "./data/wiki_data/content/"
SYMBOLS = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n,—'"
STOP_WORDS = stopwords.words('english')
LEMMATIZER = WordNetLemmatizer()
STEMMER = SnowballStemmer("english")


def get_directory_text(directory):
    (_, _, filenames) = next(os.walk(directory))
    text_data = []
    labels = []
    for filename in filenames:
        with open(directory + filename, "r") as f:
            text_data.append(" ".join(f.readlines()))
        labels.append(filename[:-4])
    return text_data, labels


def get_from_wiki(wikipedia_articles, summary):
    # write to the data sources
    print(wikipedia_articles)
    if summary:
        data_path = SUMMARY_PATH
    else:
        data_path = CONTENT_PATH

    for article in wikipedia_articles:
        file_topic = "_".join(article.split())
        # Use summary of wikipedia articles as the data source
        if not os.path.exists(data_path + f"{file_topic}.txt"):
            with open(data_path + f"{file_topic}.txt", "w") as f:
                if summary:
                    f.write(wikipedia.page(article).summary)
                else:
                    f.write(wikipedia.page(article).content)
    return get_directory_text(data_path)


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
