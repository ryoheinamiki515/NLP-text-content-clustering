from src.data import utils
import wikipedia
import pandas as pd
import os
import sys

def exception_handler(exception_type, exception, traceback):
    # All your trace are belong to us!
    # your format
    print("%s: %s" % (exception_type.__name__, exception))


sys.excepthook = exception_handler
wikipedia.set_lang("en")


def from_folder(folder_path):
    (_, _, filenames) = next(os.walk(folder_path))
    text_data = []
    labels = []
    for filename in filenames:
        with open(os.path.join(folder_path, filename), "r") as f:
            text_data.append(" ".join(f.readlines()))
        labels.append(filename[:-4])
    print("Data Labels:", labels)
    return text_data, labels


def from_random_wiki(num_articles, summary=False):
    # generate random articles
    some_wikipedia_articles = wikipedia.random(num_articles)
    text_data = []
    labels = []
    for article in some_wikipedia_articles:
        while True:
            try:
                if summary:
                    text = wikipedia.page(article).summary
                else:
                    text = wikipedia.page(article).content
                if len(text) < 1000:
                    article = wikipedia.random(1)
                    continue
                text_data.append(text)
                break
            except wikipedia.DisambiguationError:
                article = wikipedia.random(1)
        labels.append(article)
    print("Data Labels:", labels)
    return text_data, labels


def from_wiki(article_names, summary=False):
    text_data = []
    labels = []
    for article in article_names:
        if summary:
            text_data.append(wikipedia.page(article).summary)
        else:
            text_data.append(wikipedia.page(article).content)
        labels.append(article)
    print("Data Labels:", labels)
    return text_data, labels


def clean_data(text_data):
    original_words = []
    changed_words = []
    data = []
    for text in text_data:
        cleaned_text, vocabulary, stemmed_lemmatized_words = utils.clean_text(text)
        original_words.extend(vocabulary)
        changed_words.extend(stemmed_lemmatized_words)
        data.append(cleaned_text)
    changed_to_original = pd.Series(original_words, index=changed_words)
    return data, changed_to_original

