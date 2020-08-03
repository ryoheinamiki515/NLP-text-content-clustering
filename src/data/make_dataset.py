from src.data import utils
import wikipedia
import pandas as pd


def from_folder(folder_path):
    text_data, labels = utils.get_directory_text(folder_path)
    return text_data, labels


def from_random_wiki(num_articles, summary=False):
    # generate random articles
    some_wikipedia_articles = wikipedia.random(num_articles)
    text_data, labels = utils.get_from_wiki(some_wikipedia_articles, summary)
    return text_data, labels


def from_wiki(article_names, summary=False):
    text_data, labels = utils.get_from_wiki(article_names, summary)
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
