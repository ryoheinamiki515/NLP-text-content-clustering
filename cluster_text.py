import test_environment
from src.data import make_dataset
from src.visualization import visualize
from src.models import train_model
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CS440 MP1 Search')

    parser.add_argument('--source', dest="source", type=str, required=True,
                        choices=["wiki", "rand_wiki", "from_path"],
                        help='source of text data')
    parser.add_argument('--article_names', dest="article_names", type=str, default=[], nargs='+',
                        help='list of Wikipedia articles to analyze')
    parser.add_argument('--summary', dest="summary", type=bool, default=False,
                        help='Choose whether to use the whole wiki article or just the summary')
    parser.add_argument('--num_articles', dest="num_articles", type=int, default=5,
                        help='Number of random Wikipedia articles to generate')
    parser.add_argument('--path', dest="path", type=str, default="",
                        help='Path of input text folder')

    args = parser.parse_args()

    source = args.source
    article_names = args.article_names
    summary = args.summary
    num_articles = args.num_articles
    path = args.path

    test_environment.main()

    if source == "wiki":
        text_data, labels = make_dataset.from_wiki(article_names, summary)
    elif source == "rand_wiki":
        text_data, labels = make_dataset.from_random_wiki(num_articles, summary)
    else:
        text_data, labels = make_dataset.from_folder(path)

    clean_data, changed_to_original = make_dataset.clean_data(text_data)
    X, X_embedded, feature_names, n_clusters = visualize.visualize_initial(clean_data)
    model = train_model.create_model(X, n_clusters)
    visualize.print_results(model, labels, changed_to_original, feature_names)
    visualize.visualize_final(model, X_embedded, labels)




