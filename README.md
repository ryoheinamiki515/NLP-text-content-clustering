# Text Document Clustering

Using NLP to cluster text together. Uses Wikipedia articles in example

## Process
Here are the steps used to cluster the text:
###### Preprocessing
1. Collect raw text
2. Cast text to lowercase
3. Remove symbols: !\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n,—' (configurable in src --> data --> utils.py)
4. Tokenize
5. Remove stop words using NLTK defualt English stop words (configurable in src --> data --> utils.py)
6. Lemmatize
7. Stem
8. Join back into full text
9. Repeat for all text
###### Model Creation and Clustering
10. Generate the tf-idf table using scikit-learn tf-idf
11. Create a 2D visualization of the tf-idf data using scikit-learn learn TSNE
12. Prompt user for number of clusters for KMeans
13. Apply KMeans clustering to the data
14. Print the cluster assignments as well as the most prominent words in each cluster (as defined by cluster center values)
15. Show the final labeling on the visualization

