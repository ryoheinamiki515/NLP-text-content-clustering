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
5. Lemmatize
6. Stem
7. Join back into full text
8. Repeat for all text
###### Model Creation and Clustering
9. Generate the tf-idf table using sci-kit learn tf-idf
10. Create a 2D visualization of the tf-idf data using sci-kit learn TSNE
11. Prompt user for number of clusters for KMeans
12. Apply KMeans clustering to the data
13. Print the cluster assignments as well as the most prominent words in each cluster (as defined by cluster center values)
14. Show the final labeling on the visualization

