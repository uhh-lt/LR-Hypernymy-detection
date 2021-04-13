# LR-Hypernymy-detection
Low Resource Hypernymy Detection

## Directory Structure
The `data/resources` folder maintains the datasets used in the experiments. 
* The `original` folder has the English (en) EVALution, ROOT9 and bless datasets. Here, each sub-folder contains both the full dataset and the dataset with disjoint vocabulary (to analyse lexical memorization).
* The `transformed` folder has the translated versions, into Hindi, Bengali and Amharic, of the Baroni, EVALution, ROOT9 and bless datasets. Here, each sub-folder contains both the full dataset and the dataset with disjoint vocabulary (to analyse lexical memorization).

## Embeddings
The network embeddings, deepwalk, MNMF and Role2Vec, computed over the Distributional Thesaurus (DT) have been prepared using the [karateclub](https://github.com/benedekrozemberczki/karateclub) package. These representations have been maintained [here](http://ltdata1.informatik.uni-hamburg.de/LR-Hypernymy-detection/).
