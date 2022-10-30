# LR-Hypernymy-detection

This is the official implementation of [Hypernymy Detection for Low-Resource Languages: A Study for Hindi, Bengali, and Amharic](https://dl.acm.org/doi/10.1145/3490389).

Numerous attempts for hypernymy relation (e.g. dog 'is-a' animal) detection have been made for resourceful languages like English, whereas efforts made for low-resource languages are scarce primarily due to lack of gold standard datasets and suitable distributional models. Therefore, we introduce four gold standard datasets for hypernymy detection for each of the two languages, namely Hindi and Bengali, and two gold standard datasets for Amharic. Another major contribution of this work is to prepare distributional thesaurus (DT) embeddings for all three languages using three different network embedding methods (DeepWalk, role2vec, M-NMF) for the first time on these languages and to show their utility for hypernymy detection. Posing this problem as a binary classification task, we experiment with supervised classifiers like Support Vector Machine, Random Forest, etc., and show that these classifiers fed with DT embeddings can obtain promising results while evaluated against proposed gold standard datasets, specifically in an experimental setup that counteracts lexical memorization. We further incorporate DT embeddings and pre-trained fastText embeddings together using two different hybrid approaches, both of which produce an excellent performance. Additionally, we validate our methodology on gold-standard English datasets as well, where we reach a comparable performance to state-of-the-art models for hypernymy detection.

## Directory Structure
The `data/resources` folder maintains the datasets used in the experiments. 
* The `original` folder has the English (en) EVALution, ROOT9 and bless datasets. Here, each sub-folder contains both the full dataset and the dataset with disjoint vocabulary (to analyse lexical memorization).
* The `transformed` folder has the translated versions, into Hindi, Bengali and Amharic, of the Baroni, EVALution, ROOT9 and bless datasets. Here, each sub-folder contains both the full dataset and the dataset with disjoint vocabulary (to analyse lexical memorization).

The classifiers used in our study are available in `src` directory.

## Embeddings
The network embeddings, deepwalk, MNMF and Role2Vec, computed over the Distributional Thesaurus (DT) have been prepared using the [karateclub](https://github.com/benedekrozemberczki/karateclub) package. These representations have been maintained [here](http://ltdata1.informatik.uni-hamburg.de/LR-Hypernymy-detection/).

If you use these resources and methods then please cite the following paper:

```
@article{10.1145/3490389,
author = {Jana, Abhik and Venkatesh, Gopalakrishnan and Yimam, Seid Muhie and Biemann, Chris},
title = {Hypernymy Detection for Low-Resource Languages: A Study for Hindi, Bengali, and Amharic},
year = {2022},
issue_date = {July 2022},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {21},
number = {4},
issn = {2375-4699},
url = {https://doi.org/10.1145/3490389},
doi = {10.1145/3490389},
journal = {ACM Trans. Asian Low-Resour. Lang. Inf. Process.},
month = {mar},
articleno = {67},
numpages = {21}
}
```
