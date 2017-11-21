from naive_bayes import naive_bayes
from preprocessing import *

# dataset = read_and_filter_dataset(preprocess=True,nrows=5000,save_csv=True)
dataset = read_and_filter_dataset(use_preprocessed=True)

plot_statistics(dataset)

naive_bayes(dataset)
