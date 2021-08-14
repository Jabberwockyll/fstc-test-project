import os
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import yaml

from fewshot.data.loaders import load_or_cache_data

from fewshot.embeddings.word_embeddings import (
    load_word_vector_model,
    get_topk_w2v_vectors,
    get_word_embeddings
)
from fewshot.embeddings.sentence_embeddings import (
    load_transformer_model_and_tokenizer,
    get_sentence_embeddings,
)

from fewshot.models.on_the_fly import OLS_with_l2_regularization

from fewshot.eval import predict_and_score, simple_topk_accuracy

from fewshot.utils import (
    fewshot_filename,
    torch_load,
    torch_save,
    pickle_load,
    pickle_save,
    to_tensor,
    to_list,
)

scores_file = "../results/accuracy.json"
params = yaml.safe_load(open("params.yaml"))["prepare"]

nltk.download('stopwords')
nltk.download('punkt')

reddit_data = load_or_cache_data("data/reddit", "reddit")

model, tokenizer = load_transformer_model_and_tokenizer("deepset/sentence_bert")
w2v_model = load_word_vector_model(small=True) #, cache_dir="data"

# Get w2v embeddings and the vocabulary of most frequent words
w2v_vectors, vocabulary = get_topk_w2v_vectors(w2v_model, k=100000, return_word_list=True)
vocabulary_filename = fewshot_filename("my_data/w2v", "w2v_vocab_sbert_embeddings.pt")

try:
    sbert_vectors = torch_load(vocabulary_filename)['embeddings']
except:
    # get sbert embeddings for each word in vocabulary
    sbert_vectors = get_sentence_embeddings(vocabulary, 
                                            model, 
                                            tokenizer, 
                                            output_filename=vocabulary_filename)


baseline_reddit_acc, reddit_preds = predict_and_score(reddit_data, linear_maps=None, return_predictions=True)
baseline_reddit_acc_top3 = simple_topk_accuracy(reddit_data.labels, reddit_preds)

# if above cells have already been run, no need to recompute sbert embeddings
try:    
    sbert_vectors_20k = sbert_vectors[:params["kwords"]]
    w2v_vectors_20k = to_tensor(w2v_vectors[:params["kwords"]])

# otherwise, compute sbert embeddings from the top 20k w2v words
except:
    vocabulary_filename = fewshot_filename("my_data/w2v", "w2v_vocab20k_sbert_embeddings.pt")
    try:
        # Load 20k sbert vectors if we've already created them
        sbert_vectors_20k = torch_load(vocabulary_filename)['embeddings']
    except:
        # get sbert embeddings for each word in vocabulary
        w2v_vectors_20k, vocabulary = get_topk_w2v_vectors(w2v_model, k=params["kwords"], return_word_list=True)
        sbert_vectors_20k = get_sentence_embeddings(vocabulary, 
                                                    model, 
                                                    tokenizer, 
                                                    output_filename=vocabulary_filename)

Zmap_w2v = OLS_with_l2_regularization(sbert_vectors_20k, to_tensor(w2v_vectors[:params["kwords"]]), 
                                                                        alpha=params["alpha"])
# We save this Zmap for use in our few-shot application
torch_save(Zmap_w2v, fewshot_filename("data/maps", "Zmap_20k_w2v_words_alpha0.pt"))

zw2v_reddit_acc, reddit_preds = predict_and_score(reddit_data, linear_maps=[Zmap_w2v], return_predictions=True)
zw2v_reddit_acc_top3 = simple_topk_accuracy(reddit_data.labels, reddit_preds)

with open(scores_file, "w") as fd:
    json.dump({"baseline_acc": baseline_reddit_acc, 
        "basline_acc_top3": baseline_reddit_acc_top3,
        "zw2v_acc": zw2v_reddit_acc,
        "zw2v_acc_top3": zw2v_reddit_acc_top3}, fd, indent=4)