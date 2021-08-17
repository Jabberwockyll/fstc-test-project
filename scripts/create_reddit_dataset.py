import string
from collections import Counter
import pickle

from datasets import load_dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk import FreqDist, word_tokenize
import nltk
from nltk.corpus import stopwords
import yaml

from fewshot.data.loaders import _create_dataset_from_df
from fewshot.utils import pickle_save

DATADIR = f"data/reddit"

params = yaml.safe_load(open("params.yaml"))["prepare"]


def download_reddit(curated_subreddits, max_examples=60000):
    #dataset = load_dataset("reddit", split='train', cache_dir="/media/jonathob/DATA/.cache/huggingface/datasets", keep_in_memory=False)
    dataset = load_dataset("reddit", split='train', keep_in_memory=False)

    dataset.shuffle()
    dataset = dataset.select(np.arange(100000))

    print("loaded")

    #dataset = dataset['train']

    # identify unique subreddits
    unique_subreddits, counts = np.unique(dataset['subreddit'], return_counts=True)

    # count posts in each subreddit; sort by post counts
    top_subreddits = [sr for c, sr in sorted(zip(counts, unique_subreddits), reverse=True)]


    # top10: select top10 subreddits accoring the number of posts
    top10_subreddits = top_subreddits[:10]
    relevant_subreddits = list(set(top10_subreddits).union(set(curated_subreddits)))

    # next we create a mask that will select out posts from the relevant_subreddits
    subreddit_mask = np.zeros(len(dataset))

    # The top 3 categories each contain between 100k and 600k posts -- an order of magnitude more 
    # than any other popular subreddit. 
    # Truncate these 3 categories to a max of 60K posts each
    askreddit = 0
    relationships = 0
    lol = 0

    for i, sub in enumerate(dataset['subreddit']):
        if sub in relevant_subreddits:
            if sub == 'AskReddit' and askreddit <= max_examples:
                subreddit_mask[i] = 1
                askreddit += 1
            elif sub == 'relationships' and relationships <= max_examples:
                subreddit_mask[i] = 1
                relationships += 1
            elif sub == 'leagueoflegends' and lol <= max_examples:
                subreddit_mask[i] = 1
                lol += 1
            elif sub not in ['AskReddit', 'relationships', 'leagueoflegends']:    
                subreddit_mask[i] = 1
            else:
                continue

    subreddit_mask = subreddit_mask == 1

    # this file lives on a local machine
    dataset = pd.DataFrame(dataset)

    # create the subset
    subset = dataset[subreddit_mask]
    subset.to_csv(DATADIR + "/reddit_subset.pd")


def preprocess(random_state=42):
    df = pd.read_csv(DATADIR + "/reddit_subset.pd")

    # map the subreddit names to a standardized format to create category names
    df['category'] = df['subreddit'].map({
        'atheism': 'atheism',
        'funny': 'funny',
        'sex': 'sex',
        'Fitness': 'fitness',
        'AdviceAnimals': 'advice animals',
        'trees': 'trees',
        'personalfinance': 'personal finance', 
        'relationships': 'relationships',
        'relationship_advice': 'relationship advice',
        'tifu': 'tifu',
        'politics': 'politics',
        'gaming': 'gaming', 
        'worldnews': 'world news',
        'technology': 'technology',
        'leagueoflegends': 'league of legends',
        'AskReddit': 'ask reddit'
        })

    df['category'] = pd.Categorical(df.category)
    df['label'] = df.category.cat.codes

    df.dropna(inplace=True)

    # here we perform a stratified split on `subreddit` though
    # we could just as easily split on `category`
    X_train, X_test, y_train, y_test = train_test_split(df, df['subreddit'], 
                                                        test_size=.1, 
                                                        random_state=random_state, 
                                                        stratify=df['subreddit'])

    X_train.to_csv("data/reddit/reddit_subset_train.csv")
    X_test.to_csv("data/reddit/reddit_subset_test.csv")

    return df


def most_frequent_words(df, n_words=100000):
    nltk.download('stopwords')
    nltk.download('punkt')

    df.dropna(inplace=True)
    corpus = ''

    for summary in df.summary:
        try:
            corpus += summary
        except:
            print(summary) # throw actual exception here?

    thing = word_tokenize(corpus)

    stop1 = list(string.punctuation) + ["``", "''", "..."] #
    stop2 = stopwords.words("english") + list(string.punctuation) + ["``", "''", "..."]
    words1 = [word for word in thing if word not in stop1]
    words2 = [word for word in thing if word not in stop2]

    word_freq1 = Counter(words1).most_common(n_words)
    most_common_words1, counts = [list(c) for c in zip(*word_freq1)]

    word_freq2 = Counter(words2).most_common(n_words)
    most_common_words2, counts = [list(c) for c in zip(*word_freq2)]

    most_common_words = {"no_punc": most_common_words1, "no_punc_no_stop": most_common_words2}

    pickle.dump(most_common_words, open("data/reddit/most_common_words.pkl", "wb"))


def subsample(curated_subreddits, test_size=1300, train_size = 1000, random_state=42):
    df_train = pd.read_csv("data/reddit/reddit_subset_train.csv")

    curated_subreddits = ['relationships', 'trees', 'gaming', 'funny', 'politics', \
            'sex', 'Fitness', 'worldnews', 'personalfinance', 'technology']

    sample = (
        df_train[df_train.subreddit.isin(curated_subreddits)]
        .groupby('category', group_keys=False)
        .apply(lambda x: x.sample(min(len(x), train_size * 2), random_state=random_state))
    )

    X_train, X_valid, y_train, y_valid = train_test_split(sample, sample['subreddit'], 
                                                        test_size=.5, 
                                                        random_state=random_state, 
                                                        stratify=sample['subreddit'])
                                                    
    X_train.groupby('subreddit')['subreddit'].count()

    X_train.to_csv("data/reddit/reddit_subset_train1000.csv")
    X_valid.to_csv("data/reddit/reddit_subset_valid1000.csv")

    df_test = pd.read_csv("data/reddit/reddit_subset_test.csv")

    df_reddit_test = (
    df_test[df_test.subreddit.isin(curated_subreddits)]
    .groupby('category', group_keys=False)
    .apply(lambda x: x.sample(min(len(x), test_size), random_state=random_state))
    .assign(
        category = lambda df: pd.Categorical(df.category),
        label = lambda df: df.category.cat.codes
        )
    )

    import pdb;pdb.set_trace()

    # save the .csv version of this test set
    df_reddit_test.to_csv("data/reddit/reddit_subset_test1300.csv")

    filename = "data/reddit/reddit_dataset_1300.pkl"

    # Cast the pandas df as a Dateset object
    reddit_test_data = _create_dataset_from_df(df_reddit_test, 'summary')

    # Compute SentenceBERT embeddings for each example
    reddit_test_data.calc_sbert_embeddings()
    pickle_save(reddit_test_data, filename)

print("Downlaoading Reddit dataset")
download_reddit(params["curated_subreddits"], max_examples=params["max_examples"])

print("Preprocessing Reddit dataset")
df = preprocess(random_state=params["random_seed"])

print("Computing most frequent words")
most_frequent_words(df, n_words=params["n_most_common_words"])

print("Subsampling Reddit dataset")
subsample(params["curated_subreddits"], test_size=params["sample_size_test"], train_size=params["sample_size_train"],
                                        random_state=params["random_seed"])