import pandas as pd
import numpy as np
import text_normalizer as tn
import model_evaluation_utils as meu
from nltk.corpus import sentiwordnet as swn

def analyze_sentiment_sentiwordnet_lexicon(review, verbose=False):
    # tokenize and POS tag text tokens
    tagged_text = [(token.text, token.tag_) for token in tn.nlp(review)]
    pos_score = neg_score = token_count = obj_score = 0
    # get wordnet synsets based on POS tags
    # get sentiment scores if synsets are found
    for word, tag in tagged_text:
        ss_set = None
        if 'NN' in tag and list(swn.senti_synsets(word, 'n')):
            ss_set = list(swn.senti_synsets(word, 'n'))[0]
        elif 'VB' in tag and list(swn.senti_synsets(word, 'v')):
            ss_set = list(swn.senti_synsets(word, 'v'))[0]
        elif 'JJ' in tag and list(swn.senti_synsets(word, 'a')):
            ss_set = list(swn.senti_synsets(word, 'a'))[0]
        elif 'RB' in tag and list(swn.senti_synsets(word, 'r')):
            ss_set = list(swn.senti_synsets(word, 'r'))[0]
        # if senti-synset is found
        if ss_set:
            # add scores for all found synsets
            pos_score += ss_set.pos_score()
            neg_score += ss_set.neg_score()
            obj_score += ss_set.obj_score()
            token_count += 1

    # aggregate final scores
    final_score = pos_score - neg_score

    norm_final_score = round(float(final_score) / token_count, 2)
    final_sentiment = 'positive' if norm_final_score >= 0 else 'negative'

    return final_sentiment

def readFromCSV():
    dataset = pd.read_csv(r'CSVs/tweets_main.csv')

    reviews = np.array(dataset['review'])
    sentiments = np.array(dataset['sentiment'])

    # extract data for model evaluation
    test_reviews = reviews[:5000]
    test_sentiments = sentiments[:5000]
    #sample_review_ids = [7626, 3533, 13010]

    # normalize dataset
    norm_test_reviews = tn.normalize_corpus(test_reviews)

    predicted_sentiments = [analyze_sentiment_sentiwordnet_lexicon(review, verbose=False) for review in
                            norm_test_reviews]

    meu.display_model_performance_metrics(true_labels=test_sentiments, predicted_labels=predicted_sentiments, classes=['positive', 'negative'])

readFromCSV()