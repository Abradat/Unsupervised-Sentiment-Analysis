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
    try :
        norm_final_score = round(float(final_score) / token_count, 2)
    except:
        return 0.0
    final_sentiment = 'positive' if norm_final_score >= 0 else 'negative'
    if verbose:
        norm_obj_score = round(float(obj_score) / token_count, 2)
        norm_pos_score = round(float(pos_score) / token_count, 2)
        norm_neg_score = round(float(neg_score) / token_count, 2)
        # to display results in a nice table
        sentiment_frame = pd.DataFrame([[final_sentiment, norm_obj_score, norm_pos_score,
                                         norm_neg_score, norm_final_score]],
                                       columns=pd.MultiIndex(levels=[['SENTIMENT STATS:'],
                                                                     ['Predicted Sentiment', 'Objectivity',
                                                                      'Positive', 'Negative', 'Overall']],
                                                             labels=[[0, 0, 0, 0, 0], [0, 1, 2, 3, 4]]))
        print(sentiment_frame)

    return norm_final_score

def generatScoreCsv(predictedSentiments, dates):
    file = open("results.csv", 'w')
    file.write("score,created_at\n")
    for cnt in range(len(predictedSentiments)):
        file.write(str(predictedSentiments[cnt]) + ',' + str(dates[cnt]) + '\n')
    file.close()

def readFromTweet():
    dataset = pd.read_csv(r'CSVs/tweet.csv')

    reviews = np.array(dataset['text'])
    dates = np.array(dataset['created_at'])
    test_reviews = reviews
    test_dates = dates


    sample_review_ids = [430, 200, 470]

    # normalize dataset

    # Starting to normalize the data
    norm_test_reviews = tn.normalize_corpus(test_reviews, html_stripping=False)
    # End of normalizing

    awesome = list(swn.senti_synsets('awesome', 'a'))[0]
    print('Positive Polarity Score:', awesome.pos_score())
    print('Negative Polarity Score:', awesome.neg_score())
    print('Objective Score:', awesome.obj_score())

    predicted_sentiments = [analyze_sentiment_sentiwordnet_lexicon(review, verbose=False) for review in
                            norm_test_reviews]

    for s in predicted_sentiments:
        print(s)
    generatScoreCsv(predicted_sentiments, test_dates)

def parseDate(date):
    tmp = date.split(' ')[0]
    tmp = tmp.split('-')
    return tmp

def drawPolt():
    dataset = pd.read_csv(r'CSVs/results.csv')

    scores = np.array(dataset['score'])
    dates = np.array(dataset['created_at'])

    mainDict = {}
    for cnt in range(len(dates)):
        tmpDate = parseDate(dates[cnt])
        builtDate = tmpDate[2] + '-' + tmpDate[0] + '-01'
        if(builtDate in mainDict) :
            mainDict[builtDate] += scores[cnt]
        else :
            mainDict[builtDate] = scores[cnt]

    file = open("FinalResult.csv", 'w')
    file.write('score,created_at\n')
    for key in mainDict:
        file.write(str(mainDict[key])+',' + str(key) + '\n')
    file.close()
    #print(mainDict)

def drawPlot2():
    dataset = pd.read_csv(r'CSVs/results.csv')

    scores = np.array(dataset['score'])
    dates = np.array(dataset['created_at'])

    mainDict = {}
    for cnt in range(len(dates)):
        tmpDate = parseDate(dates[cnt])
        builtDate = tmpDate[2] + '-' + tmpDate[0] + '-' + tmpDate[1]
        if(builtDate in mainDict) :
            mainDict[builtDate] += scores[cnt]
        else :
            mainDict[builtDate] = scores[cnt]
    file = open("FinalResult2.csv", 'w')
    file.write('score,created_at\n')
    for key in mainDict:
        file.write(str(mainDict[key])+',' + str(key) + '\n')
    file.close()

#readFromTweet()
#drawPolt()
drawPlot2()