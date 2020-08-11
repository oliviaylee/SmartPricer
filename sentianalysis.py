import sys
import csv
import pandas as pd
from monkeylearn import MonkeyLearn

# Instantiate the client by replacing <YOUR API TOKEN HERE> with your MonkeyLearn account API key
ml = MonkeyLearn('<YOUR API TOKEN HERE>')

# Training Set from here: https://github.com/monkeylearn/sentiment-analysis-benchmark/blob/master/data/train_products.csv
# Test Set from here: https://github.com/monkeylearn/sentiment-analysis-benchmark/blob/master/data/test_products.csv
# Alternatively, gather tweets by keyword from here: https://github.com/feconroses/gather-tweets-from-stream
def main():
    args = sys.argv[1:]
    filename = args[1]
    df1 = pd.read_csv(filename, 
                    index_col='Tweet',  
                    header=0, 
                    names=['Tweet', 'Sentiment'])
    tweets = df1['Tweet']
    response = ml.classifiers.classify(model_id, tweets)
    
    sentiRating = []
    for i in range(len(response)):
        sentiRating.append(response[i]["classifications"]["confidence"])
    s1 = pd.Series(sentiRating, name='Rating')
    df2 = pd.concat([df1, s1], axis=1)
    df2.to_csv('tweets_rating')

if __name__ == "__main__":
    main()
