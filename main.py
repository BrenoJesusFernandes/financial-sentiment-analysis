import os
import sys
import transformers
from loguru import logger
import pandas as pd


def log_config():
    if os.getenv('SHOW_LOGS') == '0':
        logger.configure(handlers=[{'sink': 'log/errors.log',
                                    'format': '{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}'}])
    else:
        logger.configure(handlers=[{'sink': sys.stdout},
                                   {'sink': 'log/errors.log',
                                    'format': '{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}'}])


@logger.catch
def run():

    finbert = transformers.BertForSequenceClassification.from_pretrained('ProsusAI/finbert', num_labels=3)
    tokenizer = transformers.BertTokenizer.from_pretrained('ProsusAI/finbert')

    nlp = transformers.pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)

    sentences = ["there is a shortage of capital, and we need extra financing",
                 "growth is strong and we have plenty of liquidity",
                 "there are doubts about our finances",
                 "profits are flat"]

    results = nlp(sentences)
    pd.DataFrame(results).to_excel('results.xlsx', index=False)


if __name__ == '__main__':

    try:
        log_config()
        run()
    except Exception as error:
        print(error)
