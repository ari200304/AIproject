import pandas as pd
train1 = pd.read_csv('Fake.csv')

train2 = pd.read_csv('True.csv')
#labeling fake news 0
train1['label']=0
#true as 1
train2['label']=1
#concatinating the two dataframes
frames = [train1, train2]
train = pd.concat(frames)

# Firstly, fill all the null spaces with a space
train = train.fillna(' ')
train['total']=train['title']+' '+train['text']+' '+train['subject']+' '+train['date']

#Natural language processing
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
#nltk.download('stopwords')
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
import re
#nltk.download('punkt')
for index, row in train.iterrows():
    filter_sentence = ''
    sentence = row['total']
    # Cleaning the sentence with regex
    sentence = re.sub(r'[^\w\s]', '', sentence)
    # Tokenization
    words = nltk.word_tokenize(sentence,language="english")
    # Stopwords removal
    words = [w for w in words if not w in stop_words]
    # Lemmatization
    for words in words:
        filter_sentence = filter_sentence  + ' ' +str(lemmatizer.lemmatize(words)).lower()
        train.loc[index, 'total'] = filter_sentence
