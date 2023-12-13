## Utilities for NLP.

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

# @data: string
# Returns a NLP processed string.

def nlp_ize(data):
    # Tokenize the string.

    data = word_tokenize(data)

    # Take all the alpha words (means no numeric or special characters).

    data = [item for item in data if item.isalpha()]

    # Lowecase all the tokens.

    data = [item.lower() for item in data]

    # Remove all the stopwords like and, or. Because they are not necessary for the model to 
    # understand the context.

    stop_words = set(stopwords.words('english'))

    data = [item for item in data if item not in stop_words]

    # Stemming.

    ps = PorterStemmer()

    data = [ps.stem(item) for item in data]

    # Done with NLP.

    return ' '.join(data)
