import pandas as pd

# %% import twitter data
data = pd.read_csv(r"gender_classifier.csv",encoding = "latin1")
data = pd.concat([data.gender,data.description],axis=1)
data.dropna(axis=0,inplace= True)
data.gender = [1 if each=="female" else 0 for each in data.gender]

# %% cleaning data
# regular expression RE for ex. "[a-zA-Z]"
import re
first_description = data.description[4]
description = re.sub("[^a-zA-Z]"," ",first_description)
description = description.lower()

# %% Stopwords (irrelavent words)
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
description_split = description.split()
nltk.download('punkt') # for word tokenize error
description = nltk.word_tokenize(description) #shouldn't --> should n't

# %%
description = [word for word in description if not word in set(stopwords.words("english"))]

# %% lemmatization
import nltk as nlp
nltk.download('wordnet')
lemma = nlp.WordNetLemmatizer()
description = [lemma.lemmatize(word) for word in description]
description = " ".join(description) # again doing sentence

# %% all data
description_list = []
for description in data.description:
    description = re.sub("[^a-zA-Z]"," ",description)
    description = description.lower()
    description = nltk.word_tokenize(description) #shouldn't --> should n't
    #description = [word for word in description if not word in set(stopwords.words("english"))]
    lemma = nlp.WordNetLemmatizer()
    description = [lemma.lemmatize(word) for word in description]
    description = " ".join(description) # again doing sentence
    description_list.append(description)

# %% bag of words
from sklearn.feature_extraction.text import CountVectorizer # it is a function for create bag of words
max_features = 5000
count_vectorizer = CountVectorizer(max_features= max_features,stop_words="english")
sparce_matrix = count_vectorizer.fit_transform(description_list).toarray()
print(" => {} Most using words {}".format(max_features,count_vectorizer.get_feature_names())) 

# %% native bayes
y = data.iloc[:,0].values #male or female classes
x = sparce_matrix
#train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.1,random_state = 42)
from sklearn.naive_bayes import GaussianNB
nb =GaussianNB()
nb.fit(x_train,y_train)

# %% prediction
y_pred = nb.predict(x_test)
print("Accuracy : ",nb.score(y_pred.reshape(-1,1),y_test))