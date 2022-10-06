import re
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from rake_nltk import Rake
import textstat
import gensim
from gensim.utils import simple_preprocess
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import joblib
import pickle

device = torch.device('cuda:0')

stopwords_ = set(stopwords.words('english'))

print("Document information extraction.\n")

filename = "text.txt"
with open(filename, encoding="utf8") as myfile:
    text=" ".join(line.rstrip() for line in myfile)

    
#number of words in the text
num_words = len(text.split())
print("A) Total number of words:",num_words)

#number of sentences in the text
num_sents = len(sent_tokenize(text))
print("B) Total number of sentences:",num_sents)

# average sentence length
average_sent_len = int(num_words / num_sents)
print("C) Average sentence length:", average_sent_len)

#clean text
#remove stopwords
filtered_list=[]
words = word_tokenize(text)
for w in words:
    if w.lower() not in stopwords_:
        filtered_list.append(w)
filtered_text1 = " ".join(filtered_list)

#make everything lowercase
filtered_text2 = filtered_text1.lower()

#remove punctuation
filtered_text3 = re.sub(r'[^\w\s]', '', filtered_text2) #keep alphanumeric character or whitespace


#Flesch Reading Ease Score
textscore = textstat.flesch_reading_ease(text)
if textscore > 90:
    reading_ease_score = str(textscore) +", Very easy to read. Easily understood by an average 11-year-old student."
if 80 < textscore <= 90:
    reading_ease_score = str(textscore) +", Easy to read. Can be used as conversational for consumers."
if 70 < textscore <= 80:
    reading_ease_score = str(textscore) +", Fairly easy to read."
if 60 < textscore <= 70:
    reading_ease_score = str(textscore) + ", Easily understood by 13- to 15-year-old students."
if 50 < textscore <= 60:
    reading_ease_score = str(textscore) + ", Fairly hard to read."
if 30 < textscore <= 50:
    reading_ease_score = str(textscore) + ", Hard to read."
if 0 <= textscore <= 30:
    reading_ease_score = str(textscore) + ", Very hard to read. Best understood by university graduates."
if textscore < 0:
    reading_ease_score = str(textscore) + ", The text has very complicated sentences."

print("D) Flesch reading ease score:", reading_ease_score)


#top 5 frequent words
split_it = filtered_text3.split()
Countera = Counter(split_it)
most_occur = Countera.most_common(5)
print("E) Top 5 most frequent words:",most_occur)


#top 5 key phrases
rake = Rake()
rake.extract_keywords_from_text(text)
print("F) Top 5 key phrases of the text:",rake.get_ranked_phrases()[0:5])


#tf idf words with most weights, topics
vectorizer = TfidfVectorizer()
response = vectorizer.fit_transform([filtered_text3])

feature_array = np.array(vectorizer.get_feature_names_out())
tfidf_sorting = np.argsort(response.toarray()).flatten()[::-1]

top_n = feature_array[tfidf_sorting][:10]
print("G) The document possible refers to:", top_n)


#generate Summary
model = BartForConditionalGeneration.from_pretrained("sshleifer/distilbart-cnn-12-6").to(device)
tokenizer = BartTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
inputs = tokenizer([text],return_tensors="pt", max_length=1024, truncation=True).to(device)
summary_ids = model.generate(inputs["input_ids"], max_length = (int(len(text)*0.3)), num_beams = 4)
summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print("H) Short summary about the document: ", summary)


#sentiment analysis
classifier = joblib.load('../sentiment_analysis/sentiment_model.pkl')
vectorizer = pickle.load(open("../sentiment_analysis/sentiment_vectorizer.pickle", 'rb'))

X_new = vectorizer.transform(sent_tokenize(text))
results = classifier.predict(X_new)
#print(results)
results = results.tolist()
results_sum = sum(results)
sentiment_score=""
sntscr = results_sum/len(results)

if ((sntscr > 0.5) and (sntscr <0.75)):
    print("I) The average sentiment score of the document is: ","%.2f" %sntscr, ", Slightly Positive")
    sentiment_score = "Slightly Positive"
elif (sntscr > 0.75):
    print("I) The average sentiment score of the document is: ","%.2f" %sntscr, ", Positive")
    sentiment_score = "Positive"
elif ((sntscr > 0.25) and (sntscr < 0.5)):
    print("I) The average sentiment score of the document is: ","%.2f" %sntscr, ", Slightly Negative")
    sentiment_score = "Slightly Negative"   
elif (sntscr == 0.5):
    print("I) The average sentiment score of the document is: ","%.2f" %sntscr, ", Neutral")
    sentiment_score = "Neutral"
else:
    print("I) The average sentiment score of the document is: ","%.2f" %sntscr, ", Negative")
    sentiment_score = "Negative"
