import streamlit as st
import heapq
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import torch
from transformers import BartTokenizer, BartForConditionalGeneration, AutoTokenizer, AutoModelForTokenClassification, pipeline

stopwords_ = set(stopwords.words('english'))

st.set_page_config(
    page_title="TLDR",
)

with st.form("my_form"):
    col1, col2, col3 = st.columns(3)
    col2.write("****Summarize your text in 1 click****")
    st.markdown('##')

    text = st.text_area("Paste your text here",
        height=300,
    )
    st.markdown('##')

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Choose the summarization model**")
        radio_model = st.radio(
            "Model",
            ('Key Sentences', 'AI Summary'),
            label_visibility= "collapsed",
        )

    with col2:
        st.write("**Choose the summarization size** (not applicable to AI summary model yet)")
        radio_size = st.radio(
            "size",
            ('Short', 'Detailed'),
            label_visibility= "collapsed",
        )
    st.markdown('##')
    radio_keywords = st.checkbox('Display Important Keywords')
    st.markdown('##')

    col1, col2, col3 = st.columns(3)
    submitted = col2.form_submit_button('Summarize this...')

if submitted:
    st.markdown('##')
    if radio_keywords:
        st.write("**Important Keywords**")
        @st.cache(allow_output_mutation=True,suppress_st_warning=True)
        def extract_keywords(text):
            key_tokenizer = AutoTokenizer.from_pretrained("yanekyuk/bert-uncased-keyword-extractor")
            key_model = AutoModelForTokenClassification.from_pretrained("yanekyuk/bert-uncased-keyword-extractor")
            nlp = pipeline("ner", model=key_model, tokenizer=key_tokenizer)
            result = list()
            keyword = ""
            for token in nlp(text):
                if token['entity'] == 'I-KEY':
                    keyword += token['word'][2:] if \
                    token['word'].startswith("##") else f" {token['word']}"
                else:
                    if keyword:
                        result.append(keyword)
                    keyword = token['word']
            result.append(keyword)
            return list(set(result))

        kw = extract_keywords(text)
        fkw = ', '.join(kw)
        st.write(fkw) 
        st.markdown('##')

    if radio_model == 'AI Summary':
        @st.cache(allow_output_mutation=True)
        def extract_summary():
            model = BartForConditionalGeneration.from_pretrained("sshleifer/distilbart-cnn-12-6")
            tokenizer = BartTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
            inputs = tokenizer([text],return_tensors="pt", max_length=1024, truncation=True)
            summary_ids = model.generate(inputs["input_ids"], max_length = (int(len(text)*0.7)), num_beams = 4)
            summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            return summary

        st.write("**AI generated summary about the document**")
        st.write(extract_summary())
    else:
        if radio_size == 'Detailed':
            st.write("**Extractive detailed summary about the document**")
            def summary_large(text):
                sentences = sent_tokenize(text)
                word_frequencies = {}
                for word in word_tokenize(text):
                    if word not in stopwords_:
                        if word not in word_frequencies.keys():
                            word_frequencies[word] = 1
                        else:
                            word_frequencies[word] += 1
                maximum_frequncy = max(word_frequencies.values())
                for word in word_frequencies.keys():
                    word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)
                sentence_scores = {}
                for sent in sentences:
                    for word in word_tokenize(sent.lower()):
                        if word in word_frequencies.keys():
                            if len(sent.split(' ')) < 30:
                                if sent not in sentence_scores.keys():
                                    sentence_scores[sent] = word_frequencies[word]
                                else:
                                    sentence_scores[sent] += word_frequencies[word]
                summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)
                summary = ' '.join(summary_sentences)
                return summary
            st.write(summary_large(text))
        else:
            st.write("**Extractive short summary about the document**")
            def summary_short(text):
                sentences = sent_tokenize(text)
                word_frequencies = {}
                for word in word_tokenize(text):
                    if word not in stopwords_:
                        if word not in word_frequencies.keys():
                            word_frequencies[word] = 1
                        else:
                            word_frequencies[word] += 1
                maximum_frequncy = max(word_frequencies.values())
                for word in word_frequencies.keys():
                    word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)
                sentence_scores = {}
                for sent in sentences:
                    for word in word_tokenize(sent.lower()):
                        if word in word_frequencies.keys():
                            if len(sent.split(' ')) < 30:
                                if sent not in sentence_scores.keys():
                                    sentence_scores[sent] = word_frequencies[word]
                                else:
                                    sentence_scores[sent] += word_frequencies[word]
                summary_sentences = heapq.nlargest(3, sentence_scores, key=sentence_scores.get)
                summary = ' '.join(summary_sentences)
                return summary
            st.write(summary_short(text))