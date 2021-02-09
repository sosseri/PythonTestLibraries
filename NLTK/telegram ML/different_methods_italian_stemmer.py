#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 23:15:14 2021

@author: alessandroseri
"""

import nltk
import string
import pattern

# dictionary of Italian stop-words
it_stop_words = nltk.corpus.stopwords.words('italian')
# Snowball stemmer with rules for the Italian language
ita_stemmer = nltk.stem.snowball.ItalianStemmer()

# the following function is just to get the lemma
# out of the original input word (but right now
# it may be loosing the context about the sentence
# from where the word is coming from i.e.
# the same word could either be a noun/verb/adjective
# according to the context)
def lemmatize_word(input_word):
    in_word = input_word#.decode('utf-8')
    # print('Something: {}'.format(in_word))
    word_it = pattern.it.parse(
        in_word, 
        tokenize=False,  
        tag=False,  
        chunk=False,  
        lemmata=True 
    )
    # print("Input: {} Output: {}".format(in_word, word_it))
    the_lemmatized_word = word_it.split()[0][0][4]
    # print("Returning: {}".format(the_lemmatized_word))
    return the_lemmatized_word

it_string = "Ieri sono andato in due supermercati. Oggi volevo andare all'ippodromo. Stasera mangio la pizza con le verdure."

# 1st tokenize the sentence(s)
word_tokenized_list = nltk.tokenize.word_tokenize(it_string)
print("1) NLTK tokenizer, num words: {} for list: {}".format(len(word_tokenized_list), word_tokenized_list))

# 2nd remove punctuation and everything lower case
word_tokenized_no_punct = [str.lower(x) for x in word_tokenized_list if x not in string.punctuation]
print("2) Clean punctuation, num words: {} for list: {}".format(len(word_tokenized_no_punct), word_tokenized_no_punct))

# 3rd remove stop words (for the Italian language)
word_tokenized_no_punct_no_sw = [x for x in word_tokenized_no_punct if x not in it_stop_words]
print("3) Clean stop-words, num words: {} for list: {}".format(len(word_tokenized_no_punct_no_sw), word_tokenized_no_punct_no_sw))

# 4.1 lemmatize the words
#word_tokenize_list_no_punct_lc_no_stowords_lemmatized = [lemmatize_word(x) for x in word_tokenized_no_punct_no_sw]
#print("4.1) lemmatizer, num words: {} for list: {}".format(len(word_tokenize_list_no_punct_lc_no_stowords_lemmatized), word_tokenize_list_no_punct_lc_no_stowords_lemmatized))

#Remove also apostrophe
word_tokenized_no_punct_no_sw_no_apostrophe = [x.split("'") for x in word_tokenized_no_punct_no_sw]
word_tokenized_no_punct_no_sw_no_apostrophe = [y for x in word_tokenized_no_punct_no_sw_no_apostrophe for y in x]

test = ' '.join(words for words in word_tokenized_no_punct_no_sw_no_apostrophe)
# 4.2 snowball stemmer for Italian
word_tokenize_list_no_punct_lc_no_stowords_stem = [ita_stemmer.stem(i) for i in word_tokenized_no_punct_no_sw_no_apostrophe]
print("4.2) stemmer, num words: {} for list: {}".format(len(word_tokenize_list_no_punct_lc_no_stowords_stem), word_tokenize_list_no_punct_lc_no_stowords_stem))

# difference between stemmer and lemmatizer
print(
    "For original word(s) '{}' and '{}' the stemmer: '{}' '{}' (count 1 each), the lemmatizer: '{}' '{}' (count 2)"
    .format(
        word_tokenized_no_punct_no_sw[1],
        word_tokenized_no_punct_no_sw[6],
        word_tokenized_no_punct_no_sw_no_apostrophe[1],
        word_tokenized_no_punct_no_sw_no_apostrophe[6],
        word_tokenize_list_no_punct_lc_no_stowords_stem[1],
        word_tokenize_list_no_punct_lc_no_stowords_stem[6]
    )
)
    
    
#%% 
import treetaggerwrapper 
from pprint import pprint

it_string = "Ieri sono andato in due supermercati. Oggi volevo andare all'ippodromo. Stasera mangio la pizza con le verdure."
tagger = treetaggerwrapper.TreeTagger(TAGLANG="it")
tags = tagger.tag_text(it_string)
pprint(treetaggerwrapper.make_tags(tags))
test = treetaggerwrapper.make_tags(tags)