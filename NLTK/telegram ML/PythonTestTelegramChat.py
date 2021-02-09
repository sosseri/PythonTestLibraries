#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 13:04:31 2020
Export the chat from a telegram group in json
and place the folder path in the variable path_to_json
@author: alessandroseri
"""

import os
import io, json
import numpy as np
from pandas import DataFrame
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

import matplotlib.pyplot as plt

#%%
def readFiles(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)

            inBody = False
            lines = []
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
                if inBody:
                    lines.append(line)
                elif line == '\n':
                    inBody = True
            f.close()
            message = '\n'.join(lines)
            yield path, message


def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)

    return DataFrame(rows, index=index)

def to_bool(s):
    return 1 if s == 'true' else 0

#%%
# ------------------------- PLACE HERE THE PATH OF YOUR TELEGRAM CHAT --------------------------------
path_to_json = '/Users/alessandroseri/Downloads/Telegram Desktop/ChatExport_2020-11-24/'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
print(json_files)

for index, js in enumerate(json_files):
    with open(os.path.join(path_to_json, js)) as json_file:
        json_text = json.load(json_file)
        
messages = json_text['messages']
b = DataFrame(messages)
mydata = DataFrame(messages, columns=['from','type','text'])
indxs_drop=  []
for ii in range(len(mydata)):
    if mydata['type'][ii]=='service':
        indxs_drop = indxs_drop + [ii]
    elif type(mydata['text'].values[ii])==list:
        #print(ii)
        mydata['text'].values[ii] = mydata['text'].values[ii][0]
        if type(mydata['text'].values[ii])==dict:
            mydata['text'].values[ii]=mydata['text'].values[ii]['text']

mydata=mydata.drop(indxs_drop)
del mydata['type']


my_stop_words=["a","abbastanza","abbia","abbiamo","abbiano","abbiate","accidenti","ad","adesso","affinché","agl","agli","ahime","ahimè","ai","al","alcuna","alcuni","alcuno","all","alla","alle","allo","allora","altre","altri","altrimenti","altro","altrove","altrui","anche","ancora","anni","anno","ansa","anticipo","assai","attesa","attraverso","avanti","avemmo","avendo","avente","aver","avere","averlo","avesse","avessero","avessi","avessimo","aveste","avesti","avete","aveva","avevamo","avevano","avevate","avevi","avevo","avrai","avranno","avrebbe","avrebbero","avrei","avremmo","avremo","avreste","avresti","avrete","avrà","avrò","avuta","avute","avuti","avuto","basta","ben","bene","benissimo","brava","bravo","buono","c","caso","cento","certa","certe","certi","certo","che","chi","chicchessia","chiunque","ci","ciascuna","ciascuno","cima","cinque","cio","cioe","cioè","circa","citta","città","ciò","co","codesta","codesti","codesto","cogli","coi","col","colei","coll","coloro","colui","come","cominci","comprare","comunque","con","concernente","conclusione","consecutivi","consecutivo","consiglio","contro","cortesia","cos","cosa","cosi","così","cui","d","da","dagl","dagli","dai","dal","dall","dalla","dalle","dallo","dappertutto","davanti","degl","degli","dei","del","dell","della","delle","dello","dentro","detto","deve","devo","di","dice","dietro","dire","dirimpetto","diventa","diventare","diventato","dopo","doppio","dov","dove","dovra","dovrà","dovunque","due","dunque","durante","e","ebbe","ebbero","ebbi","ecc","ecco","ed","effettivamente","egli","ella","entrambi","eppure","era","erano","eravamo","eravate","eri","ero","esempio","esse","essendo","esser","essere","essi","ex","fa","faccia","facciamo","facciano","facciate","faccio","facemmo","facendo","facesse","facessero","facessi","facessimo","faceste","facesti","faceva","facevamo","facevano","facevate","facevi","facevo","fai","fanno","farai","faranno","fare","farebbe","farebbero","farei","faremmo","faremo","fareste","faresti","farete","farà","farò","fatto","favore","fece","fecero","feci","fin","finalmente","finche","fine","fino","forse","forza","fosse","fossero","fossi","fossimo","foste","fosti","fra","frattempo","fu","fui","fummo","fuori","furono","futuro","generale","gente","gia","giacche","giorni","giorno","giu","già","gli","gliela","gliele","glieli","glielo","gliene","grande","grazie","gruppo","ha","haha","hai","hanno","ho","i","ie","ieri","il","improvviso","in","inc","indietro","infatti","inoltre","insieme","intanto","intorno","invece","io","l","la","lasciato","lato","le","lei","li","lo","lontano","loro","lui","lungo","luogo","là","ma","macche","magari","maggior","mai","male","malgrado","malissimo","me","medesimo","mediante","meglio","meno","mentre","mesi","mezzo","mi","mia","mie","miei","mila","miliardi","milioni","minimi","mio","modo","molta","molti","moltissimo","molto","momento","mondo","ne","negl","negli","nei","nel","nell","nella","nelle","nello","nemmeno","neppure","nessun","nessuna","nessuno","niente","no","noi","nome","non","nondimeno","nonostante","nonsia","nostra","nostre","nostri","nostro","novanta","nove","nulla","nuovi","nuovo","o","od","oggi","ogni","ognuna","ognuno","oltre","oppure","ora","ore","osi","ossia","ottanta","otto","paese","parecchi","parecchie","parecchio","parte","partendo","peccato","peggio","per","perche","perchè","perché","percio","perciò","perfino","pero","persino","persone","però","piedi","pieno","piglia","piu","piuttosto","più","po","pochissimo","poco","poi","poiche","possa","possedere","posteriore","posto","potrebbe","preferibilmente","presa","press","prima","primo","principalmente","probabilmente","promesso","proprio","puo","pure","purtroppo","può","qua","qualche","qualcosa","qualcuna","qualcuno","quale","quali","qualunque","quando","quanta","quante","quanti","quanto","quantunque","quarto","quasi","quattro","quel","quella","quelle","quelli","quello","quest","questa","queste","questi","questo","qui","quindi","quinto","realmente","recente","recentemente","registrazione","relativo","riecco","rispetto","salvo","sara","sarai","saranno","sarebbe","sarebbero","sarei","saremmo","saremo","sareste","saresti","sarete","sarà","sarò","scola","scopo","scorso","se","secondo","seguente","seguito","sei","sembra","sembrare","sembrato","sembrava","sembri","sempre","senza","sette","si","sia","siamo","siano","siate","siete","sig","solito","solo","soltanto","sono","sopra","soprattutto","sotto","spesso","sta","stai","stando","stanno","starai","staranno","starebbe","starebbero","starei","staremmo","staremo","stareste","staresti","starete","starà","starò","stata","state","stati","stato","stava","stavamo","stavano","stavate","stavi","stavo","stemmo","stessa","stesse","stessero","stessi","stessimo","stesso","steste","stesti","stette","stettero","stetti","stia","stiamo","stiano","stiate","sto","su","sua","subito","successivamente","successivo","sue","sugl","sugli","sui","sul","sull","sulla","sulle","sullo","suo","suoi","tale","tali","talvolta","tanto","te","tempo","terzo","th","ti","titolo","tra","tranne","tre","trenta","triplo","troppo","trovato","tu","tua","tue","tuo","tuoi","tutta","tuttavia","tutte","tutti","tutto","uguali","ulteriore","ultimo","un","una","uno","uomo","va","vai","vale","vari","varia","varie","vario","verso","vi","vicino","visto","vita","voi","volta","volte","vostra","vostre","vostri","vostro","è"]

train_indxs = np.arange(6000)#np.random.randint(0,len(mydata),int(len(mydata)*.8))
test_indxs=np.arange(len(mydata))
test_indxs = np.delete(test_indxs,train_indxs)

mydata_train = mydata[0:6000]
mydata_test = mydata[6001:-1]

print('Optimization of ngram_range in count vectorizer')
highmatch = 0
highmatch_indx = -1
for ttt in range(10):
    
    if ttt==0:
        vectorizer = CountVectorizer()
    else:
        vectorizer = CountVectorizer(ngram_range=(1, ttt), min_df=1, analyzer = 'word', stop_words=set(my_stop_words))
    
    counts = vectorizer.fit_transform(mydata_train['text'])
    classifier = MultinomialNB()
    targets = mydata_train['from'].values
    classifier.fit(counts, targets)
    
    mydata_test_counts = vectorizer.transform(mydata_test['text'])
    predictions = classifier.predict(mydata_test_counts)
    #%%
    match = (mydata_test['from']==predictions)
    match = np.array(match)
    match = match.astype(int)
    print(str(ttt)+ " words together in ngram_range. The percentage of guessed senteces is " + str(int(np.mean(match)*100)) + "%")
    if np.mean(match)>highmatch:
        highmatch = np.mean(match)
        highmatch_indx = ttt
    plt.plot(ttt,np.mean(match),'.')
#print([highmatch, highmatch_indx])

if highmatch_indx == 0:
    print('The vectorizer is used in default options.')
    vectorizer = CountVectorizer()
else:
    print('The vectorizer is using up to ' + str(highmatch_indx) + ' number of words together.')
    vectorizer = CountVectorizer(ngram_range=(1, highmatch_indx), min_df=1, analyzer = 'word', stop_words=set(my_stop_words))

counts = vectorizer.fit_transform(mydata_train['text'])
classifier = MultinomialNB()
targets = mydata_train['from'].values
classifier.fit(counts, targets)
#%%
all_actors = mydata['from'].unique()
train_indxs = mydata_test.index
for aa in range(len(all_actors)):
    aa_indx = []
    #print('selecting raws from ' + all_actors[aa])
    for ii in range(len(mydata_test)):
        if mydata_test['from'][train_indxs[ii]] == all_actors[aa]:
            aa_indx = aa_indx+[ii]
    if len(aa_indx)>1:
        aa_indx = np.array(aa_indx)
        mydata_test_actors = mydata_test.iloc[aa_indx]
        
        mydata_test_counts = vectorizer.transform(mydata_test_actors['text'])
        predictions = classifier.predict(mydata_test_counts)
        match = (mydata_test_actors['from']==predictions)
        match = np.array(match)
        match = match.astype(int)
        print('For ' + str(all_actors[aa]) + ' we have a match of ' + str(int(np.mean(match)*100)) + '% and ' + str(len(mydata_test_actors['from']==all_actors[aa])) + ' messages')
    
#%% interesting ML to add

mydata1 = mydata.iloc[0:2170]
mydata1.append(mydata.iloc[2172:2953])

mydata = mydata1

from transformers import pipeline

nlp = pipeline("sentiment-analysis")

#result = nlp("Ti odio merda")[0]
#print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
#
#result = nlp("Caterina mi ha comprato una torta, che dolcissima!")[0]
#print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

mydata['positiveness'] = ['' for i in range(len(mydata['text']))]
mydata['score'] = [0.0 for i in range(len(mydata['text']))]
mydata['positiveness_score'] = [0.0 for i in range(len(mydata['text']))]

for ii in range(len(mydata['text'])):
    if ii%100==0:
        print("Training for sentiment analysis ..." + str(int(ii/len(mydata['text'])*100)) + "% completed ")
    result = nlp(mydata['text'].values[ii])[0]
    if result['label']=='POSITIVE':
        mydata['positiveness_score'].values[ii] =  result['score']
    else:
        mydata['positiveness_score'].values[ii] =  1-result['score']
#    print(mydata['positiveness_score'].values[ii])
    mydata['positiveness'].values[ii] = result['label']
    mydata['score'].values[ii]=result['score']

for aa in all_actors:
    tocheck=mydata[mydata['from']==aa]
    value = np.mean(tocheck['positiveness_score'])
    print(aa + " has a positiveness among a min of 0 and  a max of 1 of " + str(value))
