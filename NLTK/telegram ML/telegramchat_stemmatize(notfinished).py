"""
Created on Tue Nov 24 13:04:31 2020

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






#%% ------------ place here the path of your chat ----------------
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

all_actors = mydata['from'].unique()
#%% engineer the text
import nltk
import string
import pattern
# Importing FreqDist library from nltk and passing token into FreqDist
from nltk.probability import FreqDist

my_stop_words=["a","abbastanza","abbia","abbiamo","abbiano","abbiate","accidenti","ad","adesso","affinché","agl","agli","ahime","ahimè","ai","al","alcuna","alcuni","alcuno","all","alla","alle","allo","allora","altre","altri","altrimenti","altro","altrove","altrui","anche","ancora","anni","anno","ansa","anticipo","assai","attesa","attraverso","avanti","avemmo","avendo","avente","aver","avere","averlo","avesse","avessero","avessi","avessimo","aveste","avesti","avete","aveva","avevamo","avevano","avevate","avevi","avevo","avrai","avranno","avrebbe","avrebbero","avrei","avremmo","avremo","avreste","avresti","avrete","avrà","avrò","avuta","avute","avuti","avuto","basta","ben","bene","benissimo","brava","bravo","buono","c","caso","cento","certa","certe","certi","certo","che","chi","chicchessia","chiunque","ci","ciascuna","ciascuno","cima","cinque","cio","cioe","cioè","circa","citta","città","ciò","co","codesta","codesti","codesto","cogli","coi","col","colei","coll","coloro","colui","come","cominci","comprare","comunque","con","concernente","conclusione","consecutivi","consecutivo","consiglio","contro","cortesia","cos","cosa","cosi","così","cui","d","da","dagl","dagli","dai","dal","dall","dalla","dalle","dallo","dappertutto","davanti","degl","degli","dei","del","dell","della","delle","dello","dentro","detto","deve","devo","di","dice","dietro","dire","dirimpetto","diventa","diventare","diventato","dopo","doppio","dov","dove","dovra","dovrà","dovunque","due","dunque","durante","e","ebbe","ebbero","ebbi","ecc","ecco","ed","effettivamente","egli","ella","entrambi","eppure","era","erano","eravamo","eravate","eri","ero","esempio","esse","essendo","esser","essere","essi","ex","fa","faccia","facciamo","facciano","facciate","faccio","facemmo","facendo","facesse","facessero","facessi","facessimo","faceste","facesti","faceva","facevamo","facevano","facevate","facevi","facevo","fai","fanno","farai","faranno","fare","farebbe","farebbero","farei","faremmo","faremo","fareste","faresti","farete","farà","farò","fatto","favore","fece","fecero","feci","fin","finalmente","finche","fine","fino","forse","forza","fosse","fossero","fossi","fossimo","foste","fosti","fra","frattempo","fu","fui","fummo","fuori","furono","futuro","generale","gente","gia","giacche","giorni","giorno","giu","già","gli","gliela","gliele","glieli","glielo","gliene","grande","grazie","gruppo","ha","haha","hai","hanno","ho","i","ie","ieri","il","improvviso","in","inc","indietro","infatti","inoltre","insieme","intanto","intorno","invece","io","l","la","lasciato","lato","le","lei","li","lo","lontano","loro","lui","lungo","luogo","là","ma","macche","magari","maggior","mai","male","malgrado","malissimo","me","medesimo","mediante","meglio","meno","mentre","mesi","mezzo","mi","mia","mie","miei","mila","miliardi","milioni","minimi","mio","modo","molta","molti","moltissimo","molto","momento","mondo","ne","negl","negli","nei","nel","nell","nella","nelle","nello","nemmeno","neppure","nessun","nessuna","nessuno","niente","no","noi","nome","non","nondimeno","nonostante","nonsia","nostra","nostre","nostri","nostro","novanta","nove","nulla","nuovi","nuovo","o","od","oggi","ogni","ognuna","ognuno","oltre","oppure","ora","ore","osi","ossia","ottanta","otto","paese","parecchi","parecchie","parecchio","parte","partendo","peccato","peggio","per","perche","perchè","perché","percio","perciò","perfino","pero","persino","persone","però","piedi","pieno","piglia","piu","piuttosto","più","po","pochissimo","poco","poi","poiche","possa","possedere","posteriore","posto","potrebbe","preferibilmente","presa","press","prima","primo","principalmente","probabilmente","promesso","proprio","puo","pure","purtroppo","può","qua","qualche","qualcosa","qualcuna","qualcuno","quale","quali","qualunque","quando","quanta","quante","quanti","quanto","quantunque","quarto","quasi","quattro","quel","quella","quelle","quelli","quello","quest","questa","queste","questi","questo","qui","quindi","quinto","realmente","recente","recentemente","registrazione","relativo","riecco","rispetto","salvo","sara","sarai","saranno","sarebbe","sarebbero","sarei","saremmo","saremo","sareste","saresti","sarete","sarà","sarò","scola","scopo","scorso","se","secondo","seguente","seguito","sei","sembra","sembrare","sembrato","sembrava","sembri","sempre","senza","sette","si","sia","siamo","siano","siate","siete","sig","solito","solo","soltanto","sono","sopra","soprattutto","sotto","spesso","sta","stai","stando","stanno","starai","staranno","starebbe","starebbero","starei","staremmo","staremo","stareste","staresti","starete","starà","starò","stata","state","stati","stato","stava","stavamo","stavano","stavate","stavi","stavo","stemmo","stessa","stesse","stessero","stessi","stessimo","stesso","steste","stesti","stette","stettero","stetti","stia","stiamo","stiano","stiate","sto","su","sua","subito","successivamente","successivo","sue","sugl","sugli","sui","sul","sull","sulla","sulle","sullo","suo","suoi","tale","tali","talvolta","tanto","te","tempo","terzo","th","ti","titolo","tra","tranne","tre","trenta","triplo","troppo","trovato","tu","tua","tue","tuo","tuoi","tutta","tuttavia","tutte","tutti","tutto","uguali","ulteriore","ultimo","un","una","uno","uomo","va","vai","vale","vari","varia","varie","vario","verso","vi","vicino","visto","vita","voi","volta","volte","vostra","vostre","vostri","vostro","è"]

# dictionary of Italian stop-words
it_stop_words = nltk.corpus.stopwords.words('italian')
# Snowball stemmer with rules for the Italian language
ita_stemmer = nltk.stem.snowball.ItalianStemmer()


train_indxs = np.arange(6000)#np.random.randint(0,len(mydata),int(len(mydata)*.8))
test_indxs=np.arange(len(mydata))
test_indxs = np.delete(test_indxs,train_indxs)

mydata_train = mydata[0:6000]
mydata_test = mydata[6001:-1]

messages_dict = dict.fromkeys(all_actors, []) 
all_messages = [all_actors[i] for i in range(len(all_actors))]
messages_dict = dict.fromkeys(all_messages, []) 

all_messages = pd.DataFrame(index=all_messages, columns=['text','common'])

for actr in range(len(all_actors)):
    it_string = ' '.join(word for word in mydata_train[mydata['from']==all_actors[actr]]['text'])
    
    # 1st tokenize the sentence(s)
    word_tokenized_list = nltk.tokenize.word_tokenize(it_string)
    print("1) NLTK tokenizer, num words: {} for list: {}".format(len(word_tokenized_list), word_tokenized_list))
    
    # 2nd remove punctuation and everything lower case
    word_tokenized_no_punct = [str.lower(x) for x in word_tokenized_list if x not in string.punctuation]
    print("2) Clean punctuation, num words: {} for list: {}".format(len(word_tokenized_no_punct), word_tokenized_no_punct))
    
    #Remove also apostrophe
    word_tokenized_no_punct_no_apostrophe = [x.split("'") for x in word_tokenized_no_punct]
    word_tokenized_no_punct_no_apostrophe = [y for x in word_tokenized_no_punct_no_apostrophe for y in x]
    
    # 3rd remove stop words (for the Italian language)
    word_tokenized_no_punct_no_sw = [x for x in word_tokenized_no_punct_no_apostrophe if x not in it_stop_words]
    word_tokenized_no_punct_no_sw = [x for x in word_tokenized_no_punct_no_sw if x not in my_stop_words]
    print("3) Clean stop-words, num words: {} for list: {}".format(len(word_tokenized_no_punct_no_sw), word_tokenized_no_punct_no_sw))
    
    messages_dict[all_actors[actr]] = word_tokenized_no_punct_no_sw
    all_messages['text'][all_actors[actr]]=word_tokenized_no_punct_no_sw


    # finding the frequency distinct in the tokens
    fdist = FreqDist(messages_dict[all_actors[actr]])
    # To find the frequency of top 10 words
    fdist1 = fdist.most_common(20)
    all_messages['common'][all_actors[actr]] = fdist1

    
    # To be continued
