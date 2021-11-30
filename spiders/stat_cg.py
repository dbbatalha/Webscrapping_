#Importando as bibliotecas
from textwrap import indent
import pandas as pd
import numpy as np
from wordcloud import STOPWORDS


#importanto o arquivo gerado pelo webscrapping
arquivo = 'carioca.json'
df = pd.read_json(arquivo)
df.head()

#Consultando linhas com valores faltantes e deletando as linhas

df.isnull().sum()

df.dropna(inplace=True)

df.isnull().sum()

len(df)

df.head()




################# PROCESSAMENTO ##################




# Atentar para o exemplo seguinte para entender melhor 
#len('this is text'.split())
#'this is text'.split()

#Criando novas colunas para fazer uma melhor análise

df['num_palavras'] = df['comentario_corpo'].apply(lambda x: len(str(x).split()))

df.head() #ou df.sample(5)

#definindo a funcao para contar o numero de letras usadas
def char_counts(x):
    s = x.split() #separa as palavras
    x = ''.join(s) #junta as palavras sem espaco
    return len(x) #num de letras

df['num_letr'] = df['comentario_corpo'].apply(lambda x: char_counts(str(x)))


#o tamanho médio das palavras digitadas

df['média_tam_pal'] = df['num_letr']/df['num_palavras'] #pode parecer besteira, mas isso mostra se a pessoa é velha ou nova

#selecionando a nota do comentario

df['nota']=df['comentario_nota'].apply(lambda x: str(x)[0])

#Selecionando a data da melhor forma

df['data']=df['comentario_data'].apply(lambda x: str(x)[:8])
# lembrar de como deletar uma coluna: del df['data']

#removendo as stopwords dos comentarios

from spacy.lang.en.stop_words import STOP_WORDS as stopwords

df['stopwords_num'] = df['comentario_corpo'].apply(lambda x: len([t for t in x.split() if t in stopwords]))

#Contando palavras em Upper

df['upper_num'] = df['comentario_corpo'].apply(lambda x: len([t for t in x.split() if t.isupper()]))


df.head()


# PRE PROCESSAMENTO


#colocando tudo em lower case


df['comentario_corpo_proc'] = df['comentario_corpo'].apply(lambda x: str(x).lower())

#expandindo as contracoes

contractions = { 
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how does",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so is",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
" u ": " you ",
" ur ": " your ",
" n ": " and ",
"won't": "would not",
'dis': 'this',
'bak': 'back',
'brng': 'bring'}



def cont_to_exp(x):
    if type(x) is str:
        for key in contractions:
            value = contractions[key]
            x = x.replace(key, value)
        return x
    else:
        return x

df['comentario_corpo_proc'] = df['comentario_corpo_proc'].apply(lambda x: cont_to_exp(x))

#removendo chracter special

import re

df['comentario_corpo_proc'] = df['comentario_corpo_proc'].apply(lambda x: re.sub(r'[^\w ]+', "", x))


#removendo espacos multiplos

df['comentario_corpo_proc'] = df['comentario_corpo_proc'].apply(lambda x: ' '.join(x.split())) #esse joint irá juntar as palavras com espaco simples

#removendo os acentos

import unicodedata

def remove_accented_chars(x):
    x = unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return x

df['comentario_corpo_proc'] = df['comentario_corpo_proc'].apply(lambda x: remove_accented_chars(x))

#removendo as stopwords

df['comentario_corpo_proc_sstop'] = df['comentario_corpo_proc'].apply(lambda x: ' '.join([t for t in x.split() if t not in stopwords]))

#transformando na raiz da palavra
import spacy
nlp = spacy.load('en_core_web_sm')

def make_to_base(x):
    x = str(x)
    x_list = []
    doc = nlp(x)
    
    for token in doc:
        lemma = token.lemma_
        if lemma == '-PRON-' or lemma == 'be':
            lemma = token.text

        x_list.append(lemma)
    return ' '.join(x_list)

df['comentario_corpo_proc'] = df['comentario_corpo_proc'].apply(lambda x: make_to_base(x))

#removendo as palavras mais comuns

text = ' '.join(df['comentario_corpo_proc']) #junta todos os comentarios

freq_comm = pd.Series(text).value_counts() #conta  freuquencia das palavras

common20 = freq_comm[:20] #pega as 20 palavras mais freuentes

df['comentario_corpo_proc'] = df['comentario_corpo_proc'].apply(lambda x: ' '.join([t for t in x.split() if t not in common20]))

#removendo as palavras mais raras

rare20 = freq_comm.tail(20)

df['comentario_corpo_proc'] = df['comentario_corpo_proc'].apply(lambda x: ' '.join([t for t in x.split() if t not in rare20]))

df.head()











######## MUDANDO O TIPO DAS COLUNAS #################





#ANALISE PRELIMINAR DO ARQUIVO GERADO PELO WEBSCRAPPING E DEPOIS DO PROCESSAMENTO

df.columns #o nome das colunas
df.columns.tolist()

df.count()


df.describe() #a estat. descritiva

#*Convertendo os tipos de dados*

df.info() #vai me dar o tipo de cada coluna (tam.14.6)

del df['comentario_nota']
del df['comentario_data']

df.autor_comentario = df.autor_comentario.astype('string')
df.autor_endereco = df.autor_endereco.astype('string')
df.comentario_titulo= df.comentario_titulo.astype('string')
df.comentario_corpo= df.comentario_corpo.astype('string')
df.num_palavras = df.num_palavras.astype('int32')
df.num_letr = df.num_letr.astype('int32')
df.nota = df.nota.astype('int32')
df.comentario_corpo_proc = df.comentario_corpo_proc.astype('string')
df.comentario_corpo_proc_sstop = df.comentario_corpo_proc_sstop.astype('string')

df.info() #vai me dar o tipo de cada coluna #tam 13.5







########### EXPLORACAO DE DADOS ##################









# Analisando os comentarios
#contando os valores unicos dessas colunas
# for columns in df.columns.tolist():
#     print(pd.value_counts(df[columns]))
 
#ou

pd.value_counts(df['autor_endereco'])
pd.value_counts(df['nota'])
pd.value_counts(df['data'])

#Consultando os comentarios de um pais
df.loc[df['autor_endereco'] == 'London, UK']
#consultando o comentario de um pais que tenha notas iguais a 5
df.loc[(df['autor_endereco'] == 'London, UK') & (df['nota'] == '5')]

#Conclusão inicial: podemos afirmar que a maioria de comentarios são de londrinos que dão uma média de nota 4 e 5. 






#################### GRAFICOS #####################








#ANÁLISE E VISUALIZACAO DE DADOS



#grafico em barras
import matplotlib.pyplot as plt
from nltk import FreqDist

text = ' '.join(df['comentario_corpo_proc']) #juntando todos os comentarios

text = text.split()

freq = FreqDist(text) #analisando a frequencia dos termos


palavras = freq.keys()
y_pos = np.arange(len(palavras))
contagem = freq.values()

plt.bar(y_pos, contagem, align='center', alpha=0.5)
plt.xticks(y_pos, palavras)
plt.ylabel('Frequencia')
plt.title('Frequencia das palavras na frase')

plt.show()



#word cloud 



from wordcloud import WordCloud


text = ' '.join(df['comentario_corpo_proc']) #juntando todos os comentarios

wc = WordCloud(width=800, height=400).generate(text)
plt.imshow(wc)
plt.axis('off')
plt.show() #melhorar essa nuvem de palavras




#Scatter plot



df.plot(x='data',y='nota',kind='scatter', title='Nota x País',color='r')
plt.show()


# Pizza graph


df['nota'].value_counts()
num_notas = [27, 60,8,1,1]
labels = list(df['nota'].unique().flatten())

plt.pie(num_notas, labels=labels)
plt.show()


#Pandas Profiling Report



import pandas_profiling

pandas_profiling.ProfileReport(df)

profile = pandas_profiling.ProfileReport(df)

profile.to_file("report_preview.html")




######### RESOLVENDO CLASSES DESBALANCEADAS ###########


# if len(df.loc[(df['nota'] =='4')]) < len(df.loc[(df['nota'] == '5')]):
#     diff = len(df.loc[(df['nota'] == '5')]) - len(df.loc[(df['nota'] == '4')])
#     data = df[df['nota'] == '4'].iloc[0]

#     df.append([data] * diff, ignore_index=True)

# df['nota'].value_counts()


df.head()
df['nota'].value_counts() #esse banco de dados está muito desbalanceado

df['nota']







############# ML - CLASSIFICACAO #####################






from sklearn.feature_extraction.text import CountVectorizer

# Exemplo: x = ['which book is this', 'this is book and this is math']

# cv = CountVectorizer()
# count = cv.fit_transform(x)

# count.toarray() #BoW

# cv.get_feature_names() #pega todas as palavras do texto

# bow = pd.DataFrame(count.toarray(), columns = cv.get_feature_names())

# bow


#RANDOM FOREST

#DATA PREPARATION FOR TRAINING

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['comentario_corpo_proc'])

X = X.toarray()

X_train, X_test, y_train, y_test = train_test_split(X, df['nota'], test_size = 0.2, random_state = 0)

X_train.shape, X_test.shape

#TREINANDO O CLASSIFICADOR RANDOM FOREST

clf = RandomForestClassifier(n_estimators=100, n_jobs= -1)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

confusion_matrix(y_test, y_pred)


#Relatório de classificação

print(classification_report(y_test, y_pred))

#Lembrar como interpretar precision, recall e F1-score


def predict(x):
    x = tfidf.transform([x])
    x = x.toarray()
    pred = clf.predict(x)
    return pred


predict('I didnt like so much')


#SUPORT VECTOR MACHINE
 
#TREINANDO O CLASSFICADOR SVM

clf = SVC(C = 1000, gamma = 'auto')

clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)

confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))

predict('I didnt like so much')









