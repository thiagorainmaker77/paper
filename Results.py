#--(mnb_tf)-(ext_ft)--(sgd_ft)--(sgd_w2) ---(rf_cn) -
#0.2920284135753749




import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from warnings import filterwarnings
filterwarnings('ignore')



from sklearn.pipeline import Pipeline
import Util as util
# classificadores
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier



from Extrator import  Gerar_extrator

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from mlxtend.classifier import EnsembleVoteClassifier
from itertools import combinations
from sklearn.metrics import accuracy_score
import pandas as pd


from Classifiers import  Build_Classifiers

####################### Dataset
print('...........................................................Dataset')

uri_train = 'https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/train.tsv'
uri_valid = 'https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/valid.tsv'
uri_test = 'https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/test.tsv'

df_train = pd.read_table(uri_train,
                         names=['id', 'label', 'statement', 'subject', 'speaker', 'job', 'state', 'party',
                                'barely_true_c', 'false_c', 'half_true_c', 'mostly_true_c', 'pants_on_fire_c',
                                'venue'])
df_valid = pd.read_table(uri_valid,
                         names=['id', 'label', 'statement', 'subject', 'speaker', 'job', 'state', 'party',
                                'barely_true_c', 'false_c', 'half_true_c', 'mostly_true_c', 'pants_on_fire_c',
                                'venue'])
df_test = pd.read_csv(uri_test, sep='\t',
                      names=['id', 'label', 'statement', 'subject', 'speaker', 'job', 'state', 'party',
                             'barely_true_c', 'false_c', 'half_true_c', 'mostly_true_c', 'pants_on_fire_c',
                             'venue'])




print('--------------------------( W2V )--------------------------------')
w2v = Gerar_extrator('word2vec-google-news-300')
print('--------------------------( GL  )--------------------------------')
glove = Gerar_extrator('glove-wiki-gigaword-300')
print('--------------------------( FT  )--------------------------------')
fasttext = Gerar_extrator('fasttext-wiki-news-subwords-300')
print('--------------------------( CN  )--------------------------------')
conceptnet = Gerar_extrator('conceptnet-numberbatch-17-06-300')

cv = CountVectorizer(analyzer='word', stop_words='english')
cv.fit_transform(df_train['statement'])

tfidf = TfidfVectorizer(analyzer='word', stop_words='english' )
tfidf.fit_transform(df_train['statement'])



#--(mnb_tf)-(ext_ft)--(sgd_ft)--(sgd_w2) ---(rf_cn) -


mnb_tfidf = Pipeline([
    ('1', tfidf),
    ('3', MultinomialNB(alpha=10, fit_prior=False))])

extp_fast = Pipeline([
    ('NBCV', fasttext),
    ('nb_clf', ExtraTreesClassifier(criterion='gini', n_estimators=150, random_state=42))])

sgd_ft = Pipeline([
    ('NBCV', fasttext),
    ('clf', SGDClassifier(loss='log', penalty='l2', alpha=0.0001, random_state=42))])

sgd_w2 = Pipeline([
    ('NBCV', w2v),
    ('clf', SGDClassifier(loss='log', penalty='l2', alpha=0.0001, random_state=42))])

rf_cn = Pipeline([
    ('NBCV', conceptnet),
    ('clf', RandomForestClassifier(n_estimators=150, random_state=42, criterion='gini'))])


mnb_tfidf.fit(df_train['statement'], df_train['label'])
extp_fast.fit(df_train['statement'], df_train['label'])
sgd_ft.fit(df_train['statement'], df_train['label'])
sgd_w2.fit(df_train['statement'], df_train['label'])
rf_cn.fit(df_train['statement'], df_train['label'])



mnb_tfidf_t    = mnb_tfidf.predict_proba(df_test['statement'])
mnb_tfidf_v    = mnb_tfidf.predict_proba(df_valid['statement'])

ext_fasttext_t = extp_fast.predict_proba(df_test['statement'])
ext_fasttext_v = extp_fast.predict_proba(df_valid['statement'])

sgd_fasttext_t = sgd_ft.predict_proba(df_test['statement'])
sgd_fasttext_v = sgd_ft.predict_proba(df_valid['statement'])


sgd_w2v_t      = sgd_w2.predict_proba(df_test['statement'])
sgd_w2v_v      = sgd_w2.predict_proba(df_valid['statement'])


rf_cn_t       = rf_cn.predict_proba(df_test['statement'])
rf_cn_v       = rf_cn.predict_proba(df_valid['statement'])


mnb_df_cv_v = util.build_df('mnb_cv', mnb_tfidf_v )
mnb_df_cv_t = util.build_df('mnb_cv', mnb_tfidf_t )

ext_df_ft_v = util.build_df('ext_ft', ext_fasttext_v )
ext_df_ft_t = util.build_df('ext_ft', ext_fasttext_t )

sgd_df_ft_v = util.build_df('sgd_ft', sgd_fasttext_v )
sgd_df_ft_t = util.build_df('sgd_ft', sgd_fasttext_t )


sgd_df_w2_v = util.build_df('sgd_w2', sgd_w2v_v )
sgd_df_w2_t = util.build_df('sgd_w2', sgd_w2v_t )


rf_df_cn_v = util.build_df('rf_cn', rf_cn_v )
rf_df_cn_t = util.build_df('rf_cn', rf_cn_t )


valid_prob = pd.concat([mnb_df_cv_v, ext_df_ft_v, sgd_df_ft_v, sgd_df_w2_v, rf_df_cn_v], axis=1, join='inner')
teste_prob = pd.concat([mnb_df_cv_t, ext_df_ft_t, sgd_df_ft_t, sgd_df_w2_t, rf_df_cn_t], axis=1, join='inner')

X_valid = valid_prob
y_valid = df_valid['label']

X_teste = teste_prob
y_teste = df_test['label']



mlp = MLPClassifier(activation='relu', random_state=42)
mlp.fit(X_valid, y_valid)
resultado = mlp.predict(X_teste)
score = accuracy_score(df_test['label'], resultado)
print(str(score))

print(round(score, 3))
print('-----------------------------------------------')