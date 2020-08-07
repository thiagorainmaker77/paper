
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

tfidf = TfidfVectorizer(analyzer='word', stop_words='english')
tfidf.fit_transform(df_train['statement'])


cls = Build_Classifiers(cv, tfidf, glove, fasttext, w2v, conceptnet, df_train)


cls_n = {}

print('--------------------------(Classificadores)--------------------------------')

print('--------------------------(       MNB     )--------------------------------')
######################## MNB

mnb_cv, mnb_tfidf, mnb_gl, mnb_ft, mnb_w2v, mnb_cn = cls.build_mnb()

mnb_cv_t       = mnb_cv.predict_proba(df_test['statement'])
mnb_tfidf_t    = mnb_tfidf.predict_proba(df_test['statement'])
mnb_cn_t       = mnb_cn.predict_proba(df_test['statement'])
mnb_glove_t    = mnb_gl.predict_proba(df_test['statement'])
mnb_fasttext_t = mnb_ft.predict_proba(df_test['statement'])
mnb_w2v_t      = mnb_w2v.predict_proba(df_test['statement'])


mnb_cv_v       = mnb_cv.predict_proba(df_valid['statement'])
mnb_tfidf_v    = mnb_tfidf.predict_proba(df_valid['statement'])
mnb_cn_v       = mnb_cn.predict_proba(df_valid['statement'])
mnb_glove_v    = mnb_gl.predict_proba(df_valid['statement'])
mnb_fasttext_v = mnb_ft.predict_proba(df_valid['statement'])
mnb_w2v_v      = mnb_w2v.predict_proba(df_valid['statement'])




cls_n['mnb_cv']     = 'mnb_cv'
cls_n['mnb_tf']     = 'mnb_tf'
cls_n['mnb_cn']     = 'mnb_cn'
cls_n['mnb_gl']     = 'mnb_gl'
cls_n['mnb_ft']     = 'mnb_ft'
cls_n['mnb_w2']     = 'mnb_w2'


df_cv_v = util.build_df('mnb_cv', mnb_cv_v )
df_tf_v = util.build_df('mnb_tf', mnb_tfidf_v )
df_cn_v = util.build_df('mnb_cn', mnb_cn_v )
df_gl_v = util.build_df('mnb_gl', mnb_glove_v )
df_ft_v = util.build_df('mnb_ft', mnb_fasttext_v )
df_w2_v = util.build_df('mnb_w2', mnb_w2v_v )


df_cv_t = util.build_df('mnb_cv', mnb_cv_t )
df_tf_t = util.build_df('mnb_tf', mnb_tfidf_t )
df_cn_t = util.build_df('mnb_cn', mnb_cn_t )
df_gl_t = util.build_df('mnb_gl', mnb_glove_t )
df_ft_t = util.build_df('mnb_ft', mnb_fasttext_t )
df_w2_t = util.build_df('mnb_w2', mnb_w2v_t )

valid_prob = pd.concat([df_tf_v, df_cv_v, df_cn_v, df_gl_v, df_ft_v, df_w2_v], axis=1, join='inner')
teste_prob = pd.concat([df_tf_t, df_cv_t, df_cn_t, df_gl_t, df_ft_t, df_w2_t], axis=1, join='inner')



############################# LR

print('--------------------------(       LR      )--------------------------------')

lr_cv, lr_tfidf, lr_gl, lr_ft, lr_w2v, lr_cn = cls.build_lr()

lr_cv_t       = lr_cv.predict_proba(df_test['statement'])
lr_tfidf_t    = lr_tfidf.predict_proba(df_test['statement'])
lr_cn_t       = lr_cn.predict_proba(df_test['statement'])
lr_glove_t    = lr_gl.predict_proba(df_test['statement'])
lr_fasttext_t = lr_ft.predict_proba(df_test['statement'])
lr_w2v_t      = lr_w2v.predict_proba(df_test['statement'])


lr_cv_v       = lr_cv.predict_proba(df_valid['statement'])
lr_tfidf_v    = lr_tfidf.predict_proba(df_valid['statement'])
lr_cn_v       = lr_cn.predict_proba(df_valid['statement'])
lr_glove_v    = lr_gl.predict_proba(df_valid['statement'])
lr_fasttext_v = lr_ft.predict_proba(df_valid['statement'])
lr_w2v_v      = lr_w2v.predict_proba(df_valid['statement'])




cls_n['lr_cv']     = 'lr_cv'
cls_n['lr_tf']     = 'lr_tf'
cls_n['lr_cn']     = 'lr_cn'
cls_n['lr_gl']     = 'lr_gl'
cls_n['lr_ft']     = 'lr_ft'
cls_n['lr_w2']     = 'lr_w2'


df_cv_v = util.build_df('lr_cv', lr_cv_v )
df_tf_v = util.build_df('lr_tf', lr_tfidf_v )
df_cn_v = util.build_df('lr_cn', lr_cn_v )
df_gl_v = util.build_df('lr_gl', lr_glove_v )
df_ft_v = util.build_df('lr_ft', lr_fasttext_v )
df_w2_v = util.build_df('lr_w2', lr_w2v_v )


df_cv_t = util.build_df('lr_cv', lr_cv_t )
df_tf_t = util.build_df('lr_tf', lr_tfidf_t )
df_cn_t = util.build_df('lr_cn', lr_cn_t )
df_gl_t = util.build_df('lr_gl', lr_glove_t )
df_ft_t = util.build_df('lr_ft', lr_fasttext_t )
df_w2_t = util.build_df('lr_w2', lr_w2v_t )



valid_prob = pd.concat([valid_prob, df_tf_v, df_cv_v, df_cn_v, df_gl_v, df_ft_v, df_w2_v], axis=1, join='inner')
teste_prob = pd.concat([teste_prob, df_tf_t, df_cv_t, df_cn_t, df_gl_t, df_ft_t, df_w2_t], axis=1, join='inner')


############################# EXT

print('--------------------------(       EXT     )--------------------------------')

ext_cv, ext_tfidf, ext_gl, ext_ft, ext_w2v, ext_cn = cls.build_ext()

ext_cv_t       = ext_cv.predict_proba(df_test['statement'])
ext_tfidf_t    = ext_tfidf.predict_proba(df_test['statement'])
ext_cn_t       = ext_cn.predict_proba(df_test['statement'])
ext_glove_t    = ext_gl.predict_proba(df_test['statement'])
ext_fasttext_t = ext_ft.predict_proba(df_test['statement'])
ext_w2v_t      = ext_w2v.predict_proba(df_test['statement'])


ext_cv_v       = ext_cv.predict_proba(df_valid['statement'])
ext_tfidf_v    = ext_tfidf.predict_proba(df_valid['statement'])
ext_cn_v       = ext_cn.predict_proba(df_valid['statement'])
ext_glove_v    = ext_gl.predict_proba(df_valid['statement'])
ext_fasttext_v = ext_ft.predict_proba(df_valid['statement'])
ext_w2v_v      = ext_w2v.predict_proba(df_valid['statement'])




cls_n['ext_cv']     = 'ext_cv'
cls_n['ext_tf']     = 'ext_tf'
cls_n['ext_cn']     = 'ext_cn'
cls_n['ext_gl']     = 'ext_gl'
cls_n['ext_ft']     = 'ext_ft'
cls_n['ext_w2']     = 'ext_w2'


df_cv_v = util.build_df('ext_cv', ext_cv_v )
df_tf_v = util.build_df('ext_tf', ext_tfidf_v )
df_cn_v = util.build_df('ext_cn', ext_cn_v )
df_gl_v = util.build_df('ext_gl', ext_glove_v )
df_ft_v = util.build_df('ext_ft', ext_fasttext_v )
df_w2_v = util.build_df('ext_w2', ext_w2v_v )


df_cv_t = util.build_df('ext_cv', ext_cv_t )
df_tf_t = util.build_df('ext_tf', ext_tfidf_t )
df_cn_t = util.build_df('ext_cn', ext_cn_t )
df_gl_t = util.build_df('ext_gl', ext_glove_t )
df_ft_t = util.build_df('ext_ft', ext_fasttext_t )
df_w2_t = util.build_df('ext_w2', ext_w2v_t )


valid_prob = pd.concat([valid_prob, df_tf_v, df_cv_v, df_cn_v, df_gl_v, df_ft_v, df_w2_v], axis=1, join='inner')
teste_prob = pd.concat([teste_prob, df_tf_t, df_cv_t, df_cn_t, df_gl_t, df_ft_t, df_w2_t], axis=1, join='inner')

############################# SGD

print('--------------------------(       SGD     )--------------------------------')

sgd_cv, sgd_tfidf, sgd_gl, sgd_ft, sgd_w2v, sgd_cn = cls.build_ext()

sgd_cv_t       = sgd_cv.predict_proba(df_test['statement'])
sgd_tfidf_t    = sgd_tfidf.predict_proba(df_test['statement'])
sgd_cn_t       = sgd_cn.predict_proba(df_test['statement'])
sgd_glove_t    = sgd_gl.predict_proba(df_test['statement'])
sgd_fasttext_t = sgd_ft.predict_proba(df_test['statement'])
sgd_w2v_t      = sgd_w2v.predict_proba(df_test['statement'])


sgd_cv_v       = sgd_cv.predict_proba(df_valid['statement'])
sgd_tfidf_v    = sgd_tfidf.predict_proba(df_valid['statement'])
sgd_cn_v       = sgd_cn.predict_proba(df_valid['statement'])
sgd_glove_v    = sgd_gl.predict_proba(df_valid['statement'])
sgd_fasttext_v = sgd_ft.predict_proba(df_valid['statement'])
sgd_w2v_v      = sgd_w2v.predict_proba(df_valid['statement'])




cls_n['sgd_cv']     = 'sgd_cv'
cls_n['sgd_tf']     = 'sgd_tf'
cls_n['sgd_cn']     = 'sgd_cn'
cls_n['sgd_gl']     = 'sgd_gl'
cls_n['sgd_ft']     = 'sgd_ft'
cls_n['sgd_w2']     = 'sgd_w2'


df_cv_v = util.build_df('sgd_cv', sgd_cv_v )
df_tf_v = util.build_df('sgd_tf', sgd_tfidf_v )
df_cn_v = util.build_df('sgd_cn', sgd_cn_v )
df_gl_v = util.build_df('sgd_gl', sgd_glove_v )
df_ft_v = util.build_df('sgd_ft', sgd_fasttext_v )
df_w2_v = util.build_df('sgd_w2', sgd_w2v_v )


df_cv_t = util.build_df('sgd_cv', sgd_cv_t )
df_tf_t = util.build_df('sgd_tf', sgd_tfidf_t )
df_cn_t = util.build_df('sgd_cn', sgd_cn_t )
df_gl_t = util.build_df('sgd_gl', sgd_glove_t )
df_ft_t = util.build_df('sgd_ft', sgd_fasttext_t )
df_w2_t = util.build_df('sgd_w2', sgd_w2v_t )


valid_prob = pd.concat([valid_prob, df_tf_v, df_cv_v, df_cn_v, df_gl_v, df_ft_v, df_w2_v], axis=1, join='inner')
teste_prob = pd.concat([teste_prob, df_tf_t, df_cv_t, df_cn_t, df_gl_t, df_ft_t, df_w2_t], axis=1, join='inner')


############################# RF

print('--------------------------(       RF      )--------------------------------')

rf_cv, rf_tfidf, rf_gl, rf_ft, rf_w2v, rf_cn = cls.build_ext()

rf_cv_t       = rf_cv.predict_proba(df_test['statement'])
rf_tfidf_t    = rf_tfidf.predict_proba(df_test['statement'])
rf_cn_t       = rf_cn.predict_proba(df_test['statement'])
rf_glove_t    = rf_gl.predict_proba(df_test['statement'])
rf_fasttext_t = rf_ft.predict_proba(df_test['statement'])
rf_w2v_t      = rf_w2v.predict_proba(df_test['statement'])


rf_cv_v       = rf_cv.predict_proba(df_valid['statement'])
rf_tfidf_v    = rf_tfidf.predict_proba(df_valid['statement'])
rf_cn_v       = rf_cn.predict_proba(df_valid['statement'])
rf_glove_v    = rf_gl.predict_proba(df_valid['statement'])
rf_fasttext_v = rf_ft.predict_proba(df_valid['statement'])
rf_w2v_v      = rf_w2v.predict_proba(df_valid['statement'])




cls_n['rf_cv']     = 'rf_cv'
cls_n['rf_tf']     = 'rf_tf'
cls_n['rf_cn']     = 'rf_cn'
cls_n['rf_gl']     = 'rf_gl'
cls_n['rf_ft']     = 'rf_ft'
cls_n['rf_w2']     = 'rf_w2'


df_cv_v = util.build_df('rf_cv', rf_cv_v )
df_tf_v = util.build_df('rf_tf', rf_tfidf_v )
df_cn_v = util.build_df('rf_cn', rf_cn_v )
df_gl_v = util.build_df('rf_gl', rf_glove_v )
df_ft_v = util.build_df('rf_ft', rf_fasttext_v )
df_w2_v = util.build_df('rf_w2', rf_w2v_v )


df_cv_t = util.build_df('rf_cv', rf_cv_t )
df_tf_t = util.build_df('rf_tf', rf_tfidf_t )
df_cn_t = util.build_df('rf_cn', rf_cn_t )
df_gl_t = util.build_df('rf_gl', rf_glove_t )
df_ft_t = util.build_df('rf_ft', rf_fasttext_t )
df_w2_t = util.build_df('rf_w2', rf_w2v_t )


valid_prob = pd.concat([valid_prob, df_tf_v, df_cv_v, df_cn_v, df_gl_v, df_ft_v, df_w2_v], axis=1, join='inner')
teste_prob = pd.concat([teste_prob, df_tf_t, df_cv_t, df_cn_t, df_gl_t, df_ft_t, df_w2_t], axis=1, join='inner')



############################# mlp

print('--------------------------(       MLP     )--------------------------------')


mlp_cv, mlp_tfidf, mlp_gl, mlp_ft, mlp_w2v, mlp_cn = cls.build_ext()

mlp_cv_t       = mlp_cv.predict_proba(df_test['statement'])
mlp_tfidf_t    = mlp_tfidf.predict_proba(df_test['statement'])
mlp_cn_t       = mlp_cn.predict_proba(df_test['statement'])
mlp_glove_t    = mlp_gl.predict_proba(df_test['statement'])
mlp_fasttext_t = mlp_ft.predict_proba(df_test['statement'])
mlp_w2v_t      = mlp_w2v.predict_proba(df_test['statement'])


mlp_cv_v       = mlp_cv.predict_proba(df_valid['statement'])
mlp_tfidf_v    = mlp_tfidf.predict_proba(df_valid['statement'])
mlp_cn_v       = mlp_cn.predict_proba(df_valid['statement'])
mlp_glove_v    = mlp_gl.predict_proba(df_valid['statement'])
mlp_fasttext_v = mlp_ft.predict_proba(df_valid['statement'])
mlp_w2v_v      = mlp_w2v.predict_proba(df_valid['statement'])




cls_n['mlp_cv']     = 'mlp_cv'
cls_n['mlp_tf']     = 'mlp_tf'
cls_n['mlp_cn']     = 'mlp_cn'
cls_n['mlp_gl']     = 'mlp_gl'
cls_n['mlp_ft']     = 'mlp_ft'
cls_n['mlp_w2']     = 'mlp_w2'


df_cv_v = util.build_df('mlp_cv', mlp_cv_v )
df_tf_v = util.build_df('mlp_tf', mlp_tfidf_v )
df_cn_v = util.build_df('mlp_cn', mlp_cn_v )
df_gl_v = util.build_df('mlp_gl', mlp_glove_v )
df_ft_v = util.build_df('mlp_ft', mlp_fasttext_v )
df_w2_v = util.build_df('mlp_w2', mlp_w2v_v )


df_cv_t = util.build_df('mlp_cv', mlp_cv_t )
df_tf_t = util.build_df('mlp_tf', mlp_tfidf_t )
df_cn_t = util.build_df('mlp_cn', mlp_cn_t )
df_gl_t = util.build_df('mlp_gl', mlp_glove_t )
df_ft_t = util.build_df('mlp_ft', mlp_fasttext_t )
df_w2_t = util.build_df('mlp_w2', mlp_w2v_t )


valid_prob = pd.concat([valid_prob, df_tf_v, df_cv_v, df_cn_v, df_gl_v, df_ft_v, df_w2_v], axis=1, join='inner')
teste_prob = pd.concat([teste_prob, df_tf_t, df_cv_t, df_cn_t, df_gl_t, df_ft_t, df_w2_t], axis=1, join='inner')



############################# knn
print('--------------------------(       KNN     )--------------------------------')


knn_cv, knn_tfidf, knn_gl, knn_ft, knn_w2v, knn_cn = cls.build_ext()

knn_cv_t       = knn_cv.predict_proba(df_test['statement'])
knn_tfidf_t    = knn_tfidf.predict_proba(df_test['statement'])
knn_cn_t       = knn_cn.predict_proba(df_test['statement'])
knn_glove_t    = knn_gl.predict_proba(df_test['statement'])
knn_fasttext_t = knn_ft.predict_proba(df_test['statement'])
knn_w2v_t      = knn_w2v.predict_proba(df_test['statement'])


knn_cv_v       = knn_cv.predict_proba(df_valid['statement'])
knn_tfidf_v    = knn_tfidf.predict_proba(df_valid['statement'])
knn_cn_v       = knn_cn.predict_proba(df_valid['statement'])
knn_glove_v    = knn_gl.predict_proba(df_valid['statement'])
knn_fasttext_v = knn_ft.predict_proba(df_valid['statement'])
knn_w2v_v      = knn_w2v.predict_proba(df_valid['statement'])




cls_n['knn_cv']     = 'knn_cv'
cls_n['knn_tf']     = 'knn_tf'
cls_n['knn_cn']     = 'knn_cn'
cls_n['knn_gl']     = 'knn_gl'
cls_n['knn_ft']     = 'knn_ft'
cls_n['knn_w2']     = 'knn_w2'


df_cv_v = util.build_df('knn_cv', knn_cv_v )
df_tf_v = util.build_df('knn_tf', knn_tfidf_v )
df_cn_v = util.build_df('knn_cn', knn_cn_v )
df_gl_v = util.build_df('knn_gl', knn_glove_v )
df_ft_v = util.build_df('knn_ft', knn_fasttext_v )
df_w2_v = util.build_df('knn_w2', knn_w2v_v )


df_cv_t = util.build_df('knn_cv', knn_cv_t )
df_tf_t = util.build_df('knn_tf', knn_tfidf_t )
df_cn_t = util.build_df('knn_cn', knn_cn_t )
df_gl_t = util.build_df('knn_gl', knn_glove_t )
df_ft_t = util.build_df('knn_ft', knn_fasttext_t )
df_w2_t = util.build_df('knn_w2', knn_w2v_t )


valid_prob = pd.concat([valid_prob, df_tf_v, df_cv_v, df_cn_v, df_gl_v, df_ft_v, df_w2_v], axis=1, join='inner')
teste_prob = pd.concat([teste_prob, df_tf_t, df_cv_t, df_cn_t, df_gl_t, df_ft_t, df_w2_t], axis=1, join='inner')


print('--------------------------( COMBINAÇÕES )--------------------------------')

comb = combinations(cls_n, 3)

# Print the obtained combinations
lim = 0
k = 0
arquivo = open('resultados_knn_mlp_3.txt', 'w')
for i in list(comb):
    k = k+1
    if k < 49680:
        continue
    print(k)
    feature_cols = [
        str(i[0]) + str('_0'), str(i[0]) + str('_1'), str(i[0]) + str('_2'), str(i[0]) + str('_3'),
        str(i[0]) + str('_4'), str(i[0]) + str('_5'),
        str(i[1]) + str('_0'), str(i[1]) + str('_1'), str(i[1]) + str('_2'), str(i[1]) + str('_3'),
        str(i[1]) + str('_4'), str(i[1]) + str('_5'),
        str(i[2]) + str('_0'), str(i[2]) + str('_1'), str(i[2]) + str('_2'), str(i[2]) + str('_3'),
        str(i[2]) + str('_4'), str(i[2]) + str('_5'),

    ]

    '''
     str(i[3]) + str('_0'), str(i[3]) + str('_1'), str(i[3]) + str('_2'), str(i[3]) + str('_3'),
        str(i[3]) + str('_4'), str(i[3]) + str('_5'),
        str(i[4]) + str('_0'), str(i[4]) + str('_1'), str(i[4]) + str('_2'), str(i[4]) + str('_3'),
        str(i[4]) + str('_4'), str(i[4]) + str('_5'),
    '''
    print(i[0], i[1], i[2])

    X_valid = valid_prob.loc[:, feature_cols]
    y_valid = df_valid['label']

    X_teste = teste_prob.loc[:, feature_cols]
    y_teste = df_test['label']


    arquivo.write('--(' + i[0] + ')-(' + i[1] + ')--(' + i[2] + ')- -\n')

    mlp = MLPClassifier(activation='relu', random_state=42)
    mlp.fit(X_valid, y_valid)
    resultado = mlp.predict(X_teste)
    score = accuracy_score(df_test['label'], resultado)
    arquivo.write(str(score) + '\n')


    print(round(score, 3))
    print('-----------------------------------------------')