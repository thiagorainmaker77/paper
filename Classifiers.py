from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
# classificadores
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier



class Build_Classifiers:

    def __init__(self, cv, tfidf, gl, ft, w2v, cn, train):
        self.cv = cv
        self.tfidf = tfidf
        self.gl = gl
        self.ft = ft
        self.w2v = w2v
        self.cn = cn
        self.train = train

    def build_sgd(self):
        sgd_cv = Pipeline([
            ('NBCV', self.cv),
            ('clf', SGDClassifier(loss='log', penalty='l2', alpha=0.0001, random_state=42))])

        sgd_tf = Pipeline([
            ('NBCV', self.tfidf),
            ('clf', SGDClassifier(loss='log', penalty='l2', alpha=0.0001, random_state=42))])

        sgd_w2 = Pipeline([
            ('NBCV', self.w2v),
            ('clf', SGDClassifier(loss='log', penalty='l2', alpha=0.0001, random_state=42))])

        sgd_gl = Pipeline([
            ('NBCV', self.gl),
            ('clf', SGDClassifier(loss='log', penalty='l2', alpha=0.0001, random_state=42))])

        sgd_ft = Pipeline([
            ('NBCV', self.ft),
            ('clf', SGDClassifier(loss='log', penalty='l2', alpha=0.0001, random_state=42))])

        sgd_cn = Pipeline([
            ('NBCV', self.cn),
            ('clf', SGDClassifier(loss='log', penalty='l2', alpha=0.0001, random_state=42))])

        sgd_cv.fit(self.train['statement'], self.train['label'])
        sgd_tf.fit(self.train['statement'], self.train['label'])
        sgd_w2.fit(self.train['statement'], self.train['label'])
        sgd_gl.fit(self.train['statement'], self.train['label'])
        sgd_ft.fit(self.train['statement'], self.train['label'])
        sgd_cn.fit(self.train['statement'], self.train['label'])

        return sgd_cv, sgd_tf, sgd_gl, sgd_ft, sgd_w2, sgd_cn


    def build_ext(self):
        extp_cv = Pipeline([
            ('NBCV', self.cv),
            ('nb_clf', ExtraTreesClassifier(criterion='gini', n_estimators=150, random_state=42))])

        extp_tf = Pipeline([
            ('NBCV', self.tfidf),
            ('nb_clf', ExtraTreesClassifier(criterion='gini', n_estimators=150,  random_state=42))])

        extp_w2 = Pipeline([
            ('NBCV', self.w2v),
            ('nb_clf', ExtraTreesClassifier(criterion='gini', n_estimators=150,  random_state=42))])

        extp_glove = Pipeline([
            ('NBCV', self.gl),
            ('nb_clf', ExtraTreesClassifier(criterion='gini', n_estimators=150,  random_state=42))])

        extp_fast = Pipeline([
            ('NBCV', self.ft),
            ('nb_clf', ExtraTreesClassifier(criterion='gini', n_estimators=150, random_state=42))])

        extp_cn = Pipeline([
            ('NBCV', self.cn),
            ('nb_clf', ExtraTreesClassifier(criterion='gini', n_estimators=150,  random_state=42))])

        extp_cv.fit(self.train['statement'], self.train['label'])
        extp_tf.fit(self.train['statement'], self.train['label'])
        extp_w2.fit(self.train['statement'], self.train['label'])
        extp_glove.fit(self.train['statement'], self.train['label'])
        extp_fast.fit(self.train['statement'], self.train['label'])
        extp_cn.fit(self.train['statement'], self.train['label'])

        return extp_cv, extp_tf, extp_glove, extp_fast, extp_w2, extp_cn

    def build_mlp(self):
            mlp_cv = Pipeline([
                ('svmCV', self.cv),
                ('svm_clf', MLPClassifier(activation='relu', random_state=42))
            ])

            mlp_tf = Pipeline([
                ('svmCV', self.tfidf),
                ('svm_clf', MLPClassifier(activation='relu', random_state=42))
            ])

            mlp_w2v = Pipeline([
                ('sgd_tfidf', self.w2v),
                ('sgd_clf', MLPClassifier(activation='relu', random_state=42))
            ])

            mlp_gl = Pipeline([
                ('sgd_tfidf', self.gl),
                ('sgd_clf', MLPClassifier(activation='relu', random_state=42))
            ])

            mlp_ft = Pipeline([
                ('sgd_tfidf', self.ft),
                ('sgd_clf', MLPClassifier(activation='relu', random_state=42))
            ])

            mlp_cn = Pipeline([
                ('sgd_tfidf', self.cn),
                ('sgd_clf', MLPClassifier(activation='relu', random_state=42))
            ])

            mlp_cv.fit(self.train['statement'], self.train['label'])
            mlp_tf.fit(self.train['statement'], self.train['label'])
            mlp_w2v.fit(self.train['statement'], self.train['label'])
            mlp_gl.fit(self.train['statement'], self.train['label'])
            mlp_ft.fit(self.train['statement'], self.train['label'])
            mlp_cn.fit(self.train['statement'], self.train['label'])

            return mlp_cv, mlp_tf, mlp_gl, mlp_ft, mlp_w2v, mlp_cn



    def build_knn(self):
        knn_cv = Pipeline([
            ('NBCV', self.cv),
            ('clf', KNeighborsClassifier(algorithm='auto', n_neighbors=15))])

        knn_tf = Pipeline([
            ('NBCV', self.tfidf),
            ('clf', KNeighborsClassifier(algorithm='auto', n_neighbors=25))])

        knn_w2 = Pipeline([
            ('NBCV', self.w2v),
            ('clf', KNeighborsClassifier(algorithm='auto', n_neighbors=25))])

        knn_gl = Pipeline([
            ('NBCV', self.gl),
            ('clf', KNeighborsClassifier(algorithm='auto', n_neighbors=25))])

        knn_ft = Pipeline([
            ('NBCV', self.ft),
            ('clf', KNeighborsClassifier(algorithm='auto', n_neighbors=25))])

        knn_cn = Pipeline([
            ('NBCV', self.cn),
            ('clf', KNeighborsClassifier(algorithm='auto', n_neighbors=25))])

        knn_cv.fit(self.train['statement'], self.train['label'])
        knn_tf.fit(self.train['statement'], self.train['label'])
        knn_w2.fit(self.train['statement'], self.train['label'])
        knn_gl.fit(self.train['statement'], self.train['label'])
        knn_ft.fit(self.train['statement'], self.train['label'])
        knn_cn.fit(self.train['statement'], self.train['label'])

        return knn_cv, knn_tf, knn_gl, knn_ft, knn_w2, knn_cn


    def build_lr(self):
        lr_cv = Pipeline([
            ('NBCV', self.cv),
            ('clf', LogisticRegression(penalty='l2', C=1.0, random_state=42))])

        lr_tf = Pipeline([
            ('NBCV', self.tfidf),
            ('clf', LogisticRegression(penalty='l2', C=1.0, random_state=42))])

        lr_w2v = Pipeline([
            ('NBCV', self.w2v),
            ('clf', LogisticRegression(penalty='l2', C=1.0, random_state=42))])

        lr_gl = Pipeline([
            ('NBCV', self.gl),
            ('clf', LogisticRegression(penalty='l2', C=1.0, random_state=42))])

        lr_ft = Pipeline([
            ('NBCV', self.ft),
            ('clf', LogisticRegression(penalty='l2', C=20.0, random_state=42))])

        lr_cn = Pipeline([
            ('NBCV', self.cn),
            ('clf', LogisticRegression(penalty='l2', C=20.0, random_state=42))])

        lr_ft.fit(self.train['statement'], self.train['label'])
        lr_cn.fit(self.train['statement'], self.train['label'])
        lr_cv.fit(self.train['statement'], self.train['label'])
        lr_tf.fit(self.train['statement'], self.train['label'])
        lr_w2v.fit(self.train['statement'], self.train['label'])
        lr_gl.fit(self.train['statement'], self.train['label'])

        return lr_cv, lr_tf, lr_gl, lr_ft, lr_w2v, lr_cn




    def build_mnb(self ):

        mnb_cv = Pipeline([
            ('1', self.cv),
            ('3', MultinomialNB(alpha=10, fit_prior=False))])

        mnb_tfidf = Pipeline([
            ('1', self.tfidf),
            ('3', MultinomialNB(alpha=10, fit_prior=False))])

        mnb_w2v = Pipeline([
            ('1', self.w2v),
            ('2', MinMaxScaler(feature_range=(0, 1))),
            ('3', MultinomialNB())])

        mnb_fasttext = Pipeline([
            ('1', self.ft),
            ('2', MinMaxScaler(feature_range=(0, 1))),
            ('3', MultinomialNB(alpha=1, fit_prior=True))])

        mnb_glove = Pipeline([
            ('1', self.gl),
            ('2', MinMaxScaler(feature_range=(0, 1))),
            ('3', MultinomialNB(alpha=1, fit_prior=True))])


        mnb_cn = Pipeline([
            ('1', self.cn),
            ('2', MinMaxScaler(feature_range=(0, 1))),
            ('3',  MultinomialNB(alpha=1, fit_prior=True))])


        mnb_fasttext.fit(self.train['statement'], self.train['label'])
        mnb_cn.fit(self.train['statement'], self.train['label'])
        mnb_cv.fit(self.train['statement'], self.train['label'])
        mnb_tfidf.fit(self.train['statement'], self.train['label'])
        mnb_w2v.fit(self.train['statement'], self.train['label'])
        mnb_glove.fit(self.train['statement'], self.train['label'])

        return  mnb_cv, mnb_tfidf, mnb_glove,  mnb_fasttext, mnb_w2v, mnb_cn


    def build_rf(self):
        rf_cv = Pipeline([
            ('NBCV', self.cv),
            ('clf', RandomForestClassifier(n_estimators=35, random_state=42, criterion='gini'))])

        rf_tf = Pipeline([
            ('NBCV', self.tfidf),
            ('clf', RandomForestClassifier(n_estimators=35, random_state=42, criterion='gini'))])

        rf_w2 = Pipeline([
            ('NBCV', self.w2v),
            ('clf', RandomForestClassifier(n_estimators=150, random_state=42, criterion='gini'))])

        rf_gl = Pipeline([
            ('NBCV', self.gl),
            ('clf', RandomForestClassifier(n_estimators=7, random_state=42, criterion='gini'))])

        rf_ft = Pipeline([
            ('NBCV', self.ft),
            ('clf', RandomForestClassifier(n_estimators=150, random_state=42, criterion='gini'))])

        rf_cn = Pipeline([
            ('NBCV', self.cn),
            ('clf', RandomForestClassifier(n_estimators=150, random_state=42, criterion='gini'))])

        rf_cv.fit(self.train['statement'], self.train['label'])
        rf_tf.fit(self.train['statement'], self.train['label'])
        rf_w2.fit(self.train['statement'], self.train['label'])
        rf_gl.fit(self.train['statement'], self.train['label'])
        rf_ft.fit(self.train['statement'], self.train['label'])
        rf_cn.fit(self.train['statement'], self.train['label'])

        return rf_cv, rf_tf, rf_gl, rf_ft, rf_w2, rf_cn
