#!/usr/bin/env python
# coding: utf-8

# # Webscrapping 

# In[ ]:


case_n1 = []
case_f1 = []


# In[ ]:


from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import NoSuchElementException
driver = webdriver.Chrome(ChromeDriverManager().install())
link = "https://www.justice.gov/atr/antitrust-case-filings-alpha"
driver.get(link)
case_names = []
case_fj = []
for i in range(110,154):
    try:
        case = driver.find_element_by_xpath('//*[@id="page-2"]/p['+ str(i) +']/a')
        case_names.append(case.text)
        case.click()
        field_items = driver.find_element_by_class_name('field__items')
        w = 'Final'
        if w in field_items.text.split(' '):
            try:
                final_judgment = driver.find_element_by_xpath('/html/body/section[3]/div[2]/div/div/div/article/div[1]/div[1]/div/div/p[1]/a')
                final_judgment.click()
                try:
                    f_j = driver.find_element_by_xpath('/html/body/section[3]/div[2]/div/div/div/article/div[1]/div[4]/div/div')
                    case_fj.append(f_j.text)
                    driver.back()
                    driver.back()
                except NoSuchElementException:
                    case_fj.append('')
                    driver.back()
                    driver.back()
            except NoSuchElementException:
                case_fj.append('')
                driver.back()
        else:
            case_fj.append('')
            driver.back()
    except NoSuchElementException:
        pass


# In[ ]:


print(len(case_names), len(case_fj), len(case_n1), len(case_f1))


# In[ ]:


case_n1 += case_names
case_f1 += case_fj


# In[ ]:


c = 0
for i in case_f1:
    if i != '':
        c+= 1
print(c)


# # Importing Packages
# 

# In[ ]:


get_ipython().system(' pip install graphviz')


# In[1]:


import pandas as pd 
import numpy as np
import graphviz
import scipy.sparse
import string
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import pos_tag 
nltk.download('vader_lexicon')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import re
import string


# In[2]:


from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve, roc_auc_score


# # Creating a Vocabulary for antitrust

# In[4]:


anti_trust_monogram = ['cartel', 'conspiracy', 'collusion', 'tacit', 'parallelism', 'oligopolistic', 'oligopoly', 'monopoly', 'monopolistic',
                       'RPM','barriers', 'concentration','unilateral','coordinated','divestiture','bundling','tying','consignment','marginalize',
                       'bundling','metering','abuse','predation','preditory','cointegration','dominance','dominant','marketshare','coconspirator','bribery','rigging','fraud',
                       'divestitures']

anti_trust_bigram = ['market power','tacit collusion','explicit collusion','cartel stability','price fixing','monopoly power','exclusionary contract',
                     'increasing barriers', 'high concentration','unilateral effect','coordinated effect','low synergy','exclusionary conduct',
                     'exclusive dealing','double marginalize','tied sale','vertical integration','price discrimination','critical loss','diversion ratio',
                     'market allocation']


# # Importing Dataset 

# In[5]:


df = pd.read_csv('Final_jundgement_2.csv')

df1 = df.dropna(axis=0)

df1 = df1.rename(columns={'final judgement':'final_jud'})

df1.head()


# # Text Preprocessing 

# In[6]:


# set stop words
new_stop = ['document','available','page','judgement','defendant','documents','browsing',
            'formats','pdf','additional','add','address','accomplished','able','additional','trustee','asset',
            'florida','data','business','fiber','dc','product','physician','effort','theatre','dc',
            'washington','diamond','line','sheet','th','street','limited'] # creating a list of additional stop words
stop = nltk.corpus.stopwords.words('english')
stop = set(stopwords.words('english')) # stop words from NLTK
stop = stop.union(["cant"])
stop = stop.union(new_stop) # adding more words to list 

# set punctuations
exclude = set(string.punctuation)
exclude.update(['‘','’','“','”',':',';'])

# initialize lemmatizer and stemmer
lemma = WordNetLemmatizer()
ps=PorterStemmer()

def clean(doc):
    numb_free = ''.join([i for i in doc if not i.isdigit()]) # exclude digits
    punc_free = ''.join(ch for ch in numb_free if ch not in exclude) # exclude puntuations
    stop_free = " ".join([i for i in punc_free.lower().split() if i not in stop]) # exclude stopwords
    normalized = [lemma.lemmatize(word) for word in stop_free.split()] # bring word to its root
    normalized2 = " ".join([word for word in normalized if len(word) != 1])
    return normalized2


# In[7]:


clean_doc = [clean(text).split() for text in df1]
df1["clean"] = [clean(text) for text in df1["final_jud"]]
df1.head(10)


# Extracting all nouns and adjectives from the dataset to make it easier to process and reduce deminsionality. 

# In[8]:


# Creating a function to pull out nouns & adjectives from a string of text
def nouns_adj(text):
    '''Given a string of text, tokenize the text and pull out only the nouns and adjectives.'''
    is_noun_adj = lambda pos: pos[:2] == 'NN' or pos[:2] == 'JJ'
    tokenized = word_tokenize(text)
    nouns_adj = [word for (word, pos) in pos_tag(tokenized) if is_noun_adj(pos)] 
    return ' '.join(nouns_adj)


# Extracted all nouns and adjectives from the documents. 

# In[9]:


# Applying the nouns_adjective function to the transcripts to filter only on nouns and adjectives
data_nouns_adj = pd.DataFrame(df1.clean.apply(nouns_adj))
data_nouns_adj['clean'][1]


# In[10]:


# find the most occuring words in the corpus 
freq = pd.Series(''.join(df1['clean']).split()).value_counts()[:20]
freq


# # Creating an Index for Antitrust words in the document

# In[11]:


# function to count the number of specific antitrust words to identify antitrust case 
def antitrust_count(doc):
    count = 0
    words = doc.split()
    for i in range(len(words)):
        if words[i] in anti_trust_monogram:
           count += 1
    return count


# In[12]:


df1['word_count'] = [len(text.split()) for text in df1.clean] # count the number of word for each article
df1['antitust_uncertain_count'] = [antitrust_count(text) for text in df1.clean] # count with an specific antitrust word
df1.head()


# # Creating an outcome Label for the document.

# A label is created based on the antitrust vocabulary. A threshold is also set at 10. Therefore if the number of antitrust words in the created vocabulary appear less than the threshold, then the document is lablled 0 otherwise labelled 1 

# In[13]:


Outcome = [] # empty list 

for i, row in df1.iterrows(): # looping over created dataset to determine outcome 
  if row['antitust_uncertain_count'] <= 10:
    Outcome.append('0')
  else:
    Outcome.append('1')

df1['Label'] = np.array(Outcome)

df1.head()


# # Feature Engineering 

# In[14]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


# ###Bag of Words - Vectorization. 
# Creating a vector for the frequency of tokens in the documents. 

# In[15]:


vectorizer = CountVectorizer()

matrix = vectorizer.fit_transform(df1.clean)
counts = pd.DataFrame(matrix.toarray(),
                      columns=vectorizer.get_feature_names())

# Show us the top 10 most common words
counts.T.sort_values(by=0, ascending=False).head()


# # TF - IDF Transformation 

# Standardizing the bag of words matrix 

# In[16]:


cvna = CountVectorizer(min_df = 3,max_df=.8) # creating a count vectorizer object 
data_cvna = cvna.fit_transform(df1.clean)
data_dtmna = pd.DataFrame(data_cvna.toarray(), columns=cvna.get_feature_names())
data_dtmna.index = df1.index
data_dtmna


tfidf=TfidfTransformer() # transform a count matrix to a normalized tf or tf-idf representation
tfidf_mat=tfidf.fit_transform(data_dtmna)
tfidf_matr=pd.DataFrame(tfidf_mat.toarray(),columns=cvna.get_feature_names()) 
tfidf_matr.head()


# # Classification and Model Fitting

# Using the TF-IDF matrix as a feature and the label of the document as the target variable. 

# In[17]:


from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


accuracy = []
FP_rate = []
recall = []

X = tfidf_matr
y = df1["Label"]
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3, random_state=0)


# # Model 1 - KNN

# In[18]:


k_range = list(range(1, 26))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))
knn_pred1 = y_pred = knn.predict(X_test)


# In[19]:


import matplotlib.pyplot as plt

# allow plots to appear within the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# plot the relationship between K and testing accuracy
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')


# Confusion Matrix 

# In[20]:


y_predicted = knn.predict(np.array(X_test))
cm = confusion_matrix(y_test, y_predicted)
FP_rate.append(cm[0,1]/(cm[0,1]+cm[0,0]))
recall.append(cm[1,1]/(cm[1,0]+cm[1,1]))
pd.DataFrame(cm,
             index = ["Actual: Neg","Actual: Pos"],
             columns = ["Predicted: Neg","Predicted: Pos"])


# In[21]:


knn_auc1 = roc_auc_score(y_test,knn_pred1)
print(knn_auc1)


# # Model 2 - Logit 

# In[22]:


from sklearn.linear_model import LogisticRegression
logit = LogisticRegression().fit(X_train,y_train)
acc = logit.score(X_test,y_test)
accuracy.append(acc)
print("accuracy:", round(acc,4))


# Confusion Matrix 

# In[23]:


y_predicted = logit.predict(X_test)
lr_pred1 = y_predicted = logit.predict(X_test)
cm = confusion_matrix(y_test, y_predicted)
FP_rate.append(cm[0,1]/(cm[0,1]+cm[0,0]))
recall.append(cm[1,1]/(cm[1,0]+cm[1,1]))
pd.DataFrame(cm,
             index = ["Actual: Neg","Actual: Pos"],
             columns = ["Predicted: Neg","Predicted: Pos"])


# In[24]:


lr_auc1 = roc_auc_score(y_test,lr_pred1)
print(lr_auc1)


# # Model 3 - Naive Bayes 

# In[25]:


gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
gnb_pred1 = y_pred = gnb.fit(X_train, y_train).predict(X_test)


print("training:",gnb.score(X_train,y_train))
print("test:",gnb.score(X_test,y_test))


# Confusion Matrix 

# In[26]:


y_pred = gnb.fit(X_train, y_train).predict(X_test)
cm = confusion_matrix(y_test, y_predicted)
FP_rate.append(cm[0,1]/(cm[0,1]+cm[0,0]))
recall.append(cm[1,1]/(cm[1,0]+cm[1,1]))
pd.DataFrame(cm,
             index = ["Actual: Neg","Actual: Pos"],
             columns = ["Predicted: Neg","Predicted: Pos"])


# In[27]:


gnb_auc1 = roc_auc_score(y_test,y_pred)
print(gnb_auc1)


# # Model 4 - Decision Tree 

# In[28]:


dt = tree.DecisionTreeClassifier(random_state=0)  
dt = dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
dt_pred1 = y_pred = dt.predict(X_test)

# print training and test accuracy rate
print("training:", dt.score(X_train,y_train))
print("test:", dt.score(X_test,y_test))

# construct confusion matrix
cm = confusion_matrix(y_test, y_pred)

# record FP, recall rate
accuracy.append(dt.score(X_test,y_test))

accuracy


# In[30]:


# get top 5 importange features
importance = dt.feature_importances_
ind = np.argsort(-importance)[:10]

# summarize feature importance
for i,v in enumerate(importance[ind]):
    print('Feature: %0d, Score: %.5f' % (i,v))
    
# plot feature importance
plt.figure(figsize=(12,4))
plt.bar(X.columns[ind], importance[ind])
plt.show()


# In[31]:


#plot the decision tree
plt.figure(figsize=(12,8))
a = tree.plot_tree(dt,fontsize=10, max_depth = 3) 


# In[32]:


dt_auc1 = roc_auc_score(y_test,dt_pred1)
print(dt_auc1)


# # Model 5 - Random Forest

# In[33]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state = 0, max_depth = 5, min_samples_leaf = 4)
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
rf_pred1 = y_pred = rf.predict(X_test)

# construct confusion matrix
cm = confusion_matrix(y_test, y_pred)

# record FP, recall rate
accuracy.append(rf.score(X_test,y_test))
cm


# In[34]:


rf_auc1 = roc_auc_score(y_test,rf_pred1)
print(rf_auc1)


# In[35]:


# get top 5 importange features
importance = rf.feature_importances_
ind = np.argsort(-importance)[:10]

# summarize feature importance
for i,v in enumerate(importance[ind]):
    print('Feature: %0d, Score: %.5f' % (i,v))
    
# plot feature importance
plt.figure(figsize=(12,4))
plt.bar(X.columns[ind], importance[ind])
plt.show()


# In[36]:


train_scores = []
test_scores = []

#Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
mnb=MultinomialNB().fit(X_train,y_train)
train_scores.append(mnb.score(X_train,y_train))
test_scores.append(mnb.score(X_test,y_test))
print("finished Naive Bayes")

# Logistic classifier (could take some time)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(max_iter=100).fit(X_train,y_train)
train_scores.append(lr.score(X_train,y_train))
test_scores.append(lr.score(X_test,y_test))
print("finished logistic")

# Decision Tree Classifiers
from sklearn import tree
dt=tree.DecisionTreeClassifier().fit(X_train,y_train)
train_scores.append(dt.score(X_train,y_train))
test_scores.append(dt.score(X_test,y_test))
print("finished tree")

# Random Forest classifier
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier().fit(X_train,y_train)
train_scores.append(rf.score(X_train,y_train))
test_scores.append(rf.score(X_test,y_test))
print("finished random forest")

#knn classifier
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=7,n_jobs=-1).fit(X_train,y_train)
train_scores.append(knn.score(X_train,y_train))
test_scores.append(knn.score(X_test,y_test))
print("finished knn")

pd.DataFrame({"Training Accuracy":train_scores,"Test Accuracy":test_scores},
             index=["Naive Bayes","Logistic","Decision Tree","Random Forest","KNN"])


# In[37]:


accuracies = cross_val_score(estimator = rf, X=X_train, y=y_train, cv=5)
print(accuracies)


# # Computing ROC and AUC 

# In[38]:


r_probs = [0 for _ in range(len(y_test))]
r_auc = roc_auc_score(y_test, r_probs)


# In[39]:


rf_auc1 = roc_auc_score(y_test, rf_pred1)
nb_auc1 = roc_auc_score(y_test, gnb_pred1)
lr_auc1 = roc_auc_score(y_test, lr_pred1)
dt_auc1 = roc_auc_score(y_test, dt_pred1)
knn_auc1 = roc_auc_score(y_test, knn_pred1)


# In[40]:


r_fpr, r_tpr, _ = roc_curve(y_test, r_probs)
lr_fpr,lr_tpr, _ = roc_curve(y_test, lr_pred1)
dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_pred1)
knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_pred1)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_pred1)
gnb_fpr, nb_tpr, _ = roc_curve(y_test, gnb_pred1)


# In[41]:


plt.plot(r_fpr, r_tpr, linestyle='--', label='Random prediction (AUROC = %0.3f)' % r_auc)
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logit (AUROC = %0.3f)' % lr_auc)
plt.plot(dt_fpr, dt_tpr, marker='.', label='Decision Tree (AUROC = %0.3f)' % dt_auc)
plt.plot(knn_fpr, knn_tpr, marker='.', label='KNN (AUROC = %0.3f)' % knn_auc)
plt.plot(rf_fpr, rf_tpr, marker='.', label='Random Forest (AUROC = %0.3f)' % rf_auc)
plt.plot(gnb_fpr, nb_tpr, marker='.', label='Naive Bayes (AUROC = %0.3f)' % nb_auc)

# Title
plt.title('ROC Plot')
# Axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# Show legend
plt.legend() # 
# Show plot
plt.show()


# # Financial Data 

# In[42]:


data = pd.read_csv('eco421_data.csv')
data.head()


# Creating features for financial data 

# In[43]:


data_features = data.drop(columns = ['case names','Label'])
data_features = data_features.dropna()
data_features.head()

target = data['Label']
target = target.dropna()


# In[44]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_features, target, test_size=0.2)


# # Model 1 - KNN 

# In[46]:


k_range = list(range(1, 26))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))
knn_pred = y_pred = knn.predict(X_test)


# In[47]:


import matplotlib.pyplot as plt

# allow plots to appear within the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# plot the relationship between K and testing accuracy
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')


# In[48]:


knn_auc = roc_auc_score(y_test,knn_pred)
print(knn_auc)


# # Model 2 - Logit 

# In[49]:


from sklearn.linear_model import LogisticRegression
logit = LogisticRegression().fit(X_train,y_train)
acc = logit.score(X_test,y_test)
y_predicted = logit.predict(X_test)
lr_pred = y_predicted
accuracy.append(acc)
print("accuracy:", round(acc,4))


# In[50]:


lr_auc = roc_auc_score(y_test,lr_pred)
print(lr_auc)


# # Model 3 - Naive Bayes 

# In[51]:


gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
gnb_pred = y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("training:",gnb.score(X_train,y_train))
print("test:",gnb.score(X_test,y_test))


# In[52]:


gnb_auc = roc_auc_score(y_test,y_pred)
print(gnb_auc)


# # Model 4 - Decision Tree 

# In[53]:


dt = tree.DecisionTreeClassifier(random_state=0)  
dt = dt.fit(X_train, y_train)
y_predict = dt.predict(X_test)

dt_pred = y_predict = dt.predict(X_test)
# print training and test accuracy rate
print("training:", dt.score(X_train,y_train))
print("test:", dt.score(X_test,y_test))

# construct confusion matrix
cm = confusion_matrix(y_test, y_pred)

# record FP, recall rate
accuracy.append(dt.score(X_test,y_test))

accuracy


# In[54]:


dt_auc = roc_auc_score(y_test,dt_pred)
print(dt_auc)


# In[55]:


# get top 5 importange features
importance = dt.feature_importances_
ind = np.argsort(-importance)[:5]

# summarize feature importance
for i,v in enumerate(importance[ind]):
    print('Feature: %0d, Score: %.5f' % (i,v))
    
# plot feature importance
plt.figure(figsize=(12,4))
plt.bar(data_features.columns[ind], importance[ind])
plt.show()


# In[56]:


#plot the decision tree
plt.figure(figsize=(12,8))
a = tree.plot_tree(dt,fontsize=10, max_depth = 3) 


# # Model 5 - Random Forest

# In[57]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state = 0, max_depth = 5, min_samples_leaf = 4)
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
rf_pred = y_pred = rf.predict(X_test)

# construct confusion matrix
cm = confusion_matrix(y_test, y_pred)

# record FP, recall rate
accuracy.append(rf.score(X_test,y_test))
accuracy


# In[58]:


rf_auc = roc_auc_score(y_test,rf_pred)
print(rf_auc)


# In[59]:


# get top 5 importange features
importance = rf.feature_importances_
ind = np.argsort(-importance)[:5]

# summarize feature importance
for i,v in enumerate(importance[ind]):
    print('Feature: %0d, Score: %.5f' % (i,v))
    
# plot feature importance
plt.figure(figsize=(12,4))
plt.bar(data_features.columns[ind], importance[ind])
plt.show()


# In[60]:


train_scores = []
test_scores = []

#Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
mnb=MultinomialNB().fit(X_train,y_train)
train_scores.append(mnb.score(X_train,y_train))
test_scores.append(mnb.score(X_test,y_test))
print("finished Naive Bayes")

# Logistic classifier (could take some time)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(max_iter=100).fit(X_train,y_train)
train_scores.append(lr.score(X_train,y_train))
test_scores.append(lr.score(X_test,y_test))
print("finished logistic")

# Decision Tree Classifiers
from sklearn import tree
dt=tree.DecisionTreeClassifier().fit(X_train,y_train)
train_scores.append(dt.score(X_train,y_train))
test_scores.append(dt.score(X_test,y_test))
print("finished tree")

# Random Forest classifier
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier().fit(X_train,y_train)
train_scores.append(rf.score(X_train,y_train))
test_scores.append(rf.score(X_test,y_test))
print("finished random forest")

#knn classifier
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=7,n_jobs=-1).fit(X_train,y_train)
train_scores.append(knn.score(X_train,y_train))
test_scores.append(knn.score(X_test,y_test))
print("finished knn")

pd.DataFrame({"Training Accuracy":train_scores,"Test Accuracy":test_scores},
             index=["Naive Bayes","Logistic","Decision Tree","Random Forest","KNN"])


# # Computing AUROC and ROC curve values

# Calculate AUROC 
# 
# ROC is the receiver operating characteristic AUROC is the area under the ROC curve

# In[61]:


r_probs = [0 for _ in range(len(y_test))]
r_auc = roc_auc_score(y_test, r_probs)


# In[62]:


rf_auc = roc_auc_score(y_test, rf_pred)
nb_auc = roc_auc_score(y_test, gnb_pred)
lr_auc = roc_auc_score(y_test, lr_pred)
dt_auc = roc_auc_score(y_test, dt_pred)
knn_auc = roc_auc_score(y_test, knn_pred)


# In[63]:


r_fpr, r_tpr, _ = roc_curve(y_test, r_probs)
lr_fpr,lr_tpr, _ = roc_curve(y_test, lr_pred)
dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_pred)
knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_pred)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_pred)
gnb_fpr, nb_tpr, _ = roc_curve(y_test, gnb_pred)


# In[64]:


plt.plot(r_fpr, r_tpr, linestyle='--', label='Random prediction (AUROC = %0.3f)' % r_auc)
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logit (AUROC = %0.3f)' % lr_auc)
plt.plot(dt_fpr, dt_tpr, marker='.', label='Decision Tree (AUROC = %0.3f)' % dt_auc)
plt.plot(knn_fpr, knn_tpr, marker='.', label='KNN (AUROC = %0.3f)' % knn_auc)
plt.plot(rf_fpr, rf_tpr, marker='.', label='Random Forest (AUROC = %0.3f)' % rf_auc)
plt.plot(gnb_fpr, nb_tpr, marker='.', label='Naive Bayes (AUROC = %0.3f)' % nb_auc)

# Title
plt.title('ROC Plot')
# Axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# Show legend
plt.legend() # 
# Show plot
plt.show()

