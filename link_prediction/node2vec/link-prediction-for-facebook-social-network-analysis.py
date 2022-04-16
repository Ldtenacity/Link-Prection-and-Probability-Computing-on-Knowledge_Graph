#!/usr/bin/env python
# coding: utf-8

# # Link Prediction using Node2Vec

# # Project

# ## Importing required libraries

# In[1]:


import random
from tqdm import tqdm
import networkx as nx
from zipfile import ZipFile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)


# ## Download output files created in part-1

# In[6]:


#get_ipython().system('wget https://github.com/abcom-mltutorials/Facebook-Social-Network-Analysis/archive/master.zip -P "./content"')
#ZipFile("./content/master.zip").extractall("./content/")


# Retrieve the graph and 'fb' dataframe

# In[19]:


# We had generated these two files in part-1
# graph
G = nx.read_gpickle('./content/Facebook-Social-Network-Analysis-master/Graph.pickle')
# fb dataframe
fb = pd.read_csv('./content/Facebook-Social-Network-Analysis-master/fb.csv', index_col=[0])


# In[20]:


print(nx.info(G))


# In[21]:


fb


# ## Data rocessing

# Create an adjacency matrix

# In[22]:


# get a list of nodes in our graph
l = list(G.nodes())

# create adjacency matrix
adj_G = nx.to_numpy_matrix(G, nodelist = l)

print(str(adj_G.shape)+'\n')
adj_G


# Get non-existing edges/node-pairs from the adjacency matrix

# In[23]:



# get all node pairs which don't have an edge
non_existing_edges = []

# traverse adjacency matrix 
offset = 0
for i in tqdm(range(adj_G.shape[0])):
    for j in range(offset,adj_G.shape[1]):
        if i != j:
            if adj_G[i,j] == 0:
                    non_existing_edges.extend([(l[i],l[j])])

    offset = offset + 1


# In[24]:


# print few non_existing edges
non_existing_edges[:5]


# Select partial set of non-existing edges for balancing dataset 

# In[25]:


len(non_existing_edges)


# In[26]:


# Ramdomly select 4000 non-existing edges
nodes_4000 = sorted(random.sample(non_existing_edges, k=40000))


# Find the node pairs having a path between them

# In[27]:


non_existing_edges = [(i[0],i[1]) for i in tqdm(nodes_4000) if nx.has_path(G, i[0], i[1])]


# In[28]:


non_existing_edges[:5]


# Create a dataframe of the node pairs in 'non_existing_edges'
# 

# In[29]:


df1 = pd.DataFrame(data = non_existing_edges, columns =['Node 1', 'Node 2'])

# create a column 'Connection' with default 0 (no-connection)
df1['Connection'] = 0

df1.head()


# Getting the removable edges

# In[30]:


# Create a list of all indices of the node pairs in the fb dataframe, 
# which when removed won’t change the structure of our graph

# create a copy
fb_temp = fb.copy()

# for storing removable edges
removable_edges_indices = []

# number of connected components and nodes of G
ncc = nx.number_connected_components(G)
number_of_nodes = len(G.nodes)

# for each node pair we will be removing a node pair and creating a new graph, 
# and check if the number of connected components and the number of nodes 
# are the same as the original graph
for i in tqdm(fb.index.values):
    
       # remove a node pair and build a new graph
    G1 = nx.from_pandas_edgelist(fb_temp.drop(index= i), "Node 1", "Node 2", 
                                 create_using=nx.Graph())
    
       # If the number of connected components remain same as the original
       # graph we won't remove the edge
    if (nx.number_connected_components(G1) == ncc) and (len(G1.nodes) == number_of_nodes):
        removable_edges_indices.append(i)

        # drop the edge, so that for the next iteration the next G1 
        # is created without this edge 
        fb_temp = fb_temp.drop(index = i)


# In[31]:


# print few items
removable_edges_indices[:5]


# Create a dataframe with all the removable edges

# In[32]:


# get node pairs in fb dataframe with indices in removable_edges_indices
df2 = fb.loc[removable_edges_indices]

# create a column 'Connection' and assign default value of 1 (connected nodes)
df2['Connection'] = 1

df2.head()


# Combining df2 (removable edges) and df1 (non_existing edges) 

# In[33]:


df1 = df1.append(df2[['Node 1', 'Node 2', 'Connection']], 
                 ignore_index=True)
df1.head()


# Create subset of fb

# In[34]:


df3 = fb.drop(index=df2.index.values)

# we can assume that at a previous point of time node pairs 
# in 'df3' were the only node pairs with an edge between them, 
# such that in the future, node pairs in 'df1' with 
# 'Connection' = 1 will have a new edge
# while node pairs in 'df1' with 'Connection' = 0 
# means the node pairs don't have an edge between them 
# in our assumed future

df3.head()


# Create a graph ‘G_new’ with dataframe ‘df3’ consisting of node pairs assumed to be existing at previous point of time
# 

# In[35]:


G_new = nx.from_pandas_edgelist(df3, "Node 1", "Node 2", 
                                create_using=nx.Graph())

print(nx.info(G_new))


# ## Model Building

# Use Node2Vec for generating features

# In[36]:


#get_ipython().system('pip install node2vec')


# Create Node2Vec model with features of the nodes in graph 'G_new'

# In[37]:


from node2vec import Node2Vec

# Generating walks
node2vec = Node2Vec(G_new, dimensions=100, walk_length=16, num_walks=50)

# training the node2vec model
n2v_model = node2vec.fit(window=7, min_count=1)


# Generate edge/node pair features using node2vec model

# In[45]:


#adding up the features of the each node of a node pair 'df1' to generate the features of an edge/node pair
#storing all the features in a list
edge_features = [(n2v_model.wv[str(i)]+n2v_model.wv[str(j)]) for i,j in zip(df1['Node 1'], df1['Node 2'])]


# Import necessary libraries to create the ML model
# 

# In[46]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, auc, roc_curve, roc_auc_score,confusion_matrix

#features    
X = np.array(edge_features)    

#target
y = df1['Connection']


# Split X and y into training and test data.

# In[47]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)


# # GridSearchCV
# Apply GridSearchCV on different ML classifiers to find the best model along with its best parameters

# ### Random Forest

# In[ ]:


#classifier
clf1 = RandomForestClassifier()

# parameters
param = {'n_estimators' : [10,50,100], 'max_depth' : [5,10,15]}

# model
grid_clf_acc1 = GridSearchCV(clf1, param_grid = param)

# train the model
grid_clf_acc1.fit(X_train, y_train)

print('Grid best parameter (max. accuracy): ', grid_clf_acc1.best_params_)
print('Grid best score (accuracy): ', grid_clf_acc1.best_score_)

# alternative metric to optimize over grid parameters: AUC
grid_clf_auc1 = GridSearchCV(clf1, param_grid = param, scoring = 'roc_auc')
grid_clf_auc1.fit(X_train, y_train)
predict_proba = grid_clf_auc1.predict_proba(X_test)[:,1]

print('Test set AUC: ', roc_auc_score(y_test, predict_proba))
print('Grid best parameter (max. AUC): ', grid_clf_auc1.best_params_)
print('Grid best score (AUC): ', grid_clf_auc1.best_score_)


# ### Gradient Boost

# In[ ]:


# classifier
clf2 = GradientBoostingClassifier()

# parameters
param = {'learning_rate' : [.05,.1]}

# model
grid_clf_acc2 = GridSearchCV(clf2, param_grid = param)

# train the model
grid_clf_acc2.fit(X_train, y_train)

print('Grid best parameter (max. accuracy): ', grid_clf_acc2.best_params_)
print('Grid best score (accuracy): ', grid_clf_acc2.best_score_)

# alternative metric to optimize over grid parameters: AUC
grid_clf_auc2 = GridSearchCV(clf2, param_grid = param, scoring = 'roc_auc')
grid_clf_auc2.fit(X_train, y_train)
predict_proba = grid_clf_auc2.predict_proba(X_test)[:,1]

print('Test set AUC: ', roc_auc_score(y_test, predict_proba))
print('Grid best parameter (max. AUC): ', grid_clf_auc2.best_params_)
print('Grid best score (AUC): ', grid_clf_auc2.best_score_)


# ## MLP Classifier (A Neural Network Classifier)

# In[ ]:


# classifier
clf3 = MLPClassifier(max_iter=1000)

# scaling training and test sets
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# parameters
param = {'hidden_layer_sizes' : [10,100,[10,10]], 'activation' : ['tanh','relu'], 'solver' : ['adam','lbfgs']}

# model
grid_clf_acc3 = GridSearchCV(clf3, param_grid = param)

# train the model
grid_clf_acc3.fit(X_train_scaled, y_train)

print('Grid best parameter (max. accuracy): ', grid_clf_acc3.best_params_)
print('Grid best score (accuracy): ', grid_clf_acc3.best_score_)

# alternative metric to optimize over grid parameters: AUC
grid_clf_auc3 = GridSearchCV(clf3, param_grid = param, scoring = 'roc_auc')
grid_clf_auc3.fit(X_train_scaled, y_train)
predict_proba = grid_clf_auc3.predict_proba(X_test_scaled)[:,1]

print('Test set AUC: ', roc_auc_score(y_test, predict_proba))
print('Grid best parameter (max. AUC): ', grid_clf_auc3.best_params_)
print('Grid best score (AUC): ', grid_clf_auc3.best_score_)


# #Choosing Best Model 
# MLP Classifier with scoring ‘roc_auc’
# 

# In[ ]:


# MLP Classifier with scoring ‘roc_auc’, grid_clf_auc3, 
# gives the best accuracy hence we choose this as our final model

# get predictions of our test data
pred = grid_clf_auc3.predict(X_test_scaled)

pred[:5]


# ##Accuracy Score

# In[ ]:


accuracy_score(pred,y_test)


# ##Confusion Matrix 

# In[ ]:


confusion_matrix(pred,y_test)


# ## Roc_auc score
# Find the ROC_AUC score and plot the ROC Curve

# In[ ]:


predict_proba = grid_clf_auc3.predict_proba(X_test_scaled)[:,1]

false_positive_rate,true_positive_rate,_ = roc_curve(y_test, predict_proba)
roc_auc_score = auc(false_positive_rate,true_positive_rate)

#plotting
plt.plot(false_positive_rate,true_positive_rate)
plt.title(f'ROC Curve \n ROC AUC Score : {roc_auc_score}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')


# #Inference

# In[ ]:


df1.head()


# In[ ]:


# from the above dataframe let's choose the nodes 2 and 146
# node pair (2,146) is at index 4, hence it's features

print(f' ({df1.iloc[4,0]},{df1.iloc[4,1]}) node pair features : {X[4]}')

# its position in X_train
print(f'Index of ({df1.iloc[4,0]},{df1.iloc[4,1]}) node pair in X_train : {np.where(X_train == X[4])[0][1]}')


# In[ ]:


# probability of the two nodes forming a link or edge
# we will be using the values from X_train_scaled since 
# we have used the scaled version to train our model
predict_proba = grid_clf_auc3.predict_proba(X_train_scaled[np.where(X_train == X[4])[0][1]].reshape(1,-1))[:,1]

print(f'Probability of nodes {df1.iloc[4,0]} and {df1.iloc[4,1]} to form a link is : {float(predict_proba)*100 : .2f}%')


# Saving dataset for part-3

# In[ ]:


df1.to_csv('/content/Facebook-Social-Network-Analysis-Project-master/df1.csv, index=False')

