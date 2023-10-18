#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import time
import string
import nltk
import gensim
import matplotlib as plt
import csv
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
import re
from gensim.models import Word2Vec
import multiprocessing
import torch
import torch.nn as nn
import transformers
import torch
import torch.nn as nn
import transformers
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score
import torch.nn.functional as F
from transformers import BertTokenizer
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np
from transformers import TrainingArguments, Trainer
from transformers import pipeline
from scipy.stats import ttest_ind


# In[2]:


model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3, output_hidden_states=True)
# model.init_weights()


# In[3]:


######## Below are list indecies ##########
    # target_words indecies:
    # 0 - career_words, 1 - family_words, 2 - math_words, 3 - arts_words
    
    # attributes indecies:
    # 0 - m_attributes, 1 - f_attributes, 2 - european_american_names, 3 - african_american_names,  
    #
######### End of list incdecies #########
attribute_names = ["male attributes", "female attributes", "european american names", "african american names"]
attributes = [["male", "man", "boy", "brother", "he", "him", "his", "son"], 
              ["female", "woman", "girl", "sister", "she", "her", "hers", "daughter"], ["Adam", "Harry", "Justin", "Ryan", "Amanda", "Kristin", "Stephanie","Megan"], 
             ["Theo", "Jerome", "Leroy", "Lamar", "Tyrone",  "Ebony",   "Jasmine","Tia"],]

target_names = ["carrer words", "family words", "math words", "art words",
                "pleasant words", "unpleasant words"]

target_words = [["executive", "management", "professional", "corporation", "salary", "office", "business", "career"]
               ,["home", "parents", "children", "family", "cousins", "marriage", "wedding", "relatives"],
                ["math", "algebra", "geometry", "calculus", "equations", "computation", "numbers", "addition"],
                ["poetry", "art", "dance", "literature", "novel", "symphony", "drama", "sculpture"],
                ['loyal', 'gentle', 'honest', 'honor', 'family','freedom', 'health', 'love'], 
                ['murder','assault','hatred','divorce','jail','poverty', 'abuse','death'],]
                
# attributes[0] = male attributes, attributes[1] = female attributes, attributes[2] = european american names, attributes[3] african american names
# target_words[0] = career words, target_words[1] = family words, target_words[2] = math_words, target_words[3] = art words, target_words[4] = pleasant words, target_words[5] = unpleasant words


# In[4]:


def weat_test(target_word_1, target_word_2, attribute_words_1, attribute_words_2):
    
    
    def cosine_similarity(target_embeddings, attribute_embedding):

        target_reshaped = target_embeddings.reshape(1, -1)
        attribute_reshaped = attribute_embedding.reshape(1, -1)

        dot_product = np.dot(target_reshaped, attribute_reshaped.T)
        norm_target = np.linalg.norm(target_reshaped)
        norm_attribute = np.linalg.norm(attribute_reshaped)
        return dot_product / (norm_target * norm_attribute)
        
            
    """
    Perform the WEAT test on a target word and two sets of attribute words.
    
    Args:
    - target_word_1 (str): the target word
    - attribute_words_1 (list): a list of words for the first attribute set
    - attribute_words_2 (list): a list of words for the second attribute set
    - embeddings_model: the embeddings model used to obtain word embeddings
    
    Returns:
    - The WEAT score for the target word and the two attribute sets
    """
    
    def get_embeddings(words):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokens = tokenizer(words, return_tensors='pt', padding=True)
        embeddings = model.bert.embeddings.word_embeddings(tokens['input_ids'])
        return embeddings
    
    # get the embeddings for all the words
    target_embedding_1 = get_embeddings(target_word_1).detach().numpy()
    target_embedding_2 = get_embeddings(target_word_2).detach().numpy()
    attribute_embeddings_1 = get_embeddings(attribute_words_1).detach().numpy()
    attribute_embeddings_2 = get_embeddings(attribute_words_2).detach().numpy()

    # calculate the WEAT score
    def calc_weat_score(embeddings_x, embeddings_y, embeddings_a, embeddings_b):
        # Calculate the cosine similarities between the target and attribute word embeddings
        sim_xa = np.mean(cosine_similarity(embeddings_x, embeddings_a))
        sim_ya = np.mean(cosine_similarity(embeddings_y, embeddings_a))
        sim_xb = np.mean(cosine_similarity(embeddings_x, embeddings_b))
        sim_yb = np.mean(cosine_similarity(embeddings_y, embeddings_b))
        # Calculate the WEAT score
        weat_score = (sim_xa - sim_xb) - (sim_ya - sim_yb)
        return weat_score

    weat_score = calc_weat_score(target_embedding_1, target_embedding_2, attribute_embeddings_1, attribute_embeddings_2)
    
    target_embedding =  np.concatenate((target_embedding_1, target_embedding_2), axis=0)
    # calculate the p-value
    sample_size = len(attribute_words_1) + len(attribute_words_2)
    a1 = attribute_embeddings_1.reshape(-1, attribute_embeddings_1.shape[-1])
    a2 = attribute_embeddings_2.reshape(-1, attribute_embeddings_2.shape[-1])
    t_emb = target_embedding.reshape(-1, target_embedding.shape[-1])
    combined_embeddings = np.concatenate((a1, a2), axis=0)
    all_similarities = np.dot(combined_embeddings, t_emb.T) / np.linalg.norm(combined_embeddings, axis=1).reshape(-1, 1)
    similarities_1 = all_similarities[:len(attribute_words_1)]
    similarities_2 = all_similarities[len(attribute_words_1):]
    t, p = ttest_ind(similarities_1, similarities_2, equal_var=False)
    weat_p_value = p
    
    
    return weat_score, weat_p_value


# In[5]:


# def run_weat(attributes, targets):
#     weat_results = []
#     p_value_results = []
#     for i in range(0, len(attributes), 2):
#         for j, target_set in enumerate(targets):
            
#             weat_score, p_value = weat_test(target_set, attributes[i], attributes[i + 1])
#             weat_results.append((attribute_names[i] , attribute_names[i + 1], target_names[j], weat_score))
#             p_value_results.append(p_value)
#     return weat_results, p_value_results


# In[5]:


def run_weat(attributes, targets):
    weat_results = []
    p_value_results = []
    for i in range(0, len(attributes), 2):
        
        for j in range(0, len(targets), 2):
            weat_score, p_value = weat_test(targets[j], targets[j + 1], attributes[i], attributes[i + 1])
            weat_results.append((attribute_names[i] , attribute_names[i + 1], target_names[j], target_names[j + 1], weat_score))
            p_value_results.append(p_value)
    return weat_results, p_value_results


# In[6]:


weat_results, p_value = run_weat(attributes, target_words)


# In[7]:


df = pd.DataFrame(weat_results, columns=['Attribute 1', 'Attribute 2', 'Target Word 1', 'Target Word 2', 'WEAT Score'])


# In[8]:


m_p = []
for elements in p_value:
    m_p.append(np.mean(elements))


# In[9]:


df['p_value'] = m_p


# In[10]:


df


# In[11]:


for name, param in model.named_parameters():
        if 'bias' in name:
            torch.nn.init.constant_(param, 0.0)
        elif 'weight' in name:
            torch.nn.init.normal_(param, mean=0.0, std=0.02)


# In[12]:


weat_results, p_value = run_weat(attributes, target_words)


# In[13]:


df1 = pd.DataFrame(weat_results, columns=['Attribute 1', 'Attribute 2', 'Target Word 1', 'Target Word 2', 'WEAT Score'])


# In[14]:


m_p = []
for elements in p_value:
    m_p.append(np.mean(elements))


# In[15]:


df1['p_value'] = m_p


# In[16]:


df1


# In[31]:


model.load_state_dict(torch.load("./pure_bert/checkpoint-9914/pytorch_model.bin"))


# In[36]:


weat_results, p_value = run_weat(attributes, target_words)


# In[37]:


df2 = pd.DataFrame(weat_results, columns=['Attribute 1', 'Attribute 2', 'Target Word 1', 'Target Word 2', 'WEAT Score'])


# In[38]:


m_p = []
for elements in p_value:
    m_p.append(np.mean(elements))


# In[39]:


m_p


# In[40]:


df2['p_value'] = m_p


# In[41]:


df2


# In[ ]:




