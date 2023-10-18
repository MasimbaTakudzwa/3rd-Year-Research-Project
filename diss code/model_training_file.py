#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !pip install transformers datasets evaluate


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

# nltk.download("punkt")
# nltk.download('wordnet')


# In[2]:


m_attributes = ["man","male", "man", "boy", "brother", "he", "him", "his", "son"]
f_attributes = ["woman","female", "woman", "girl", "sister", "she", "her", "hers", "daughter"]

euro_american = ["Adam", "Harry", "Justin", "Ryan", "Amanda", "Kristin", "Stephanie","Megan"]
afr_american = ["Theo", "Jerome", "Leroy", "Lamar", "Tyrone",  "Ebony",   "Jasmine","Tia"]


# In[3]:


df = pd.read_csv('../dissdata/labeled_data.csv')

df


# # Dataset Evaluation
# 
# The dataset being obsereved here is the dataset from the Automated Hate Speech Detection and the Problem of Offensive Language paper (Davidson et al. 2017).
# The point of this dataset is to try separate instaces of hate speech from instances of purely offensive non-hate speech language. 
# 
# The problem with this other than the difficulty of getting people to be objective is getting people to agree what is hate speech and what is not. labeling a comment as offensive is an easier task as it just requires the labeler to understan why someone might see a comment as offensive. A comment I easier to judge to be objectively offensive than to be objectively hateful.
# 
# In this dataset we have the following columns: count -> (This column states the number of individuals that were present when labeling the datasets. there is a minimum of 3 people that had to be present when labelling the dataset)
# 
# hate_speech -> (a binary label (0 or 1) if the comment in the considered row is considered hate speech by the labelers)
# 
# Offensive -> (a binary label (0 or 1) if the comment in the considered row is considered offensive by the labelers)
# 
# neither -> (a binary label (0 or 1) if the comment in the considered row is considered neither hate speech or offensive language by the labelers)
# 
# Lets take a look at the number of comments in each set of classifications
# 

# ### Overlap
# 
# As we can see below there are 4390 instances where comments where individuals found the comment to be either hateful or hate speech (at least 2 different individuals designated the hate or offensive label on the same comment)

# In[ ]:


df.loc[(df["hate_speech"] > 0) & (df["offensive_language"] > 0)]


# ### Disagreement
# 
# Below you can see instances where people saw the language as hateful but at least 1 individual diagreed with this label and 
# saw the comment as neither offensive nor hateful. Lets take a closer look at one of these instances

# In[ ]:


df.loc[(df["hate_speech"] > 0) & (df["neither"] > 0)]


# In[ ]:


df["tweet"][219]


# Well this is clearly at least offensive and the use of the derogatory term to degrade drake could be seen as hate speech.
# To be fair there is only 1 individual who said neither and the rest either called it hate speech or offensive at the least. Let's look at an example where the skew is the other way (i.e. we have more saying its neither than hate or offensive)

# In[ ]:


df.loc[(df["hate_speech"] > 0) & (df["neither"] > 0) & (df["neither"] > (df["offensive_language"] + df["hate_speech"] ) )]


# In[ ]:


c1 = df["tweet"][379]
c2 = df["tweet"][395]
c3 = df["tweet"][509]
print(f"Comment 1: {c1} \n")
print(f"Comment 2: {c2} \n")
print(f"Comment 3: {c3} \n")


# ### Disagreements in the neither direction
# 
# comments 1 and 2 look like replies/ quote tweets and don't really seem offensive if they are. However comment 3 seems to skirt the line between offensive and racism if we are to just go off waht we are reading. These examples prove that objectivity is not always possible but also classification of these comments will be highly subjective. My view is that if a comment can be seen as rude, mean or hateful then it is also offensive, others may disagree. 

# ### Separating the classification values from comments
# 
# Below i define a class "ExamineData" that, at the end of the project, will store a number of functions used to access the data and observe different aspects of the data
# 
# Currently "return_column_values" is a method that returns the number of people who designated with class the tweets belong to and the overall class values of the comments. When I made the function i thought it was a binary list therefore the  variable names where create in that nature and I have not renamed them yet.
# 
# "examine_data_spread" returns the number of times a comment was designated a specific class
# 
# "output_value" is just used to return the value of the specific cell at row == comment number and column == column name this is just to try streamline the process so I don't have to type "df[column name][cell number]" every time and can instead just call instances of examine.output_value(*args) (where *args are just the required arguments)
# 
# All of these functions can be improved and their depth of investigation will be considered going forward this is just the starting point.

# In[4]:


columns = ["hate_speech", "offensive_language", "neither", "class"]

class ExamineData:
    
    def __init__(self):
        pass
        
    # Returns the values of the cells for all columns in list "columns"
    def return_column_values(self, dataframe, columns, integer, binary_list):
        binary_list= []
        for comments in dataframe[f"{columns[0]}"]:
            
            binary_list.append([])
            
        i = 0    
        for names in columns:
            
            integer = 0
            for values in dataframe[f"{names}"]:
                binary_list[integer].append(values)
                integer += 1
            
        return binary_list    
    
    def examine_data_spread(self, binary_list, column_names):
        
        """The column names need to be in order of the way 
        they were stored in the
        binary list
        count, hate, offensive language, neither, class"""
        bin_dict = {}
        
        hate = 0 
        offensive = 0
        neither = 0 
        
        
        for i in range(len(binary_list)):
            
            if binary_list[i][len(binary_list[i])-1] == 0:
                hate += 1
            if binary_list[i][len(binary_list[i])-1] == 1:
                offensive += 1
            if binary_list[i][len(binary_list[i])-1] == 2:
                neither += 1
                
        bin_dict["hate"]= hate
        bin_dict["offensive"]= offensive
        bin_dict["neither"]= neither
        
        return bin_dict
  
    def output_value(self, dataframe, comment_number, column):

        
        return dataframe[column][comment_number]
        
        """Class and count are not stored in this dict as they are not binary
        and require a different way to be observed"""
examine = ExamineData()

classification_values = examine.return_column_values(dataframe=df, columns=columns, binary_list=[], integer=0)
bin_dict = examine.examine_data_spread(classification_values, columns)              


# ### Storing the classification values and defining a data examination class
# Above our examine class will store our data evaluation methods the return_column_values method of our ExamineData class takes all the binary values of our classification columns in our data frame and returns them in an array that stores it in the same way as they were in the dataframe but in an array instead This will be useful when we need these values for train our model
# 
# Obviously comments that were hateful were likely offensive but not all offensive comments were hateful and so there is likely overlap that that I could look to furhter observe

# ### Function block
# 
# Below is a function block that contains all the senetence formating functions that make use of nltk and gensim as well as counting the number of unqiue words.

# In[5]:


from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

new_training = False  # Recreate word embeddings = True else False
print_true = False  # Print all the tweets present in the dataset



# print(df)
class PreProcessing():

    def __init__(self, tweet_array):
        
        self.tweet_array = tweet_array
        
    def bert_process(self):
        
        t = self.remove_hyperlinks(self.tweet_array)
        t = self.lowercase(t)
        t = self.remove_escape_characters(t)
        
        t = [nltk.word_tokenize(sentences) for sentences in t]
        
        return self.count_nuniques(t), t
        
    def remove_hyperlinks(self, tweet_array):
        
        url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        removed_url = [re.sub(url_pattern, ' ', tweet) for tweet in tweet_array]
        
        return removed_url
    
    def count_nuniques(self, tokenized_array):
        l = []
        
        for sentences in tokenized_array:
            
            for words in sentences:
                
                l.append(words)
                
        l = set(l)
        return len(l), l
        
    
    def lematize(self, tweet_array):
        # nltk.download('wordnet')
        lemmatizer = WordNetLemmatizer()

        lemmatized_tweet = [lemmatizer.lemmatize(words) for words in tweet_array]

        return lemmatized_tweet
    
    def lowercase(self, tweet_array):
        lower_words = [word.lower() for word in tweet_array]
        
        return lower_words
        
    def tokenize_and_remove_stop_words(self, tweet_array):
        # Split the lyrics into individual words

        words = nltk.word_tokenize(tweet_array)

        # Remove any punctuation characters from the words
        words = [word for word in words if word.isalpha()]

        # Get a list of stop words

        stop_words = [gensim.parsing.preprocessing.remove_stopwords(strings) for strings in
                      words]  # nltk.corpus.stopwords.words("english")
        ll = []
        for wordss in stop_words:
            if wordss != '':
                ll.append(wordss)

        return ll
    
    def remove_escape_characters(self, tweet_array):
        for i, s in enumerate(tweet_array):
            # Use the `translate()` method to remove all escape characters
            s = s.translate(str.maketrans('', '', string.punctuation))
            tweet_array[i] = s

        return tweet_array
    
    
df_array = list(df["tweet"])
preprocessing = PreProcessing(tweet_array=df_array)
nuniques ,formatted_tweet_list = preprocessing.bert_process()


# In[6]:


# Function for creating a dataframe of tweets out of tokenized tweets
def process_and_format(tweet_list):
    
    def rejoin_sentences(tok_sentences):
    
        rejoined_list = []

        for sentences in tok_sentences:

            sentence = ' '.join(sentences)

            rejoined_list.append(sentence)

        return rejoined_list

    
    tweet_list = rejoin_sentences(tweet_list)
    
    
    new_list = []
    for i in range(len(df["class"])):

        new_list.append([tweet_list[i],df["class"][i]])
        
    new_df = pd.DataFrame(new_list, columns=["text", "labels"])
    
    return new_df


# In[7]:



def CDA(list_of_tweets, l1, l2):
    
    """ This function takes the following inputs:
    list_of_tweets (list): list of tokenized tweets or sentences stored in a list of lists
    l1 (list): list of attributes that we are looking for or using to replace (must be same length as l2)
    l2 (list): list of attributes that we are looking for or using to replace (must be same length as l1)
    returns a df which has CDA implemented using the lists l1 and l2
    this function does not prevent coreference CDA which may result in problematic associations"""
    cda = []
    i = 0
    for sentences in list_of_tweets:
        cda.append([])
        for words in sentences:
            
    
            if (words in l1):
                
                index_l1 = l1.index(words)
                
                cda[i].append(l2[index_l1])
            
            elif (words in l2):
                
                index_l2 = l2.index(words)
                
                cda[i].append(l1[index_l2])
                
            else:
                
                cda[i].append(words)
                
        i += 1
    return cda
    


# In[8]:


cda_true = False
pretrained = False
if cda_true:
    cda_list = CDA(formatted_tweet_list, m_attributes, f_attributes)
    cda_df = process_and_format(cda_list)
    processed_df = process_and_format(formatted_tweet_list)
    processed_df = pd.concat([cda_df, processed_df])
    train_name = "train_pretrained_cda.csv"
    test_name = "test_pretrained_cda.csv"

else:
    processed_df = process_and_format(formatted_tweet_list)
    train_name = "train_pretrained_non_cda.csv"
    test_name = "test_pretrained_non_cda.csv"
    

# if not pretrained:
    
#     for name, param in model.named_parameters():
#         if 'bias' in name:
#             torch.nn.init.constant_(param, 0.0)
#         elif 'weight' in name:
#             torch.nn.init.normal_(param, mean=0.0, std=0.02)
            


# In[9]:


from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(processed_df, test_size=0.2, random_state=42 )
train_df.to_csv("train.csv"), test_df.to_csv("test.csv")


# In[10]:


train_df


# In[ ]:





# In[13]:




# implementation based on: https://huggingface.co/docs/transformers/tasks/sequence_classification 
file_1 = ['train.csv'] # train file
file_2 = ['test.csv'] # test file

def create_bert_model(train_file, test_file, pretrained=False):
    train_dataset = load_dataset('csv', data_files=train_file, delimiter=",")
    test_dataset = load_dataset('csv', data_files=test_file, delimiter=",")
    print(train_dataset["train"])
    print(test_dataset["train"])
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized_df1 = train_dataset.map(preprocess_function, batched=True)
    tokenized_df2 = test_dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    accuracy = evaluate.load("accuracy")
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)
    
    id2label = {0: "Hate", 1: "Offensive", 2: "Neither"}
    label2id = {"Hate": 0, "Offensive": 1,"Neither":2,}
    
    
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=3, id2label=id2label, label2id=label2id
    )
    
    if not pretrained:
        for name, param in model.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
    
    training_args = TrainingArguments(
    output_dir="pure_bert",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_df1["train"],
        eval_dataset=tokenized_df2["train"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

create_bert_model(file_1, file_2)
# tweet_dataset = load_dataset("../dissdata/labeled_data.csv")


# In[ ]:





# In[ ]:


classifier = pipeline("sentiment-analysis", model="pretrained_bert/checkpoint-9914")


# In[ ]:


text = "Man you are a sack of shit."
classifier(text)


# In[ ]:


# word_csv_files = ["MusicalInstruments_Weapons_Pleasant_Unpleasant.csv", "Science_Arts_Male_Female.csv", "Names_Female_Male.csv", "Male_Female_Career_Family.csv", "Math_Arts_Male_Female.csv", "Flowers_Insects_Pleasant_Unpleasant.csv", "EuropeanAmerican_AfricanAmerican_Pleasant_Unpleasant_2.csv", "EuropeanAmerican_AfricanAmerican_Pleasant_Unpleasant.csv", "Careers_Female_Male.csv"]
# # 


# In[ ]:


# model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)


# In[ ]:


# torch.load("./modelfile/mymodel/pytorch_model.bin")


# In[ ]:


# model.load_state_dict(torch.load("./modelfile/mymodel/pytorch_model.bin"))


# In[ ]:



# professions = [" technician", "accountant", "supervisor", "engineer", "worker", "educator", "clerk",
#                "counselor", "inspector", "mechanic", "manager", "therapist", "administrator", "salesperson", 
#                "receptionist", "librarian", "advisor", "pharmacist", "janitor", "psychologist", "physician", 
#                "carpenter", "nurse", "investigator", "bartender", "specialist", "electrician", "officer", 
#                "pathologist", "teacher", "lawyer", "planner", "practitioner", "plumber", "instructor", "surgeon",
#                "veterinarian", "paramedic", "examiner", "chemist", "machinist", "appraiser", "nutritionist", 
#                "architect", "hairdresser", "baker", "programmer", "paralegal", "hygienist", "scientist"]



# In[ ]:


# f_csim = []
# m_csim = []


# for pair in pairs1:
    
#     tokens = tokenizer(pair, return_tensors='pt', padding=True)
#     embeddings = embed_layer(tokens['input_ids']).detach().numpy()

#     sim = np.dot(embeddings[0][0], embeddings[0][1])/(np.linalg.norm(embeddings[0][0]) * np.linalg.norm(embeddings[0][1]))
# #     print(f"Cosine similarity between '{pair[0]}' and '{pair[1]}': {sim:2f}")
#     f_csim.append([pair[0], pair[1], sim])
    
    
# for pair in pairs2:
    
#     tokens = tokenizer(pair, return_tensors='pt', padding=True)
#     embeddings = embed_layer(tokens['input_ids']).detach().numpy()

#     sim = np.dot(embeddings[0][0], embeddings[0][1])/(np.linalg.norm(embeddings[0][0]) * np.linalg.norm(embeddings[0][1]))
# #     print(f"Cosine similarity between '{pair[0]}' and '{pair[1]}': {sim:2f}")
#     m_csim.append([pair[0], pair[1], sim])
    


# In[ ]:


# words_to_find = [("male attributes" ,["male", "man", "boy", "brother", "he", "him", "his", "son",])
#                  ("female attributes" ,["female", "woman", "girl", "sister", "she", "her", "hers", "daughter",])
#                  ("european american names" ,["Adam", "Harry", "Justin", "Ryan", "Amanda", "Kristin", "Stephanie","Megan",])
#                  ("african american names", ["Theo", "Jerome", "Leroy", "Lamar", "Tyrone",  "Ebony",   "Jasmine","Tia",])
#                  ]

# replacement_words = ["female", "woman", "girl", "sister", "she", "her", "hers", 
#                      "male", "man", "boy", "brother", "he", "him", "his", "son",
#                      "Theo", "Jerome", "Leroy", "Lamar", "Tyrone",  "Ebony",   "Jasmine","Tia",
#                      "Adam", "Harry", "Justin", "Ryan", "Amanda", "Kristin", "Stephanie","Megan",
#                  ]


# In[ ]:


print(formatted_tweet_list[0])


# In[ ]:


# # List of words to find and replacement word
# words = [("male attributes" ,["male", "man", "boy", "brother", "he", "him", "his", "son",])
#                  ("female attributes" ,["female", "woman", "girl", "sister", "she", "her", "hers", "daughter",])
#                  ("european american names" ,["Adam", "Harry", "Justin", "Ryan", "Amanda", "Kristin", "Stephanie","Megan",])
#                  ("african american names", ["Theo", "Jerome", "Leroy", "Lamar", "Tyrone",  "Ebony",   "Jasmine","Tia",])]



# # List to store the modified tweets
# modified_tweets = []

# # Iterate through each tweet
# for pairs in words:
#     words_to_find = pairs[1]
#     for tweet in formatted_tweet_list:
#         # Check if the tweet contains any of the words to find
#         if any(word in tweet for word in words_to_find):
#             # Clone the tweet and replace wordA with the replacement word
#             modified_tweet = tweet.replace('wordA', replacement_word)
#             # Add the modified tweet to the list of modified tweets
#             modified_tweets.append(modified_tweet)

#         # Add the original tweet to the list of modified tweets
#         modified_tweets.append(tweet)

            


# In[ ]:





# In[ ]:




