#!/usr/bin/env python
# coding: utf-8

# In[9]:


## Loading and Preprocessing of the data


# In[11]:


import pandas as pd

# df_courses = pd.read_csv('data/courses.csv', encoding='utf-8')#, sep=";")#, on_bad_lines='skip', engine='python')
df_certificates = pd.read_json('C:/Users/rvoca/OneDrive/Escritorio/TFG/MOOCs/coursera/certificates.json')
df_courses = pd.read_json('C:/Users/rvoca/OneDrive/Escritorio/TFG/MOOCs/coursera/courses.json')
df_mastertracks = pd.read_json('C:/Users/rvoca/OneDrive/Escritorio/TFG/MOOCs/coursera/mastertracks.json')
df_projects = pd.read_json('C:/Users/rvoca/OneDrive/Escritorio/TFG/MOOCs/coursera/projects.json')
df_specialization = pd.read_json('C:/Users/rvoca/OneDrive/Escritorio/TFG/MOOCs/coursera/specializations.json')

df_courses


# In[2]:


skills = [elem for elem_list in df_courses["skills"].to_list() for elem in elem_list]
skills = list(filter(None, skills))

# Remove weird symbols
# def replaceSymbols(originalText):
#     badSymbols = "ÀÁÂÃÄàáâãäªÈÉÊËèéêëÍÌÎÏíìîïÒÓÔÕÖòóôõöÙÚÛÜùúûüÑñÇç§"
#     goodSymbols   = "AAAAAaaaaaAEEEEeeeeIIIIiiiiOOOOOooooOUUUUuuuuNnCcS"
#     replacedText = ""                                 
#     for c in originalText:                                  # for each chatacter c in the text
#         p = badSymbols.find(c)                              # find position in our list of weird characters (if it is not in the string, the result will be negative)
#         replacedText += goodSymbols[p] if p>=0 else c       # if it is in our string of problematic characters, we use the "good" version of it; otherwise, we add the original one
# 
#     return replacedText

# skills = [replaceSymbols(x) for x in skills]
skills


# In[3]:


df_esco = pd.read_csv('data/skills_en_ESCO1.1.csv')#, sep=";")#, on_bad_lines='skip', engine='python')
df_esco


# In[4]:


escoLabels = df_esco["preferredLabel"].to_list()
escoLabels # 13896 labels!
#len(escoLabels)


# # ##  Attempt 1: HF -> zero shot

# # Warning: the model is 2GB, takes around 40 min to download

# # In[4]:


# from transformers import pipeline
# classifier = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli")

# #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 

# # In[30]:


# # we will classify the Russian translation of, "Who are you voting for in 2020?"
# sequence_to_classify = skills[0]
# # we can specify candidate labels in Russian or any other language above:
# candidate_labels = escoLabels
# classifier(sequence_to_classify, candidate_labels)
# # {'labels': ['politics', 'Europe', 'public health'],
# #  'scores': [0.9048484563827515, 0.05722189322113991, 0.03792969882488251],
# #  'sequence': 'За кого вы голосуете в 2020 году?'}


# ## Atttemp 2: HF -> Sentence Classification

# In[66]:



from sentence_transformers import SentenceTransformer, util
import torch
# import pandas as pd

# We load the model and encode the esco labels (~1 min)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(escoLabels)

# We define the function that gets the list of skills and 
# returns the most probable correspondance to ESCO (label, probability and URI)
def mapESCO(skills):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    listLabel = []
    listSim = []
    listUri = []
    df_skills = pd.DataFrame(skills, columns=["originalSkill"])
    for skill in skills:
        #Compute embedding for the skill
        embedding_1= model.encode(skill, convert_to_tensor=True)    # we encode the skill  
        sim = util.pytorch_cos_sim(embedding_1, embeddings)         # get the cosine similarity
        listSim.append(torch.max(sim).item())                      # get the maximum (the closest one)
        listLabel.append(escoLabels[torch.argmax(sim).item()])      # get the label of the maxiumum
        listUri.append(df_esco["conceptUri"][torch.argmax(sim).item()])  
    df_skills["esco_label"] = listLabel
    df_skills["esco_sim"] = listSim
    df_skills["esco_uri"] = listUri
    return df_skills
        


# Executing all the skills (~6k), it takes around 6 min

# In[67]:


df_skills = mapESCO(skills)
df_skills


# In[75]:


df_skills.to_csv("app2.csv")



# ## Attempt 3: previous model with more fields concatenated as a single text

# In[76]:


# df_esco["all"] = df_esco["preferredLabel"].astype(str) + " " + df_esco["altLabels"].astype(str).str.replace("\\n", " ")# + " "  + df_esco["description"].astype(str)
# escoText = df_esco["all"].to_list()
# escoText


# In[77]:

# from sentence_transformers import SentenceTransformer, util
# import pandas as pd

# # We load the model and encode the esco labels (~7 min)
# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# embeddingsText = model.encode(escoText)

# # We define the function that gets the list of skills and 
# # returns the most probable correspondance to ESCO (label, probability and URI)
# def mapESCOmoreText(skills):
#     model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
#     listLabel = []
#     listSim = []
#     listUri = []
#     df_skills = pd.DataFrame(skills, columns=["originalSkill"])
#     for skill in skills:
#         #Compute embedding for the skill
#         embedding_1= model.encode(skill, convert_to_tensor=True)    # we encode the skill  
#         sim = util.pytorch_cos_sim(embedding_1, embeddingsText)     # get the cosine similarity
#         listSim.append(torch.max(sim).item())                      # get the maximum (the closest one)
#         listLabel.append(escoLabels[torch.argmax(sim).item()])      # get the label of the maxiumum
#         listUri.append(df_esco["conceptUri"][torch.argmax(sim).item()])  
#     df_skills["esco_label"] = listLabel
#     df_skills["esco_sim"] = listSim
#     df_skills["esco_uri"] = listUri
#     return df_skills
        


# # Executing all the skills (~6k), it takes around 5 min

# # In[78]:


# df_skills_full = mapESCOmoreText(skills)
# df_skills_full


# # In[79]:


# df_skills_full.to_csv("app3.csv")


# # ## Attempt 4: Multilingual model

# # In[5]:


# df_esco["all"] = df_esco["preferredLabel"].astype(str) + " " + df_esco["altLabels"].astype(str).str.replace("\\n", " ")# + " "  + df_esco["description"].astype(str)
# escoText = df_esco["all"].to_list()
# escoText


# # In[6]:


# from sentence_transformers import SentenceTransformer, util
# import torch
# import pandas as pd
# from sentence_transformers import SentenceTransformer

# # We load the model and encode the esco labels (~7 min)
# model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
# embeddingsText = model.encode(escoText)

# # We define the function that gets the list of skills and 
# # returns the most probable correspondance to ESCO (label, probability and URI)
# def mapESCOmulti(skills):
#     model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
#     listLabel = []
#     listSim = []
#     listUri = []
#     df_skills = pd.DataFrame(skills, columns=["originalSkill"])
#     for skill in skills:
#         #Compute embedding for the skill
#         embedding_1= model.encode(skill, convert_to_tensor=True)    # we encode the skill  
#         sim = util.pytorch_cos_sim(embedding_1, embeddingsText)     # get the cosine similarity
#         listSim.append(torch.max(sim).item())                      # get the maximum (the closest one)
#         listLabel.append(escoLabels[torch.argmax(sim).item()])      # get the label of the maxiumum
#         listUri.append(df_esco["conceptUri"][torch.argmax(sim).item()])  
#     df_skills["esco_label"] = listLabel
#     df_skills["esco_sim"] = listSim
#     df_skills["esco_uri"] = listUri
#     return df_skills
        


# # In[7]:


# df_skills_multi = mapESCOmulti(skills)
# df_skills_multi


# # In[8]:


# df_skills_multi.to_csv("app4.csv")


# # In[10]:


# #utf8_decode("Ã¨Â¾\x8f")

