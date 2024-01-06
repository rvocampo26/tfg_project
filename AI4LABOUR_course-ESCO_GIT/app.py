from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import torch
import pandas as pd

class App():
    # Name of model
    name_model= 'sentence-transformers/all-MiniLM-L6-v2'
    # We load the model and encode the esco labels (~1 min)
    model = SentenceTransformer(name_model)
    
    def __init__(self) -> None:
        self.model = SentenceTransformer('distilbert-base-nli-stsb') # Load the model

    def prepare_dataset (path_Esco,model):
        #df_esco = pd.read_csv('data/skills_en_ESCO1.1.csv')#, sep=";")#, on_bad_lines='skip', engine='python')
        df_esco = pd.read_csv(path_Esco)#, sep=";")#, on_bad_lines='skip', engine='python')
        escoLabels = df_esco["preferredLabel"].to_list()
        embeddings = model.encode(escoLabels)
        return embeddings
    
    def prepare_text():
        text = ""
    
    def map_Esco(skills, name_model, embeddings, escoLabels):
        model = SentenceTransformer(name_model)
        listLabel1 = []
        listSim1 = []
        df_skills = pd.DataFrame(skills, columns=["originalSkill"])
        for skill in skills:
            #Compute embedding for the skill
            embedding_1= model.encode(skill, convert_to_tensor=True)    # we encode the skill  
            sim = util.pytorch_cos_sim(embedding_1, embeddings)         # get the cosine similarity
            listSim1.append(torch.topk(sim,1)[0][0][0].item())
            listLabel1.append(escoLabels[torch.topk(sim,1)[1][0][0]])
        print(listLabel1)
        print(listSim1)
        df_skills["esco_label1"] = listLabel1
        df_skills["esco_sim1"] = listSim1
        return df_skills
    
    