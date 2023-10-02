# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 12:21:54 2023

@author: rvoca
"""

import pandas as pd
import nlkt
import spacy
#Leemos fichero de habilidades
skills = pd.read_csv("C:/Users/rvoca/OneDrive/Escritorio/TFG/AI4LABOUR_course-ESCO-main/data/skills_en_ESCO1.1.csv")

#Leemos fichero de cursos 
courses = pd.read_csv("C:/Users/rvoca/OneDrive/Escritorio/TFG/AI4LABOUR_course-ESCO-main/courses.csv")

#skill name column: preferredLabel
#Others name column: altLabels
names_skills = skills[["preferredLabel","altLabels"]]


courses_skills = courses[["skills"]]
courses_list = courses_s 

# 1º buscamos el nombre de las habilidades los cursos
# 2º si no se encuentra, probamos a buscar sinónimos
# 3º si no se encuentra, pasamos a comprobar la similitud del texto.
