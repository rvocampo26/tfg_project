import json
import pandas as pd

# Cargar el contenido del archivo JSON
with open("C:/Users/rvoca/OneDrive/Escritorio/tfg_project/MOOCs/coursera/courses.json", 'r') as file:
    data = json.load(file)

# Filtrar los elementos con "language" igual a "english"
filtered_data = [] 
for item in data:
    print("____")
    print(item)
    if item['language'] == "English":
        filtered_data.append(item)
        print(item)
print("JSON queda")
print(filtered_data)
# Sobrescribir el archivo JSON con los elementos filtrados
with open("C:/Users/rvoca/OneDrive/Escritorio/tfg_project/MOOCs/coursera/courses.json", 'w') as file:
    json.dump(filtered_data, file, indent=2)

print("Elementos filtrados y guardados en el archivo 'data.json'.")

