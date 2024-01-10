from transformers import BertTokenizer

# Cargar el tokenizador de BERT preentrenado
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Ejemplo de tokenización
texto_ejemplo = "Hola, ¿cómo estás?"
tokens = tokenizer.tokenize(texto_ejemplo)

print("Texto original:", texto_ejemplo)
print("Tokens:", tokens)

with open("test.txt", "w") as archivo:
    for elem in tokens:
        archivo.write(str(elem)+"\n")