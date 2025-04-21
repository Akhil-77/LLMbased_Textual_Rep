import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import re

def compile_prompt(x):
    pattern = r'\[(\d+)-(\d+)\]'
    prompt = (
        f"Tell me about the customer with a credit score of {x['CreditScore']} who is {x['Age']} years old "
        f"from {x['Geography']} with {x['Tenure']} years of banking tenure. "
        f"The customer has a balance of {x['Balance']} and owns {x['NumOfProducts']} product(s). "
        f"The customer is {'active' if x['IsActiveMember'] == 1 else 'inactive'} and "
        f"{'has' if x['HasCrCard'] == 1 else 'does not have'} a credit card. "
        f"The estimated salary of the customer is {x['EstimatedSalary']}."
    )
    return prompt

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
    return embeddings

def generate_features_from_df(df):
    prompts = df.apply(compile_prompt, axis=1).tolist()    
    embeddings = get_bert_embeddings(prompts)
    
    return embeddings, prompts

if __name__ == "__main__":
    df = pd.read_csv("train.csv")

    embeddings, prompts = generate_features_from_df(df)
    
    for i in range(5):
        print(f"Prompt: {prompts[i]}")
        print(f"Embedding: {embeddings[i]}")
