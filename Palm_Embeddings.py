import os
import pandas as pd
from google.cloud import aiplatform
from google.protobuf import struct_pb2
import json

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "...json"

aiplatform.init(project="...", location="us-central1")

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

def generate_embeddings_with_palm(prompts):
    endpoint = "..."

    instances = [{"content": prompt} for prompt in prompts]
    parameters = {"temperature": 0.7}

    payload = {
        "instances": instances,
        "parameters": parameters
    }

    response = aiplatform.gapic.PredictionServiceClient().predict(
        endpoint=endpoint,
        instances=instances,
        parameters=parameters
    )

    embeddings = response.predictions
    return embeddings

def generate_features_from_df(df):
    prompts = df.apply(compile_prompt, axis=1).tolist()
    embeddings = generate_embeddings_with_palm(prompts)

    return embeddings, prompts

if __name__ == "__main__":
    df = pd.read_csv("train.csv")

    embeddings, prompts = generate_features_from_df(df)

    for i in range(5):
        print(f"Prompt: {prompts[i]}")
        print(f"Embedding: {embeddings[i]}")
