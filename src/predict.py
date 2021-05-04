import numpy as np

import torch
import joblib

import config
import dataset
import engine
from model import EntityModel

import json

if __name__ == "__main__":
    meta_data = joblib.load("meta.bin")
    enc_tag = meta_data["enc_tag"]

    num_tag = len(list(enc_tag.classes_))

    sentence = """
    Hung uses the ADNI dataset for his research at the University.
    """
    tokenized_sentence = config.TOKENIZER.encode(
       sentence
    )
    
    sentence = sentence.split() # This is currently the starting point for my dataset
    print(sentence)
    print(len(sentence))
    print(tokenized_sentence)
    print(len(tokenized_sentence))

    print(config.TOKENIZER.decode(tokenized_sentence))
    print(len(config.TOKENIZER.decode(tokenized_sentence).split()))

    test_dataset = dataset.EntityDataset(
        texts=[sentence], 
        tags=[[0] * len(sentence)]
    )

    device = torch.device("cuda")
    model = EntityModel(num_tag)
    model.load_state_dict(torch.load(config.MODEL_PATH))
    model.to(device)

    with torch.no_grad():
        data = test_dataset[0]
        for k, v in data.items():
            data[k] = v.to(device).unsqueeze(0)
        tag, _ = model(**data)

        print(
            enc_tag.inverse_transform(tag.argmax(2).cpu().numpy().reshape(-1)
                )[:len(tokenized_sentence)])
        print(len(enc_tag.inverse_transform(tag.argmax(2).cpu().numpy().reshape(-1)
                )[:len(tokenized_sentence)]))