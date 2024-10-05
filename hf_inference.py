from huggingface_hub import InferenceClient
import os

client = InferenceClient(
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
    token=os.environ["HF_KEY"],
)

client.text_classification("Today is a great day")