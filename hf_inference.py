from huggingface_hub import InferenceClient

client = InferenceClient(
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
    token="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",
)

client.text_classification("Today is a great day")