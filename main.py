import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


# Initialize the client with your endpoint and API key
client = OpenAI(
    base_url=require_env("EMBEDDING_API_ENDPOINT"),
    api_key=require_env("EMBEDDING_API_KEY"),
)

# Generate embeddings
response = client.embeddings.create(
    model="text-embedding-3-large-897052",
    input=["Azure AI Foundry is great for RAG."],
)
print(response.data[0].embedding)
