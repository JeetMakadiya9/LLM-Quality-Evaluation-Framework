from src.config.settings import OPENAI_API_KEY, MODELS
from openai import OpenAI

print("API key loaded:", bool(OPENAI_API_KEY))
print("Models:", MODELS)

client = OpenAI(api_key=OPENAI_API_KEY)

response = client.responses.create(
    model=MODELS["Model_A"],
    input="Say hello in one short sentence."
)

print("Response:")
print(response.output_text)