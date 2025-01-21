import nest_asyncio

nest_asyncio.apply()

from llama_index.llms.lmstudio import LMStudio
from llama_index.core.base.llms.types import ChatMessage, MessageRole

llm = LMStudio(
    model_name="Hermes-2-Pro-Llama-3-8B",
    base_url="http://localhost:1234/v1",
    temperature=0.7,
)

response = llm.complete("Hey there, what is 2+2?")
print(str(response))