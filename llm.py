import os
from openai.lib.azure import AsyncAzureOpenAI

from dotenv import load_dotenv


async def get_openai_answer(question: str) -> str:
    # Load the environment variables from .env file
    load_dotenv()

    # Access the environment variables
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("OPENAI_API_VERSION")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    openai_deployment = os.getenv("AZURE_OPENAI_API_DEPLOYMENT")

    _open_ai_llm = AsyncAzureOpenAI(api_key=api_key,
                                    api_version=api_version,
                                    azure_endpoint=azure_endpoint)

    try:
        # Send the question to the Azure OpenAI service and get the response
        response = await _open_ai_llm.chat.completions.create(
            model=openai_deployment,
            temperature=1,
            messages=[
                {"role": "system", "content": "You are an AI assistant."},
                {"role": "user", "content": question}
            ]
        )

        print(f"Total tokens: {response.usage.total_tokens}")
        print(f"Response message role: {response.choices[0].message.role}")

        # Extract the answer from the response
        answer = response.choices[0].message.content
        return answer

    except Exception as e:
        raise RuntimeError(f"Failed to get response from Azure OpenAI service: {str(e)}")
