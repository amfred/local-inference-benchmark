# Start up the local ollama server before using this
# ollama pull granite3-dense:8b
# ollama serve

import ollama
import timeit
import logging
import asyncio
import os
import openai
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("granite-from-ollama")

MODEL_FOR_CHAT="granite3-dense:8b"
MODEL_FOR_COMPLETIONS="granite3-moe:3b"

PROMPT_FOR_CHAT="Generate a function in python that will calculate the first N numbers in a Fibonacci sequence"
PROMPT_FOR_COMPLETIONS="# A function in python that will calculate the first N numbers in a Fibonacci sequence"

RUNS_FOR_CHAT=5
RUNS_FOR_COMPLETIONS=5

LOCAL_OLLAMA_URL="http://127.0.0.1:11434/api/chat"

GRANITE3_MAAS_KEY=os.getenv("GRANITE3_MAAS_KEY")
GRANITE3_MAAS_MODEL=os.getenv("GRANITE3_MAAS_MODEL")
GRANITE3_MAAS_URL=os.getenv("GRANITE3_MAAS_URL")
GRANITE3_MAAS_CHAT_API=GRANITE3_MAAS_URL+"/v1/"

GRANITE_CODE_KEY=os.getenv("GRANITE_API_KEY")
GRANITE_CODE_MODEL=os.getenv("GRANITE_MODEL")
GRANITE_CODE_URL=os.getenv("GRANITE_API_URL")
GRANITE_CODE_CHAT_API=GRANITE_CODE_URL+"/v1/"

async def run_ollama_tests(model: str, prompt:str, iterations: int):
  timeValues = [0]*iterations

  for i in range(iterations):
    start = timeit.default_timer()
    response = ollama.chat(model, messages=[
      {
        'role': 'user',
        'content': prompt,
      },
    ])
    inferenceTime = timeit.default_timer() - start
    timeValues[i] = inferenceTime
    logger.debug(f"Response #{i}: {response['message']['content']}")

  return timeValues

async def run_openai_tests(url: str, apiKey: str, modelToUse: str, prompt:str, iterations: int):
  timeValues = [0]*iterations

  openai.api_key = apiKey
  openai.base_url = url
  openai.http_client = httpx.Client(verify=False)

  for i in range(iterations):
    start = timeit.default_timer()
    completion = openai.chat.completions.create(
      model=modelToUse,
      messages=[
        {
          "role": "user",
          "content": prompt
        }
      ],
      temperature=0,
      max_tokens=1000,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
    inferenceTime = timeit.default_timer() - start
    timeValues[i] = inferenceTime
    logger.debug(f"Response #{i}: {completion.choices[0].message.content}")

  return timeValues

async def run():
    
    # OpenShift AI running VLLM in a dev cluster with gen-2 Granite-Code
    timeValues = await run_openai_tests(GRANITE_CODE_CHAT_API, GRANITE_CODE_KEY, GRANITE_CODE_MODEL, PROMPT_FOR_CHAT, 1)
    logger.info(f"Total response time for the first {GRANITE_CODE_MODEL} on Dev chat request: {sum(timeValues)}")

    timeValues = await run_openai_tests(GRANITE_CODE_CHAT_API, GRANITE_CODE_KEY, GRANITE_CODE_MODEL, PROMPT_FOR_CHAT, RUNS_FOR_CHAT)
    logger.info(f"Total response time over {RUNS_FOR_CHAT} {GRANITE_CODE_MODEL} on Dev chat requests: {sum(timeValues)}")

    # OpenShift AI running VLLM in our Model-as-a-Service cluster with Granite 3
    timeValues = await run_openai_tests(GRANITE3_MAAS_CHAT_API, GRANITE3_MAAS_KEY, GRANITE3_MAAS_MODEL, PROMPT_FOR_CHAT, 1)
    logger.info(f"Total response time for the first {GRANITE3_MAAS_MODEL} on Maas chat request: {sum(timeValues)}")

    timeValues = await run_openai_tests(GRANITE3_MAAS_CHAT_API, GRANITE3_MAAS_KEY, GRANITE3_MAAS_MODEL, PROMPT_FOR_CHAT, RUNS_FOR_CHAT)
    logger.info(f"Total response time over {RUNS_FOR_CHAT} {GRANITE3_MAAS_MODEL} on MaaS chat requests: {sum(timeValues)}")

    # Local ollama tests with Granite 3
    timeValues = await run_ollama_tests(MODEL_FOR_CHAT, PROMPT_FOR_CHAT, 1)
    logger.info(f"Total response time for the first {MODEL_FOR_CHAT} on ollama chat request: {sum(timeValues)}")

    timeValues = await run_ollama_tests(MODEL_FOR_CHAT, PROMPT_FOR_CHAT, RUNS_FOR_CHAT)
    logger.info(f"Total response time over {RUNS_FOR_CHAT} {MODEL_FOR_CHAT} on ollama chat requests: {sum(timeValues)}")

    timeValues = await run_ollama_tests(MODEL_FOR_COMPLETIONS, PROMPT_FOR_COMPLETIONS, 1)
    logger.info(f"Total response time for the first {MODEL_FOR_COMPLETIONS} on ollama completion request: {sum(timeValues)}")

    timeValues = await run_ollama_tests(MODEL_FOR_COMPLETIONS, PROMPT_FOR_COMPLETIONS, RUNS_FOR_COMPLETIONS)
    logger.info(f"Total response time over {RUNS_FOR_COMPLETIONS} {MODEL_FOR_COMPLETIONS} on ollama completion requests: {sum(timeValues)}")

    # Interleaving requests for two different models on local ollama
    start = timeit.default_timer()
    timeValues = await run_ollama_tests(MODEL_FOR_CHAT, PROMPT_FOR_CHAT, 1)
    timeValues = await run_ollama_tests(MODEL_FOR_COMPLETIONS, PROMPT_FOR_COMPLETIONS, 1)
    timeValues = await run_ollama_tests(MODEL_FOR_CHAT, PROMPT_FOR_CHAT, 1)
    timeValues = await run_ollama_tests(MODEL_FOR_COMPLETIONS, PROMPT_FOR_COMPLETIONS, 1)
    timeValues = await run_ollama_tests(MODEL_FOR_CHAT, PROMPT_FOR_CHAT, 1)
    inferenceTime = timeit.default_timer() - start
    logger.info(f"Total response time over several ollama calls to different models: {inferenceTime}")

if __name__ == "__main__":
    asyncio.run(run())
