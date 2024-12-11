# Start up the local ollama server before using this
# ollama pull granite3-dense:8b
# ollama serve

# Runs all tests by default
# To only run the local tests, export BENCHMARK_SERVER=false
# To only run the server tests, export BENCHMARK_OLLAMA=false

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
MODEL_FOR_COMPLETIONS="granite3-dense:2b"

PROMPT_FOR_CHAT="Generate a function in python that will calculate the first N numbers in a Fibonacci sequence"
PROMPT_FOR_COMPLETIONS="# A function in python that will calculate the first N numbers in a Fibonacci sequence"

RUNS_FOR_CHAT=5
RUNS_FOR_COMPLETIONS=5

LOCAL_OLLAMA_URL="http://127.0.0.1:11434/api/chat"

# These URLs and API keys are only needed for the server suite of tests

GRANITE3_MAAS_KEY=os.getenv("GRANITE3_MAAS_KEY")
GRANITE3_MAAS_MODEL=os.getenv("GRANITE3_MAAS_MODEL")
GRANITE3_MAAS_URL=os.getenv("GRANITE3_MAAS_URL")
GRANITE3_MAAS_CHAT_API=GRANITE3_MAAS_URL+"/v1/"

GRANITE_CODE_KEY=os.getenv("GRANITE_API_KEY")
GRANITE_CODE_MODEL=os.getenv("GRANITE_MODEL")
GRANITE_CODE_URL=os.getenv("GRANITE_API_URL")
GRANITE_CODE_CHAT_API=GRANITE_CODE_URL+"/v1/"

BENCHMARK_SERVER = os.getenv("BENCHMARK_SERVER", 'True').lower() not in ('false', '0', 'f')
BENCHMARK_OLLAMA = os.getenv("BENCHMARK_OLLAMA", 'True').lower() not in ('false', '0', 'f')

TTL_EVICT_NOW = 0
TTL_NEVER_EVICT = -1
TTL_DEFAULT = "5m"
TTL_DAY = "24h"

async def run_ollama_tests(model: str, prompt:str, iterations: int, ttl: int|str ):
  timeValues = [0]*iterations

  for i in range(iterations):
    start = timeit.default_timer()
    response = ollama.chat(
       model, 
       messages=[
        {
          'role': 'user',
          'content': prompt,
        }],
        keep_alive=ttl
      )
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
    
    if BENCHMARK_SERVER:
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
    else:
       logger.info("Skipping server benchmarks")

    if BENCHMARK_OLLAMA:
      # Local ollama tests with Granite 3

      # Load each model into memory for TTL_DEFAULT minutes
      timeValues = await run_ollama_tests(MODEL_FOR_CHAT, PROMPT_FOR_CHAT, 1, TTL_DEFAULT)
      logger.info(f"Total response time for the first {MODEL_FOR_CHAT} on ollama chat request: {sum(timeValues)}")

      timeValues = await run_ollama_tests(MODEL_FOR_COMPLETIONS, PROMPT_FOR_COMPLETIONS, 1, TTL_DEFAULT)
      logger.info(f"Total response time for the first {MODEL_FOR_COMPLETIONS} on ollama completion request: {sum(timeValues)}")

      # Interleave requests for two different models on local ollama with TTL_DEFAULT
      logger.info("Making several ollama calls to different models")
      start = timeit.default_timer()
      for i in range(RUNS_FOR_CHAT):
        timeValues = await run_ollama_tests(MODEL_FOR_CHAT, PROMPT_FOR_CHAT, 1, TTL_DEFAULT)
        logger.info(f"Total response time for the #{i} request to {MODEL_FOR_CHAT} on ollama: {sum(timeValues)}")
        timeValues = await run_ollama_tests(MODEL_FOR_COMPLETIONS, PROMPT_FOR_COMPLETIONS, 1, TTL_DEFAULT)
        logger.info(f"Total response time for the #{i} request to {MODEL_FOR_COMPLETIONS} on ollama: {sum(timeValues)}")
      inferenceTime = timeit.default_timer() - start
      logger.info(f"Total response time over several ollama calls to different models: {inferenceTime}; average: { (inferenceTime / (RUNS_FOR_CHAT * 2)) }")

      for keep_alive in [TTL_DEFAULT, TTL_EVICT_NOW, TTL_NEVER_EVICT]:
        logger.info(f"Testing with keep_alive = {keep_alive}")
        timeValues = await run_ollama_tests(MODEL_FOR_CHAT, PROMPT_FOR_CHAT, RUNS_FOR_CHAT, keep_alive)
        logger.info(f"Total response time over {RUNS_FOR_CHAT} {MODEL_FOR_CHAT} on ollama chat requests: {sum(timeValues)}; average: { (sum(timeValues) / RUNS_FOR_CHAT) })")
        timeValues = await run_ollama_tests(MODEL_FOR_COMPLETIONS, PROMPT_FOR_COMPLETIONS, RUNS_FOR_COMPLETIONS, keep_alive)
        logger.info(f"Total response time over {RUNS_FOR_CHAT} {MODEL_FOR_COMPLETIONS} on ollama completion requests: {sum(timeValues)}; average: { (sum(timeValues) / RUNS_FOR_COMPLETIONS ) }")
         
      logger.info(f"Setting the keep_alive for {MODEL_FOR_COMPLETIONS} to {TTL_DEFAULT}")
      timeValues = await run_ollama_tests(MODEL_FOR_COMPLETIONS, PROMPT_FOR_COMPLETIONS, 1, TTL_DEFAULT)

    else:
       logger.info("Skipping ollama benchmarks")

    logger.info("Tests complete")

if __name__ == "__main__":
    asyncio.run(run())
