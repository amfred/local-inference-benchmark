# Start up the local ollama server before using this
# ollama pull granite3-dense:8b
# ollama serve

import ollama
import timeit
import logging
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("granite-from-ollama")

MODEL_FOR_CHAT="granite3-dense:8b"
MODEL_FOR_COMPLETIONS="granite3-moe:3b"

PROMPT_FOR_CHAT="Generate a function in python that will calculate the first N numbers in a Fibonacci sequence"
PROMPT_FOR_COMPLETIONS="# A function in python that will calculate the first N numbers in a Fibonacci sequence"

RUNS_FOR_CHAT=5
RUNS_FOR_COMPLETIONS=5

OLLAMA_URL="http://127.0.0.1:11434/api/chat"

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
    logger.info(f"Response #{i}: {response['message']['content']}")

  return timeValues

async def run():

    timeValues = await run_ollama_tests(MODEL_FOR_CHAT, PROMPT_FOR_CHAT, 1)
    logger.info(f"Total response time over {RUNS_FOR_CHAT} ollama chat runs: {sum(timeValues)}")

    timeValues = await run_ollama_tests(MODEL_FOR_CHAT, PROMPT_FOR_CHAT, RUNS_FOR_CHAT)
    logger.info(f"Total response time over {RUNS_FOR_CHAT} ollama chat runs: {sum(timeValues)}")

    timeValues = await run_ollama_tests(MODEL_FOR_COMPLETIONS, PROMPT_FOR_COMPLETIONS, 1)
    logger.info(f"Total response time over {RUNS_FOR_CHAT} ollama chat runs: {sum(timeValues)}")

if __name__ == "__main__":
    asyncio.run(run())
