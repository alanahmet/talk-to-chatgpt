
import openai

def get_response(text):
  openai.api_key = "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"

  response = openai.Completion.create(
    model="text-davinci-003",
    prompt=text,
    temperature=0.7,
    max_tokens=128,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
  )

  return response.choices[0].get("text")
