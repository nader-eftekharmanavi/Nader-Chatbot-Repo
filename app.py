import os
from pathlib import Path

import gradio as gr
from openai import OpenAI
from pypdf import PdfReader



api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in Hugging Face Secrets.")

client = OpenAI(api_key=api_key)

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "Master.pdf"
SUMMARY_PATH = BASE_DIR / "Summary.txt"

if not PDF_PATH.exists():
    raise FileNotFoundError(f"PDF file not found: {PDF_PATH}")

if not SUMMARY_PATH.exists():
    raise FileNotFoundError(f"Summary file not found: {SUMMARY_PATH}")

reader = PdfReader(str(PDF_PATH))
linkedin = ""

for page in reader.pages:
    text = page.extract_text()
    if text:
        linkedin += text + "\n"

with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
    summary = f.read()

name = "Nader"

system_prompt = f"""You are acting as {name}. You are answering questions on {name}'s website,
particularly questions related to {name}'s career, background, skills and experience.
Your responsibility is to represent {name} for interactions on the website as faithfully as possible.
You are given a summary of {name}'s background and LinkedIn profile which you can use to answer questions.
Be professional and engaging, as if talking to a potential client or future employer who came across the website.
If you don't know the answer, say so.
## Summary:
{summary}
## LinkedIn Profile:
{linkedin}
With this context, please chat with the user, always staying in character as {name}.
"""


def chat(message, history):
    messages = [{"role": "system", "content": system_prompt}]

    for item in history:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})

    if isinstance(message, dict):
        user_content = message.get("content", "")
    else:
        user_content = message

    messages.append({"role": "user", "content": user_content})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    return response.choices[0].message.content

    

demo = gr.ChatInterface(
    fn=chat,
    title="Nader Chatbot",
    description="Ask questions about Nader's background, career, skills, and experience."
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )