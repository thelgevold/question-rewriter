import os

from llama_index.core import PromptTemplate, Settings
from llama_index.llms.ollama import Ollama


OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "question-rewriter-qwen3-0.6b")
SYSTEM_PROMPT = (
    "You rewrite a conversational sequence of homeowner questions and answers "
    "into one self-contained question. Keep the output as a single question, "
    "preserve concrete entities, dates, and service details when needed, and "
    "do not answer the question."
)
USER_TASK_INTRO = (
    "Rewrite the following conversation into one standalone question that "
    "makes sense without the earlier chat history."
)
Settings.llm_rewrite = Ollama(
    model=MODEL_NAME,
    base_url=OLLAMA_HOST,
    temperature=0,
    context_window=4096,
    request_timeout=120.0,
    thinking=False,
    additional_kwargs={"num_predict": 128},
)


def build_user_prompt(conversation: list[dict[str, str]]) -> str:
    lines = [f"{message['role'].title()}: {message['content']}" for message in conversation]
    return (
        f"{USER_TASK_INTRO}\n\n"
        f"Conversation:\n" + "\n".join(lines) + "\n\n"
        "Standalone question:"
    )


def main() -> None:
    conversation_thread = [
        {"role": "user", "content": "when did we install the driveway security camera"},
        {"role": "assistant", "content": "We installed the driveway security camera on 2024-09-17."},
        {"role": "user", "content": "how much did it cost"},
        {"role": "assistant", "content": "It cost $640."},
        {"role": "user", "content": "who did it"},
    ]

    print("\nConversation Thread", flush=True)
    print("-------------------", flush=True)
    for message in conversation_thread:
        print(f"{message['role'].title()}: {message['content']}", flush=True)

    print("\nSending conversation to Ollama...", flush=True)
    raw_prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{build_user_prompt(conversation_thread)}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    rewrite = Settings.llm_rewrite.predict(PromptTemplate(raw_prompt)).strip()
    print(f"\nModel rewrite: {rewrite}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("Unable to reach Ollama.", flush=True)
        print(f"Checked host: {OLLAMA_HOST}", flush=True)
        print(f"Underlying error: {error}", flush=True)
        raise
