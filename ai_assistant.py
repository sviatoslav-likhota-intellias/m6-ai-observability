import json
import os
import random
import re
import sys
from dotenv import load_dotenv
from langfuse.openai import openai
from langfuse import observe, propagate_attributes
from langfuse import Langfuse



# Add the repo root to path so we can import from shared/
sys.path.append(os.path.abspath(".."))
from shared.shopbot_config import SHOPBOT_SYSTEM_PROMPT

load_dotenv(".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()
langfuse = Langfuse()

LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Model comes from .env (e.g. a Groq model); the OpenAI client also reads OPENAI_BASE_URL from env.
MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")

print("Setup complete — Langfuse client initialised")
print(f"Traces will appear at: {LANGFUSE_BASE_URL}")

JUDGE_PROMPT = """You are evaluating a customer support answer for an online store.
Store facts (ground truth):
{facts}

Customer question: {question}
Bot answer: {answer}

Rate the answer's helpfulness from 1 (useless or wrong) to 5 (accurate, complete, concise).
Respond with JSON only: {{"score": <1-5>, "reason": "<one short sentence>"}}"""


def fact_check(answer: str, expected_patterns: list[str]) -> bool:
    """Rule-based check: every expected fact (regex) must appear in the answer."""
    return all(re.search(p, answer, re.IGNORECASE) for p in expected_patterns)


@observe(name="llm-judge", as_type="evaluator")
def judge_helpfulness(question: str, answer: str) -> dict:
    """LLM-as-judge: ask the model to rate the answer 1-5 with a short reason."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": JUDGE_PROMPT.format(
                facts=SHOPBOT_SYSTEM_PROMPT, question=question, answer=answer
            ),
        }],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


# Tags are applied via propagate_attributes below (newer SDKs don't accept tags/type on @observe).
@observe(name="chatbot")
def ask_bot(user_message: str, expected_patterns: list[str] | None = None) -> str:
    properties = {"feature": "bot", "channel": "cli"}

    # Attach metadata to the observed function.
    langfuse.update_current_span(metadata=properties)

    # Propagate standard IDs, tags, and metadata to nested observations.
    with propagate_attributes(
        user_id="user_" + str(random.randint(1000, 9999)),
        session_id="session_" + str(random.randint(1000, 9999)),
        tags=["shopbot", "support"],
        metadata=properties,
    ):
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SHOPBOT_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        answer = response.choices[0].message.content

        # Score 1: rule-based fact check (BOOLEAN) — does the answer contain the expected store fact?
        if expected_patterns:
            passed = fact_check(answer, expected_patterns)
            langfuse.score_current_trace(
                name="fact_check",
                value=1 if passed else 0,
                data_type="BOOLEAN",
                comment=f"expected: {expected_patterns}",
            )

        # Score 2: LLM-as-judge helpfulness (NUMERIC 1-5) with the judge's reasoning as comment.
        verdict = judge_helpfulness(user_message, answer)
        langfuse.score_current_trace(
            name="helpfulness",
            value=float(verdict["score"]),
            data_type="NUMERIC",
            comment=verdict.get("reason"),
        )

    print(f"Q: {user_message}\n   fact_check={passed if expected_patterns else 'n/a'}, "
          f"helpfulness={verdict['score']} ({verdict.get('reason')})")
    return answer


# (question, regexes the answer must contain to pass the fact check)
TEST_CASES = [
    ("What is your return policy?", [r"30\s*days?"]),
    ("What is your refund method?", [r"original\s+payment"]),
    ("What are your customer support hours?", [r"9\s*a\.?m", r"6\s*p\.?m"]),
    ("Do you offer free shipping?", [r"\$\s*75"]),
]

for question, expected in TEST_CASES:
    ask_bot(question, expected)

# Make sure all buffered traces are sent before the script exits.
langfuse.flush()