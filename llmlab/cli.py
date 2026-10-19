"""Entry point: ask one question, print the answer and what it cost.

Loose mode shows the raw reply. Strict mode adds the output contract and
parses the result, so the two runs can be compared on tokens, not vibes.
"""
import argparse
import json

from llmlab import client, prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask an LLM and measure the prompt.")
    parser.add_argument("question")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="ask for a JSON object only and parse it",
    )
    parser.add_argument("--model", default=client.MODEL)
    args = parser.parse_args()

    messages = (
        prompts.strict_messages(args.question)
        if args.strict
        else prompts.build_messages(args.question)
    )
    raw = client.chat(messages, model=args.model)
    text = raw["choices"][0]["message"]["content"]

    print("--- system prompt ---")
    print(messages[0]["content"])
    print("--- answer ---")
    print(text)
    if args.strict:
        try:
            print("--- parsed ---")
            print(json.dumps(prompts.extract_json(text)))
        except (ValueError, TypeError) as err:
            print(f"--- parsed FAILED: {err}")
    print("--- usage ---")
    print(prompts.usage_totals(raw))


if __name__ == "__main__":
    main()