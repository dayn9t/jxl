from loguru import logger
from openai import OpenAI


def main() -> None:
    client = OpenAI()

    response = client.responses.create(
        model="gpt-4o", input="Write a one-sentence bedtime story about a unicorn."
    )

    logger.info("{}", response.output_text)


if __name__ == "__main__":
    main()
