"""OpenAI Responses transport; credentials never enter prompts or archives."""

import base64
import os


def create_client(env_file, *, timeout=180, trust_env=False):
    from dotenv import dotenv_values
    import httpx
    from openai import OpenAI

    values = {**dotenv_values(env_file), **os.environ}
    if not values.get("BASE_URL") or not values.get("API_KEY"):
        raise ValueError("BASE_URL and API_KEY must be set in the env file or environment")
    return OpenAI(
        base_url=values["BASE_URL"],
        api_key=values["API_KEY"],
        timeout=timeout,
        max_retries=0,
        http_client=httpx.Client(trust_env=trust_env, timeout=timeout),
    )


def image_content(path):
    mime = "image/png" if path.suffix == ".png" else "image/jpeg"
    return {
        "type": "input_image",
        "image_url": f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}",
        "detail": "original",
    }


def request(client, *, model, instructions, messages, tools, reasoning, max_output_tokens):
    return client.responses.create(
        model=model,
        instructions=instructions,
        input=messages,
        tools=tools,
        parallel_tool_calls=False,
        store=False,
        include=["reasoning.encrypted_content"],
        reasoning={"effort": reasoning},
        max_output_tokens=max_output_tokens,
    )


def retryable(error: Exception) -> bool:
    from openai import APIConnectionError
    from openai import APIStatusError

    return isinstance(error, APIConnectionError) or (
        isinstance(error, APIStatusError) and (error.status_code in (400, 408, 409, 429) or error.status_code >= 500)
    )
