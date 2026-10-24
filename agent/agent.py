"""Шаг 1. Первый запрос к модели — без инструментов.

Скрипт отправляет ровно один запрос в OpenAI-совместимый API
(по умолчанию RouterAI, поддерживается также OpenRouter) и печатает
три вещи, которые требует задание:

* message.content  — текст ответа модели;
* finish_reason    — причина завершения генерации (ожидается "stop");
* usage            — количество входных и выходных токенов.

Настройки берутся из agent/.env (см. agent/.env.example).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import APIStatusError, OpenAI

# Чтобы русский текст корректно печатался в консоли Windows.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ENV_PATH = Path(__file__).with_name(".env")
QUESTION = "Что такое токен?"

DEFAULT_BASE_URL = "https://routerai.ru/api/v1"
DEFAULT_MODEL = "deepseek/deepseek-v4-flash"

# Имена переменных, в которых может лежать ключ, в порядке приоритета.
API_KEY_VARS = ("ROUTERAI_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY")


def main() -> int:
    load_dotenv(ENV_PATH)

    api_key = next((os.getenv(name) for name in API_KEY_VARS if os.getenv(name)), None)
    if not api_key:
        print(
            f"Ошибка: в {ENV_PATH} не задан ключ API.\n"
            "Скопируйте agent/.env.example в agent/.env и впишите свой ключ "
            "в ROUTERAI_API_KEY (или OPENROUTER_API_KEY).",
            file=sys.stderr,
        )
        return 2

    base_url = os.getenv("BASE_URL", DEFAULT_BASE_URL)
    model = os.getenv("MODEL", DEFAULT_MODEL)

    client = OpenAI(base_url=base_url, api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": QUESTION}],
        )
    except APIStatusError as exc:
        print(f"Ошибка API (HTTP {exc.status_code}): {exc.message}", file=sys.stderr)
        if exc.status_code == 403:
            print(
                "\nПодсказка: 403 «Access denied by security policy» OpenRouter отдаёт\n"
                "запросам из заблокированного региона (ключ тут ни при чём).\n"
                "Варианты: использовать RouterAI (BASE_URL=https://routerai.ru/api/v1)\n"
                "или включить VPN/прокси и задать в agent/.env\n"
                "HTTPS_PROXY=http://127.0.0.1:<порт> (httpx подхватит её сам).",
                file=sys.stderr,
            )
        elif exc.status_code == 401:
            print("\nПодсказка: ключ неверный или отозван — проверьте ключ в agent/.env.", file=sys.stderr)
        elif exc.status_code == 402:
            print("\nПодсказка: на балансе провайдера закончились средства.", file=sys.stderr)
        return 1

    choice = response.choices[0] if response.choices else None
    if choice is None or choice.message is None:
        error = getattr(response, "error", None)
        print("Модель не вернула ответ.", file=sys.stderr)
        if error:
            print(f"Причина: {error.get('message', error)}", file=sys.stderr)
        else:
            print(f"Сырой ответ: {response.model_dump()}", file=sys.stderr)
        return 1

    message = choice.message
    finish_reason = choice.finish_reason
    usage = response.usage

    print("=== message.content ===")
    print(message.content)
    print()
    print("=== finish_reason ===")
    print(finish_reason)
    print()
    print("=== usage ===")
    if usage is None:
        print("usage отсутствует в ответе")
    else:
        print(f"prompt_tokens:     {usage.prompt_tokens}")
        print(f"completion_tokens: {usage.completion_tokens}")
        print(f"total_tokens:      {usage.total_tokens}")

    if finish_reason != "stop":
        print(f"\nВнимание: ожидался finish_reason = stop, получено {finish_reason!r}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
