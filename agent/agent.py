"""Шаг 2. Чат с историей.

Скрипт держит диалог с моделью в цикле input(). Вся переписка хранится
в одном списке messages:

* первым элементом всегда идёт системный промпт (роль "system");
* каждый вопрос пользователя добавляется как {"role": "user", ...};
* каждый ответ модели добавляется как {"role": "assistant", ...}.

Список целиком отправляется в модель на каждом шаге, поэтому она видит
предыдущие реплики и может отвечать с учётом контекста.

Запуск:
    python agent.py           — интерактивный чат;
    python agent.py --once    — один запрос из шага 1 (проверка content,
                               finish_reason и usage без истории).

Команды в чате:
    /reset, /clear  — очистить историю (системный промпт остаётся);
    /exit, /quit    — выйти (то же самое делают exit, quit и Ctrl+D).

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
# Ввод тоже в UTF-8: консоль Windows и так работает в UTF-8, а при
# перенаправлении stdin это избавляет от ошибок декодирования.
sys.stdin.reconfigure(encoding="utf-8", errors="replace")

ENV_PATH = Path(__file__).with_name(".env")
QUESTION = "Что такое токен?"

DEFAULT_BASE_URL = "https://routerai.ru/api/v1"
DEFAULT_MODEL = "deepseek/deepseek-v4-flash"
DEFAULT_SYSTEM_PROMPT = (
    "Ты — ассистент проекта Computer Monitoring System. "
    "Отвечай на русском языке, кратко и по делу."
)

# Имена переменных, в которых может лежать ключ, в порядке приоритета.
API_KEY_VARS = ("ROUTERAI_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY")

EXIT_COMMANDS = {"/exit", "/quit", "exit", "quit", "выход"}
RESET_COMMANDS = {"/reset", "/clear"}


def print_api_error(exc: APIStatusError) -> None:
    """Печатает причину ошибки API и подсказку по частым кодам."""
    print(f"Ошибка API (HTTP {exc.status_code}): {exc.message}", file=sys.stderr)
    if exc.status_code == 401:
        print("Подсказка: ключ неверный или отозван — проверьте ключ в agent/.env.", file=sys.stderr)
    elif exc.status_code == 402:
        print("Подсказка: на балансе провайдера закончились средства.", file=sys.stderr)
    elif exc.status_code == 403:
        print(
            "Подсказка: 403 «Access denied by security policy» OpenRouter отдаёт\n"
            "запросам из заблокированного региона (ключ тут ни при чём).\n"
            "Варианты: использовать RouterAI (BASE_URL=https://routerai.ru/api/v1)\n"
            "или включить VPN/прокси и задать в agent/.env\n"
            "HTTPS_PROXY=http://127.0.0.1:<порт> (httpx подхватит её сам).",
            file=sys.stderr,
        )


def request_completion(client: OpenAI, model: str, messages: list[dict]) -> tuple[str, str, object] | None:
    """Отправляет всю историю messages и возвращает (текст, finish_reason, usage).

    Возвращает None, если запрос не удался: сообщение об ошибке уже напечатано.
    """
    try:
        response = client.chat.completions.create(model=model, messages=messages)
    except APIStatusError as exc:
        print_api_error(exc)
        return None

    choice = response.choices[0] if response.choices else None
    if choice is None or choice.message is None:
        # Некоторые провайдеры отвечают HTTP 200 с телом-ошибкой:
        # у такого ответа choices отсутствует, а причина лежит в поле error.
        error = getattr(response, "error", None)
        print("Модель не вернула ответ.", file=sys.stderr)
        if error:
            print(f"Причина: {error.get('message', error)}", file=sys.stderr)
        else:
            print(f"Сырой ответ: {response.model_dump()}", file=sys.stderr)
        return None

    return choice.message.content or "", choice.finish_reason or "unknown", response.usage


def format_usage(usage: object) -> str:
    """Компактная строка про израсходованные токены."""
    if usage is None:
        return "usage: нет данных"
    return (
        f"usage: prompt={usage.prompt_tokens} "
        f"completion={usage.completion_tokens} total={usage.total_tokens}"
    )


def run_once(client: OpenAI, model: str) -> int:
    """Шаг 1: один запрос без истории, с печатью content/finish_reason/usage."""
    result = request_completion(client, model, [{"role": "user", "content": QUESTION}])
    if result is None:
        return 1
    content, finish_reason, usage = result

    print("=== message.content ===")
    print(content)
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


def run_chat(client: OpenAI, model: str, system_prompt: str) -> int:
    """Шаг 2: цикл input(), история переписки в списке messages."""
    # Первым в списке всегда идёт системный промпт.
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    print(f"Модель: {model}")
    print("Чат с историей. Команды: /reset — очистить историю, /exit — выход.")
    print()

    while True:
        try:
            question = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            # Ctrl+D / Ctrl+C — завершаем чат без трейсбека.
            print()
            break

        if not question:
            continue
        if question.lower() in EXIT_COMMANDS:
            break
        if question.lower() in RESET_COMMANDS:
            messages = [{"role": "system", "content": system_prompt}]
            print("История очищена.\n")
            continue

        # Вопрос попадает в историю до запроса, чтобы модель видела его
        # вместе со всеми предыдущими репликами.
        messages.append({"role": "user", "content": question})

        result = request_completion(client, model, messages)
        if result is None:
            # Запрос не удался — убираем вопрос, чтобы история осталась согласованной.
            messages.pop()
            print("Реплика не добавлена в историю, попробуйте ещё раз.\n")
            continue

        answer, finish_reason, usage = result

        print(f"Агент: {answer}")
        print(f"[{format_usage(usage)}, finish_reason={finish_reason}, в истории {len(messages) + 1} сообщ.]")
        print()

        # Ответ модели тоже становится частью истории.
        messages.append({"role": "assistant", "content": answer})

        if finish_reason not in ("stop", "length"):
            print(f"Внимание: неожиданный finish_reason = {finish_reason!r}", file=sys.stderr)

    print(f"Чат завершён. Сообщений в истории (вместе с системным): {len(messages)}")
    return 0


def main() -> int:
    # load_dotenv по умолчанию не перезаписывает уже заданные переменные
    # окружения, поэтому .env подхватывается как значения по умолчанию.
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
    system_prompt = os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)

    client = OpenAI(base_url=base_url, api_key=api_key)

    if "--once" in sys.argv[1:]:
        return run_once(client, model)
    return run_chat(client, model, system_prompt)


if __name__ == "__main__":
    raise SystemExit(main())
