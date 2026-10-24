"""Лабораторная работа №2. Агент с доступом к LLM.

Режимы запуска:
    python agent.py           — интерактивный чат с инструментами (шаги 2–5);
    python agent.py --once    — один запрос без инструментов (шаг 1).

Агент умеет работать с файлами в каталоге agent/workspace через инструменты
list_files, read_file и write_file, а также вести журнал мониторинга
компьютеров инструментами record_metric и list_computers (данные — в
agent/workspace/telemetry.json). Модель сама решает, когда их вызвать: ответ
приходит с tool_calls, агент выполняет вызовы и возвращает результаты модели,
пока она не даст обычный текстовый ответ (не длиннее MAX_STEPS шагов).

Настройки берутся из agent/.env (см. agent/.env.example).
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
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
    "Отвечай на русском языке, кратко и по делу. "
    "Показатели компьютеров фиксируй через record_metric, а состояние парка "
    "машин смотри через list_computers. "
    "Для работы с прочими файлами используй list_files, read_file и write_file."
)

# Имена переменных, в которых может лежать ключ, в порядке приоритета.
API_KEY_VARS = ("ROUTERAI_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY")

EXIT_COMMANDS = {"/exit", "/quit", "exit", "quit", "выход"}
RESET_COMMANDS = {"/reset", "/clear"}

# Согласием на запись считается только явный ответ; Enter и всё остальное — отказ.
CONFIRM_YES = {"y", "yes", "д", "да"}

# Рабочая папка агента: инструменты не выходят за её пределы.
WORKSPACE = Path(__file__).with_name("workspace")

# Ограничения, чтобы ответ инструмента не раздувал контекст.
MAX_ITEMS = 200
MAX_FILE_CHARS = 4000
MAX_STEPS_DEFAULT = 8

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": (
                "Показать список файлов и папок в рабочей директории агента. "
                "Используй, чтобы узнать, какие файлы доступны."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Относительный путь внутри рабочей папки. По умолчанию '.'",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Прочитать текстовый файл из рабочей директории агента. "
                "Возвращает содержимое (длинный файл обрезается)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Относительный путь к файлу внутри рабочей папки.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Создать или перезаписать текстовый файл в рабочей директории агента. "
                "Изменяет файловую систему, поэтому пользователь подтверждает запись "
                "вручную; при отказе файл остаётся без изменений."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Относительный путь к файлу внутри рабочей папки.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Полное новое содержимое файла.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_metric",
            "description": (
                "Записать новый замер компьютера в журнал мониторинга. "
                "Вызывай, когда пользователь сообщает показатели машины (CPU, RAM, диск, "
                "температуру) или просит зафиксировать её состояние. "
                "Проценты — от 0 до 100, температура — в градусах Цельсия (0..150). "
                "Состояние ok/warning/critical считается по порогам автоматически."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "computer": {
                        "type": "string",
                        "description": "Имя компьютера, например WS-01.",
                    },
                    "cpu_percent": {
                        "type": "number",
                        "description": "Загрузка процессора, % (0..100).",
                    },
                    "ram_percent": {
                        "type": "number",
                        "description": "Использование оперативной памяти, % (0..100).",
                    },
                    "disk_percent": {
                        "type": "number",
                        "description": "Занято дискового пространства, % (0..100).",
                    },
                    "temperature_c": {
                        "type": "number",
                        "description": "Температура процессора, °C (0..150).",
                    },
                },
                "required": [
                    "computer",
                    "cpu_percent",
                    "ram_percent",
                    "disk_percent",
                    "temperature_c",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_computers",
            "description": (
                "Показать последние замеры компьютеров и их состояние по порогам. "
                "Вызывай, когда нужно узнать, какие машины на связи, у кого превышены "
                "пороги или кто находится в критическом состоянии."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["all", "ok", "warning", "critical"],
                        "description": (
                            "Фильтр по состоянию: all — все машины (по умолчанию), "
                            "ok — в норме, warning — превышены пороги, critical — критично."
                        ),
                    }
                },
                "required": [],
            },
        },
    },
]


# --- инструменты -----------------------------------------------------------


def _resolve_in_workspace(raw_path: str) -> Path:
    """Приводит путь к абсолютному и проверяет, что он внутри WORKSPACE."""
    candidate = (WORKSPACE / raw_path).resolve()
    workspace = WORKSPACE.resolve()
    if candidate != workspace and workspace not in candidate.parents:
        raise ValueError(f"путь вне рабочей папки: {raw_path}")
    return candidate


def list_files(path: str = ".") -> str:
    """Возвращает список файлов и папок внутри рабочей директории."""
    target = _resolve_in_workspace(path)
    if not target.exists():
        raise FileNotFoundError(f"путь не найден: {path}")
    if not target.is_dir():
        raise NotADirectoryError(f"это не папка: {path}")

    entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    lines = [
        f"{entry.name}/" if entry.is_dir() else entry.name
        for entry in entries[:MAX_ITEMS]
    ]
    if not lines:
        return "(папка пуста)"
    if len(entries) > MAX_ITEMS:
        lines.append(f"... ещё {len(entries) - MAX_ITEMS} элементов")
    return "\n".join(lines)


def read_file(path: str) -> str:
    """Возвращает содержимое текстового файла из рабочей директории."""
    target = _resolve_in_workspace(path)
    if not target.exists():
        raise FileNotFoundError(f"файл не найден: {path}")
    if not target.is_file():
        raise IsADirectoryError(f"это папка, а не файл: {path}")

    text = target.read_text(encoding="utf-8", errors="replace")
    if len(text) > MAX_FILE_CHARS:
        text = text[:MAX_FILE_CHARS] + f"\n... (обрезано, всего {len(text)} символов)"
    return text or "(файл пуст)"


def write_file(path: str, content: str) -> str:
    """Создаёт или перезаписывает файл, но только после подтверждения.

    Запись меняет файловую систему, поэтому перед ней печатается путь и
    фрагмент содержимого, а затем задаётся вопрос «Разрешить? [y/N]».
    Подтверждением считается только явное «y»/«да»; Enter, «n», Ctrl+D
    и Ctrl+C означают отказ, и файл остаётся без изменений.
    """
    target = _resolve_in_workspace(path)
    if target.is_dir():
        raise IsADirectoryError(f"это папка, записать файл нельзя: {path}")

    exists = target.exists()
    action = "перезаписать" if exists else "создать"
    done = "перезаписан" if exists else "создан"

    print()
    print(f"[подтверждение] {action} файл: {path}")
    if exists:
        print(f"  сейчас в файле {target.stat().st_size} байт")
    print(f"  новый размер: {len(content)} символов")
    print("  --- начало содержимого ---")
    for line in content.splitlines()[:10] or [""]:
        print(f"  | {line}")
    if len(content.splitlines()) > 10:
        print(f"  | ... ещё {len(content.splitlines()) - 10} строк")
    print("  --- конец ---")

    try:
        answer = input("Разрешить? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        # Нет ввода (Ctrl+D/Ctrl+C) — это не согласие.
        print()
        answer = ""

    if answer not in CONFIRM_YES:
        return f"ОТКАЗ: пользователь не разрешил запись файла '{path}'. Файл не изменён."

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"OK: файл '{path}' {done} ({len(content)} символов)."


# --- инструменты проекта: журнал мониторинга (шаг 5) -----------------------

# Все данные инструментов проекта лежат в JSON-файле внутри workspace/.
TELEMETRY_FILE = WORKSPACE / "telemetry.json"
MAX_RECORDS_PER_COMPUTER = 50
MAX_COMPUTER_NAME = 64

# Пороги: превышение — предупреждение, сильное превышение — критично.
THRESHOLDS = {
    "cpu_percent": (80.0, 90.0),
    "ram_percent": (80.0, 90.0),
    "disk_percent": (85.0, 95.0),
    "temperature_c": (70.0, 85.0),
}
METRIC_LABELS = {
    "cpu_percent": "cpu",
    "ram_percent": "ram",
    "disk_percent": "disk",
    "temperature_c": "temp",
}
ALLOWED_STATUSES = ("all", "ok", "warning", "critical")


def _fmt_number(value: float) -> str:
    """42.0 → «42», 41.5 → «41.5» — чтобы результат был коротким."""
    return f"{value:.0f}" if float(value).is_integer() else f"{value:.1f}"


def _check_number(field: str, value: object, low: float, high: float) -> float:
    """Проверяет, что значение — число в допустимом диапазоне."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} должен быть числом, получено {value!r}")
    if not low <= value <= high:
        raise ValueError(
            f"{field} должен быть в диапазоне {_fmt_number(low)}..{_fmt_number(high)}, "
            f"получено {_fmt_number(value)}"
        )
    return float(value)


def _load_telemetry() -> dict:
    """Читает telemetry.json; отсутствие файла означает пустой журнал."""
    if not TELEMETRY_FILE.exists():
        return {"computers": {}}

    try:
        data = json.loads(TELEMETRY_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"файл telemetry.json повреждён ({exc.msg}), исправьте или удалите его"
        ) from exc

    if not isinstance(data, dict) or not isinstance(data.get("computers"), dict):
        raise ValueError('файл telemetry.json имеет неверный формат, ожидался {"computers": {...}}')
    return data


def _save_telemetry(data: dict) -> None:
    """Записывает журнал обратно в telemetry.json (данные инструмента, без подтверждения)."""
    WORKSPACE.mkdir(exist_ok=True)
    TELEMETRY_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _evaluate(record: dict) -> str:
    """Состояние машины по одному замеру: ok / warning / critical."""
    state = "ok"
    for field, (warn, crit) in THRESHOLDS.items():
        value = record.get(field)
        if value is None:
            continue
        if value >= crit:
            return "critical"
        if value >= warn:
            state = "warning"
    return state


def _describe(record: dict) -> str:
    """Короткая строка показателей: «cpu 34%, ram 62%…»."""
    parts = []
    for field, label in METRIC_LABELS.items():
        value = record.get(field)
        if value is None:
            continue
        unit = "°C" if field == "temperature_c" else "%"
        parts.append(f"{label} {_fmt_number(value)}{unit}")
    return ", ".join(parts) if parts else "нет показателей"


def record_metric(computer: str, cpu_percent, ram_percent, disk_percent, temperature_c) -> str:
    """Сохраняет замер компьютера в telemetry.json и возвращает короткий итог."""
    name = (computer or "").strip()
    if not name:
        raise ValueError("computer не должен быть пустым")
    if len(name) > MAX_COMPUTER_NAME:
        raise ValueError(f"computer слишком длинный, максимум {MAX_COMPUTER_NAME} символов")

    record = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "computer": name,
        "cpu_percent": _check_number("cpu_percent", cpu_percent, 0, 100),
        "ram_percent": _check_number("ram_percent", ram_percent, 0, 100),
        "disk_percent": _check_number("disk_percent", disk_percent, 0, 100),
        "temperature_c": _check_number("temperature_c", temperature_c, 0, 150),
    }

    data = _load_telemetry()
    records = data["computers"].setdefault(name, [])
    records.append(record)
    # Держим только последние замеры, чтобы файл не рос бесконечно.
    del records[:-MAX_RECORDS_PER_COMPUTER]
    _save_telemetry(data)

    return (
        f"OK: {name} — {_describe(record)}; состояние {_evaluate(record)} "
        f"(записей: {len(records)})"
    )


def list_computers(status: str = "all") -> str:
    """Показывает последние замеры компьютеров, при необходимости с фильтром."""
    status = (status or "all").strip().lower()
    if status not in ALLOWED_STATUSES:
        allowed = ", ".join(ALLOWED_STATUSES)
        raise ValueError(f"status должен быть одним из: {allowed}; получено {status!r}")

    computers = _load_telemetry()["computers"]
    if not computers:
        return "Журнал пуст: ни один компьютер ещё не присылал замеры."

    lines = []
    for name in sorted(computers):
        records = computers[name]
        if not records:
            continue
        last = records[-1]
        state = _evaluate(last)
        if status != "all" and state != status:
            continue
        lines.append(f"{name}: {state} — {_describe(last)} ({last.get('timestamp', '—')})")

    if not lines:
        return f"Нет компьютеров в состоянии {status}."
    return "\n".join([f"Компьютеров в выборке: {len(lines)}", *lines[:MAX_ITEMS]])


TOOL_FUNCTIONS = {
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file,
    "record_metric": record_metric,
    "list_computers": list_computers,
}


def execute_tool(name: str, raw_args: str) -> str:
    """Выполняет инструмент по имени и JSON-аргументам.

    Любая ошибка (кривой JSON, неизвестный инструмент, исключение внутри
    функции) возвращается строкой «ОШИБКА: ...», а не выбрасывается наружу:
    агент должен продолжить работу и рассказать модели, что пошло не так.
    """
    try:
        args = json.loads(raw_args) if raw_args and raw_args.strip() else {}
    except json.JSONDecodeError as exc:
        return f"ОШИБКА: не удалось разобрать аргументы JSON ({exc.msg}): {raw_args!r}"

    if not isinstance(args, dict):
        return f"ОШИБКА: аргументы должны быть JSON-объектом, получено {type(args).__name__}"

    function = TOOL_FUNCTIONS.get(name)
    if function is None:
        available = ", ".join(sorted(TOOL_FUNCTIONS))
        return f"ОШИБКА: неизвестный инструмент {name!r}. Доступны: {available}"

    try:
        return function(**args)
    except TypeError as exc:
        return f"ОШИБКА: неверные аргументы для {name}: {exc}"
    except ValueError as exc:
        # Ошибки валидации инструменты формулируют сами — класс исключения не нужен.
        return f"ОШИБКА: {exc}"
    except Exception as exc:  # noqa: BLE001 — любую ошибку возвращаем строкой
        return f"ОШИБКА: {type(exc).__name__}: {exc}"


# --- работа с API ----------------------------------------------------------


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
    """Запрос без инструментов (шаг 1). Возвращает (текст, finish_reason, usage)."""
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


def preview(text: str, limit: int = 300) -> str:
    """Однострочный фрагмент для лога."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[:limit] + "…"


# --- агентный цикл (шаг 3) -------------------------------------------------


def run_agent(client: OpenAI, model: str, messages: list[dict], max_steps: int = MAX_STEPS_DEFAULT) -> str | None:
    """Цикл «запрос → инструменты → ответ», не длиннее max_steps шагов.

    messages изменяется на месте: в него добавляются сообщения ассистента
    с tool_calls и сообщения с ролью tool. Возвращает финальный текст ответа
    или None, если запрос к API не удался.
    """
    for step in range(1, max_steps + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
            )
        except APIStatusError as exc:
            print_api_error(exc)
            return None

        choice = response.choices[0] if response.choices else None
        if choice is None or choice.message is None:
            error = getattr(response, "error", None)
            print("Модель не вернула ответ.", file=sys.stderr)
            if error:
                print(f"Причина: {error.get('message', error)}", file=sys.stderr)
            else:
                print(f"Сырой ответ: {response.model_dump()}", file=sys.stderr)
            return None

        message = choice.message
        tool_calls = message.tool_calls or []

        # В лог — имя инструмента, аргументы и usage: это идёт в отчёт.
        print(
            f"[шаг {step}/{max_steps}] {format_usage(response.usage)} "
            f"finish_reason={choice.finish_reason}"
        )

        # Ответ ассистента сохраняется в историю целиком (с tool_calls),
        # иначе модель не увидит связки «вызов → результат».
        messages.append(message.model_dump(exclude_none=True))

        if not tool_calls:
            return message.content or "(модель вернула пустой ответ)"

        for call in tool_calls:
            name = call.function.name
            raw_args = call.function.arguments or "{}"
            print(f"[шаг {step}] инструмент: {name} | аргументы: {raw_args}")

            result = execute_tool(name, raw_args)
            print(f"[шаг {step}] результат {name}: {preview(result)}")

            tool_message = {
                "role": "tool",
                "tool_call_id": call.id,
                "content": result,
            }
            messages.append(tool_message)

            # AGENT_DEBUG=1 — печатать точное сообщение, которое получит модель.
            if os.getenv("AGENT_DEBUG"):
                print(
                    f"[шаг {step}] модели отправлено: "
                    f"{json.dumps(tool_message, ensure_ascii=False)}"
                )

    # Лимит шагов исчерпан — честно сообщаем об этом и модели, и пользователю.
    print(f"Достигнут лимит MAX_STEPS={max_steps}, агент остановлен.", file=sys.stderr)
    return (
        f"Не удалось завершить задачу за {max_steps} шагов. "
        "Попробуйте уточнить запрос или увеличить MAX_STEPS."
    )


# --- режимы запуска --------------------------------------------------------


def run_once(client: OpenAI, model: str) -> int:
    """Шаг 1: один запрос без истории и без инструментов."""
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


def run_chat(client: OpenAI, model: str, system_prompt: str, max_steps: int) -> int:
    """Шаги 2–3: цикл input(), история в messages, инструменты через run_agent."""
    # Первым в списке всегда идёт системный промпт.
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    print(f"Модель: {model}")
    print(f"Рабочая папка: {WORKSPACE}")
    print("Команды: /reset — очистить историю, /exit — выход.")
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

        answer = run_agent(client, model, messages, max_steps)
        if answer is None:
            # Запрос не удался — убираем вопрос, чтобы история осталась согласованной.
            messages.pop()
            print("Реплика не добавлена в историю, попробуйте ещё раз.\n")
            continue

        print(f"Агент: {answer}")
        print(f"[в истории {len(messages)} сообщ.]")
        print()

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
    try:
        max_steps = int(os.getenv("MAX_STEPS", str(MAX_STEPS_DEFAULT)))
    except ValueError:
        print("Ошибка: MAX_STEPS должен быть целым числом.", file=sys.stderr)
        return 2

    WORKSPACE.mkdir(exist_ok=True)
    client = OpenAI(base_url=base_url, api_key=api_key)

    if "--once" in sys.argv[1:]:
        return run_once(client, model)
    return run_chat(client, model, system_prompt, max_steps)


if __name__ == "__main__":
    raise SystemExit(main())
