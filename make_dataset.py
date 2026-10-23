import os
import json
import time

from typing import Union

from google import genai
from google.genai import types

from pydantic import BaseModel, Field


# ==================================================
# Setting
# ==================================================


API_KEY = "GEMINI_API_KEY"

MODEL_NAME = "gemini-2.5-flash"

LOOP_COUNT = 5

WAIT_SECONDS = 60

PROMPT_FILE = "prompts/schedule_request_prompt.txt"

DATASET_FILE = "dataset/dataset.json"


# ==================================================
# Gemini
# ==================================================

client = genai.Client(
    api_key=API_KEY
)


# ==================================================
# Schema
# ==================================================

class DialoguePair(BaseModel):

    instruction: str = Field()
    output: str = Field()

    action_required: bool = Field()

    action_trigger: str = Field()

    action_argument: Union[int, str] = Field()


class NPCDataSet(BaseModel):

    dataset: list[DialoguePair]


# ==================================================
# Prompt Loader
# ==================================================

def load_prompt(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return f.read()


sys_instruction = load_prompt(
    "prompts/sys_instruction.txt"
)

user_prompt = load_prompt(
    PROMPT_FILE
)


# ==================================================
# Validation Rules
# ==================================================

FORBIDDEN_PATTERNS = [

    "요.",
    "요?",
    "요!",
    "습니다",
    "습니까",
    "세요",
    "계세요",
    "드릴게요",
    "해주세요",
]

REQUIRED_KEYS = {

    "instruction",
    "output",
    "action_required",
    "action_trigger",
    "action_argument"
}


# ==================================================
# Validators
# ==================================================

def validate_schema(item):

    return REQUIRED_KEYS.issubset(
        item.keys()
    )


def validate_length(item):

    instruction = item["instruction"].strip()

    output = item["output"].strip()

    return (
        len(instruction) >= 2
        and len(output) >= 5
    )


def validate_tone(item):

    output = item["output"]

    return not any(
        pattern in output
        for pattern in FORBIDDEN_PATTERNS
    )


def validate_action(item):

    output = item["output"]

    if "{current_time}" in output:

        return (
            item["action_required"] is True
            and item["action_trigger"]
            == "GET_CURRENT_TIME"
        )

    return True


def validate_repetition(item):

    output = item["output"]

    keywords = [

        "메인 모니터",
        "타임라인 스케줄러",
        "디지털 시계"
    ]

    hit_count = sum(
        keyword in output
        for keyword in keywords
    )

    return hit_count <= 1


def is_duplicate(
    item,
    seen_instruction,
    seen_output
):

    return (
        item["instruction"]
        in seen_instruction

        or

        item["output"]
        in seen_output
    )


# ==================================================
# Dataset Filter
# ==================================================

def filter_dataset(
    dataset,
    seen_instruction,
    seen_output
):

    validated = []

    stats = {

        "schema_fail": 0,
        "length_fail": 0,
        "tone_fail": 0,
        "action_fail": 0,
        "repetition_fail": 0,
        "duplicate_fail": 0,
        "accepted": 0
    }

    for item in dataset:

        if not validate_schema(item):

            stats["schema_fail"] += 1
            continue

        if not validate_length(item):

            stats["length_fail"] += 1
            continue

        if not validate_tone(item):

            stats["tone_fail"] += 1
            continue

        if not validate_action(item):

            stats["action_fail"] += 1
            continue

        if not validate_repetition(item):

            stats["repetition_fail"] += 1
            continue

        if is_duplicate(
            item,
            seen_instruction,
            seen_output
        ):

            stats["duplicate_fail"] += 1
            continue

        validated.append(item)

        seen_instruction.add(
            item["instruction"]
        )

        seen_output.add(
            item["output"]
        )

        stats["accepted"] += 1

    return validated, stats


# ==================================================
# Dataset Load
# ==================================================

os.makedirs(
    "dataset",
    exist_ok=True
)

existing_dataset = []

if os.path.exists(DATASET_FILE):

    with open(
        DATASET_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        try:

            existing_dataset = json.load(f)

        except json.JSONDecodeError:

            existing_dataset = []

print(
    f"기존 데이터 {len(existing_dataset)}개 로드"
)

seen_instruction = {

    item["instruction"]

    for item in existing_dataset
}

seen_output = {

    item["output"]

    for item in existing_dataset
}


# ==================================================
# Main Loop
# ==================================================

for i in range(LOOP_COUNT):

    print(
        f"\n[{i+1}/{LOOP_COUNT}] 생성 시작"
    )

    try:

        response = client.models.generate_content(

            model=MODEL_NAME,

            contents=user_prompt,

            config=types.GenerateContentConfig(

                system_instruction=sys_instruction,

                response_mime_type="application/json",

                response_schema=NPCDataSet,

                temperature=0.7
            )
        )

        json_data = json.loads(
            response.text
        )

        generated = json_data["dataset"]

        validated, stats = filter_dataset(

            generated,

            seen_instruction,

            seen_output
        )

        existing_dataset.extend(
            validated
        )

        with open(
            DATASET_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                existing_dataset,
                f,
                ensure_ascii=False,
                indent=2
            )

        print(
            f"생성: {len(generated)}"
        )

        print(
            f"통과: {stats['accepted']}"
        )

        print(
            f"중복 제거: {stats['duplicate_fail']}"
        )

        print(
            f"존댓말 제거: {stats['tone_fail']}"
        )

        print(
            f"액션 오류 제거: {stats['action_fail']}"
        )

        print(
            f"누적 데이터: {len(existing_dataset)}"
        )

        if i < LOOP_COUNT - 1:

            print(
                f"{WAIT_SECONDS}초 대기"
            )

            time.sleep(
                WAIT_SECONDS
            )

    except Exception as e:

        print(
            f"에러 발생: {e}"
        )

        if i < LOOP_COUNT - 1:

            time.sleep(60)

print(
    "\n데이터 생성 완료"
)