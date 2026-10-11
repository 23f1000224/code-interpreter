import os
import sys
import traceback
from io import StringIO
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    api_key=os.environ["AIPIPE_TOKEN"],
    base_url="https://aipipe.org/openai/v1",
)

MODEL = "gpt-4.1-nano"


class CodeRequest(BaseModel):
    code: str


class ErrorAnalysis(BaseModel):
    error_lines: List[int]


class CodeResponse(BaseModel):
    error: List[int]
    result: str


def execute_python_code(code: str) -> dict:
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        exec(code, {"__name__": "__main__"})
        output = sys.stdout.getvalue()
        return {"success": True, "output": output}

    except Exception:
        output = traceback.format_exc()
        return {"success": False, "output": output}

    finally:
        sys.stdout = old_stdout


def analyze_error_with_ai(code: str, error_output: str) -> List[int]:
    prompt = f"""Analyze this Python code and its traceback.
Identify the exact line number(s) in CODE where the error occurred.

CODE:
{code}

TRACEBACK:
{error_output}

Return only the relevant line number(s), numbered from 1 for the first line of CODE.
"""

    response = client.chat.completions.parse(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format=ErrorAnalysis,
    )

    analysis = response.choices[0].message.parsed
    return analysis.error_lines if analysis else []


@app.post("/code-interpreter", response_model=CodeResponse)
async def code_interpreter(request: CodeRequest):
    execution = execute_python_code(request.code)

    if execution["success"]:
        return CodeResponse(error=[], result=execution["output"])

    error_lines = analyze_error_with_ai(request.code, execution["output"])
    return CodeResponse(error=error_lines, result=execution["output"])