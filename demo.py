"""Demo: every minute, one data row (= a 1-minute snapshot) of one customer is sent to the model.
The previous status is passed along so the model has context.

Usage:
    python3 demo.py --people-id 2 --model llama_3.2
    python3 demo.py --people-id 2 --model qwen_2.5 --interval 5    # faster demo (5s per "minute")
"""
import argparse
import csv
import time
from datetime import datetime

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"


def rows_of(people_id, data_file):
    # Stream the file so a big dataset is not loaded into memory
    with open(data_file, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["PeopleId"] == people_id:
                yield row


def ask(model, prompt, retries=3):
    # Retry: the model sometimes starts with a newline, which the Modelfile's stop "\n" cuts to ""
    for _ in range(retries):
        r = requests.post(OLLAMA_URL, json={"model": model, "prompt": prompt, "stream": False, "think": False}, timeout=600)
        r.raise_for_status()
        answer = r.json()["response"].strip()
        if answer:
            return answer
    return "UNKNOWN: model không trả lời"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--people-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="dataset2.csv")
    parser.add_argument("--interval", type=float, default=60, help="seconds between snapshots")
    args = parser.parse_args()

    last_status = "none"
    for minute, row in enumerate(rows_of(args.people_id, args.data), 1):
        # One line, current data first: a multi-line prompt with "Input:" makes the model echo it back
        prompt = f"{','.join(row.values())} | previous status: {last_status}"
        status = ask(args.model, prompt)
        print(f"[{datetime.now():%H:%M:%S}] Customer {args.people_id} | minute {minute} | "
              f"{row['activityID']} -> {status}")
        if not status.startswith("UNKNOWN"):
            last_status = status
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
