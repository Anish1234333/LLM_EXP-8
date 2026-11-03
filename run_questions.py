import csv
import time
import requests

API = "http://127.0.0.1:8000/ask"

with open("test_questions.csv", encoding="utf-8") as f, \
     open("answers_log.csv", "w", newline="", encoding="utf-8") as out:
    reader = csv.DictReader(f)
    writer = csv.writer(out)
    writer.writerow(["id", "question", "expected_type", "http_status",
                     "status", "sources", "latency_ms", "answer"])
    for row in reader:
        q = row["question"]
        start = time.perf_counter()
        r = requests.post(API, json={"question": q}, timeout=60)
        latency = (time.perf_counter() - start) * 1000
        if r.ok:
            d = r.json()
            status, sources, answer = d["status"], ";".join(d["sources"]), d["answer"]
        else:
            status, sources, answer = f"HTTP {r.status_code}", "", ""
        writer.writerow([row["id"], q, row["expected_type"], r.status_code,
                         status, sources, f"{latency:.1f}", answer])

        print("=" * 70)
        print(f"Q{row['id']} ({row['expected_type']}): {q}")
        print(f"Status: {status} | Sources: {sources or '-'} | {latency:.0f} ms")
        print(f"Answer: {answer}")
        print()