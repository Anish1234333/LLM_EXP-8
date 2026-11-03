"""Print retrieval distances for supported and unsupported questions.
Use the output to pick MAX_DISTANCE: a value above every supported score
and below the unsupported scores (if the two groups separate at all)."""
from support_rag import get_rag

SUPPORTED = [
    "How do I block my debit card if it is lost?",
    "What documents do I need to open a savings account?",
    "How long does a cheque take to clear?",
    "How long does an NEFT transfer take?",
    "What is the minimum balance for a savings account?",
    "How do I apply for a personal loan?",
]
UNSUPPORTED = [
    "What is the interest rate on fixed deposits today?",
    "Can you approve my loan right now?",
    "Can you tell me the PIN of my debit card?",
    "What is the weather in Mumbai today?",
]

rag = get_rag()
for label, questions in [("supported", SUPPORTED), ("unsupported", UNSUPPORTED)]:
    for q in questions:
        best = rag.retrieve(q)[0][1]
        print(f"{label:12s} distance={best:.3f}  {q}")
