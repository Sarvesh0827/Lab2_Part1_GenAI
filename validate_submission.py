# Usage: python validate_submission.py [submission.csv] [Questions.csv]
import re, sys, pandas as pd
COLS = ["id", "conversation_id", "question_id", "answer", "figure_references", "citations", "status"]
STATUS = {"answered", "not_found", "conflict", "false_premise", "session_override"}
LABEL = r"(Figure|Table) ([12]|[A-C])-\d+( \(corrected\))?"
FIG = re.compile(rf"^(NONE|{LABEL}(,{LABEL})*)$")
CIT = re.compile(rf"^(NONE|DOC[1-3]:(p\d+|{LABEL})(,DOC[1-3]:(p\d+|{LABEL}))*)$")
s = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "submission.csv", dtype=str, keep_default_na=False)
q = pd.read_csv(sys.argv[2] if len(sys.argv) > 2 else "Questions.csv", dtype=str, keep_default_na=False)
assert list(s.columns) == COLS, "wrong columns or column order"
assert list(s["id"]) == list(q["id"]), "ids missing, extra or out of order"
assert (s[["conversation_id", "question_id"]].values == q[["conversation_id", "question_id"]].values).all()
assert s["status"].isin(STATUS).all(), "invalid status value"
assert (s["answer"].str.strip() != "").all(), "empty answer"
assert s["figure_references"].str.match(FIG).all(), "bad figure_references"
assert s["citations"].str.match(CIT).all(), "bad citations"
print("valid:", len(s), "rows")
