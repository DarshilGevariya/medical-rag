import json
import uuid
from pathlib import Path
import xml.etree.ElementTree as ET

def parse():
    records = []
    base = Path("MedQuAD")

    for file in base.rglob("*.xml"):
        try:
            tree = ET.parse(file)
            root = tree.getroot()

            # MedQuAD structure: QAPair / Question / Answer
            for qa in root.iter("QAPair"):
                q = qa.findtext("Question")
                a = qa.findtext("Answer")

                if q and a:
                    records.append({
                        "id": str(uuid.uuid4()),
                        "question": q.strip(),
                        "answer": a.strip(),
                        "source": str(file)
                    })

        except Exception as e:
            # skip malformed files
            continue

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    out = Path("data/processed/qa_records.jsonl")

    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Parsed {len(records)} QA pairs")

if __name__ == "__main__":
    parse()
