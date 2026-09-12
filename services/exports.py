import io
import json
import zipfile
import pandas as pd
from services.analytics import results_to_dataframe


def make_excel(results):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        results_to_dataframe(results).to_excel(writer, index=False, sheet_name="Results")
        details = []
        for result in results:
            for q in result.get("question_results", []):
                details.append({"Student": result.get("student_name"), "Roll No": result.get("roll_no"), **q})
        if details:
            pd.DataFrame(details).to_excel(writer, index=False, sheet_name="Question Details")
    return output.getvalue()


def make_json_zip(results):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("results.json", json.dumps(results, indent=2, ensure_ascii=False))
        for index, result in enumerate(results):
            student = str(result.get("student_name") or "student").replace("/", "_")
            roll = str(result.get("roll_no") or "unknown").replace("/", "_")
            z.writestr(
                f"students/{index + 1}_{roll}_{student}.json",
                json.dumps(result, indent=2, ensure_ascii=False),
            )
    return output.getvalue()
