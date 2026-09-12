import io
import json
import zipfile

import pandas as pd

from services.analytics import results_to_dataframe


def make_excel(results):
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary = results_to_dataframe(results)
        summary.to_excel(
            writer,
            index=False,
            sheet_name="Results",
        )

        details = []

        for result in results:
            for q in result.get("question_results", []):
                details.append({
                    "Student": result.get("student_name"),
                    "Roll No": result.get("roll_no"),
                    **q,
                })

        if details:
            pd.DataFrame(details).to_excel(
                writer,
                index=False,
                sheet_name="Question Details",
            )

    return output.getvalue()


def make_json_zip(results):
    output = io.BytesIO()

    with zipfile.ZipFile(
        output,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as z:

        z.writestr(
            "results.json",
            json.dumps(
                results,
                indent=2,
                ensure_ascii=False,
            ),
        )

        for index, result in enumerate(results):
            student = (
                result.get("student_name") or "student"
            ).replace("/", "_")

            roll = (
                result.get("roll_no") or "unknown"
            ).replace("/", "_")

            filename = f"{index+1}_{roll}_{student}.json"

            z.writestr(
                f"students/{filename}",
                json.dumps(
                    result,
                    indent=2,
                    ensure_ascii=False,
                ),
            )

    return output.getvalue()
