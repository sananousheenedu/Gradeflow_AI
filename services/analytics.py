import pandas as pd


def results_to_dataframe(results):
    rows = []
    for r in results:
        rows.append({
            "Roll No": r.get("roll_no"),
            "Student Name": r.get("student_name"),
            "Score": r.get("score"),
            "Max Marks": r.get("total_marks"),
            "Percentage": r.get("percentage"),
            "Grade": r.get("grade"),
            "Confidence": round(float(r.get("confidence", 0)) * 100, 1),
            "Review Required": bool(r.get("review_required")),
            "Status": "ERROR" if r.get("error") else "OK",
            "File": r.get("filename"),
        })
    return pd.DataFrame(rows)


def build_question_analytics(results):
    rows = []
    for result in results:
        for q in result.get("question_results", []):
            rows.append({"Question": q.get("question"), "Marks": float(q.get("marks_awarded", 0)), "Max Marks": float(q.get("max_marks", 0))})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    grouped = df.groupby("Question").agg(Average_Marks=("Marks", "mean"), Average_Max=("Max Marks", "mean")).reset_index()
    grouped["Average Marks"] = grouped["Average_Marks"].round(2)
    grouped["Average Max Marks"] = grouped["Average_Max"].round(2)
    grouped["Difficulty %"] = (grouped["Average_Marks"] / grouped["Average_Max"].replace(0, 1) * 100).round(1)
    return grouped[["Question", "Average Marks", "Average Max Marks", "Difficulty %"]]


def detect_review_flags(result, threshold):
    flags = []
    if float(result.get("confidence", 0)) < threshold:
        flags.append(f"AI confidence is below {threshold:.0%}.")
    if result.get("student_name") in ["", "Unknown", None]:
        flags.append("Student name could not be confidently identified.")
    if result.get("roll_no") in ["", "Unknown", None]:
        flags.append("Roll number could not be confidently identified.")
    if result.get("error"):
        flags.append("Processing error occurred.")
    return flags
