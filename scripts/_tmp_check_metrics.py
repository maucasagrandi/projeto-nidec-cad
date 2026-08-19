"""Check if any of the 10 validation sheets have numeric values in Objective Metrics."""
import openpyxl
import os

VALIDATION_DIR = "scripts/validation/41-50 Structured reviews"

METRIC_FIELDS = [
    "Quantidade de cotas",
    "Quantidade de cotas HIC",
    "Quantidade de cotas CTQ",
    "Quantidade de cotas CTQ-S",
    "Quantidade de GD&Ts",
    "Quantidade de Datums Reference",
    "Lista de datums reference",
    "Quantidade de revisões",
    "Quantidade de notas",
    "Quantidade de códigos",
]

found_any = False

for n in range(41, 51):
    path = f"{VALIDATION_DIR}/Drawing Data Extraction - Number_ {n}.xlsx"
    if not os.path.exists(path):
        path = f"{VALIDATION_DIR}/Drawing Data Extraction - Number_ {n}_.xlsx"

    wb = openpyxl.load_workbook(path)
    ws = wb["Single Drawing Data Extraction"]

    label_value = {
        row[0].strip(): (row[1] if len(row) > 1 else None)
        for row in ws.iter_rows(values_only=True)
        if row[0] and isinstance(row[0], str)
    }

    metrics_filled = {
        field: label_value[field]
        for field in METRIC_FIELDS
        if field in label_value and label_value[field] is not None
    }

    if metrics_filled:
        found_any = True
        print(f"TEST {n} — has metric values:")
        for field, val in metrics_filled.items():
            print(f"  {field}: {val}")
    else:
        print(f"TEST {n} — no metric values (all None)")

if not found_any:
    print("\nConclusion: NONE of the 10 sheets have Objective Metrics filled.")
