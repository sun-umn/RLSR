"""Create the local MBPP eval dataset (Task D, programming OOD).

Uses the `mbpp` full config's test split (500 problems, task ids 11-510).
MBPP convention: the prompt includes the reference tests so the model knows the
expected function name/signature. Execution fields (`test_list`,
`test_setup_code`) are carried as extra columns. Saved under RLCR/data/mbpp.
"""

from datasets import load_dataset, DatasetDict


def main():
    ds = load_dataset("mbpp")["test"]
    print(f"Loaded mbpp (full) test split: {len(ds)} problems")

    def to_row(ex):
        tests = "\n".join(ex["test_list"])
        problem = (
            f"{ex['text']}\n\nYour solution must pass these tests:\n{tests}\n\n"
            "Inside your <answer> tags, output ONLY the complete Python function "
            "definition (including any imports you need), with no explanations outside the code."
        )
        return {
            "problem": problem,
            "answer": ex["code"],  # reference only (not used for grading)
            "source": "mbpp",
            "test_list": ex["test_list"],
            "test_setup_code": ex.get("test_setup_code") or "",
            "task_id": ex["task_id"],
        }

    out = ds.map(to_row, remove_columns=ds.column_names)
    dd = DatasetDict({"test": out})
    dd.save_to_disk("RLCR/data/mbpp")
    print("Saved RLCR/data/mbpp:", dd)


if __name__ == "__main__":
    main()
