"""Create the local MedQA 4-options eval dataset (extended OOD, MCQ).

Builds from the public GBaker/MedQA-USMLE-4-options (verified row-for-row
identical to the original Jin et al. data_clean US/4_options test split:
1273/1273 questions, answer letters, and option dicts match exactly).
Follows the house pattern: `problem` = question with options rendered in,
`answer` = the gold letter, `source` = 'medqa' for reward/check routing;
save_to_disk under RLCR/data/medqa_4options.
"""

from datasets import load_dataset, DatasetDict


def format_question_with_options(question, options):
    formatted = question.strip()
    formatted += "\n\nOptions:"
    for letter in sorted(options.keys()):
        formatted += f"\n{letter}. {options[letter]}"
    return formatted


def to_row(ex):
    return {
        "problem": format_question_with_options(ex["question"], ex["options"]),
        "question_raw": ex["question"],
        "options": ex["options"],
        "answer": ex["answer_idx"],   # gold letter — what MCQ grading compares
        "answer_text": ex["answer"],  # full text, kept for reference
        "meta_info": ex.get("meta_info", ""),
        "source": "medqa",            # routing key for accuracy_reward/check_fn
    }


def main():
    ds = load_dataset("GBaker/MedQA-USMLE-4-options")["test"]
    print(f"Loaded GBaker/MedQA-USMLE-4-options test: {len(ds)} questions")
    out = ds.map(to_row, remove_columns=ds.column_names)
    dd = DatasetDict({"test": out})
    dd.save_to_disk("RLCR/data/medqa_4options")
    print("Saved RLCR/data/medqa_4options:", dd)


if __name__ == "__main__":
    main()
