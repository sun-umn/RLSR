import os
import importlib.util

# Explicitly load the local system_prompts.py so the RLCR submodule's copy
# (also on sys.path) can never shadow it.
_current_dir = os.path.dirname(os.path.abspath(__file__))
_sp_spec = importlib.util.spec_from_file_location(
    "local_system_prompts",
    os.path.join(_current_dir, "system_prompts.py")
)
_local_sp = importlib.util.module_from_spec(_sp_spec)
_sp_spec.loader.exec_module(_local_sp)
get_sys_prompt = _local_sp.get_sys_prompt


def process_dataset(dataset, script_args):
    sys_prompt = get_sys_prompt(script_args.sys_prompt_name)

    if script_args.task_spec == "gen":
        dataset = make_generation_dataset(dataset, sys_prompt)

    return dataset


def make_generation_dataset(dataset, sys_prompt):
    def make_generation_conversation(example):
        if 'question' in example.keys():
            user_format = (
                f"\n\nPROBLEM: {example['question']}\n\n"
                )
        else:
            user_format = (
                    f"\n\nPROBLEM: {example['problem']}\n\n"
                    )
        return {
            "prompt": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_format},
            ],
            "chat_template_kwargs": {},
        }

    dataset = dataset.map(make_generation_conversation)
    return dataset
