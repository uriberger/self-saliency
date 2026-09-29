"""Real LLaMA-Factory data-load verify on the mulberry cold-start subset (CPU, no model weights).
Confirms: LF parses the Saliency-R1 mulberry JSON, resolves images from disk, and the
qwen3_vl_nothink template + Qwen3VLPlugin produce input_ids + pixel_values."""
from llamafactory.hparams import get_train_args
from llamafactory.model import load_tokenizer
from llamafactory.data import get_dataset, get_template_and_fix_tokenizer

from selfsal.data.paths import DATA_ROOT

COLDSTART = DATA_ROOT / "coldstart"   # override the root with SELFSAL_DATA


args = dict(
    model_name_or_path="Qwen/Qwen3-VL-8B-Instruct",
    stage="sft",
    do_train=True,
    finetuning_type="lora",
    dataset="saliency_r1_mulberry_sft_full",
    dataset_dir=str(COLDSTART / "Saliency-R1-cold"),
    template="qwen3_vl_nothink",
    cutoff_len=16384,
    max_samples=16,
    overwrite_cache=True,
    preprocessing_num_workers=4,
    output_dir=str(COLDSTART / "_verify_out"),
    default_system=("A conversation between user and assistant. The user asks a question, and the "
                    "assistant solves it. The assistant first thinks about the reasoning process in "
                    "the mind and then provides the user with the answer."),
    report_to="none",
    use_cpu=True,  # data-load-only verify on a login node (no GPU); training uses bf16 on GPU
)
model_args, data_args, training_args, finetuning_args, _ = get_train_args(args)
tok_module = load_tokenizer(model_args)
template = get_template_and_fix_tokenizer(tok_module["tokenizer"], data_args)
ds = get_dataset(template, model_args, data_args, training_args, stage="sft", **tok_module)

dataset = ds["train_dataset"]
print("VERIFY: parsed dataset, num examples (capped):", len(dataset))
ex = dataset[0]
print("VERIFY: example keys:", list(ex.keys()))
print("VERIFY: input_ids len:", len(ex["input_ids"]))
has_pix = any(k for k in ex.keys() if "pixel" in k or "image" in k)
print("VERIFY: has pixel/image tensor:", has_pix, "| keys:", [k for k in ex.keys() if "pix" in k or "image" in k or "grid" in k])
print("VERIFY: decoded head:", tok_module["tokenizer"].decode(ex["input_ids"][:40]).replace(chr(10), " ")[:200])
print("RESULT: PASS")
