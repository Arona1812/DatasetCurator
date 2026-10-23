"""Offline batch inference, one GPU model load per batch. No downloads or API fallback."""
import argparse
import json
import os
import time
from pathlib import Path

from initial_captions import image_sha256, store_caption


def run(job_path):
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    os.environ["HF_HOME"] = str(Path(job["cache_dir"]) / "runtime_cache")
    os.environ["HF_MODULES_CACHE"] = str(Path(job["cache_dir"]) / "runtime_cache" / "modules")
    import torch
    from PIL import Image, ImageOps
    from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig, LlavaForConditionalGeneration

    config = job["settings"]
    engine = config["engine"]
    if engine not in {"florence2", "joycaption"}:
        raise ValueError("Local worker only supports Florence and JoyCaption")
    if not torch.cuda.is_available():
        raise RuntimeError("Local caption provider requires CUDA; choose Luna or configure a CUDA runtime.")
    torch.manual_seed(config["seed"])
    snapshot = job["weights"]["snapshot"]
    processor = AutoProcessor.from_pretrained(snapshot, trust_remote_code=engine == "florence2", local_files_only=True, use_fast=False)
    if engine == "florence2":
        model = AutoModelForCausalLM.from_pretrained(snapshot, trust_remote_code=True,
            torch_dtype=torch.float16, attn_implementation="eager", local_files_only=True).to("cuda")
    else:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["vision_tower", "multi_modal_projector"])
        model = LlavaForConditionalGeneration.from_pretrained(snapshot, torch_dtype=torch.bfloat16,
            quantization_config=quant, device_map={"": "cuda:0"}, attn_implementation="sdpa", local_files_only=True)
    model.eval()
    for index, case in enumerate(job["cases"], 1):
        if image_sha256(case["path"]) != case["image_sha256"]:
            raise RuntimeError("Image changed during initial captioning")
        started = time.perf_counter()
        with Image.open(case["path"]) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        if engine == "florence2":
            inputs = processor(text=config["prompt"], images=image, return_tensors="pt").to("cuda", torch.float16)
        else:
            messages = [{"role": "system", "content": "You are a helpful image captioner."},
                        {"role": "user", "content": config["prompt"]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=config["max_new_tokens"], do_sample=False,
                                    num_beams=3 if engine == "florence2" else 1, use_cache=True)
        tokens = output[0] if engine == "florence2" else output[0][inputs["input_ids"].shape[-1]:]
        raw = processor.decode(tokens, skip_special_tokens=engine != "florence2", clean_up_tokenization_spaces=False)
        caption = processor.post_process_generation(raw, task=config["prompt"], image_size=image.size)[config["prompt"]] if engine == "florence2" else raw.strip()
        store_caption(job["cache_dir"], case["image_sha256"], config,
                      {"status": "ok", "caption": caption, "raw_output": raw,
                       "hit_token_limit": len(tokens) >= config["max_new_tokens"],
                       "output_tokens": len(tokens), "seconds": round(time.perf_counter() - started, 3)})
        print(f"Initial caption ready: {index}/{len(job['cases'])} | {Path(case['path']).name}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("job")
    run(parser.parse_args().job)
