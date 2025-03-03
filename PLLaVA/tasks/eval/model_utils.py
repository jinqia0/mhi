
import torch
import os
from peft import get_peft_model, LoraConfig, TaskType
from safetensors import safe_open
from peft import PeftModel
from tasks.eval.eval_utils import Conversation
from models.pllava import PllavaProcessor, PllavaForConditionalGeneration, PllavaConfig
from accelerate import init_empty_weights, dispatch_model, infer_auto_device_map,load_checkpoint_in_model, load_checkpoint_and_dispatch
from accelerate.utils import get_balanced_memory

from transformers import StoppingCriteria, AutoConfig
class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(
        self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
            return False
        else:
            outputs = self.tokenizer.batch_decode(
                output_ids[:, self.start_len:], skip_special_tokens=True
            )
            flag = True
            for output in outputs:
                for keyword in self.keywords:
                    if keyword not in output:
                        flag = False
                        return False
            return flag


def load_pllava(repo_id, num_frames, use_lora=False, weight_dir=None, lora_alpha=32, 
               use_multi_gpus=False, pooling_shape=(16,12,12), device_map="auto"):
    # ========== 动态显存计算 ==========
    def calculate_video_mem_per_gpu():
        frame_mem = 672 * 672 * 3 * 4  # 单帧内存（float32）
        return (frame_mem * num_frames * 8) / 1e9  # 预估每卡视频处理内存（GB）

    video_mem_per_gpu = calculate_video_mem_per_gpu()
    available_mem = [torch.cuda.get_device_properties(i).total_memory / 1e9 
                    for i in range(torch.cuda.device_count())]
    max_memory = {i: f"{min(80, avail * 0.95 - video_mem_per_gpu):.0f}GB"  # A800预留5%显存
                 for i, avail in enumerate(available_mem)}

    # ========== 模型配置 ==========
    kwargs = {
        'num_frames': num_frames,
        'torch_dtype': torch.bfloat16,
        'device_map': device_map if use_multi_gpus else None
    }
    
    if num_frames == 0:
        kwargs.update(pooling_shape=(0,12,12))
        
    config = AutoConfig.from_pretrained(
        repo_id if not use_lora else weight_dir,
        pooling_shape=pooling_shape,
        **kwargs
    )

    # ========== 空权重初始化+分布式加载 ==========
    with init_empty_weights():
        model = PllavaForConditionalGeneration(config)
    
    model = load_checkpoint_and_dispatch(
        model,
        repo_id,
        device_map=device_map,
        max_memory=max_memory if use_multi_gpus else None,
        no_split_module_classes=["LlamaDecoderLayer"],
        dtype=torch.bfloat16
    )

    # ========== 处理器加载 ==========
    try:
        processor = PllavaProcessor.from_pretrained(repo_id)
    except Exception as e:
        processor = PllavaProcessor.from_pretrained('llava-hf/llava-1.5-7b-hf')

    # ========== 动态LoRA配置 ==========
    if use_lora and weight_dir is not None:
        print("[LoRA] Initializing adapter...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 扩展目标层
            r=128,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            fan_in_fan_out=True,  # 适配模型并行
            modules_to_save=["embed_tokens", "lm_head"]  # 保持关键层完整
        )
        model.language_model = get_peft_model(model.language_model, peft_config)
        print(f"[LoRA] Adapter initialized with alpha={lora_alpha}")

    # ========== 分布式权重加载 ==========
    if weight_dir is not None:
        from accelerate.utils import load_checkpoint_in_model
        print(f"[Loading] Distributed weights from {weight_dir}")
        
        load_checkpoint_in_model(
            model,
            checkpoint_location=weight_dir,
            device_map=device_map,
            offload_state_dict=True,
            offload_buffers=True
        )

    # ========== 混合精度配置 ==========
    model = model.to(torch.bfloat16)
    for param in model.parameters():
        param.requires_grad_(False)  # 冻结基础模型
        
    # ========== 多卡同步检查 ==========
    if use_multi_gpus:
        from accelerate.utils import check_device_map
        check_device_map(model, device_map)
        print(f"[Device Map] Model distributed across devices: {model.hf_device_map}")

    return model, processor


def load_adapters(model, adapter_model_name_or_paths):

    for adapter_model_name_or_path in adapter_model_name_or_paths:
        if not isinstance(model, PeftModel):
            model = PeftModel.from_pretrained(model, adapter_model_name_or_path, adapter_model_name_or_path)
        else:
            model.load_adapter(adapter_model_name_or_path, adapter_model_name_or_path)

    return model


def pllava_answer(conv: Conversation, model, processor, img_list, do_sample=True, max_new_tokens=200, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, stop_criteria_keywords=None, print_res=False):
    # torch.cuda.empty_cache()
    prompt = conv.get_prompt()
    inputs = processor(text=prompt, images=img_list, return_tensors="pt")
    if inputs['pixel_values'] is None:
        inputs.pop('pixel_values')
    inputs = inputs.to(model.device)
    
    # set up stopping criteria
    if stop_criteria_keywords is not None:
        stopping_criteria = [KeywordsStoppingCriteria(stop_criteria_keywords, processor.tokenizer, inputs["input_ids"])]
    else:
        stopping_criteria= None

    with torch.no_grad():
        output_token = model.generate(**inputs, media_type='video',
                                      do_sample=do_sample, max_new_tokens=max_new_tokens, num_beams=num_beams, min_length=min_length, 
                                      top_p=top_p, repetition_penalty=repetition_penalty, length_penalty=length_penalty, temperature=temperature, 
                                      stopping_criteria=stopping_criteria,)
        output_text = processor.batch_decode(output_token, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    if print_res: # debug usage
        print('### PROMPTING LM WITH: ', prompt)
        print('### LM OUTPUT TEXT:  ', output_text)
    if conv.roles[-1] == "<|im_start|>assistant\n":
        split_tag = "<|im_start|> assistant\n"
    else:
        split_tag = conv.roles[-1]
    output_text = output_text.split(split_tag)[-1]
    ending = conv.sep if isinstance(conv.sep, str) else conv.sep[1]
    output_text = output_text.removesuffix(ending).strip()
    conv.messages[-1][1] = output_text
    return output_text, conv

