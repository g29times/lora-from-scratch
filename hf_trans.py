# HF
# https://zhuanlan.zhihu.com/p/576691638

# Train
# PEFT Megatron DeepSpeed https://zhuanlan.zhihu.com/p/619426866
# https://github.com/liguodongiot/llm-action
# https://huggingface.co/blog/zh/megatron-training
# https://github.com/microsoft/Megatron-DeepSpeed
# OpenMMLab FSDP ColossalAI DeepSpeed https://zhuanlan.zhihu.com/p/645564540
# https://github.com/facebookresearch/llama-recipes/

# Use
# https://huggingface.co/docs/transformers/pipeline_tutorial
# https://huggingface.co/docs/transformers/tasks/language_modeling#inference
# https://boinc-ai.gitbook.io/transformers/api/main-classes/auto-classes/natural-language-processing/automodelforcausallm

prompt = "public void hello_world() {"
my_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# case 1 pipeline
# from transformers import pipeline

# pipe = pipeline("text-generation", model=my_model)
# outputs = pipe(prompt)
# print(outputs[0]["generated_text"])

# case 2 AutoModelForCausalLM
# Tokenize the text and return the input_ids as PyTorch tensors:
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(my_model)
inputs = tokenizer(prompt, return_tensors="pt").input_ids

# Use the generate() method to generate text.
from transformers import AutoModelForCausalLM
# pip install accelerate
model = AutoModelForCausalLM.from_pretrained(my_model, device_map="auto")
# use gpu
inputs = inputs.to('cuda')
outputs = model.generate(inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
# Decode the generated token ids back into text:
text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(text)