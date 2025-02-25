# A system default prompt (Pang et al., 2024)
SYSTEM_PROMPT_IRPO = """Your task is to answer the question below. Give step by step reasoning before you answer, and when you’re ready to answer, please use the format 'The answer is'"""

# A system be concise prompt (Renze and Guven, 2024)
SYSTEM_PROMPT_CONCISE = """Your task is to answer the question below. Give step by step reasoning before you answer, and when you’re ready to answer, please use the format 'The answer is'. Be concise."""

# A system prompt for budget estimation (Han et al., 2024)
SYSTEM_PROMPT_BUDGET_ESTIMATION = """Task: Analyze the given question and estimate the minimum number of tokens required to generate a complete and accurate response. Please give the response by strictly following this format: [[budget]], for example, Budget: [[12]]."""

# A system prompt for fixed budget (Nayab et al., 2024)
SYSTEM_PROMPT_FIXED_BUDGET = """Let’s think a bit step by step and limit the answer length to 100 words."""

# A system Hand Crafted 1 (ours)
SYSTEM_PROMPT_SUMMARY = """Your task is to answer the question below. Give step by step reasoning before you answer, and when you’re ready to answer, please use the format 'The answer is'. You don’t need unnecessary explanations; you can solve problems using only essential words and expressions. Summarize your thought process as simply as possible and provide your answer. Do not generate only the final answer."""

# A system Hand Crafted 2 (ours)
SYSTEM_PROMPT_COMPACT = """Your task is to answer the question below. Give step by step reasoning before you answer, and when you’re ready to answer, please use the format 'The answer is'. Use less words and more compact expressions to be concise."""

# A system Hand Crafted 3 (ours)
SYSTEM_PROMPT_SHORT = """Give concise step by step reasoning before you answer. Only retain key steps such as names, objects, numbers and mathematical operations. Use short plain words. Don't use any formatting such as emphasis, lists, or enumeration. Make sure that the intermediate results are presented in the order that they are calculated. After your reasoning, please use the format 'The answer is' to answer."""

# A system Hand Crafted 4 (ours)
SYSTEM_PROMPT_SHORT2 = """Your task is to answer the question below. Carefully solve the problem step by step, while using *as few words are possible*. Be careful about your choice of words. Use only necessary and essential steps. Avoid extra words. Avoid repetition. Avoid verbose statements. Avoid introductory remarks. When you’re ready to answer, please use the format 'The answer is'."""

# A zero-shot user prompt
ZERO_SHOT_PROMPT = """Question: {question}
Solution:"""

# A zero-shot user prompt for budget estimation
ZERO_SHOT_BUDGET_ESTIMATION_PROMPT = """Question: {question}"""

# LLaMa chat template
LLAMA_CHAT_TEMPLATE = """{{- bos_token }}

{#- This block extracts the system message #}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content']|trim %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = "" %}
{%- endif %}

{#- System message #}
{{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
{{- system_message }}
{{- "<|eot_id|>" }}

{%- for message in messages %}
    {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}
{%- endfor %}

{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}"""
