import os

# —— 在任何导入前：强制 transformers 用 eager 注意力，并禁用 FA2 + 放开 MPS 内存上限 ——
os.environ["TRANSFORMERS_ATTENTION_IMPLEMENTATION"] = "eager"
os.environ["HF_USE_FLASH_ATTENTION_2"] = "0"
# 让 MPS 不再卡在 80% 高水位线，避免 8B + KV cache 直接 OOM
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import gradio as gr
import spaces
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, pipeline

# 再显式设置一次（有些环境下仅靠 env 变量不生效）
try:
    from transformers import set_attn_implementation
    set_attn_implementation("eager")
except Exception:
    pass

import torch
from threading import Thread

# 在加载 marker/surya 之前，给 surya 的 MBART 注意力表做兜底（避免 'sdpa' 键缺失）
try:
    from surya.model.ordering.decoder import MBART_ATTENTION_CLASSES
    if "sdpa" not in MBART_ATTENTION_CLASSES and "eager" in MBART_ATTENTION_CLASSES:
        MBART_ATTENTION_CLASSES["sdpa"] = MBART_ATTENTION_CLASSES["eager"]
except Exception:
    pass

from marker.convert import convert_single_pdf
from marker.output import markdown_exists, save_markdown, get_markdown_filepath
from marker.pdf.utils import find_filetype
from marker.pdf.extract_text import get_length_of_text
from marker.models import load_all_models
from marker.settings import settings
from marker.logger import configure_logging
from surya.settings import settings as surya_settings
import traceback
import re


# --------------------
# marker/surya 基本设置
# --------------------
configure_logging()
MAX_PAGES = 30
MIN_LENGTH = 200
settings.EXTRACT_IMAGES = False
settings.DEBUG = False
settings.PDFTEXT_CPU_WORKERS = 1
settings.DETECTOR_POSTPROCESSING_CPU_WORKERS = 1
settings.OCR_PARALLEL_WORKERS = 1
surya_settings.IN_STREAMLIT = True

# —— 这里会加载 surya 的各个模型，必须在上面完成注意力实现设置 ——
model_refs = load_all_models()
metadata = {}

# --------------------
# 准备 LLM（禁用 accelerate 的 offload，彻底避免 bfloat16→MPS）
# --------------------
model_name = "maxidl/Llama-OpenReviewer-8B"

use_mps = torch.backends.mps.is_available()
device = torch.device("mps" if use_mps else "cpu")
# MPS 下必须用 float16（macOS < 14 不支持 bfloat16）
dtype = torch.float16 if use_mps else torch.float32

# 不传 device_map/offload_state_dict/max_memory，避免 accelerate 注入 hooks
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
)

# 显式把整个模型放到目标设备与 dtype
model.to(device=device, dtype=dtype)

# 彻底排查/转换潜在的 bfloat16 buffer
with torch.no_grad():
    for p in model.parameters():
        if p.dtype == torch.bfloat16:
            p.data = p.data.to(dtype)
    for b in model.buffers():
        if b.dtype == torch.bfloat16:
            b.data = b.data.to(dtype)

# —— 内存关键：禁用 KV cache，降低生成期内存占用 ——
model.config.use_cache = False
if getattr(model, "generation_config", None) is not None:
    model.generation_config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name)

# 确保有 pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# 统一使用“标量 int”ID（避免 pad/eos 为 list 触发的比较错误）
def _scalar_id(x):
    if isinstance(x, (list, tuple)):
        return int(x[0])
    return int(x)

EOS_ID = _scalar_id(tokenizer.eos_token_id)
PAD_ID = _scalar_id(tokenizer.pad_token_id)
BOS_ID = _scalar_id(tokenizer.bos_token_id) if tokenizer.bos_token_id is not None else None

# 同步到 model 配置（标量）
model.config.eos_token_id = EOS_ID
model.config.pad_token_id = PAD_ID
if BOS_ID is not None:
    model.config.bos_token_id = BOS_ID

# 同步到 generation_config（标量）
if getattr(model, "generation_config", None) is not None:
    model.generation_config.eos_token_id = EOS_ID
    model.generation_config.pad_token_id = PAD_ID
    if BOS_ID is not None:
        model.generation_config.bos_token_id = BOS_ID

# 建议左侧 padding（decoder-only + 截断时保留末尾上下文更稳）
tokenizer.padding_side = "left"

streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, decode_kwargs=dict(skip_special_tokens=True))

# --------------------
# 论文检测分类器（可选，失败则降级为规则判断）
# --------------------
try:
    paper_classifier = pipeline(
        "text-classification",
        model="fabriceyhc/bert-base-uncased-arxiv-classification",
        device=0 if torch.cuda.is_available() else -1,
        truncation=True,
        max_length=512
    )
    AI_CLASSIFIER_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not load AI classifier: {e}")
    print("Falling back to rule-based detection only")
    paper_classifier = None
    AI_CLASSIFIER_AVAILABLE = False

def init_zero_shot_classifier():
    try:
        from transformers import pipeline
        classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if torch.cuda.is_available() else -1
        )
        return classifier
    except Exception as e:
        print(f"Could not initialize zero-shot classifier: {e}")
        return None

if not AI_CLASSIFIER_AVAILABLE:
    zero_shot_classifier = init_zero_shot_classifier()
    if zero_shot_classifier:
        AI_CLASSIFIER_AVAILABLE = True
else:
    zero_shot_classifier = None

def ai_check_paper(text):
    """
    用 AI 模型判断是否为科研论文
    返回 (is_paper, confidence, ai_reason)
    """
    if not AI_CLASSIFIER_AVAILABLE:
        return None, 0, "AI classifier not available"
    max_chars = 2000
    if len(text) > max_chars * 2:
        text_sample = text[:max_chars] + "\n...\n" + text[-max_chars:]
    else:
        text_sample = text[:max_chars*2]
    try:
        if zero_shot_classifier and not paper_classifier:
            labels = [
                "academic research paper",
                "scientific article", 
                "technical report",
                "business document",
                "news article",
                "blog post",
                "other document"
            ]
            result = zero_shot_classifier(
                text_sample,
                candidate_labels=labels,
                hypothesis_template="This text is a {}."
            )
            top_label = result['labels'][0]
            top_score = result['scores'][0]
            paper_labels = {"academic research paper", "scientific article", "technical report"}
            if top_label in paper_labels:
                if top_score > 0.7:
                    return True, top_score, f"AI detected: {top_label} (confidence: {top_score:.2f})"
                elif top_score > 0.5:
                    return True, 0.6, f"AI detected: likely {top_label} (confidence: {top_score:.2f})"
                else:
                    return False, top_score, "AI detected: uncertain document type"
            else:
                return False, 1-top_score, f"AI detected: {top_label}, not a research paper"
        elif paper_classifier:
            result = paper_classifier(text_sample)[0]
            label = result['label'].lower()
            score = result['score']
            paper_keywords = ['cs', 'math', 'physics', 'eess', 'econ', 'stat', 'q-bio']
            is_paper = any(keyword in label for keyword in paper_keywords)
            if is_paper:
                return True, score, f"AI detected: {label} paper (confidence: {score:.2f})"
            else:
                return False, 1-score, "AI detected: not a research paper"
    except Exception as e:
        print(f"AI classification error: {e}")
        return None, 0, "AI classification failed"
    return None, 0, "AI check not performed"

# --------------------
# Prompt
# --------------------
SYSTEM_PROMPT_TEMPLATE = """You are an expert reviewer for AI conferences. You follow best practices and review papers according to the reviewer guidelines.

Reviewer guidelines:
1. Read the paper: It's important to carefully read through the entire paper, and to look up any related work and citations that will help you comprehensively evaluate it. Be sure to give yourself sufficient time for this step.
2. While reading, consider the following:
    - Objective of the work: What is the goal of the paper? Is it to better address a known application or problem, draw attention to a new application or problem, or to introduce and/or explain a new theoretical finding? A combination of these? Different objectives will require different considerations as to potential value and impact.
    - Strong points: is the submission clear, technically correct, experimentally rigorous, reproducible, does it present novel findings (e.g. theoretically, algorithmically, etc.)?
    - Weak points: is it weak in any of the aspects listed in b.?
    - Be mindful of potential biases and try to be open-minded about the value and interest a paper can hold for the community, even if it may not be very interesting for you.
3. Answer four key questions for yourself, to make a recommendation to Accept or Reject:
    - What is the specific question and/or problem tackled by the paper?
    - Is the approach well motivated, including being well-placed in the literature?
    - Does the paper support the claims? This includes determining if results, whether theoretical or empirical, are correct and if they are scientifically rigorous.
    - What is the significance of the work? Does it contribute new knowledge and sufficient value to the community? Note, this does not necessarily require state-of-the-art results. Submissions bring value to the community when they convincingly demonstrate new, relevant, impactful knowledge (incl., empirical, theoretical, for practitioners, etc).
4. Write your review including the following information: 
    - Summarize what the paper claims to contribute. Be positive and constructive.
    - List strong and weak points of the paper. Be as comprehensive as possible.
    - Clearly state your initial recommendation (accept or reject) with one or two key reasons for this choice.
    - Provide supporting arguments for your recommendation.
    - Ask questions you would like answered by the authors to help you clarify your understanding of the paper and provide the additional evidence you need to be confident in your assessment.
    - Provide additional feedback with the aim to improve the paper. Make it clear that these points are here to help, and not necessarily part of your decision assessment.

Your write reviews in markdown format. Your reviews contain the following sections:

# Review

{review_fields}

Your response must only contain the review in markdown format with sections as defined above.
"""

USER_PROMPT_TEMPLATE = """Review the following paper:

{paper_text}
"""

REVIEW_FIELDS = """## Summary
Briefly summarize the paper and its contributions. This is not the place to critique the paper; the authors should generally agree with a well-written summary.

## Novelty
Please assign the paper a numerical rating on the following scale to indicate the novelty and originality of the work. Consider whether the paper presents new ideas, methods, or perspectives that have not been explored before. Choose from the following:
4: excellent - Highly original work with groundbreaking ideas or completely novel approaches
3: good - Significant new contributions with clear advances over existing work
2: fair - Some new elements but largely incremental improvements or combinations of existing ideas
1: poor - Little to no novelty, mostly reproducing existing work or trivial variations

## Novelty Explanation
IMPORTANT: Focus ONLY on novelty aspects. DO NOT discuss soundness, presentation, or general contribution here.
Please provide specific justification for your novelty score by addressing:
- What specific new concepts, methods, or approaches does this paper introduce?
- How do these differ from existing work in the field? Cite specific prior work for comparison.
- Are the differences substantial or incremental?
- Is this addressing a problem in a genuinely new way, or applying known methods to a new domain?
DO NOT repeat content from other sections. DO NOT discuss writing quality, experimental rigor, or implementation details here.

## Soundness
Please assign the paper a numerical rating on the following scale to indicate the soundness of the technical claims, experimental and research methodology and on whether the central claims of the paper are adequately supported with evidence. Choose from the following:
4: excellent
3: good
2: fair
1: poor

## Soundness Explanation
IMPORTANT: Focus ONLY on technical correctness and methodological rigor. DO NOT discuss novelty or writing quality here.
Please provide specific reasons for your soundness score by addressing:
- Are the technical claims mathematically/logically correct?
- Is the experimental methodology rigorous and appropriate?
- Are the experiments sufficient to support the claims?
- Are there any methodological flaws or missing controls?
- Is the statistical analysis (if any) appropriate and correctly executed?
DO NOT repeat content from other sections. DO NOT discuss the novelty of the approach or presentation quality here.

## Presentation
Please assign the paper a numerical rating on the following scale to indicate the quality of the presentation. This should take into account the writing style and clarity, as well as contextualization relative to prior work. Choose from the following:
4: excellent
3: good
2: fair
1: poor

## Presentation Explanation
IMPORTANT: Focus ONLY on writing quality, clarity, and organization. DO NOT discuss technical merit or novelty here.
Please explain your presentation score by addressing:
- Is the paper well-organized and easy to follow?
- Are the main ideas clearly explained?
- Are figures, tables, and visualizations effective and well-designed?
- Is the related work section comprehensive and fair?
- Are mathematical notations consistent and clear?
- Is the language precise and grammatically correct?
DO NOT repeat content from other sections. DO NOT discuss the novelty of ideas or soundness of methods here.

## Contribution
Please assign the paper a numerical rating on the following scale to indicate the quality of the overall contribution this paper makes to the research area being studied. Are the questions being asked important? Does the paper bring a significant originality of ideas and/or execution? Are the results valuable to share with the broader ICLR community? Choose from the following:
4: excellent
3: good
2: fair
1: poor

## Contribution Explanation
IMPORTANT: Focus on the OVERALL IMPACT and SIGNIFICANCE to the field. This is different from novelty.
Please justify your contribution score by explaining:
- Why is this work important for the field?
- What practical or theoretical impact could this have?
- Who would benefit from this work and how?
- Does this open new research directions or close important gaps?
- How significant are the improvements over baselines (if applicable)?
Consider both immediate utility and long-term impact. DO NOT simply repeat the novelty assessment here.

## Strengths
List the main strengths of the paper. Be specific and provide evidence. Each strength should be a separate bullet point. Focus on what the paper does well across all dimensions (novelty, soundness, presentation, contribution). Avoid generic statements.

## Weaknesses
List the main weaknesses of the paper. Be specific, constructive, and actionable. Each weakness should be a separate bullet point with suggestions for improvement where possible. Focus on significant issues that affect the paper's validity or impact.

## Questions
List specific questions for the authors that could clarify ambiguities or address concerns. Number each question. These should be questions where the answer could potentially change your assessment of the paper.

## Flag For Ethics Review
If there are ethical issues with this paper, please flag the paper for an ethics review and select area of expertise that would be most useful for the ethics reviewer to have. Please select all that apply. Choose from the following:
No ethics review needed.
Yes, Discrimination / bias / fairness concerns
Yes, Privacy, security and safety
Yes, Legal compliance (e.g., GDPR, copyright, terms of use)
Yes, Potentially harmful insights, methodologies and applications
Yes, Responsible research practice (e.g., human subjects, data release)
Yes, Research integrity issues (e.g., plagiarism, dual submission)
Yes, Unprofessional behaviors (e.g., unprofessional exchange between authors and reviewers)
Yes, Other reasons (please specify below)

## Details Of Ethics Concerns
Please provide details of your concerns. If no ethics review is needed, write "N/A".

## Rating
Please provide an "overall score" for this submission. Choose from the following:
1: strong reject
3: reject, not good enough
5: marginally below the acceptance threshold
6: marginally above the acceptance threshold
8: accept, good paper
10: strong accept, should be highlighted at the conference

## Overall Justification
Provide a comprehensive justification for your overall rating that:
- Synthesizes the assessments from all dimensions (novelty, soundness, presentation, contribution)
- Explains how you weighted different aspects in arriving at your final score
- Clearly states whether the strengths outweigh the weaknesses or vice versa
- Indicates what would need to change for a different rating
This should be a holistic assessment, not a repetition of individual sections.

"""

# --------------------
# 论文判定
# --------------------
def is_research_paper(text, use_ai=True):
    if not text or len(text) < MIN_LENGTH:
        return False, 0, "Text is too short to be a research paper"
    text_lower = text.lower()
    indicators = {
        'abstract': bool(re.search(r'\babstract\b', text_lower)),
        'introduction': bool(re.search(r'\bintroduction\b', text_lower)),
        'conclusion': bool(re.search(r'\bconclusion\b', text_lower)),
        'references': bool(re.search(r'\b(references|bibliography)\b', text_lower)),
        'methodology': bool(re.search(r'\b(method|methodology|approach|algorithm|model)\b', text_lower)),
        'results': bool(re.search(r'\b(results|experiments|evaluation|analysis)\b', text_lower)),
        'citations': bool(re.search(r'\[[\d,\s]+\]|\(\w+,?\s*\d{4}\)', text)),
        'figures_tables': bool(re.search(r'\b(figure\s*\d+|table\s*\d+|fig\.\s*\d+)\b', text_lower)),
        'academic_terms': bool(re.search(r'\b(propose|present|demonstrate|evaluate|contribution|novel|state-of-the-art)\b', text_lower))
    }
    indicator_count = sum(indicators.values())
    non_paper_indicators = []
    if re.search(r'\b(invoice|receipt)\b', text_lower) and re.search(r'\b(total|amount|payment|billing)\b', text_lower):
        non_paper_indicators.append(True)
    if re.search(r'\bpurchase order\b', text_lower):
        non_paper_indicators.append(True)
    if re.search(r'\b(dear\s+(sir|madam|customer)|sincerely|best regards|yours truly)\b', text_lower):
        non_paper_indicators.append(True)
    if re.search(r'\b(chapter\s+\d+|lesson\s+\d+|exercise\s+\d+)\b', text_lower) and indicator_count < 3:
        non_paper_indicators.append(True)
    if re.search(r'<html|<body|<div|<script|<!DOCTYPE', text_lower):
        non_paper_indicators.append(True)
    if re.search(r'\b(ingredients|recipe|preparation|cooking time|servings)\b', text_lower) and not re.search(r'\b(algorithm|method|experiment)\b', text_lower):
        non_paper_indicators.append(True)
    if any(non_paper_indicators) and indicator_count < 6:
        return False, 0, "Content appears to be a non-academic document"

    ai_result = None
    ai_confidence = 0
    ai_reason = ""
    if use_ai and AI_CLASSIFIER_AVAILABLE:
        ai_result, ai_confidence, ai_reason = ai_check_paper(text)

    if indicator_count == 9:
        rule_decision = True
        rule_confidence = 0.9
        rule_reason = f"Found all {indicator_count}/9 academic paper indicators"
    elif indicator_count >= 6:
        rule_decision = True
        rule_confidence = 0.6
        missing = [k for k, v in indicators.items() if not v]
        rule_reason = f"Found only {indicator_count}/9 indicators. Missing: {', '.join(missing)}"
    else:
        rule_decision = False
        rule_confidence = 0
        missing = [k for k, v in indicators.items() if not v]
        rule_reason = f"Found only {indicator_count}/9 indicators. Missing: {', '.join(missing[:4])}"

    if ai_result is not None:
        combined_confidence = (rule_confidence * 0.6) + (ai_confidence * 0.4)
        if rule_decision and ai_result:
            if combined_confidence >= 0.9:
                return True, 0.9, f"High confidence: {rule_reason}. {ai_reason}"
            else:
                return True, 0.6, f"Warning: {rule_reason}. {ai_reason}"
        elif not rule_decision and not ai_result:
            return False, 0, f"Not a research paper. {rule_reason}. {ai_reason}"
        else:
            if combined_confidence >= 0.5:
                return True, 0.6, f"Mixed signals: {rule_reason}. {ai_reason}"
            else:
                return False, 0, f"Likely not a research paper. {rule_reason}. {ai_reason}"
    else:
        if rule_decision:
            if rule_confidence >= 0.9:
                return True, 0.9, f"High confidence: {rule_reason}"
            else:
                return True, 0.6, f"Warning: {rule_reason}"
        else:
            return False, 0, f"Does not appear to be a research paper. {rule_reason}"

def create_messages(review_fields, paper_text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(review_fields=review_fields)},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(paper_text=paper_text)},
    ]
    return messages

@spaces.GPU()
def convert_file(filepath):
    full_text, images, out_metadata = convert_single_pdf(
            filepath, model_refs, metadata=metadata, max_pages=MAX_PAGES
    )
    return full_text

def process_file(file):
    print(file.name)
    filepath = file.name
    try:
        if MIN_LENGTH:
            filetype = find_filetype(filepath)
            if filetype == "other":
                raise ValueError()

            length = get_length_of_text(filepath)
            if length < MIN_LENGTH:
                raise ValueError()
        paper_text = convert_file(filepath)
        paper_text = paper_text.strip()
        if not len(paper_text) > MIN_LENGTH:
            raise ValueError()
    except spaces.zero.gradio.HTMLError as e:
        print(e)
        return "Error. GPU quota exceeded. Please return later.", False
    except Exception as e:
        print(traceback.format_exc())
        print(f"Error converting {filepath}: {e}")
        return "Error processing pdf", False
    
    is_paper, confidence, reason = is_research_paper(paper_text, use_ai=True)
    if not is_paper:
        return f"⚠️ **Not a Research Paper**\n\nThe uploaded document does not appear to be a research paper.\n\nReason: {reason}\n\nPlease upload a proper academic/research paper with sections like Abstract, Introduction, Methodology, Results, and References.", False
    
    if confidence < 0.9:
        paper_text = f"⚠️ **Warning**: {reason}. \n\nThe document may be incomplete or missing key sections. Proceeding with review generation...\n\n---\n\n{paper_text}"
    
    return paper_text, True

# —— 限制最大输入 tokens，避免长论文导致显存爆 ——
MAX_INPUT_TOKENS = 2048

@spaces.GPU(duration=190)
def generate(paper_text, review_template):
    # Quick sanity check
    is_paper, confidence, reason = is_research_paper(paper_text, use_ai=False)
    if not is_paper:
        return f"⚠️ Cannot generate review: {reason}"
    
    messages = create_messages(review_template, paper_text)
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors='pt'
    )

    # 截断到 MAX_INPUT_TOKENS（保留最后一段上下文更有用）
    if input_ids.shape[1] > MAX_INPUT_TOKENS:
        input_ids = input_ids[:, -MAX_INPUT_TOKENS:]

    input_ids = input_ids.to(device)

    generation_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=512,     # ↓ 内存关键：降低新生成 token 上限
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        pad_token_id=PAD_ID,
        eos_token_id=EOS_ID,
        use_cache=False,        # ↓ 内存关键：关闭 KV cache
    )

    # 显式 attention_mask
    attention_mask = torch.ones_like(input_ids)
    generation_kwargs["attention_mask"] = attention_mask

    print(f"input_ids shape: {input_ids.shape}")
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        yield generated_text.replace("<|eot_id|>", "")

# --------------------
# UI
# --------------------
title = """<h1 align="center">OpenReviewer</h1>
<div align="center">Using <a href="https://huggingface.co/maxidl/Llama-OpenReviewer-8B" target="_blank"><code>Llama-OpenReviewer-8B</code></a> - Built with Llama</div>
"""

description = """This is an online demo featuring [Llama-OpenReviewer-8B](https://huggingface.co/maxidl/Llama-OpenReviewer-8B), a large language model that generates high-quality reviews for machine learning and AI papers.

## Demo Guidelines

1. Upload your paper as a PDF file. Alternatively you can paste the full text of your paper in markdown format below. We do **not** store your data. User data is kept in ephemeral storage during processing.

2. Once you upload a PDF, it will be converted to markdown and **validated to ensure it's a research paper**. This takes some time as it runs multiple transformer models to parse the layout and extract text and tables. Check out [marker](https://github.com/VikParuchuri/marker/tree/master) for details.

3. Having obtained a markdown version of your paper and confirmed it's a valid research paper, you can now click *Generate Review*.

Take a look at the Review Template to properly interpret the generated review. You can also change the review template before generating in case you want to generate a review with a different schema and aspects.

To obtain more than one review, just generate again.

**GPU quota:** If exceeded, either sign in with your HF account or come back later. Your quota has a half-life of 2 hours.

"""

theme = gr.themes.Default(primary_hue="gray", secondary_hue="blue", neutral_hue="slate")
with gr.Blocks(theme=theme) as demo:
    title = gr.HTML(title)
    description = gr.Markdown(description)
    
    with gr.Row():
        file_input = gr.File(file_types=[".pdf"], file_count="single")
        validation_status = gr.Markdown("", visible=False)
    
    paper_text_field = gr.Textbox("Upload a pdf or paste the full text of your paper in markdown format here.", label="Paper Text", lines=20, max_lines=20, autoscroll=False)
    
    with gr.Accordion("Review Template", open=False):
        review_template_description = gr.Markdown("We use the ICLR 2025 review template by default, but you can modify the template below as you like.")
        review_template_field = gr.Textbox(label=" ",lines=20, max_lines=20, autoscroll=False, value=REVIEW_FIELDS)
    
    generate_button = gr.Button("Generate Review", interactive=False)
    
    def handle_file_upload(file):
        if file is None:
            return "", gr.update(visible=False), gr.update(interactive=False)
        text, is_valid = process_file(file)
        if is_valid:
            is_paper, confidence, reason = is_research_paper(text, use_ai=False)
            if confidence >= 0.9:
                status_msg = "✅ **Document validated**: This appears to be a complete research paper."
            else:
                status_msg = f"⚠️ **Warning**: {reason}\n\nThe document may be incomplete or missing key sections of a research paper."
            return text, gr.update(value=status_msg, visible=True), gr.update(interactive=True)
        else:
            return text, gr.update(value="❌ **Validation failed**: Please upload a research paper.", visible=True), gr.update(interactive=False)
    
    def handle_text_change(text):
        if not text or len(text) < 200:
            return gr.update(interactive=False), gr.update(visible=False)
        
        is_paper, confidence, reason = is_research_paper(text, use_ai=True)
        if is_paper:
            if confidence >= 0.9:
                status = "✅ **Text validated**: This appears to be a complete research paper."
            else:
                status = f"⚠️ **Warning**: {reason}\n\nThe document may be incomplete or missing key sections."
            return gr.update(interactive=True), gr.update(value=status, visible=True)
        else:
            return gr.update(interactive=False), gr.update(value=f"❌ **Not a research paper**: {reason}", visible=True)
    
    file_input.upload(handle_file_upload, file_input, [paper_text_field, validation_status, generate_button])
    paper_text_field.change(handle_text_change, paper_text_field, [generate_button, validation_status])
    
    review_field = gr.Markdown("\n\n\n\n\n", label="Review")
    generate_button.click(
        fn=lambda: gr.update(interactive=False), 
        inputs=None, 
        outputs=generate_button
    ).then(
        generate, 
        [paper_text_field, review_template_field], 
        review_field
    ).then(
        fn=lambda: gr.update(interactive=True), 
        inputs=None, 
        outputs=generate_button
    )
    
    demo.title = "OpenReviewer"

if __name__ == "__main__":
    demo.launch()
