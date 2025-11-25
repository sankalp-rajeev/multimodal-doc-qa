"""
Professional Demo UI for Multimodal Document QA
Streamlit version - Shows predictions from BERT, LayoutLMv3, LayoutLMv3+Vision, and Multi-Task
"""

import streamlit as st
import torch
from pathlib import Path
import sys
from PIL import Image
import io
from openai import OpenAI
from dotenv import load_dotenv
import os
from about import show_about_page

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from transformers import (
    BertForQuestionAnswering,
    BertTokenizerFast,
    LayoutLMv3ForQuestionAnswering,
    LayoutLMv3TokenizerFast,
    LayoutLMv3ImageProcessor,
    BertConfig,
    LayoutLMv3Config,
)
from src.models.multitask_model import LayoutLMv3ForMultiTask
import pytesseract
from pytesseract import Output
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

load_dotenv()

def reformulate_question_with_ai(natural_question: str, ocr_text: str) -> str:
    """
    Convert natural language question to FUNSD-style field label
    Uses OpenAI GPT to understand intent and extract form field
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.warning("⚠️ No OpenAI API key found. Using original question.")
        return natural_question
    
    try:
        client = OpenAI(api_key=api_key)
        
        prompt = f"""You are helping convert natural language questions into form field labels for document question answering.

The document contains this text:
{ocr_text[:800]}

The user asked: "{natural_question}"

Your task: Convert this natural language question to a SHORT form field label (1-5 words) that would appear on the form.

Examples:
- "What is the date?" → "Date"
- "Who is the company?" → "Name of Corporation"
- "What's the address?" → "Address"
- "What is the purpose?" → "Purpose"
- "Who is the treasurer?" → "Name of Treasurer"

Return ONLY the field label, nothing else. Be concise."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that converts questions to form field labels."},
                {"role": "user", "content": prompt}
            ]
        )
        
        reformulated = response.choices[0].message.content.strip()
        reformulated = reformulated.strip('"\'')  # Remove quotes if present
        
        # Show what happened
        st.info(f"🤖 AI converted: '{natural_question}' → '{reformulated}'")
        
        return reformulated
        
    except Exception as e:
        st.warning(f"⚠️ AI reformulation failed: {str(e)}")
        st.caption("Using original question instead")
        return natural_question
    
    
# BIO Label mapping
LABEL2ID = {
    "O": 0,
    "B-ANSWER": 1,
    "I-ANSWER": 2,
    "B-QUESTION": 3,
    "I-QUESTION": 4,
    "B-HEADER": 5,
    "I-HEADER": 6,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


# Page config
st.set_page_config(
    page_title="Multimodal Document QA",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        color: #2c3e50;
    }
    .sub-header {
        text-align: center;
        color: #7f8c8d;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    .model-card {
        padding: 1.5rem;
        border-radius: 12px;
        border: 3px solid #ddd;
        margin-bottom: 1rem;
        background-color: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .model-1 {
        border-color: #CD7F32;
    }
    .model-2 {
        border-color: #C0C0C0;
    }
    .model-3 {
        border-color: #FFD700;
    }
    .model-4 {
        border-color: #9370DB;
    }
    .model-title {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 0.3rem;
        color: #2c3e50;
    }
    .model-caption {
        font-size: 0.9rem;
        color: #7f8c8d;
        margin-bottom: 1rem;
    }
    .answer-box {
        font-size: 1.3rem;
        font-weight: bold;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 8px;
        background-color: #f8f9fa;
        border-left: 4px solid #3498db;
        color: #2c3e50;
    }
    .confidence {
        font-size: 1rem;
        color: #27ae60;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    .bio-tags {
            font-size: 0.95rem;
            margin-top: 1rem;
            padding: 1.2rem;
            background-color: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #9370DB;
            color: #2c3e50;
            line-height: 2.2;  /* Increased from 1.8 */
        }
        .bio-tags b {
            color: #5a5a5a;
            font-weight: 700;
        }
    .bio-label {
        font-weight: 600;
        color: #7f8c8d;
    }
</style>
""", unsafe_allow_html=True)


EXAMPLES = [
    {
        "image_path": "demo/examples/91814768_91814769.png",
        "question": "2. Name of Corporation",  # Exact form field label
        "name": "Example 1: Corporate Form"
    },
    {
        "image_path": "demo/examples/91814768_91814769.png",
        "question": "Date",  # Simple field label
        "name": "Example 2: Date Field"
    },
    {
        "image_path": "demo/examples/91814768_91814769.png",
        "question": "Purpose",  # Simple field label
        "name": "Example 3: Purpose Field"
    },
    {
        "image_path": "demo/examples/87528380.png",
        "question": "to",  # From "PROPOSAL to Lorillard Corporation"
        "name": "Example 4: Recipient"
    },
]

@st.cache_resource
def load_models():
    """Load all models once and cache them"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    models = {}
    
    with st.spinner("🔄 Loading models... This may take a minute..."):
        # Load BERT
        bert_path = project_root / "models/bert_baseline/best_model"
        bert_config = BertConfig.from_json_file(str(bert_path / "config.json"))
        models['bert_model'] = BertForQuestionAnswering.from_pretrained(
            str(bert_path),
            config=bert_config,
            local_files_only=True,
            trust_remote_code=False
        ).to(device).eval()
        models['bert_tokenizer'] = BertTokenizerFast.from_pretrained(
            str(bert_path),
            local_files_only=True
        )
        
        # Load LayoutLMv3
        layout_path = project_root / "models/layoutlmv3/best_model"
        layout_config = LayoutLMv3Config.from_json_file(str(layout_path / "config.json"))
        models['layout_model'] = LayoutLMv3ForQuestionAnswering.from_pretrained(
            str(layout_path),
            config=layout_config,
            local_files_only=True,
            trust_remote_code=False
        ).to(device).eval()
        models['layout_tokenizer'] = LayoutLMv3TokenizerFast.from_pretrained(
            "microsoft/layoutlmv3-base"
        )
        
        # Load LayoutLMv3 + Vision
        vision_path = project_root / "models/layoutlmv3_vision/best_model"
        vision_config = LayoutLMv3Config.from_json_file(str(vision_path / "config.json"))
        models['vision_model'] = LayoutLMv3ForQuestionAnswering.from_pretrained(
            str(vision_path),
            config=vision_config,
            local_files_only=True,
            trust_remote_code=False
        ).to(device).eval()
        models['image_processor'] = LayoutLMv3ImageProcessor.from_pretrained(
            "microsoft/layoutlmv3-base"
        )
        
        # Load Multi-Task Model
        multitask_path = project_root / "models/multitask/best_model"
        multitask_config = LayoutLMv3Config.from_json_file(str(multitask_path / "config.json"))
        models['multitask_model'] = LayoutLMv3ForMultiTask.from_pretrained(
            str(multitask_path),
            config=multitask_config,
            local_files_only=True,
            trust_remote_code=False
        ).to(device).eval()
        
        models['device'] = device
    
    return models


def extract_text_and_boxes(image):
    """Extract text and bounding boxes using OCR"""
    ocr_data = pytesseract.image_to_data(image, output_type=Output.DICT)
    
    words = []
    boxes = []
    img_width, img_height = image.size
    
    for i in range(len(ocr_data['text'])):
        text = ocr_data['text'][i].strip()
        if text:
            x = ocr_data['left'][i]
            y = ocr_data['top'][i]
            w = ocr_data['width'][i]
            h = ocr_data['height'][i]
            
            # Normalize to [0, 1000]
            x0 = int((x / img_width) * 1000)
            y0 = int((y / img_height) * 1000)
            x1 = int(((x + w) / img_width) * 1000)
            y1 = int(((y + h) / img_height) * 1000)
            
            words.append(text)
            boxes.append([x0, y0, x1, y1])
    
    context = " ".join(words)
    return context, words, boxes


def predict_bert(models, question, context):
    """Model 1: BERT prediction"""
    inputs = models['bert_tokenizer'](
        question,
        context,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    inputs = {k: v.to(models['device']) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = models['bert_model'](**inputs)
    
    start_idx = outputs.start_logits.argmax().item()
    end_idx = outputs.end_logits.argmax().item()
    
    if end_idx < start_idx:
        end_idx = start_idx
    
    start_probs = torch.softmax(outputs.start_logits, dim=1)
    end_probs = torch.softmax(outputs.end_logits, dim=1)
    confidence = (start_probs[0, start_idx] * end_probs[0, end_idx]).item()
    
    answer_tokens = inputs['input_ids'][0][start_idx:end_idx+1]
    answer = models['bert_tokenizer'].decode(answer_tokens, skip_special_tokens=True).strip()
    
    return answer if answer else "[NO ANSWER]", confidence


def predict_layoutlmv3(models, question, words, boxes):
    """Model 2: LayoutLMv3 prediction"""
    q_words = question.strip().split()
    q_boxes = [[0, 0, 0, 0]] * len(q_words)
    
    all_words = q_words + words
    all_boxes = q_boxes + boxes
    
    encoded = models['layout_tokenizer'](
        all_words,
        boxes=all_boxes,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    encoded = {k: v.to(models['device']) for k, v in encoded.items()}
    
    with torch.no_grad():
        outputs = models['layout_model'](**encoded)
    
    start_idx = outputs.start_logits.argmax().item()
    end_idx = outputs.end_logits.argmax().item()
    
    if end_idx < start_idx:
        end_idx = start_idx
    
    start_probs = torch.softmax(outputs.start_logits, dim=1)
    end_probs = torch.softmax(outputs.end_logits, dim=1)
    confidence = (start_probs[0, start_idx] * end_probs[0, end_idx]).item()
    
    answer_tokens = encoded['input_ids'][0][start_idx:end_idx+1]
    answer = models['layout_tokenizer'].decode(answer_tokens, skip_special_tokens=True).strip()
    
    return answer if answer else "[NO ANSWER]", confidence


def predict_layoutlmv3_vision(models, question, words, boxes, image):
    """Model 3: LayoutLMv3 + Vision prediction"""
    q_words = question.strip().split()
    q_boxes = [[0, 0, 0, 0]] * len(q_words)
    
    all_words = q_words + words
    all_boxes = q_boxes + boxes
    
    encoded = models['layout_tokenizer'](
        all_words,
        boxes=all_boxes,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    pixel_values = models['image_processor'](
        image,
        return_tensors="pt",
        apply_ocr=False
    ).pixel_values
    
    encoded['pixel_values'] = pixel_values
    encoded = {k: v.to(models['device']) for k, v in encoded.items()}
    
    with torch.no_grad():
        outputs = models['vision_model'](**encoded)
    
    start_idx = outputs.start_logits.argmax().item()
    end_idx = outputs.end_logits.argmax().item()
    
    if end_idx < start_idx:
        end_idx = start_idx
    
    start_probs = torch.softmax(outputs.start_logits, dim=1)
    end_probs = torch.softmax(outputs.end_logits, dim=1)
    confidence = (start_probs[0, start_idx] * end_probs[0, end_idx]).item()
    
    answer_tokens = encoded['input_ids'][0][start_idx:end_idx+1]
    answer = models['layout_tokenizer'].decode(answer_tokens, skip_special_tokens=True).strip()
    
    return answer if answer else "[NO ANSWER]", confidence

def predict_multitask(models, question, words, boxes, image):
    """Model 4: Multi-Task prediction (QA + BIO tagging)"""
    q_words = question.strip().split()
    q_boxes = [[0, 0, 0, 0]] * len(q_words)
    
    all_words = q_words + words
    all_boxes = q_boxes + boxes
    
    encoding = models['layout_tokenizer'](
        all_words,
        boxes=all_boxes,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    pixel_values = models['image_processor'](
        image,
        return_tensors="pt",
        apply_ocr=False
    ).pixel_values
    
    input_ids = encoding['input_ids'].to(models['device'])
    bbox = encoding['bbox'].to(models['device'])
    attention_mask = encoding['attention_mask'].to(models['device'])
    pixel_values = pixel_values.to(models['device'])
    
    with torch.no_grad():
        outputs = models['multitask_model'](
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            pixel_values=pixel_values
        )
    
    # QA prediction
    start_idx = outputs.start_logits.argmax().item()
    end_idx = outputs.end_logits.argmax().item()
    
    if end_idx < start_idx:
        end_idx = start_idx
    
    start_probs = torch.softmax(outputs.start_logits, dim=1)
    end_probs = torch.softmax(outputs.end_logits, dim=1)
    confidence = (start_probs[0, start_idx] * end_probs[0, end_idx]).item()
    
    answer_tokens = input_ids[0][start_idx:end_idx+1]
    answer = models['layout_tokenizer'].decode(answer_tokens, skip_special_tokens=True).strip()
    
    # BIO prediction
    sequence_output = models['multitask_model'].layoutlmv3(
        input_ids=input_ids,
        bbox=bbox,
        attention_mask=attention_mask,
        pixel_values=pixel_values
    ).last_hidden_state
    
    bio_logits = models['multitask_model'].classifier(sequence_output)
    predicted_labels = bio_logits.argmax(dim=-1)[0]
    
    # Map to word-level labels
    word_ids = encoding.word_ids(batch_index=0)
    predicted_bio = []
    shift = len(q_words)
    
    for i in range(len(words)):
        word_id_to_find = shift + i
        for j, w_id in enumerate(word_ids):
            if w_id == word_id_to_find:
                label_id = predicted_labels[j].item()
                predicted_bio.append((words[i], ID2LABEL[label_id]))
                break
        else:
            predicted_bio.append((words[i], "O"))
    
    # OPTION 3: Extract and organize entities
    entity_counts = {
        "ANSWER": 0,
        "QUESTION": 0,
        "HEADER": 0
    }
    
    entities = {
        "ANSWER": [],
        "QUESTION": [],
        "HEADER": []
    }
    
    current_span = []
    current_type = None
    
    for word, tag in predicted_bio:
        if tag.startswith("B-"):
            # Save previous span
            if current_span and current_type:
                entity_text = " ".join(current_span)
                if entity_text not in entities[current_type]:  # Avoid duplicates
                    entities[current_type].append(entity_text)
                entity_counts[current_type] += 1
            # Start new span
            current_type = tag.split("-")[1]
            current_span = [word]
        elif tag.startswith("I-") and current_span:
            current_span.append(word)
        else:
            # End span
            if current_span and current_type:
                entity_text = " ".join(current_span)
                if entity_text not in entities[current_type]:
                    entities[current_type].append(entity_text)
                entity_counts[current_type] += 1
            current_span = []
            current_type = None
    
    # Save last span
    if current_span and current_type:
        entity_text = " ".join(current_span)
        if entity_text not in entities[current_type]:
            entities[current_type].append(entity_text)
        entity_counts[current_type] += 1
    
    # Format summary
    summary_parts = []
    
    # Show counts
    total = sum(entity_counts.values())
    if total > 0:
        summary_parts.append(f" <b>{total} entities detected</b>")
    
    # Show top entities by type
    if entities["ANSWER"]:
        top_answers = entities["ANSWER"][:3]
        answers_text = ", ".join(top_answers)
        if len(answers_text) > 100:
            answers_text = answers_text[:97] + "..."
        summary_parts.append(f" <b>Answers:</b> {answers_text}")
    
    if entities["QUESTION"]:
        top_questions = entities["QUESTION"][:3]
        questions_text = ", ".join(top_questions)
        if len(questions_text) > 100:
            questions_text = questions_text[:97] + "..."
        summary_parts.append(f" <b>Fields:</b> {questions_text}")
    
    if entities["HEADER"]:
        top_headers = entities["HEADER"][:2]
        headers_text = ", ".join(top_headers)
        if len(headers_text) > 80:
            headers_text = headers_text[:77] + "..."
        summary_parts.append(f" <b>Headers:</b> {headers_text}")
    
    bio_summary = "<br>".join(summary_parts) if summary_parts else "No entities detected"
    
    return (answer if answer else "[NO ANSWER]", confidence, bio_summary)

def process_document(models, image, question, use_ai_reformulation=True):
    """Process document and return results"""
    try:
        # Extract text and boxes
        context, words, boxes = extract_text_and_boxes(image)
        
        if not words:
            st.error("❌ No text found in image")
            return None
        
        # Show extracted text
        with st.expander("📋 Extracted Text", expanded=False):
            st.text_area("Extracted Text", context, height=150, disabled=True, label_visibility="hidden")
            st.caption(f"**{len(words)} words detected**")
        
        # AI Question Reformulation (if enabled)
        if use_ai_reformulation:
            reformulated_question = reformulate_question_with_ai(question, context)
        else:
            reformulated_question = question
            st.caption(f"Using original question: '{question}'")
        
        # Get predictions with reformulated question
        bert_ans, bert_conf = predict_bert(models, reformulated_question, context)
        layout_ans, layout_conf = predict_layoutlmv3(models, reformulated_question, words, boxes)
        vision_ans, vision_conf = predict_layoutlmv3_vision(models, reformulated_question, words, boxes, image)
        multitask_ans, multitask_conf, bio_tags = predict_multitask(models, reformulated_question, words, boxes, image)
        
        return {
            'bert': (bert_ans, bert_conf),
            'layout': (layout_ans, layout_conf),
            'vision': (vision_ans, vision_conf),
            'multitask': (multitask_ans, multitask_conf, bio_tags)
        }
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)
        return None

def display_results(results):
    """Display model predictions"""
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Model 1
        st.markdown("""
        <div class="model-card model-1">
            <div class="model-title">🥉 Model 1: BERT</div>
            <div class="model-caption">Text Only - 25.62% F1</div>
            <div class="answer-box"> {}</div>
            <div class="confidence">✓ Confidence: {:.1%}</div>
        </div>
        """.format(results["bert"][0], results["bert"][1]), unsafe_allow_html=True)
        
        # Model 3
        st.markdown("""
        <div class="model-card model-3">
            <div class="model-title">🥇 Model 3: LayoutLMv3 + Vision</div>
            <div class="model-caption">Text + Layout + Vision - 49.77% F1</div>
            <div class="answer-box"> {}</div>
            <div class="confidence">✓ Confidence: {:.1%}</div>
        </div>
        """.format(results["vision"][0], results["vision"][1]), unsafe_allow_html=True)
    
    with col2:
        # Model 2
        st.markdown("""
        <div class="model-card model-2">
            <div class="model-title">🥈 Model 2: LayoutLMv3</div>
            <div class="model-caption">Text + Layout - 49.34% F1</div>
            <div class="answer-box"> {}</div>
            <div class="confidence">✓ Confidence: {:.1%}</div>
        </div>
        """.format(results["layout"][0], results["layout"][1]), unsafe_allow_html=True)
        
        # Model 4
        st.markdown("""
        <div class="model-card model-4">
            <div class="model-title"> Model 4: Multi-Task</div>
            <div class="model-caption">QA: 43.85% F1 | BIO Tagging: 68.88% F1</div>
            <div class="answer-box"> {}</div>
            <div class="confidence">✓ Confidence: {:.1%}</div>
            <div class="bio-tags">{}</div>
        </div>
        """.format(results["multitask"][0], results["multitask"][1], results["multitask"][2]), unsafe_allow_html=True)
    
    st.success("✅ All models processed successfully!")
    
def show_evaluation_tab():
    """Display comprehensive evaluation metrics and visualizations"""
    
    st.header(" Model Evaluation & Analysis")
    
    st.markdown("""
    This page presents a comprehensive evaluation of our four models on the FUNSD dataset.
    All models were trained on **120 clean documents** (no test set contamination) and evaluated on **19 validation documents**.
    """)
    
    st.markdown("---")
    
    # ============================================
    # SECTION 1: OVERALL PERFORMANCE COMPARISON
    # ============================================
    st.subheader("1️⃣ Overall Performance Comparison")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Performance bar chart
        import pandas as pd
        import plotly.graph_objects as go
        
        # Data
        models_data = {
            'Model': ['Model 1:\nBERT', 'Model 2:\nLayoutLMv3', 'Model 3:\n+Vision', 'Model 4:\nMulti-Task'],
            'F1 Score': [25.62, 49.34, 49.77, 43.85],
            'Exact Match': [22.58, 43.45, 43.07, 37.76],
            'Color': ['#CD7F32', '#C0C0C0', '#FFD700', '#9370DB']
        }
        
        fig = go.Figure()
        
        # F1 bars
        fig.add_trace(go.Bar(
            name='F1 Score',
            x=models_data['Model'],
            y=models_data['F1 Score'],
            marker_color=models_data['Color'],
            text=[f"{val:.1f}%" for val in models_data['F1 Score']],
            textposition='outside',
        ))
        
        fig.update_layout(
            title="Question Answering Performance (F1 Score)",
            xaxis_title="Model",
            yaxis_title="F1 Score (%)",
            yaxis_range=[0, 60],
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.markdown("###  Key Metrics")
        
        st.metric("Best QA Model", "LayoutLMv3", "49.34% F1")
        st.metric("Layout Improvement", "+92%", "vs. BERT baseline")
        st.metric("Vision Contribution", "+0.4%", "minimal benefit")
        st.metric("Multi-Task BIO F1", "68.88%", "entity recognition")
    
    st.markdown("---")
    
    # ============================================
    # SECTION 2: DETAILED METRICS TABLE
    # ============================================
    st.subheader("2️⃣ Detailed Performance Metrics")
    
    metrics_df = pd.DataFrame({
        'Model': [
            '🥉 Model 1: BERT',
            '🥈 Model 2: LayoutLMv3',
            '🥇 Model 3: LayoutLMv3 + Vision',
            ' Model 4: Multi-Task'
        ],
        'Features': [
            'Text Only',
            'Text + Layout',
            'Text + Layout + Vision',
            'Text + Layout + Vision'
        ],
        'Parameters': ['110M', '133M', '193M', '125M'],
        'F1 Score (%)': [25.62, 49.34, 49.77, 43.85],
        'Exact Match (%)': [22.58, 43.45, 43.07, 37.76],
        'Training Time': ['2h', '2.5h', '5h', '15h'],
        'Additional Task': ['—', '—', '—', 'BIO: 68.88%']
    })
    
    st.dataframe(
        metrics_df,
        hide_index=True,
        width='stretch',
        column_config={
            "F1 Score (%)": st.column_config.ProgressColumn(
                "F1 Score (%)",
                format="%.2f%%",
                min_value=0,
                max_value=100,
            ),
        }
    )
    
    st.markdown("---")
    
    # ============================================
    # SECTION 3: FEATURE IMPORTANCE ANALYSIS
    # ============================================
    st.subheader("3️⃣ Feature Importance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Feature contribution chart
        feature_data = {
            'Feature': ['Text\nOnly', '+ Layout\nFeatures', '+ Vision\nFeatures'],
            'F1 Score': [25.62, 49.34, 49.77],
            'Improvement': [0, 23.72, 0.43]
        }
        
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(
            x=feature_data['Feature'],
            y=feature_data['F1 Score'],
            mode='lines+markers+text',
            marker=dict(size=15, color=['#CD7F32', '#C0C0C0', '#FFD700']),
            line=dict(width=3, color='#3498db'),
            text=[f"{val:.1f}%" for val in feature_data['F1 Score']],
            textposition='top center',
        ))
        
        fig2.update_layout(
            title="Incremental Feature Contribution",
            xaxis_title="Feature Set",
            yaxis_title="F1 Score (%)",
            yaxis_range=[0, 60],
            height=400,
        )
        
        st.plotly_chart(fig2, width='stretch')
    
    with col2:
        st.markdown("### 📈 Feature Analysis")
        
        st.success("""
        **✅ Layout Features: Critical (+92%)**
        - Bounding boxes capture spatial relationships
        - Forms have structured layouts
        - Biggest performance jump
        """)
        
        st.warning("""
        **⚠️ Vision Features: Minimal (+0.4%)**
        - FUNSD documents are black/white scans
        - No color or font variation to learn from
        - Layout already captures spatial info
        """)
        
        st.info("""
        **💡 Key Insight:**
        For form documents, **layout > vision**
        """)
    
    st.markdown("---")
    
    # ============================================
    # SECTION 4: MULTI-TASK LEARNING ANALYSIS
    # ============================================
    st.subheader("4️⃣ Multi-Task Learning Analysis")
    
    st.markdown("""
    **Model 4** performs two tasks simultaneously:
    1. **Span Extraction** (Question Answering)
    2. **BIO Tagging** (Named Entity Recognition)
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "QA Performance",
            "43.85% F1",
            "-5.5% vs Model 2",
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            "BIO Performance",
            "68.88% F1",
            "Entity Recognition"
        )
    
    with col3:
        st.metric(
            "Trade-off Ratio",
            "1:12",
            "5% loss → 69% gain"
        )
    
    # BIO tagging per-class performance
    st.markdown("#### BIO Tagging Detailed Results")
    
    bio_data = pd.DataFrame({
        'Entity Type': ['O (Outside)', 'B-ANSWER', 'I-ANSWER', 'B-QUESTION', 'I-QUESTION', 'B-HEADER', 'I-HEADER'],
        'Precision (%)': [90.64, 67.61, 60.23, 77.02, 73.12, 53.44, 18.70],
        'Recall (%)': [85.32, 76.62, 65.21, 78.29, 75.44, 48.38, 44.67],
        'F1 Score (%)': [87.90, 71.84, 62.62, 77.65, 74.26, 50.78, 26.37],
        'Support': [62334, 7803, 8931, 8636, 10455, 1205, 1097]
    })
    
    st.dataframe(
        bio_data,
        hide_index=True,
        width='stretch',
        column_config={
            "F1 Score (%)": st.column_config.ProgressColumn(
                "F1 Score (%)",
                format="%.2f%%",
                min_value=0,
                max_value=100,
            ),
        }
    )
    
    st.markdown("---")
    
    from metrics_advanced import show_advanced_metrics
    show_advanced_metrics()
    


def main():
    # Header
    st.markdown('<div class="main-header">📄 Multimodal Document QA</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">CIS-583 Deep Learning Project | University of Michigan-Dearborn</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # CREATE TABS
    tab1, tab2, tab3 = st.tabs(["Live Demo", " Evaluation & Metrics", "ℹ️ About"])
    
    # =========================================================================
    # TAB 1: LIVE DEMO
    # =========================================================================
    with tab1:
        # Sidebar
        with st.sidebar:
            st.header(" Model Performance")
            st.metric("🥉 BERT (Text Only)", "25.62% F1")
            st.metric("🥈 LayoutLMv3 (Text+Layout)", "49.34% F1")
            st.metric("🥇 LayoutLMv3+Vision", "49.77% F1")
            st.metric("💜 Multi-Task (Dual)", "43.85% QA + 68.88% BIO")
            
            st.markdown("---")
            
            # AI Question Reformulation Toggle
            use_ai = st.checkbox(
                "🤖 AI Question Reformulation",
                value=True,
                help="Use GPT to convert natural language questions to form field labels"
            )
            
            st.markdown("---")
            st.markdown("### 💡 Key Findings")
            st.info("**Layout** features provide 92% improvement")
            st.info("**Multi-task** enables entity recognition")
        
        # Load models
        models = load_models()
        st.success("✅ All 4 models loaded successfully!")
        
        # Initialize session state
        if 'selected_example' not in st.session_state:
            st.session_state.selected_example = None
        if 'uploaded_image' not in st.session_state:
            st.session_state.uploaded_image = None
        if 'question_text' not in st.session_state:
            st.session_state.question_text = ""
        
        # Main content
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📤 Input")
            
            # Tab for upload vs examples
            input_tab1, input_tab2 = st.tabs(["📁 Upload Image", "📚 Try Examples"])
            
            with input_tab1:
                uploaded_file = st.file_uploader(
                    "Upload Document Image",
                    type=['png', 'jpg', 'jpeg'],
                    help="Upload a document image",
                    key="file_uploader"
                )
                
                if uploaded_file:
                    st.session_state.uploaded_image = Image.open(uploaded_file).convert("RGB")
                    st.session_state.selected_example = None
            
            with input_tab2:
                for i, example in enumerate(EXAMPLES):
                    col_img, col_text = st.columns([1, 2])
                    
                    with col_img:
                        try:
                            example_img_path = project_root / example['image_path']
                            if example_img_path.exists():
                                example_img = Image.open(example_img_path)
                                st.image(example_img, width='stretch')
                        except:
                            st.warning("Image not found")
                    
                    with col_text:
                        st.markdown(f"**{example['name']}**")
                        st.caption(f"*{example['question']}*")
                        if st.button(f"Load Example {i+1}", key=f"example_{i}", width='stretch'):
                            try:
                                example_img_path = project_root / example['image_path']
                                st.session_state.selected_example = i
                                st.session_state.uploaded_image = Image.open(example_img_path).convert("RGB")
                                st.session_state.question_text = example['question']
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                    
                    st.markdown("---")
            
            # Display current image
            if st.session_state.uploaded_image:
                st.image(st.session_state.uploaded_image, caption="Current Document", width='stretch')
            
            # Question input
            question = st.text_input(
                "❓ Your Question",
                value=st.session_state.question_text,
                placeholder="e.g., What is the brand?",
                key="question_input"
            )
            
            st.session_state.question_text = question
            
            # Submit button
            process_button = st.button("🚀 Get Answers from All Models", type="primary", width='stretch')
        
        with col2:
            st.subheader("Results")
            
            if process_button:
                if st.session_state.uploaded_image is None:
                    st.error("⚠️ Please upload an image or select an example!")
                elif not question.strip():
                    st.error("⚠️ Please enter a question!")
                else:
                    with st.spinner("🔄 Processing with all 4 models..."):
                        results = process_document(
                            models, 
                            st.session_state.uploaded_image, 
                            question,
                            use_ai_reformulation=use_ai
                        )
                        if results:
                            display_results(results)
    
    # =========================================================================
    # TAB 2: EVALUATION & METRICS
    # =========================================================================
    with tab2:
        show_evaluation_tab()
    
    # =========================================================================
    # TAB 3: ABOUT
    # =========================================================================
    with tab3:
        show_about_page()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <b>Project:</b> Multimodal Document Question Answering on FUNSD Dataset<br>
        <b>Student:</b> Sankalp | <b>Course:</b> CIS-583 Deep Learning<br>
        <b>Models:</b> 4 architectures compared (BERT, LayoutLMv3, +Vision, Multi-Task)
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()