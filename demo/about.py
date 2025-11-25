"""
About page for Multimodal Document QA Demo
Technical project overview and achievements
"""

import streamlit as st


def show_about_page():
    """Display comprehensive project information"""

    # Project Title & Overview
    st.markdown("""
    # Multimodal Document Question Answering
    ### CIS-583 Deep Learning Course Project
    **University of Michigan-Dearborn | Fall 2025**

    ---
    """)

    # ============================================
    # PROJECT MOTIVATION
    # ============================================
    st.header(" Project Motivation")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        Document understanding is a critical challenge in enterprise AI systems. Organizations process millions 
        of forms, invoices, and reports daily—but extracting structured information from these documents remains 
        difficult due to their visual complexity and varied layouts.
        
        **The Challenge:**
        - Traditional NLP models treat documents as plain text, losing spatial structure
        - Layout information (tables, forms, headers) carries semantic meaning
        - Visual features (fonts, colors, logos) provide additional context
        
        **Our Goal:**
        Systematically evaluate how different modalities (text, layout, vision) contribute to document 
        question answering performance, and explore multi-task learning for comprehensive document understanding.
        """)

    with col2:
        st.info("""
        **Dataset**  
        FUNSD (Form Understanding in Noisy Scanned Documents)
        
        - 199 real-world forms
        - 120 training documents
        - 19 validation documents
        - 9,707 entities annotated
        - Noisy scanned images
        """)

    st.markdown("---")

    # ============================================
    # EXPERIMENTAL DESIGN
    # ============================================
    st.header(" Experimental Design")

    st.markdown("We designed a **four-model progression** to systematically isolate the contribution of each modality:")

# Model progression cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='padding: 1.5rem; background-color: white; border: 3px solid #CD7F32; border-radius: 8px; margin-bottom: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
            <h3 style='color: #2c3e50;'>🥉 Model 1: BERT Baseline</h3>
            <p style='color: #2c3e50;'><b>Architecture:</b> BERT-base (110M parameters)</p>
            <p style='color: #2c3e50;'><b>Input:</b> Text only (OCR-extracted)</p>
            <p style='color: #2c3e50;'><b>Purpose:</b> Establish baseline without spatial information</p>
            <p style='color: #2c3e50;'><b>Hypothesis:</b> Text alone is insufficient for form understanding</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='padding: 1.5rem; background-color: white; border: 3px solid #FFD700; border-radius: 8px; margin-bottom: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
            <h3 style='color: #2c3e50;'>🥇 Model 3: LayoutLMv3 + Vision</h3>
            <p style='color: #2c3e50;'><b>Architecture:</b> LayoutLMv3-base + Image Encoder (193M parameters)</p>
            <p style='color: #2c3e50;'><b>Input:</b> Text + Bounding Boxes + Raw Image</p>
            <p style='color: #2c3e50;'><b>Purpose:</b> Test if visual features improve beyond layout</p>
            <p style='color: #2c3e50;'><b>Hypothesis:</b> Vision adds minimal value for B&W forms</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='padding: 1.5rem; background-color: white; border: 3px solid #C0C0C0; border-radius: 8px; margin-bottom: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
            <h3 style='color: #2c3e50;'>🥈 Model 2: LayoutLMv3</h3>
            <p style='color: #2c3e50;'><b>Architecture:</b> LayoutLMv3-base (133M parameters)</p>
            <p style='color: #2c3e50;'><b>Input:</b> Text + Bounding Boxes</p>
            <p style='color: #2c3e50;'><b>Purpose:</b> Isolate layout contribution</p>
            <p style='color: #2c3e50;'><b>Hypothesis:</b> Spatial structure is critical for forms</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='padding: 1.5rem; background-color: white; border: 3px solid #9370DB; border-radius: 8px; margin-bottom: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
            <h3 style='color: #2c3e50;'>💜 Model 4: Multi-Task Learning</h3>
            <p style='color: #2c3e50;'><b>Architecture:</b> LayoutLMv3 + Dual Task Heads (125M parameters)</p>
            <p style='color: #2c3e50;'><b>Tasks:</b> Span Extraction + BIO Entity Tagging</p>
            <p style='color: #2c3e50;'><b>Purpose:</b> Explore task synergy and trade-offs</p>
            <p style='color: #2c3e50;'><b>Hypothesis:</b> Complementary tasks share useful representations</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ============================================
    # TECHNICAL IMPLEMENTATION
    # ============================================
    st.header("Technical Implementation")

    tab1, tab2, tab3 = st.tabs(["Data Pipeline", "Model Architecture", "Training Setup"])

    # ---------------------- Tab 1 ----------------------
    with tab1:
        st.markdown("""
        ### Data Processing Pipeline

        **1. Dataset Preparation**
        - Loaded FUNSD annotations (JSON format)
        - Discovered and removed test set contamination (36 documents, 762 QA pairs)
        - Final split: 120 train / 19 validation documents

        **2. OCR & Layout Extraction**
        ```python
        # Tesseract OCR for text extraction
        words, boxes = extract_text_and_boxes(image)

        # Normalize bounding boxes to [0, 1000]
        normalized_boxes = normalize_coordinates(boxes, img_width, img_height)
        ```

        **3. BIO Label Generation (Model 4)**
        - Generated token-level BIO tags (7 classes: O, B/I-ANSWER, B/I-QUESTION, B/I-HEADER)

        **4. Data Augmentation**
        - None applied due to dataset size
        """)

    # ---------------------- Tab 2 ----------------------
    with tab2:
        st.markdown("""
        ### Model Architectures
        
        **BERT Baseline (Model 1)**
        ```
        Input: [CLS] Question [SEP] Document [SEP]
              ↓
        BERT Encoder
              ↓
        QA Head (start/end logits)
        ```

        **LayoutLMv3 (Models 2, 3, 4)**
        ```
        Inputs:
        - Text tokens
        - Bounding boxes
        - Image (Model 3 only)
              ↓
        LayoutLMv3 Encoder
              ↓
        Task Head(s)
        ```

        **Multi-Task Architecture (Model 4)**
        ```
        Shared LayoutLMv3 Encoder
              ↓
        Dropout
              ↓
        ┌────────────┬──────────────┐
        │   QA Head   │   BIO Head   │
        └────────────┴──────────────┘
        ```
        """)

    # ---------------------- Tab 3 ----------------------
    with tab3:
        st.markdown("""
        ### Training Configuration

        **Hyperparameters (Models 1–3)**
        - LR: 5e-5
        - Batch size: 16
        - Epochs: 50 (early stopping)
        - Weight decay: 0.01

        **Model 4 (Multi-Task)**
        - LR: 3e-5
        - Batch size: 8
        - Loss: QA + BIO (equal weighting)
        """)

    st.markdown("---")

    # ============================================
    # KEY ACHIEVEMENTS
    # ============================================
    st.header(" What We Achieved")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.success("""
        ### Primary Goals
        - Systematic ablation study
        - Clean, reproducible pipeline
        """)

    with col2:
        st.success("""
        ### Novel Contributions
        - Multi-task learning on FUNSD
        - Real-time demo system
        """)

    with col3:
        st.success("""
        ### Technical Skills
        - End-to-end ML pipeline
        - Research rigor & analysis
        """)

    st.markdown("---")

    # ============================================
    # CHALLENGES
    # ============================================
    st.header("Challenges & What Didn't Work")

    st.markdown("""
    ### 1️⃣ Data Contamination  
    - 36 test documents leaked into training  
    - Required full retraining  

    ### 2️⃣ Vision Features Underperformed  
    - Nearly no benefit on B&W forms  

    ### 3️⃣ Multi-Task Complexity  
    - BIO head dominated gradients  
    - QA F1 dropped ~5.5%  

    ### 4️⃣ Small Dataset  
    - Performance ceiling around 50% F1  

    ### 5️⃣ Extractive QA is Hard  
    - Strict span boundary predictions  
    """)

    st.markdown("---")

    # ============================================
    # FUTURE WORK
    # ============================================
    st.header("Future Directions")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Model Improvements
        - Tune loss weights
        - Larger models
        - Augmentation
        """)

    with col2:
        st.markdown("""
        ### Expanded Scope
        - More datasets
        - Real-world deployment
        - Attention visualization
        """)

    st.markdown("---")

    # ============================================
    # TECH STACK
    # ============================================
    st.header(" Technical Stack")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **Deep Learning**
        - PyTorch
        - Transformers
        - CUDA
        """)

    with col2:
        st.markdown("""
        **Visualization**
        - Streamlit
        - Plotly
        - Matplotlib
        """)

    with col3:
        st.markdown("""
        **AI Services**
        - OpenAI API
        - RTX 4080 GPU
        """)

    st.markdown("---")

    # ============================================
    # ACKNOWLEDGMENTS
    # ============================================
    st.header("Acknowledgments")

    st.markdown("""
    **Dataset:** FUNSD  
    **Models:** BERT, LayoutLMv3  
    **Course:** CIS-583  
    """)

    st.markdown("---")

    # Contact
    st.markdown("""
        <div style='text-align: center; padding: 2rem; background-color: #f8f9fa; border-radius: 10px; border: 2px solid #ddd;'>
            <h3 style='color: #2c3e50; margin-bottom: 1rem;'>📧 Contact</h3>
            <p style='color: #2c3e50; font-size: 1.05rem; margin: 0.5rem 0;'><b>Name:</b> Sankalp Rajeev</p>
            <p style='color: #2c3e50; font-size: 1.05rem; margin: 0.5rem 0;'><b>University:</b> University of Michigan-Dearborn</p>
            <p style='color: #2c3e50; font-size: 1.05rem; margin: 0.5rem 0;'><b>Email:</b> srajeev@umich.edu</p>
            <p style='color: #2c3e50; font-size: 1.05rem; margin: 0.5rem 0;'><b>Course:</b> CIS-583 Deep Learning</p>
            <p style='color: #2c3e50; font-size: 1.05rem; margin: 0.5rem 0;'><b>Semester:</b> Fall 2025</p>
        </div>
        """, unsafe_allow_html=True)
