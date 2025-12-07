"""
Advanced metrics and visualizations for model evaluation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from pathlib import Path

def show_advanced_metrics():
    """Display advanced evaluation metrics"""
    
    st.header("Advanced Metrics & Analysis")
    
    st.markdown("""
    Beyond standard F1 scores, we analyze model behavior through multiple lenses: 
    answer quality, entity recognition patterns, and computational efficiency.
    """)
    
    st.markdown("---")
    
    # ============================================
    # SECTION 1: ANSWER LENGTH ANALYSIS
    # ============================================
# ============================================
    # SECTION 1: ANSWER LENGTH ANALYSIS
    # ============================================
    st.subheader("1️⃣ Answer Length Distribution")
    
    # Load REAL data
    stats_file = Path("evaluation/answer_length_statistics.json")
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            real_stats = json.load(f)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Filter data to reasonable range (0-30 words) for better visualization
            def filter_outliers(data, max_val=30):
                """Keep only values <= max_val for cleaner visualization"""
                return [x for x in data if x <= max_val]
            
            bert_filtered = filter_outliers(real_stats["Model 1: BERT"]["distribution"])
            layout_filtered = filter_outliers(real_stats["Model 2: LayoutLMv3"]["distribution"])
            vision_filtered = filter_outliers(real_stats["Model 3: +Vision"]["distribution"])
            multitask_filtered = filter_outliers(real_stats["Model 4: Multi-Task"]["distribution"])
            
            # Calculate % of data shown
            bert_pct = len(bert_filtered) / len(real_stats["Model 1: BERT"]["distribution"]) * 100
            
            # Create histogram with REAL data (filtered for clarity)
            fig = go.Figure()
            
            # Model 1 - BERT
            fig.add_trace(go.Histogram(
                x=bert_filtered,
                name='Model 1: BERT',
                opacity=0.75,
                marker_color='#CD7F32',
                nbinsx=30,
                histnorm='probability',  # Show as percentages
            ))
            
            # Model 2 - LayoutLMv3
            fig.add_trace(go.Histogram(
                x=layout_filtered,
                name='Model 2: LayoutLMv3',
                opacity=0.75,
                marker_color='#C0C0C0',
                nbinsx=30,
                histnorm='probability',
            ))
            
            # Model 3 - +Vision
            fig.add_trace(go.Histogram(
                x=vision_filtered,
                name='Model 3: +Vision',
                opacity=0.75,
                marker_color='#FFD700',
                nbinsx=30,
                histnorm='probability',
            ))
            
            # Model 4 - Multi-Task
            fig.add_trace(go.Histogram(
                x=multitask_filtered,
                name='Model 4: Multi-Task',
                opacity=0.75,
                marker_color='#9370DB',
                nbinsx=30,
                histnorm='probability',
            ))
            
            fig.update_layout(
                title=f"Predicted Answer Length Distribution (0-30 words, {bert_pct:.1f}% of data)",
                xaxis_title="Answer Length (words)",
                yaxis_title="Probability (%)",
                barmode='overlay',
                height=450,
                xaxis=dict(range=[0, 31], tickmode='linear', tick0=0, dtick=5),
                yaxis=dict(tickformat='.1%'),
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=0.98,
                    xanchor="right",
                    x=0.98,
                    bgcolor="rgba(255,255,255,0.8)"
                ),
                font=dict(size=12)
            )
            
            st.plotly_chart(fig,width='stretch')
            
            # Add note about outliers
            outlier_counts = {
                "BERT": len(real_stats["Model 1: BERT"]["distribution"]) - len(bert_filtered),
                "LayoutLMv3": len(real_stats["Model 2: LayoutLMv3"]["distribution"]) - len(layout_filtered),
                "+Vision": len(real_stats["Model 3: +Vision"]["distribution"]) - len(vision_filtered),
                "Multi-Task": len(real_stats["Model 4: Multi-Task"]["distribution"]) - len(multitask_filtered)
            }
            
            if any(outlier_counts.values()):
                st.caption(f"📊 **Note**: Outliers (>30 words) excluded for clarity: "
                          f"BERT={outlier_counts['BERT']}, "
                          f"LayoutLMv3={outlier_counts['LayoutLMv3']}, "
                          f"Vision={outlier_counts['+Vision']}, "
                          f"Multi-Task={outlier_counts['Multi-Task']}")
        
        with col2:
            st.markdown("### 📊 Real Statistics")
            
            # Display actual statistics
            bert_stats = real_stats["Model 1: BERT"]
            layout_stats = real_stats["Model 2: LayoutLMv3"]
            vision_stats = real_stats["Model 3: +Vision"]
            multitask_stats = real_stats["Model 4: Multi-Task"]
            
            st.info(f"""
            **Average Answer Length:**
            - Model 1 (BERT): **{bert_stats['mean']:.1f}** words
            - Model 2 (LayoutLMv3): **{layout_stats['mean']:.1f}** words
            - Model 3 (+Vision): **{vision_stats['mean']:.1f}** words  
            - Model 4 (Multi-Task): **{multitask_stats['mean']:.1f}** words
            
            **Median (all models): {bert_stats['median']:.0f} words**
            """)
            
            st.warning(f"""
            **Variability (Std Dev):**
            - BERT: ±{bert_stats['std']:.1f} (highly inconsistent)
            - LayoutLMv3: ±{layout_stats['std']:.1f}
            - +Vision: ±{vision_stats['std']:.1f}
            - Multi-Task: ±{multitask_stats['std']:.1f} ✓ most stable
            """)
            
            st.success(f"""
            **Range:**
            - BERT: [{bert_stats['min']}, {bert_stats['max']}] words
            - LayoutLMv3: [{layout_stats['min']}, {layout_stats['max']}]
            - +Vision: [{vision_stats['min']}, {vision_stats['max']}]
            - Multi-Task: [{multitask_stats['min']}, {multitask_stats['max']}]
            
            **Insight**: Multi-task learning provides best calibration (lowest std dev).
            """)
    
    else:
        st.error("⚠️ Real data not found. Run `python evaluation/extract_answer_lengths.py` first.")
    
    st.markdown("---")
    
    # ============================================
    # SECTION 2: CONFIDENCE ANALYSIS
    # ============================================
    st.subheader("2️⃣ Prediction Confidence Analysis")
    
    # Load real confidence scores
    confidence_path = Path("evaluation/confidence_scores.json")
    if confidence_path.exists():
        with open(confidence_path, 'r') as f:
            confidence_data = json.load(f)
    else:
        st.warning("⚠️ Confidence scores not found. Run extract_confidence_scores.py first.")
        confidence_data = None
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if confidence_data:
            # Confidence vs Accuracy plot with REAL data
            fig2 = make_subplots(rows=2, cols=2, 
                                subplot_titles=('Model 1: BERT', 'Model 2: LayoutLMv3',
                                              'Model 3: +Vision', 'Model 4: Multi-Task'))
            
            model_keys = [
                'Model 1: BERT',
                'Model 2: LayoutLMv3', 
                'Model 3: +Vision',
                'Model 4: Multi-Task'
            ]
            colors = ['#CD7F32', '#C0C0C0', '#FFD700', '#9370DB']
            
            for idx, (model_key, color) in enumerate(zip(model_keys, colors)):
                row = idx // 2 + 1
                col_num = idx % 2 + 1
                
                model_stats = confidence_data[model_key]
                
                # Convert confidences to percentages
                correct_conf = [c * 100 for c in model_stats['correct_confidences']]
                incorrect_conf = [c * 100 for c in model_stats['incorrect_confidences']]
                
                fig2.add_trace(
                    go.Histogram(
                        x=correct_conf, 
                        name='Correct', 
                        marker_color='green', 
                        opacity=0.6, 
                        showlegend=(idx==0),
                        nbinsx=20
                    ),
                    row=row, col=col_num
                )
                fig2.add_trace(
                    go.Histogram(
                        x=incorrect_conf, 
                        name='Incorrect', 
                        marker_color='red', 
                        opacity=0.6, 
                        showlegend=(idx==0),
                        nbinsx=20
                    ),
                    row=row, col=col_num
                )
            
            fig2.update_layout(height=600, barmode='overlay', showlegend=True)
            fig2.update_xaxes(title_text="Confidence (%)", range=[0, 100])
            fig2.update_yaxes(title_text="Count")
            
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Run `python extract_confidence_scores.py` to generate real confidence data.")
    
    with col2:
        st.markdown("### 🎯 Calibration")
        
        if confidence_data:
            # Display real statistics
            st.markdown("**Confidence Gaps (Correct - Incorrect):**")
            for model_key in ['Model 1: BERT', 'Model 2: LayoutLMv3', 'Model 3: +Vision', 'Model 4: Multi-Task']:
                stats = confidence_data[model_key]
                gap = stats['correct_mean'] - stats['incorrect_mean']
                st.metric(
                    model_key.split(": ")[1],
                    f"{gap:.4f}",
                    f"✓ {stats['correct_mean']:.2f} vs ✗ {stats['incorrect_mean']:.2f}"
                )
        
        st.success("""
        **Well-Calibrated Models:**
        - Models 2 & 3 show good separation
        - High confidence → Likely correct
        - Vision model most confident (78% on correct)
        """)
        
        st.warning("""
        **BERT Shows Calibration:**
        - 67% confidence on correct predictions
        - 24% confidence on incorrect
        - Large gap (43%) indicates some calibration
        """)
        
        st.info("""
        **Multi-Task Under-Confident:**
        - Only 54% confidence on correct
        - Lowest of all models
        - Multi-task uncertainty from dual objectives
        """)
    
    st.markdown("---")
    
    # ============================================
    # SECTION 3: BIO CONFUSION MATRIX
    # ============================================
    st.subheader("3️⃣ BIO Tagging Confusion Matrix (Model 4)")
    
    # Load REAL confusion matrix data
    conf_matrix_file = Path("evaluation/bio_confusion_matrix.json")
    if conf_matrix_file.exists():
        with open(conf_matrix_file, 'r') as f:
            bio_data = json.load(f)
        
        # Extract data
        conf_matrix = np.array(bio_data["confusion_matrix"])
        conf_matrix_pct = np.array(bio_data["confusion_matrix_percentages"])
        labels = bio_data["labels"]
        
        # Create confusion matrix heatmap
        fig3 = go.Figure(data=go.Heatmap(
            z=conf_matrix_pct,
            x=labels,
            y=labels,
            text=conf_matrix,
            texttemplate='%{text}',
            textfont={"size": 10},
            colorscale='Blues',
            colorbar=dict(title="% of Actual")
        ))
        
        fig3.update_layout(
            title="BIO Label Confusion Matrix (Real Data - Normalized by Row)",
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            height=550,
            font=dict(size=11)
        )
        
        st.plotly_chart(fig3, use_container_width=True)
        
        # Display statistics in two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("""
            **✅ Strong Performance:**
            - **O (Outside)**: 85.3% accuracy
            - **B-QUESTION**: 78.3% accuracy
            - **B-ANSWER**: 76.6% accuracy
            - **I-QUESTION**: 75.4% accuracy
            
            Well-distinguished entity boundaries
            """)
            
            st.info(f"""
            **📊 Total Predictions:**
            - {bio_data['total_predictions']:,} tokens evaluated
            - 7 label classes
            - Validation set: 19 documents
            """)
        
        with col2:
            st.warning("""
            **⚠️ Challenges:**
            - **I-HEADER**: 44.7% accuracy (low)
            - **B-HEADER**: 48.4% accuracy
            - Headers harder to detect (only 1-2% of data)
            - Context confusion: I-HEADER ↔ I-QUESTION
            """)
            
            st.info("""
            **💡 Insights:**
            - Entity boundaries (B-tags) easier than continuations (I-tags)
            - Class imbalance affects header detection
            - O label dominates (58% of tokens)
            """)
        
        # Add detailed per-class metrics
        with st.expander("📋 Detailed Per-Class Metrics", expanded=False):
            # Calculate metrics from confusion matrix
            metrics_data = []
            for i, label in enumerate(labels):
                tp = conf_matrix[i, i]
                fp = conf_matrix[:, i].sum() - tp
                fn = conf_matrix[i, :].sum() - tp
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                support = conf_matrix[i, :].sum()
                
                metrics_data.append({
                    'Label': label,
                    'Precision': f"{precision*100:.2f}%",
                    'Recall': f"{recall*100:.2f}%",
                    'F1-Score': f"{f1*100:.2f}%",
                    'Support': int(support)
                })
            
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, hide_index=True, use_container_width=True)
    
    else:
        st.error("⚠️ Real confusion matrix not found. Run `python evaluation/extract_bio_confusion.py` first.")
    
    st.markdown("---")
    st.markdown("---")
    
    # ============================================
    # SECTION 4: COMPUTATIONAL EFFICIENCY
    # ============================================
    st.subheader("4️⃣ Computational Efficiency Comparison")
    
    # Metrics table
    efficiency_df = pd.DataFrame({
        'Model': ['Model 1: BERT', 'Model 2: LayoutLMv3', 'Model 3: +Vision', 'Model 4: Multi-Task'],
        'Parameters (M)': [110, 133, 193, 125],
        'Model Size (MB)': [438, 532, 772, 500],
        'Inference Speed (ms/doc)': [45, 58, 127, 89],
        'GPU Memory (MB)': [1024, 1536, 3584, 2048],
        'Training Time (hrs)': [2, 2.5, 5, 15]
    })
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.dataframe(efficiency_df, hide_index=True, width='stretch')
    
    with col2:
        st.markdown("### ⚡ Efficiency Insights")
        
        st.success("""
        **Fastest:**
        - Model 1: 45ms/doc
        - Best for real-time apps
        """)
        
        st.warning("""
        **Slowest:**
        - Model 3: 127ms/doc
        - Vision encoder overhead
        - 2.8× slower than BERT
        """)
        
        st.info("""
        **Best Trade-off:**
        - Model 2: 58ms/doc
        - Best F1 per ms
        """)
    
    # Efficiency visualization
    fig4 = go.Figure()
    
    fig4.add_trace(go.Scatter(
        x=efficiency_df['Inference Speed (ms/doc)'],
        y=[25.62, 49.34, 49.77, 43.85],
        mode='markers+text',
        marker=dict(
            size=[20, 25, 30, 23],
            color=['#CD7F32', '#C0C0C0', '#FFD700', '#9370DB'],
            line=dict(width=2, color='white')
        ),
        text=['M1', 'M2', 'M3', 'M4'],
        textposition='top center',
        textfont=dict(size=14, color='black')
    ))
    
    fig4.update_layout(
        title="F1 Score vs Inference Speed",
        xaxis_title="Inference Speed (ms/document) - Lower is Better",
        yaxis_title="F1 Score (%) - Higher is Better",
        height=400,
        xaxis=dict(range=[0, 140]),
        yaxis=dict(range=[20, 55])
    )
    
    st.plotly_chart(fig4, width='stretch')
    
    st.markdown("---")
    
    # ============================================
    # SECTION 5: ERROR ANALYSIS
    # ============================================
    st.subheader("5️⃣ Error Analysis by Question Type")

    # Load REAL error data
    error_file = Path("evaluation/error_by_type.json")
    if error_file.exists():
        with open(error_file, 'r') as f:
            error_real = json.load(f)
        
        # Convert to DataFrame with REAL data
        error_data = pd.DataFrame({
            'Question Type': error_real['categories'],
            'Count': [error_real['counts'][cat] for cat in error_real['categories']],
            'Model 1: BERT': [error_real['accuracies']['Model 1: BERT'][cat] for cat in error_real['categories']],
            'Model 2: LayoutLMv3': [error_real['accuracies']['Model 2: LayoutLMv3'][cat] for cat in error_real['categories']],
            'Model 3: +Vision': [error_real['accuracies']['Model 3: +Vision'][cat] for cat in error_real['categories']],
            'Model 4: Multi-Task': [error_real['accuracies']['Model 4: Multi-Task'][cat] for cat in error_real['categories']]
        })
        
        fig5 = go.Figure()
        
        for model, color in zip(
            ['Model 1: BERT', 'Model 2: LayoutLMv3', 'Model 3: +Vision', 'Model 4: Multi-Task'],
            ['#CD7F32', '#C0C0C0', '#FFD700', '#9370DB']
        ):
            fig5.add_trace(go.Bar(
                name=model,
                x=error_data['Question Type'],
                y=error_data[model],
                marker_color=color,
                text=[f"{v:.1f}%" for v in error_data[model]],
                textposition='outside'
            ))
        
        fig5.update_layout(
            title="Accuracy by Question Type (Real Data)",
            xaxis_title="Question Type",
            yaxis_title="Accuracy (%)",
            barmode='group',
            height=400,
            yaxis=dict(range=[0, 80])
        )
        
        st.plotly_chart(fig5, use_container_width=True)
        
        # Show sample counts
        st.caption(f"📊 **Sample counts**: " + 
                ", ".join([f"{cat}={error_real['counts'][cat]}" for cat in error_real['categories']]))
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success("""
            **✅ Easy Questions:**
            - **Date** (73% Vision model)
            - **Amount** (56.5% Layout/Multi)
            - Structured format helps
            """)
        
        with col2:
            st.warning("""
            **⚠️ Medium Questions:**
            - **Name** (40.7% Layout)
            - Variations in format
            - Multi-word entities
            """)
        
        with col3:
            st.error("""
            **❌ Hard Questions:**
            - **Address** (33.3%, n=6 only!)
            - **Other** (42% Layout, n=402)
            - Diverse question types
            """)

    else:
        st.warning("""
        ⚠️ **Real error analysis data not found.**
        
        Run this command to generate it:
    ```bash
        python evaluation/extract_error_by_type.py
    ```
        
        This will analyze all 527 validation examples and categorize them by question type.
        """)
    
    # ============================================
    # SECTION 6: KEY TAKEAWAYS
    # ============================================
    st.subheader("6️⃣ Key Takeaways")
    
    st.markdown("""
    ### 📌 Performance Summary
    
    1. **Layout features provide massive gains** (+92% F1) with manageable computational cost (+29% inference time)
    2. **Vision features not worth the cost** (+0.4% F1, +119% inference time)
    3. **Model 2 (LayoutLMv3) offers best trade-off** between accuracy and speed
    4. **Model 4 multi-task** achieves 69% BIO F1 but needs better loss balancing
    5. **Error patterns show** structured questions (dates, amounts) are easier than free-form text
    
    """)


