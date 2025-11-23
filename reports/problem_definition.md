**Problem Definition -- Multimodal Document Question Answering Using Layout-Aware Transformers**

Many real-world business processes rely on information stored in semi-structured documents such as forms, invoices, and receipts. These documents often come as scanned images or PDFs, where information is organized spatially in boxes, tables, and key--value fields. Traditional OCR-based pipelines extract text from these documents, but they typically lose layout structure and require brittle, rule-based post-processing to answer even simple questions such as "What is the invoice number?" or "What is the document date?".

In this project, we study the problem of **document question answering (DocQA)** on scanned forms. Given a document image and a natural language question, the goal is to extract a short textual answer from the document. Formally, the input to the system is:

-   A **document image** DDD representing a scanned form or receipt.

-   A **question** qqq in natural language, such as "What is the company name?" or "What is the total amount?".

The output is:

-   An **answer string** aaa that appears in the document, for example "ABC Corporation" or "$123.45".

We model this as an **extractive question answering** problem: the answer must be a span present in the document text. The challenges include noisy OCR output, diverse document layouts, and the need to reason over both textual content and spatial layout.

To address this, we will implement and compare two deep learning approaches:

1.  A **text-only baseline** based on BERT-style extractive QA, which receives the question and a flattened OCR text sequence as input.

2.  A **layout-aware model** based on LayoutLMv3, which incorporates both text and 2D layout information (bounding boxes) from the document.

We will use the **FUNSD dataset**, which consists of scanned forms with annotations for text fields, their bounding boxes, and semantic labels such as questions and answers. The dataset allows us to derive question--answer pairs from annotated key--value relationships and to evaluate how well each model can answer questions about form fields.

For evaluation, we will follow standard QA metrics:

-   **Exact Match (EM):** the percentage of predictions that match the ground truth answer string exactly (after normalization).

-   **F1 score:** the token-level F1 between predicted and ground truth answers, which accounts for partial overlaps.

The central research question of this project is:

> *Does incorporating layout information via a layout-aware transformer (LayoutLMv3) provide a significant improvement over a text-only BERT baseline on document QA tasks?*