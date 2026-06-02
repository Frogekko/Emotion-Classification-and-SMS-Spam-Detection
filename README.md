# NLP Project Suite: Emotion & Spam Classification

This repository contains the work for two distinct Natural Language Processing (NLP) projects which are part of the course-work of the Programming 2 course at University of Inland Norway. The suite covers both complex multi-label classification using transformers and classic binary classification with feature engineering.

1.  **Multi-Label Emotion Classification:** A BERT-based model trained to identify 28 different emotions from text.
2.  **SMS Spam Detection:** A system comparing classic ML models with custom features to classify messages as spam or ham.

Interactive web demos were developed for both classifiers using Gradio.

---

## Projects in Action
![Emotion Classification Demo](./assets/emotion_demo.gif)

![Spam Detection Demo](./assets/spam_demo.gif)

---

## Key Features & Results

### Emotion Classification
* **Model:** Utilized the `bert-base-uncased` transformer model for sequence classification.
* **Dataset:** Trained on the **full 211,225-sample GoEmotions dataset**, going beyond the recommended 1k subset to handle real-world scale and imbalance.
* **Techniques:** Implemented **class weighting** in the `BCEWithLogitsLoss` function and **per-label threshold tuning** on the validation set to optimize F1 scores for each of the 28 emotions, effectively addressing class imbalance.
* **Performance:** Achieved a **Micro F1 of 0.4375** and **Macro F1 of 0.3666** on the test set, with strong performance on frequent emotions like 'gratitude' (0.80 F1) and 'love' (0.64 F1).

### SMS Spam Detection
* **Approach:** Combined classic machine learning with **custom feature engineering** based on Exploratory Data Analysis (EDA).
* **Features:** Extracted Bag of Words (Binary, TF, TF-IDF) representations and added custom features like word count, punctuation count, uppercase word count, and presence of spam keywords.
* **Model Comparison:** Evaluated Logistic Regression, Naive Bayes, and Support Vector Machines (SVM).
* **Performance:** The **SVM classifier** using Binary Bag of Words combined with custom features achieved the best result with **98.5% accuracy** and a low number of false negatives (14 out of 1115 test samples).

---

## How it Works & Tech Stack

### Emotion Classification
The system uses a pre-trained `bert-base-uncased` model fine-tuned for multi-label sequence classification. Input text is tokenized, padded/truncated to 128 tokens, and fed into the BERT model. The output logits pass through a sigmoid function, and predictions are made based on individually tuned thresholds for each of the 28 emotion classes. Class weights are used during training to mitigate data imbalance.

### SMS Spam Detection
Input SMS messages undergo preprocessing including tokenization and lemmatization using NLTK with POS tagging. Features are extracted using various Bag of Words methods (Binary BoW performed best) and combined with engineered features (length, counts, keywords) identified during EDA. These combined features are then used to train and evaluate classic ML classifiers, with SVM yielding the top performance.

### Tech Stack
* **Machine Learning:** `PyTorch`, `Scikit-Learn`
* **NLP / Transformers:** `Transformers (HuggingFace)`, `NLTK`, `tokenizers`
* **Data Handling & Analysis:** `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`
* **Web Demo:** `Gradio`
* **Environment:** `Conda`, `Pip`

---

## How to Run

### Setup
The Conda method is recommended for full environment replication using the provided `environment.yml` file.

1.  **Clone the repository.**
2.  **Create & Activate Conda Environment:**
    ```bash
    conda env create -f environment.yml
    conda activate nlp_suite
    ```
    *(Alternatively, use `pip install -r requirements.txt` in a Python 3.12+ environment)*
3.  **Download NLTK Data (if needed for Spam Detection):** Uncomment and run the `nltk.download()` lines in `sms_spam_detection.ipynb` if you haven't used NLTK before.

### Running the Projects

#### Emotion Classification
1.  **Training (Optional):** Run the `emotion_classification.ipynb` notebook. Ensure `goemotions.csv` is in the same directory. This will generate `best_emotion_model.pt`.
2.  **Run Demo:** Execute the Gradio interface script. Ensure `best_emotion_model.pt` is present.
    ```bash
    python emotion_classification_interface.py
    ```

#### SMS Spam Detection
1.  **Training & EDA:** Run the `sms_spam_detection.ipynb` notebook. Ensure `spam.csv` and `sms_spam_utils.py` are in the same directory. This will generate `sms_spam_model.pkl`.
2.  **Run Demo:** Execute the Gradio interface script. Ensure `sms_spam_model.pkl` and `sms_spam_utils.py` are present.
    ```bash
    python sms_spam_interface.py
    ```

---

## Contributors

* [Pedro Torrão](https://www.linkedin.com/in/pedro-torrao/)
