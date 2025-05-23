### SMS Spam Detection Gradio Interface ###
# This script was mostly generated by Gemini 2.5, with some modifications. Prompt included as PDF file.

import pickle
import numpy as np
import gradio as gr
from sms_spam_utils import preprocess_text, count_words, count_punctuation, count_uppercase_words

# Load Model and Data
try:
    with open('sms_spam_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
        model = model_data['model']
        vectorizer = model_data['vectorizer']
        spam_keywords = model_data['spam_keywords']
except FileNotFoundError:
    print("ERROR: sms_spam_model.pkl not found. Please train and save the model first.")
    model, vectorizer, spam_keywords = None, None, []
except Exception as e:
    print(f"Error loading model: {e}")
    model, vectorizer, spam_keywords = None, None, []

# Prediction Function to classify SMS as spam or ham
def predict_spam(message_text):
    if model is None or vectorizer is None:
        return "Model not loaded. Please check setup.", None
    # Check if themodel and vectorizer are loaded
    if not message_text.strip():
        return "Please enter a message to classify.", None

    # 1. Preprocess the input message (clean, tokenize, lemmatize)
    cleaned_text = preprocess_text(message_text)

    # 2. Transform using the loaded vectorizer (BoW features)
    bow_features = vectorizer.transform([cleaned_text]).toarray()

    # 3. Create extra features (consistent with training)
    extra_features_values = np.array([
        count_words(message_text),          
        count_punctuation(message_text),    
        count_uppercase_words(message_text),
        # Spam keyword presence (based on cleaned text, like in training)
        *[1 if keyword in cleaned_text.split() else 0 for keyword in spam_keywords]
    ]).reshape(1, -1) # Ensure it's a 2D array for hstack

    # 4. Combine features
    combined_features = np.hstack([bow_features, extra_features_values])

    # 5. Predict using SVM model
    prediction_val = model.predict(combined_features)[0]
    probability_scores = None
    if hasattr(model, "predict_proba"):
        probability_scores = model.predict_proba(combined_features)[0]
        # For binary classification with LabelEncoder (0=ham, 1=spam typically)
        # prob_spam = probability_scores[1] # Assuming 1 is the spam class
        # prob_ham = probability_scores[0] # Assuming 0 is the ham class

    # 6. Format Output with label and confidence
    result_label = "🚨 SPAM 🚨" if prediction_val == 1 else "✅ HAM (Legitimate)"
    
    confidence_info = ""
    if probability_scores is not None:
        try:
            predicted_class_index = int(prediction_val)
            confidence = probability_scores[predicted_class_index] * 100
            confidence_info = f"(Confidence: {confidence:.2f}%)"
        except Exception:
            pass # Skip if confidence calculation fails

    final_output_text = f"{result_label} {confidence_info}"
    
    # Create styled HTML output for Gradio
    if prediction_val == 1: # Spam
        styled_output = f"""
        <div style="padding: 15px; border-radius: 10px; background-color: #FFD2D2; color: #D8000C; border: 1px solid #D8000C;">
            <h3 style="margin-top:0; color: #D8000C;">🚨 Prediction: SPAM 🚨</h3>
            <p style="font-size: 1.1em;">{confidence_info}</p>
        </div>
        """
    else: # Ham
        styled_output = f"""
        <div style="padding: 15px; border-radius: 10px; background-color: #DFF2BF; color: #4F8A10; border: 1px solid #4F8A10;">
            <h3 style="margin-top:0; color: #4F8A10;">✅ Prediction: HAM (Legitimate) ✅</h3>
            <p style="font-size: 1.1em;">{confidence_info}</p>
        </div>
        """
    return styled_output


# Gradio Interface
with gr.Blocks(theme=gr.themes.Soft(), css="footer {display: none !important}") as app:
    gr.Markdown(
        """
        # 📧 SMS Spam Detector 🕵️‍♂️
        Enter an SMS message below to classify it as **Spam** or **Ham** (Legitimate).
        The model uses linguistic features and keywords to make its prediction.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            message_input = gr.Textbox(
                label="✉️ Your SMS Message:",
                placeholder="e.g., Congratulations! You've won a $1000 Walmart gift card. Go to http://example.com to claim now.",
                lines=5,
                show_label=True
            )
            with gr.Row():
                clear_button = gr.Button(value="🧹 Clear")
                submit_button = gr.Button(value="🔍 Classify Message", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### 📊 Prediction Result:")
            output_display = gr.Markdown(value="<p style='text-align:center; color:grey;'>Results will appear here...</p>")


    gr.Examples(
        examples=[
            ["Congratulations! You've been selected for a FREE cruise. Call 0800 123 456 to claim now!"],
            ["Hi, can you pick up some milk on your way home? Thanks!"],
            ["LIMITED OFFER: Get 50% off your next phone bill! Reply YES to opt-in. T&Cs apply."],
            ["Yo, what's up? Wanna grab coffee later?"],
            ["You are a winner of our $10,000 cash prize! Visit www.winyourprize.com to claim."],
            ["Meeting rescheduled to 3 PM tomorrow. Please confirm."]
        ],
        inputs=[message_input],
        outputs=[output_display],
        fn=predict_spam,
        cache_examples=False
    )
    
    gr.Markdown(
        """
        ---
        *Powered by a Machine Learning model trained on textual features.*
        *The model uses a combination of Bag-of-Words and additional features to classify messages as spam or ham.*
        """
    )

    # Button Actions
    submit_button.click(fn=predict_spam, inputs=message_input, outputs=output_display)
    clear_button.click(lambda: [None, "<p style='text-align:center; color:grey;'>Results will appear here...</p>"], inputs=None, outputs=[message_input, output_display])

# Launch the Gradio app if model loaded successfully
if __name__ == "__main__":
    if model is not None and vectorizer is not None:
        app.launch(debug=True)
    else:
        print("Could not launch Gradio app because model or vectorizer failed to load.")