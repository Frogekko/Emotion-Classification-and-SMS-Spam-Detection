import pickle
import numpy as np
import gradio as gr
from sms_spam_utils import preprocess_text, count_words, count_punctuation, count_uppercase_words

# Load Model and Associated Data
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

# Prediction Function
def predict_spam(message_text):
    if model is None or vectorizer is None:
        return "Model not loaded. Please check setup.", None

    if not message_text.strip():
        return "Please enter a message to classify.", None

    # 1. Preprocess the input message (same as training)
    cleaned_text = preprocess_text(message_text)

    # 2. Transform using the loaded vectorizer (BoW features)
    bow_features = vectorizer.transform([cleaned_text]).toarray()

    # 3. Create extra features (consistent with training)
    # Ensure these utils functions are robust to empty cleaned_text if preprocess_text can return that
    extra_features_values = np.array([
        count_words(message_text),          # Based on original message, but count_words preprocesses
        count_punctuation(message_text),    # Based on original message
        count_uppercase_words(message_text),# Based on original message
        # Spam keyword presence (based on cleaned text, like in training)
        *[1 if keyword in cleaned_text.split() else 0 for keyword in spam_keywords]
    ]).reshape(1, -1) # Ensure it's a 2D array for hstack

    # 4. Combine features
    combined_features = np.hstack([bow_features, extra_features_values])

    # 5. Predict
    prediction_val = model.predict(combined_features)[0]
    probability_scores = None
    if hasattr(model, "predict_proba"):
        probability_scores = model.predict_proba(combined_features)[0]
        # For binary classification with LabelEncoder (0=ham, 1=spam typically)
        # prob_spam = probability_scores[1] # Assuming 1 is the spam class
        # prob_ham = probability_scores[0] # Assuming 0 is the ham class

    # 6. Format Output
    result_label = "🚨 SPAM 🚨" if prediction_val == 1 else "✅ HAM (Legitimate)"
    
    confidence_info = ""
    if probability_scores is not None:
        # Find the index of the predicted class in the model's classes_ attribute
        # This assumes your label_encoder encoded ham as 0 and spam as 1,
        # and model.classes_ would be [0, 1]
        # If model.classes_ is not directly available or differs, adjust accordingly.
        # For SVM with linear kernel, predict_proba might need probability=True during training.
        # If not available, we just show the label.
        try:
            # Assuming your model was trained with labels where spam=1, ham=0
            predicted_class_index = int(prediction_val) # 0 or 1
            confidence = probability_scores[predicted_class_index] * 100
            confidence_info = f"(Confidence: {confidence:.2f}%)"
        except Exception: # Broad except if classes_ or indexing fails
            pass # Silently ignore if proba details are hard to get

    final_output_text = f"{result_label} {confidence_info}"
    
    # For styled output using HTML in Markdown
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
            # Using Markdown for styled output
            output_display = gr.Markdown(value="<p style='text-align:center; color:grey;'>Results will appear here...</p>")


    gr.Examples(
        examples=[
            ["URGENT! You have won a 1 week FREE membership in our £100,000 Prize Draw. Text WORD to 88008 to claim your prize."],
            ["Hey, are we still on for dinner tonight at 7? Let me know!"],
            ["Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"],
            ["Ok lar... Joking wif u oni..."],
            ["WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only."]
        ],
        inputs=[message_input],
        outputs=[output_display], # Output to the Markdown component
        fn=predict_spam, # Function to run for examples
        cache_examples=False # Depending on whether you want to cache or not
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


if __name__ == "__main__":
    if model is not None and vectorizer is not None:
        app.launch(debug=True)
    else:
        print("Could not launch Gradio app because model or vectorizer failed to load.")