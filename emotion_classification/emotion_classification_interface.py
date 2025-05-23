### Emotion Classification Gradio Interface ###
# This script was mostly generated by Gemini 2.5, with some modifications. Prompt included as PDF file.

import torch
import gradio as gr
from transformers import BertTokenizer, BertForSequenceClassification

# Load Model and Tokenizer
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=28)
    model.load_state_dict(torch.load('best_emotion_model.pt', map_location=device)) # Load the model weights
    model.to(device)
    model.eval()
    # Emotion labels
    emotion_labels = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
                      'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 
                      'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 
                      'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 
                      'relief', 'remorse', 'sadness', 'surprise', 'neutral']
    # Optimal thresholds for each emotion
    optimal_thresholds = [0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1, 0.2, 0.6, 0.1, 0.3, 0.3, 0.1, 0.3, 0.1, 0.1, 0.1, 0.2, 0.3, 0.2, 0.2]
except FileNotFoundError:
    print("ERROR: best_emotion_model.pt not found. Please train and save the model first.")
    model, tokenizer, emotion_labels, optimal_thresholds = None, None, [], []
except Exception as e:
    print(f"Error loading model: {e}")
    model, tokenizer, emotion_labels, optimal_thresholds = None, None, [], []

# Prediction Function
def predict_emotions(text):
    if model is None or tokenizer is None:
        return "Model not loaded. Please check setup.", None

    if not text.strip():
        return "Please enter a text to classify.", None

    # 1. Tokenize input
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 2. Predict emotions using the model
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
        preds = [prob > optimal_thresholds[i] for i, prob in enumerate(probs)]
    
    # 3. Extract predicted emotions and their confidence scores
    predicted_emotions = [(emotion_labels[i], probs[i]) for i, pred in enumerate(preds) if pred]
    if not predicted_emotions:
        predicted_emotions = [('neutral', 1.0)]  # Default to neutral if no emotions predicted

    # 4. Determine sentiment for styling (simplified: positive, negative, or neutral)
    positive_emotions = ['admiration', 'amusement', 'approval', 'caring', 'excitement', 'gratitude', 
                        'joy', 'love', 'optimism', 'pride', 'relief']
    negative_emotions = ['anger', 'annoyance', 'disappointment', 'disapproval', 'disgust', 
                        'embarrassment', 'fear', 'grief', 'nervousness', 'remorse', 'sadness']
    
    sentiment = 'neutral'
    for emotion, _ in predicted_emotions:
        if emotion in positive_emotions:
            sentiment = 'positive'
            break
        elif emotion in negative_emotions:
            sentiment = 'negative'
            break

    # 5. Format output with emotion labels and confidence scores
    emotion_list = ', '.join([f"{emotion} ({prob*100:.2f}%)" for emotion, prob in predicted_emotions])
    result_label = f"Predicted Emotions: {emotion_list}"

    # Create styled HTML output for Gradio
    if sentiment == 'positive':
        styled_output = f"""
        <div style="padding: 15px; border-radius: 10px; background-color: #DFF2BF; color: #4F8A10; border: 1px solid #4F8A10;">
            <h3 style="margin-top:0; color: #4F8A10;">😊 {result_label} 😊</h3>
        </div>
        """
    elif sentiment == 'negative':
        styled_output = f"""
        <div style="padding: 15px; border-radius: 10px; background-color: #FFD2D2; color: #D8000C; border: 1px solid #D8000C;">
            <h3 style="margin-top:0; color: #D8000C;">😔 {result_label} 😔</h3>
        </div>
        """
    else:
        styled_output = f"""
        <div style="padding: 15px; border-radius: 10px; background-color: #E6E6E6; color: #333333; border: 1px solid #333333;">
            <h3 style="margin-top:0; color: #333333;">😐 {result_label} 😐</h3>
        </div>
        """
    
    return styled_output

# Gradio Interface
with gr.Blocks(theme=gr.themes.Soft(), css="footer {display: none !important}") as app:
    gr.Markdown(
        """
        # 😊 Emotion Classifier 🥳
        Enter a short text below to classify its emotions (e.g., joy, anger, neutral).
        The model uses BERT to predict multiple emotions with confidence scores.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="✍️ Your Text:",
                placeholder="e.g., I'm so happy to see you!",
                lines=5,
                show_label=True
            )
            with gr.Row():
                clear_button = gr.Button(value="🧹 Clear")
                submit_button = gr.Button(value="🔍 Classify Emotions", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### 📊 Prediction Result:")
            output_display = gr.Markdown(value="<p style='text-align:center; color:grey;'>Results will appear here...</p>")

    gr.Examples(
        examples=[
            ["That game hurt."],
            ["Man I love nature."],
            ["Everyone likes pizza."],
            ["That looks amazing!"],
            ["I'm so sad about the news..."],
            ["You did great, don't worry."]
        ],
        inputs=[text_input],
        outputs=[output_display],
        fn=predict_emotions,
        cache_examples=False
    )
    
    gr.Markdown(
        """
        ---
        *Powered by a BERT model trained on the GoEmotions dataset.*
        *This model can classify 28 different emotions, including joy, anger, and sadness.*
        """
    )

    # Button Actions
    submit_button.click(fn=predict_emotions, inputs=text_input, outputs=output_display)
    clear_button.click(
        lambda: [None, "<p style='text-align:center; color:grey;'>Results will appear here...</p>"],
        inputs=None,
        outputs=[text_input, output_display]
    )

# Launch the Gradio app if model loaded successfully
if __name__ == "__main__":
    if model is not None and tokenizer is not None:
        app.launch(debug=True)
    else:
        print("Could not launch Gradio app because model or tokenizer failed to load.")