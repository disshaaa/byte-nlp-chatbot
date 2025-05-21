import gradio as gr
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import nltk

# Initialize models
try:
    # Translation
    translation_en_hi = pipeline("translation_en_to_hi", model="Helsinki-NLP/opus-mt-en-hi")
    translation_en_fr = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
    translation_fr_en = pipeline("translation_fr_to_en", model="Helsinki-NLP/opus-mt-fr-en")
    translation_en_es = pipeline("translation_en_to_es", model="Helsinki-NLP/opus-mt-en-es")
    translation_es_en = pipeline("translation_es_to_en", model="Helsinki-NLP/opus-mt-es-en")
    
    # Sentiment Analysis
    sentiment_pipe = pipeline("text-classification", model="finiteautomata/bertweet-base-sentiment-analysis")
    
    # Grammar Correction
    grammar_tokenizer = AutoTokenizer.from_pretrained("vennify/t5-base-grammar-correction")
    grammar_model = AutoModelForSeq2SeqLM.from_pretrained("vennify/t5-base-grammar-correction")
    
    # Load fine-tuned model
    paraphrase_model = T5ForConditionalGeneration.from_pretrained("Vamsi/T5_Paraphrase_Paws")
    paraphrase_tokenizer = T5Tokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
    

    # T5 for QA, paraphrasing, etc.
    t5_model = T5ForConditionalGeneration.from_pretrained("./t5-small-finetuned-cnn-epoch5").to(
    "cuda" if torch.cuda.is_available() else "cpu")
    t5_tokenizer = T5Tokenizer.from_pretrained("./t5-small-finetuned-cnn-epoch5")
    
except Exception as e:
    print(f"Error loading models: {e}")
    raise

def detect_task(message):
    message_lower = message.lower()
    
    if any(word in message_lower for word in ["नमस्ते", "धन्यवाद", "कैसे"]):
        return "Hindi Chat"
    elif "context:" in message_lower and "question:" in message_lower:
        return "Question Answering"
    elif "translate" in message_lower:
        return "Translation"
    elif "summarize" in message_lower or "summary" in message_lower:
        return "Summarization"
    elif "sentiment" in message_lower or "feeling" in message_lower:
        return "Sentiment Analysis"
    elif "paraphrase" in message_lower or "rewrite" in message_lower:
        return "Paraphrasing"
    elif "grammar" in message_lower or "correct this" in message_lower:
        return "Grammar Correction"
    elif any(word in message_lower for word in ["hello", "hi", "hey"]):
        return "English Chat"
    else:
        return "Summarization"
    
def handle_task(message, task=None):
    # If no task specified, auto-detect
    if task is None or task == "Auto-detect":
        task = detect_task(message)
    
    # Handle greetings and generic responses
    greetings = {
        "hello": "Hello! How can I assist you today?",
        "hi": "Hi there! What can I do for you?",
        "how are you": "I'm doing well, thank you! How about you?",
        "नमस्ते": "नमस्ते! मैं आपकी कैसे मदद कर सकता हूँ?",
        "धन्यवाद": "आपका स्वागत है!"
    }
    
    lower_msg = message.lower()
    if lower_msg in greetings:
        return greetings[lower_msg]
    
    # Task-specific handling
    try:
        if task == "Translation":
            text = message.replace("translate:", "").strip().lower()
            if "english to hindi:" in text:
                return translation_en_hi(text.replace("english to hindi:", "").strip())[0]['translation_text']
            elif "hindi to english:" in text:
                return translation_fr_en(text.replace("hindi to english:", "").strip())[0]['translation_text']
            elif "english to french:" in text:
                return translation_en_fr(text.replace("english to french:", "").strip())[0]['translation_text']
            elif "french to english:" in text:
                return translation_fr_en(text.replace("french to english", "").strip())[0]['translation_text']
            elif "english to spanish:" in text:
                return translation_en_es(text.replace("english to spanish:", "").strip())[0]['translation_text']
            elif "spanish to english:" in text:
                return translation_es_en(text.replace("spanish to english:", "").strip())[0]['translation_text']
            else:
                return "Please specify direction: e.g., 'translate: english to french: [your sentence]'"
    
        elif task == "Sentiment Analysis":
            text_to_analyze = message.replace("sentiment:", "").strip()
            result = sentiment_pipe(text_to_analyze)[0]
            return f"Sentiment: {result['label']} (confidence: {result['score']:.2f})"
            
        elif task == "Question Answering":
            try:
                # Expecting input format: "question: <question text> context: <context text>"
                text = message.replace("question:", "").strip()
                inputs = t5_tokenizer(f"question: {text} </s>", return_tensors="pt", truncation=True, padding=True).to(t5_model.device)
                outputs = t5_model.generate(**inputs, max_length=200, num_beams=5, early_stopping=True)
                return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
            except Exception as e:
                return f"Please use format: 'question: [question] context: [text]{str(e)}"

        elif task == "Paraphrasing":
            text = message.replace("paraphrase:", "").strip()
            input_text = f"paraphrase: {text} </s>"
            inputs = paraphrase_tokenizer([input_text], return_tensors="pt", truncation=True, padding=True).to(paraphrase_model.device)
            
            outputs = paraphrase_model.generate(
                **inputs,
                max_length=200,
                num_beams=5,
                num_return_sequences=1,
                temperature=1.0
            )
            return paraphrase_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        elif task == "Summarization":
            inputs = t5_tokenizer(
                f"summarize: {message}",
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(t5_model.device)
            
            outputs = t5_model.generate(
                **inputs,
                max_length=150,
                min_length=30,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
            
            return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

            
        elif task == "Grammar Correction":
            input_text = "grammar: " + message
            inputs = grammar_tokenizer(input_text, return_tensors="pt").to(grammar_model.device)
            outputs = grammar_model.generate(**inputs, max_length=128, num_beams=5)
            return grammar_tokenizer.decode(outputs[0], skip_special_tokens=True)

            
        elif task in ["Hindi Chat", "English Chat"]:
            return "I'm a multilingual chatbot. Please ask me anything!"
            
        else:
            return "I'm not sure how to handle this request. Please try being more specific."
            
    except Exception as e:
        return f"Error processing your request: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="🧠 BYTE - NLP Chatbot") as demo:
    gr.Markdown(
        """
        <div style="text-align: center; margin-bottom: 20px;">
            <h1 style="font-size: 2.2em; color: #ffffff;">
                🧠 BYTE - NLP Chatbot
            </h1>
        </div>
        <div style="font-size: 1.05em; padding: 15px 25px;">
            <p>A clean, powerful assistant powered by T5. Just type in your request and BYTE will help you with:</p>
            <ul>
                <li>Summarization of long paragraphs</li>
                <li>Grammar correction</li>
                <li>Paraphrasing sentences</li>
                <li>Sentiment analysis</li>
                <li>Translation (to/from English → Hindi, French, Spanish)</li>
                <li>Question answering and chat</li>
            </ul>
        </div>  
        <div style="font-size: 1.05em; padding: 15px 25px;">
            <p><strong>Examples:</strong></p>
            <ul>
                <li>summarize: The government has recently announced...</li>
                <li>grammar: she go to school everyday</li>
                <li>paraphrase: The weather is nice today</li>
                <li>translate: english to hindi: I am happy to see you</li>
                <li>sentiment: I love this product</li>
                <li>question: What is the capital of France? context: Capital of France is Paris. </li>
            </ul>
        </div>
        """
    )
    
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Your Message", placeholder="Type your message or task...")
    clear = gr.Button("Clear Chat")
    
    def respond(message, chat_history):
        response = handle_task(message)
        chat_history.append((message, response))
        return "", chat_history

    
    msg.submit(lambda m, c: respond(m, c), [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch()