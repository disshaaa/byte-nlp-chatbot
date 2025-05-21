# 🧠 BYTE: A Unified NLP Chatbot using T5 and Transformers

BYTE is an intelligent, multilingual chatbot built using the T5 Transformer architecture. It performs multiple NLP tasks — including summarization, grammar correction, paraphrasing, sentiment analysis, translation, and question answering — using a prompt-based, unified text-to-text approach. It features a clean and interactive Gradio-based UI.

---

## 🚀 Features

- 📝 Summarization (fine-tuned T5 on CNN/DailyMail)
- 🧠 Question Answering (T5-style QA with context)
- 🔁 Paraphrasing (using Vamsi/T5_Paraphrase_Paws)
- ✅ Grammar Correction (vennify/t5-base-grammar-correction)
- ❤️ Sentiment Analysis (pretrained DistilBERT via Transformers pipeline)
- 🌐 Translation (English ↔ Hindi, French, Spanish using MarianMT)
- 💬 Basic multilingual chat support (Hello/Namaste detection)

---

## 🗂️ Project Structure

```bash
├── app.py                     # Main Gradio chatbot application
├── finetune_summarize.py      # Script to fine-tune T5 on CNN/DailyMail summarization
├── finetune_sst2.py           # Script to fine-tune T5 on SST2 sentiment classification
├── requirements.txt           # Python dependencies
├── README.md                  # You are here :)
```
---

## 🧪 Examples

Use the following prompt styles:
- summarize: The government has recently announced new policies...
- paraphrase: The weather is nice today.
- grammar: she go to school everyday
- translate: english to hindi: I am happy to see you
- translate: english to french: How are you doing?
- sentiment: I love this product!
- question: What is the capital of France? context: Capital of France is Paris.

---

## ⚙️ Installation & Running the App

1. Clone the Repository
   ```bash
   git clone https://github.com/yourusername/byte-nlp-chatbot.git
   cd byte-nlp-chatbot
   ```
2. Install Dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Run the App
   ```bash
   python app.py
   ```
The Gradio interface will launch in your browser at http://localhost:7860.

---

## 📚 Model Details

| Task               | Model Used                                      | Fine-Tuned |
|--------------------|-------------------------------------------------|------------|
| Summarization      | t5-small (fine-tuned on CNN/DailyMail)          | ✅         |
| Paraphrasing       | Vamsi/T5_Paraphrase_Paws                         | ❌         |
| Grammar Correction | vennify/t5-base-grammar-correction              | ❌         |
| Sentiment          | finiteautomata/bertweet-base-sentiment-analysis | ❌         |
| Translation        | Helsinki-NLP MarianMT                           | ❌         |
| Question Answering | t5-small (prompt-engineered with context)       | ✅         |

---

## 🛠 Future Work

- Add document-level summarization and QA
- Enable speech-to-text and TTS for voice interface
- Integrate BERTScore and human evaluation metrics for quality checking
- Extend support to more Indian languages (e.g., Tamil, Bengali)

---

## 🤝 Acknowledgements

- 🤗 Hugging Face Transformers & Datasets
- 📄 T5 paper: "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (Raffel et al.)
- 🌐 Google mT5 and MarianMT
- 🧠 Vamsi's T5 models and Vennify’s grammar correction model

---

## 📃 License

This project is open-source and free to use under the MIT License.

---

