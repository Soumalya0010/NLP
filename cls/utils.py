from transformers import BartTokenizer, BartForSequenceClassification
import torch

# Load the fine-tuned model and tokenizer
tokenizer = BartTokenizer.from_pretrained('./fine_tuned_model')
model = BartForSequenceClassification.from_pretrained('./fine_tuned_model')
model.eval()

# Define the labels
labels = ["Remembering", "Understanding", "Applying", "Analyzing", "Evaluating", "Creating"]

def determine_taxonomy_level(question):
    # Tokenize the question
    inputs = tokenizer(question, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
    
    # Move inputs to the same device as the model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    inputs = {key: val.to(device) for key, val in inputs.items()}
    model.to(device)

    # Get the model's predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the predicted label
    predicted_label = torch.argmax(outputs.logits, dim=1).item()
    confidence = torch.softmax(outputs.logits, dim=1).max().item()

    return labels[predicted_label], confidence

# Example usage
question = "Explain the process of photosynthesis."
taxonomy_level, confidence = determine_taxonomy_level(question)
print(f"Predicted taxonomy level: {taxonomy_level} with confidence {confidence:.2f}")
