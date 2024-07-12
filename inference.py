import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import argparse

MAX_INPUT_LENGTH = 128
MAX_OUTPUT_LENGTH = 64
MODEL_NAME = "google-t5/t5-base"
BEAM_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_description(input_text):
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model.load_state_dict(torch.load("tags_to_description_model.pth"))
    model.eval()
    input_encoding = tokenizer(input_text, max_length=MAX_INPUT_LENGTH, padding="max_length", truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_encoding.input_ids,
            attention_mask=input_encoding.attention_mask,
            max_length=MAX_OUTPUT_LENGTH,
            num_beams=BEAM_SIZE,
            early_stopping=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate description from input tags")
    parser.add_argument("input_tags", type=str, help="Input tags separated by commas")
    
    args = parser.parse_args()
    
    generated_description = generate_description(args.input_tags)
    
    print(f"Input: {args.input_tags}")
    print(f"Generated description: {generated_description}")