from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_from_disk
import torch
import json
import re


def clean_and_extract_values(text):
    """
    Removes unnecessary symbols like [], '', (), ensures spaces are retained for readability,
    and keeps only one space between words while preserving numerical values correctly.
    """
    text = re.sub(r'\(([^)]*\d+[^)]*)\)', r'\1', text)  # Remove parentheses but keep content inside if it contains numbers
    text = re.sub(r'[\[\]"]', ' ', text)  # Replace brackets and double quotes with spaces
    text = re.sub(r'\s*:\s*', ': ', text)  # Normalize colons
    text = re.sub(r'\s*,\s*', ', ', text)  # Normalize commas
    text = re.sub(r'\s*\'\s*', ' ', text)  # Replace single quotes with spaces
    text = re.sub(r'(?<=\d)\s*,\s*(?=\d{3})', '', text)  # Ensure numbers like 2,000 remain intact
    text = re.sub(r'\s+', ' ', text)  # Ensure only one space between words
    return text.strip()

def find_mentioned_criteria(generated_text, criteria_list):
    """
    Identifies which criteria from the predefined list are mentioned in the generated output.
    """
    mentioned_criteria = {}
    
    for criterion in criteria_list:
        pattern = rf'"{criterion}":\s*"?(.*?)"?(,|$)'
        match = re.search(pattern, generated_text)
        if match:
            mentioned_criteria[criterion] = match.group(1).strip()
    
    return mentioned_criteria

def evaluate_t5():
    MODEL_PATH = "fine_tuned_models/checkpoint-20000"  # Change to latest checkpoint
    BASE_T5_MODEL = "t5-base"  # Ensure tokenizer is from original base model

    # 🔄 Load model and tokenizer
    #print("🔄 Loading model and tokenizer...")
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
    tokenizer = T5Tokenizer.from_pretrained(BASE_T5_MODEL, legacy=True)  # Base tokenizer

    model.eval()  # Set model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
    model.to(device)

    # 🔄 Load validation dataset (optional)
    dataset_path = "./synthetic_prompts/tokenized_synthetic_travel_data"
    dataset = load_from_disk(dataset_path)["validation"]

    # Sample prompt for testing
    sample_prompt = dataset[59]["prompt"]    
    sample_prompt = clean_and_extract_values(sample_prompt)
    print(sample_prompt)
    #print(f"\n🎯 **Sample Prompt:**\n{sample_prompt}")

    criteria_list = [
        "departure_location", "departure_month", "return_month", "budget", "weather_preference",
        "destination_type", "travel_companions", "preferred_activities", "food_preference", "travel_duration",
        "accommodation_preference", "transportation_mode", "transportation_preference", "season", "event_interest",
        "safety_preference", #"accessibility_needs",
          "language_preference", "visa_requirement", "travel_theme",
        "sustainability_focus", "trip_intensity", "cultural_preference", "shopping_style", "internet_availability",
        "luxury_rating", "pet_friendly", "wellness_activities", "adventure_level", "nightlife_preferences",
        "currency_preference", "insurance_preference", "travel_addon"
    ]

    # Tokenize input
    inputs = tokenizer(sample_prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move tensors to correct device

    # 🔄 Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=1024,  # Ensure full structured output
            num_beams=5,  # Beam search for structured output
            #early_stopping=True,
            repetition_penalty=1.2,  # Reduce repetition
            #temperature=0.7,  # Controls randomness
        )

    # Decode generated output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    #print("\n✅ **Raw Model Output:**")
    #print(generated_text)

    # Clean up and structure output
    cleaned_output = find_mentioned_criteria(generated_text, criteria_list)
    #print("\n✅ **Parsed JSON Output:**")
    #print(json.dumps(cleaned_output, indent=2))
    
    #output_data = f"Sample Input:\n{sample_prompt}\n\nRaw Generated Output:\n{generated_text}\n\nParsed JSON Output:\n{json.dumps(cleaned_output, indent=2)}"
    #print(cleaned_output)
    with open("user_preference.json", "w", encoding="utf-8") as file:
        json.dump(cleaned_output, file, indent=2)

if __name__ == "__main__":
    evaluate_t5()
