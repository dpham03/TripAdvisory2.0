import random
import json

# Define expanded categories for diversity
budgets = ["under $500", "$1000", "$2000", "$5000", "over $10,000"]
weather_prefs = ["hot", "cold", "tropical", "mild", "rainy", "snowy"]
durations = ["weekend trip", "one week", "two weeks", "one month", "long-term stay"]
companions = ["solo", "partner", "family of four", "friends", "business trip"]
dest_types = ["urban", "coastal", "mountainous", "desert", "rural", "tropical island"]
activities = [
    ["hiking", "camping", "wildlife safari"],
    ["beach", "scuba diving", "snorkeling"],
    ["shopping", "nightlife", "music festivals"],
    ["cultural experiences", "historical sites", "museums"],
    ["family-friendly activities", "theme parks", "zoo visits"]
]
food_prefs = ["street food", "fine dining", "seafood", "vegetarian-friendly", "local delicacies"]
accommodations = ["hostel", "budget hotel", "luxury resort", "Airbnb", "private villa"]
flights = ["economy", "business class", "first class", "direct flights"]
accessibility = ["wheelchair access", "dietary restrictions", "child-friendly"]
safety_prefs = ["low-crime area", "female-friendly", "family-friendly"]
transportation = ["public transport", "rental car", "cycling", "walking"]
seasons = ["summer", "winter", "spring", "fall", "Christmas", "cherry blossom season"]
events = ["Oktoberfest", "Carnival in Rio", "F1 races", "Coachella"]
languages = ["English-speaking", "French-speaking", "Spanish-speaking"]
visa_reqs = ["visa-free", "requires visa", "electronic travel authorization"]

# Generate synthetic dataset
synthetic_data = {"data": []}

for _ in range(10_000):  # Generate 10,000 synthetic samples
    prompt = f"I want to visit a {random.choice(dest_types)} location with {random.choice(weather_prefs)} weather. "
    prompt += f"My budget is {random.choice(budgets)}, and I prefer {random.choice(food_prefs)}. "
    prompt += f"I will be traveling {random.choice(companions)} for {random.choice(durations)}, and I enjoy {random.choice(activities)}. "
    prompt += f"I prefer staying in a {random.choice(accommodations)} and using {random.choice(transportation)} for transport. "
    prompt += f"My trip is in {random.choice(seasons)}, and I am interested in {random.choice(events)}. "
    prompt += f"I want {random.choice(safety_prefs)} locations and {random.choice(accessibility)} support. "
    prompt += f"The local language should be {random.choice(languages)}, and I need a {random.choice(visa_reqs)} destination."

    expected_output = {
        "budget": random.choice(budgets),
        "weather_preference": random.choice(weather_prefs),
        "destination_type": random.choice(dest_types),
        "travel_companions": random.choice(companions),
        "preferred_activities": random.choice(activities),
        "food_preference": random.choice(food_prefs),
        "travel_duration": random.choice(durations),
        "accommodation_preference": random.choice(accommodations),
        "transportation_preference": random.choice(transportation),
        "season": random.choice(seasons),
        "event_interest": random.choice(events),
        "safety_preference": random.choice(safety_prefs),
        "accessibility_needs": random.choice(accessibility),
        "language_preference": random.choice(languages),
        "visa_requirement": random.choice(visa_reqs),
    }

    synthetic_data["data"].append({"prompt": prompt, "output": expected_output})

# Save dataset
with open("expanded_synthetic_travel_data.json", "w") as f:
    json.dump(synthetic_data, f, indent=4)

print("Expanded synthetic travel data generated and saved!")
