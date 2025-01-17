# Recommendation-System
Group 1's Senior Project

Input
User Uploaded Images:
	Image 1: A photo of a quiet, tropical beach with palm trees.
	Image 2: A sunset over a mountain range.
User Text Input:
"I want a relaxing vacation under $2000 with peaceful beaches, scenic views, good weather in March, and nearby restaurants. The location should have affordable flights and hotels. I also enjoy hiking and photography."
Processing Steps
Step 1: Analyze Images

	Extract features using CLIP:
    	Image 1: Embedding suggests “beach,” “tropical,” “quiet.”
    	Image 2: Embedding suggests “mountain,” “sunset,” “scenic.”

Step 2: Parse Text

	Extracted criteria from text input:
    	Budget: $2000.
    	Preferences: Relaxing, peaceful beach, hiking, scenic views, photography.
    	Weather: Good weather in March.
    	Additional: Restaurants, affordable flights/hotels.

Step 3: Query Knowledge Base

	Visual Matching:
    	Match image embeddings with database image features:
        	Top matches: Bali (Indonesia), Phuket (Thailand), and Huaraz (Peru).
	Textual Matching:
    	Match keywords like "peaceful beach," "scenic views," "good weather," and "March":
        	Filtered destinations: Bali, Phuket.
	Final Match:
    	Combine visual and textual scores, filter by budget and weather:
        	Final recommendation: Bali, Indonesia.

Step 4: Fetch Live Data

	Use APIs to fetch additional dynamic information:
    	Cheapest Flight: $750 round trip (via Skyscanner).
    	Hotel: $70 per night for a beachfront villa (via Booking.com).
    	Weather: Sunny, 27°C average in March (via OpenWeatherMap).

Step 5: LLM-Generated Explanation

	Input Prompt to LLM:

	Based on the user's budget of $2000, preference for a peaceful beach, scenic views, hiking opportunities, and good weather in March, explain why Bali is an ideal vacation spot. Highlight its unique features, such as beaches, restaurants, and photography spots, and how it aligns with their budget.

	Generated Output: "Bali is an excellent destination for your relaxing vacation under $2000. With tranquil beaches like Nusa Dua and scenic mountain views in Ubud, Bali offers the perfect blend of relaxation and adventure. The weather in March is ideal, with sunny days averaging 27°C. Bali also caters to food lovers with a variety of beachfront restaurants serving delicious local and international cuisines. Affordable hotels and flights make Bali a budget-friendly choice, while hiking trails and picturesque landscapes make it a paradise for photography enthusiasts."

Step 6: Curate Images of the Destination

	Retrieve high-quality images of Bali:
    	Beachfront at Nusa Dua.
    	Sunset at Uluwatu Temple.
    	Lush green rice terraces in Ubud.

UI: after the user input, make a world map, where there are several red dots that users can hover, when they hover to that red dot, information about the location appears: images, link to the cheapest flight (maybe SkyScanner API), current weather…., and maybe a suggested tour.
We can also try to extract whether they are travelling alone, couple, family… -> suggest tour accordingly
