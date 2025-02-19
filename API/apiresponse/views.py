# django tools
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_protect, csrf_exempt
from .models import EmbeddingConfig

# utilities
import json
from constants import KnownDirs, HTTP, Config
import os
    
# core functionality
from FlightScraper import SearchFlights

# database and embedding tools
from src.embedding_extract.implicit_user_embedding import get_user_overall_embedding
from src.faiss_indexing.extract_city import recommend_cities

# debugging tools
import time

os.makedirs(KnownDirs.IMAGE_DIR, exist_ok=True)

@csrf_exempt
def upload_image(request):
    """ Post the images """
    if not (request.method == "POST" and request.FILES.get("image")):
            return JsonResponse({"error": "No image uploaded."}, status=HTTP.BAD_REQUEST)

    image = request.FILES["image"]
    image_path = os.path.join(KnownDirs.IMAGE_DIR, image.name)
    
    if not image.name.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
        return JsonResponse({"error": "Invalid image format. Only .jpg, .jpeg, .png, .gif, and .bmp are allowed."}, status=HTTP.BAD_REQUEST)

    with open(image_path, "wb+") as dest:
        for chunk in image.chunks():
            dest.write(chunk)

    return JsonResponse({"error": f"Image '{image.name}' uploaded."}, status=HTTP.CREATED)
    
@csrf_exempt
def upload_prompt(request):
    """ Post the user prompt """
    if not request.method == "POST":
        return JsonResponse({"error": "Invalid request method."}, status=HTTP.METHOD_NOT_ALLOWED)
    try:
        data = json.loads(request.body)
        prompt = data.get("prompt", "")
        
        with open(KnownDirs.TEXT_FILE_PATH, "w") as f:
            f.write(prompt)
            
        return JsonResponse({"success": f"Prompt saved!"}, status=HTTP.CREATED)
    
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON format."}, status=HTTP.BAD_REQUEST)
    
@csrf_exempt
def set_alpha_beta(request):
    """ Post the alpha value (find and set the beta value) """
    if not request.method == "POST":
        return JsonResponse({"error": "Invalid request method."}, status=HTTP.METHOD_NOT_ALLOWED)
    try:
        data = json.loads(request.body)
        alpha = float(data.get("alpha", 0))
        beta = float(data.get("beta", 0))

        if not (alpha + beta == 1 and 0 <= alpha <= 1 and 0 <= beta <= 1):
            return JsonResponse({"error": "Alpha and Beta must sum to 1."}, status=HTTP.BAD_REQUEST)

        config, _ = EmbeddingConfig.objects.get_or_create(id=1)
        config.alpha, config.beta = alpha, beta
        config.save()

        return JsonResponse({"success": f"Alpha set to {alpha}, beta set to {beta}."}, status=HTTP.OK)
    except (ValueError, json.JSONDecodeError):
        return JsonResponse({"error": "Invalid alpha/beta values."}, status=HTTP.BAD_REQUEST)

@csrf_exempt
def recommend_cities(request):
    if not request.method == "GET":
        return JsonResponse({"error": "Invalid request method."}, status=HTTP.METHOD_NOT_ALLOWED)
    
    image_path = KnownDirs.IMAGE_DIR
    prompt_path = KnownDirs.TEXT_FILE_PATH
    
    # cursory checks for populated paths
    prompt_exists = os.path.exists(prompt_path)
    images_exist = os.path.exists(image_path) and any(
        filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')) for filename in os.listdir(image_path)
    )
    
    # if nothing was populated, give up
    if not prompt_exists and not images_exist:
        return JsonResponse({"error": "No images or prompt to read"}, status=HTTP.BAD_REQUEST)

    a,b = Config.ALPHA_DEFAULT, Config.BETA_DEFAULT
    try:
        grabbedConfig = EmbeddingConfig.objects.first()
        if grabbedConfig:
            a,b = grabbedConfig.alpha, grabbedConfig.beta        
        if images_exist and not prompt_path:
            a,b = Config.IMAGE_ONLY_AB
        elif prompt_exists and not images_exist:
            a,b = Config.PROMPT_ONLY_AB
    except:
        return JsonResponse({"error": "Error fetching alpha/beta from database"}, status=HTTP.INTERNAL_SERVER_ERROR)
    
    try:
        user_embedding = get_user_overall_embedding(image_path, prompt_path, a, b)

        recommended_cities = recommend_cities(user_embedding, top_k=Config.TOP_K)

        return JsonResponse({"recommended_cities": recommended_cities}, status=HTTP.OK)

    except Exception as e:
        return JsonResponse({"error": "Error processing the embeddings"}, status=HTTP.INTERNAL_SERVER_ERROR)
    

        
@csrf_exempt
def find_airport_path(request):
    """
    Given a list of cities, return the optimal airport path between cities in sequence.
    """
    if not request.method == 'POST':
        return JsonResponse({"error": "Invalid request method."}, status=HTTP.METHOD_NOT_ALLOWED)
    try:
        # Parse JSON body
        data = json.loads(request.body)
        cities = data.get('cities', [])

        if len(cities) < 2:
            return JsonResponse({"error": "At least two cities are required."}, status=HTTP.BAD_REQUEST)

        # Initialize FlightScraper logic
        graph = SearchFlights(fetch_from_web=True)

        # Find the flight path
        path_result = graph.find_path_between_multiple_cities(cities)

        # If the function returns an error string
        if isinstance(path_result, str):
            return JsonResponse({"error": path_result}, status=HTTP.NOT_FOUND)

        # Return a structured response
        return JsonResponse({"city_airport_paths": path_result}, status=HTTP.OK)
    
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON input."}, status=HTTP.BAD_REQUEST)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=HTTP.INTERNAL_SERVER_ERROR)

@csrf_exempt
def find_two_city_path(request):
    """
    Given a list of cities, find the fastest airport route from the first city to each other city.
    """
    if not request.method == 'POST':
        return JsonResponse({"error": "Invalid request method."}, status=HTTP.METHOD_NOT_ALLOWED)
    try:
        # Parse JSON body
        data = json.loads(request.body)
        cities = data.get('cities', [])

        if len(cities) < 2:
            return JsonResponse({"error": "At least two cities are required."}, status=HTTP.BAD_REQUEST)
        
        # Initialize FlightScraper logic
        graph = SearchFlights(fetch_from_web=True)

        city1 = cities[0]
        tpath = {}
        for city2 in cities[1:]:
            # Find the flight path
            path_result = graph.find_path_between_cities(city1, city2)
            tpath.update( { city2 : path_result } )
            
        # Return a structured response
        return JsonResponse({"cities":tpath}, status=HTTP.OK)

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON input."}, status=HTTP.BAD_REQUEST)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=HTTP.INTERNAL_SERVER_ERROR)