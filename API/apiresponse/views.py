from django.http import JsonResponse
from django.views.decorators.csrf import csrf_protect, csrf_exempt
import json
from FlightScraper import SearchFlights

@csrf_exempt
def find_airport_path(request):
    """
    Given a list of cities in a POST request, return the optimal airport path with city-airport mapping.
    """
    if request.method == 'POST':
        try:
            # Parse JSON body
            data = json.loads(request.body)
            cities = data.get('cities', [])

            if len(cities) < 2:
                return JsonResponse({"error": "At least two cities are required."}, status=400)

            # Initialize FlightScraper logic
            graph = SearchFlights(fetch_from_web=True)

            # Find the flight path
            path_result = graph.find_path_between_multiple_cities(cities)

            # If the function returns an error string
            if isinstance(path_result, str):
                return JsonResponse({"error": path_result}, status=404)

            # Return a structured response
            return JsonResponse({"city_airport_paths": path_result}, status=200)

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON input."}, status=400)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)