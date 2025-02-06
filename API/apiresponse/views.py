from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from FlightScraper import AirportFlightGraph

@csrf_exempt  # Disable CSRF protection for testing purposes
def find_airport_path(request):
    """
    Given a list of cities in a POST request, return the optimal airport path.
    """
    if request.method == 'POST':
        try:
            # Parse JSON body to get the cities list
            data = json.loads(request.body)  # Manually parse the JSON body
            cities = data.get('cities', [])

            if len(cities) < 2:
                return JsonResponse({"error": "At least two cities are required."}, status=400)

            # Initialize the FlightScraper logic
            graph = AirportFlightGraph(fetch_from_web=True)

            # Use the logic to find the path
            path = graph.find_path_between_multiple_cities(cities)

            if isinstance(path, str) and "No valid path found" in path:
                return JsonResponse({"error": path}, status=404)

            # Return the optimal path as a JSON response
            return JsonResponse({"path": path}, status=200)

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
