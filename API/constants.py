""" FlightScraper.py constants """
class FlightScraper:
    URL_AIRPORTS = 'https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat'
    URL_ROUTES = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/routes.dat"

    URL_AIRPORTS_FILE_NAME = 'airports.dat'
    URL_ROUTES_FILE_NAME = 'routes.dat'
    
    DATA_CSV_FLIGHT_DETAILS_FORMAT = ["id", "name", "city", "country", "iata", "icao", "lat", "lon", "alt", "timezone", "dst", "tz", "type", "source"]
    DATA_CSV_FLIGHT_DETAILS_COLUMNS = [1, 2, 3, 4]  
    DATA_CSV_ROUTES_DETAILS_FORMAT = ["airline", "airline_id", "source_airport", "source_airport_id", "destination_airport", "destination_airport_id", "codeshare", "stops", "equipment"]
    DATA_CSV_ROUTES_DETAILS_COLUMNS = [2, 4] 
    
class HTTP:
    OK = 200
    NOT_FOUND = 404
    BAD_REQUEST = 400
    INTERNAL_SERVER_ERROR = 500