import pandas as pd
import networkx as nx
import pickle
import os
import requests
from constants import FlightScraper as FS
import Pickler
from functools import lru_cache

class SearchFlights:
    def __init__(self, fetch_from_web=True, airports_url=FS.URL_AIRPORTS, routes_url=FS.URL_ROUTES, graph_pkl_file='Pickles/airport_flight_graph.pkl', airports_pkl_file='Pickles/airports_data.pkl', routes_pkl_file = 'Pickles/routes_data.pkl'):
        """ Tool to search for optimal paths between cities """
        self.airports_url = airports_url
        self.routes_url = routes_url
        
        self.graph_pkl_file = graph_pkl_file
        self.airports_pkl_file = airports_pkl_file
        self.routes_pkl_file = routes_pkl_file
        
        self.graph, self.city_translator = self._load_or_build_graph(fetch_from_web)

    def _load_or_build_graph(self, fetch_from_web):
        """Loads the graph from pickle if available; otherwise, fetches or builds it."""
        # Load up the data - either from pickle or download it from scratch
        self.airports_df = self._fetch_airports_data(fetch_from_web)
        self.routes_df = self._fetch_routes_data(fetch_from_web)
        
        # verify the data is legit
        if self.airports_df is None or self.routes_df is None:
            raise Exception("Link is not working, no pkl stored")
        
        # check to see if we need to build the graph at all
        graph = Pickler.load_pkl(self.graph_pkl_file)
        if graph is not None: return (graph, self._create_airport_city_map())
        
        # Build and save the graph
        graph = self._build_flight_graph()
        Pickler.store_pkl(self.graph_pkl_file, graph)
        return (graph, self._create_airport_city_map())

    def _fetch_airports_data(self, fetch_from_web):
        """ Downloads airport data and stores it as a pickle file. """
        return Pickler.conditionally_fetch_from_web(fetch_from_web, self.airports_pkl_file, self.airports_url, FS.DATA_CSV_FLIGHT_DETAILS_FORMAT, FS.DATA_CSV_FLIGHT_DETAILS_COLUMNS)
        
    def _fetch_routes_data(self, fetch_from_web):
        """ Downloads route data between airports and stores it as a pickle file. """
        return Pickler.conditionally_fetch_from_web(fetch_from_web, self.routes_pkl_file, self.routes_url, FS.DATA_CSV_ROUTES_DETAILS_FORMAT, FS.DATA_CSV_ROUTES_DETAILS_COLUMNS)

    def _build_flight_graph(self):
        """ build a graph from the routes dataframe """
        graph = nx.DiGraph()
        for _, row in self.routes_df.iterrows():
            source, destination = row["source_airport"], row["destination_airport"]
            if source and destination and source != "\\N" and destination != "\\N":
                graph.add_edge(source, destination)
        return graph

    def find_path_between_cities(self, src_city, dst_city):
        """ optimal flight path finder between two cities """
        # Translate cities to IATA codes using the loaded airports_df
        src_airports = self.airports_df[self.airports_df["city"].str.lower() == src_city.lower()]["iata"].tolist()
        dst_airports = self.airports_df[self.airports_df["city"].str.lower() == dst_city.lower()]["iata"].tolist()

        # Check if we found any airports for the given cities
        if not src_airports:
            return f"No airports found for source city '{src_city}'."
        if not dst_airports:
            return f"No airports found for destination city '{dst_city}'."
        
        # Try to find the shortest path using all combinations of source and destination airports
        shortest_path = None
        for src in src_airports:
            for dst in dst_airports:
                try:
                    path = nx.shortest_path(self.graph, source=src, target=dst)
                    # If no shortest path has been found yet or the new path is shorter, update
                    if shortest_path is None or len(path) < len(shortest_path):
                        shortest_path = path
                except nx.NetworkXNoPath:
                    continue  # Try the next combination of source and destination
                except nx.NodeNotFound:
                    continue  # Skip invalid airports if any

        if shortest_path is not None:
            return shortest_path
        return f"No valid path found between the cities {src_city} and {dst_city}."

    def find_path_between_multiple_cities(self, cities):
        """ Finds an optimal flight path through multiple cities using airport connections. """
        if len(cities) < 2:
            return "At least two cities are required to find a path."

        # Convert city names to airport IATA codes
        city_airports = {
            city: self.airports_df[self.airports_df["city"].str.lower() == city.lower()]["iata"].tolist()
            for city in cities
        }

        # Check for missing airports
        for city, airports in city_airports.items():
            if not airports:
                cities.remove(city)

        # The graph search begins
        total_path = []
        total_distance = 0
        
        for i in range(len(cities) - 1): 
            # for each of the city pairs...
            src_city, dst_city = cities[i], cities[i + 1]
            src_airports, dst_airports = city_airports[src_city], city_airports[dst_city]

            shortest_path, shortest_distance = None, float('inf')

            # Pick shortest airport path between the two cities
            for src in src_airports:
                for dst in dst_airports:
                    path, distance = self._get_shortest_airport_path(src, dst)
                    if distance < shortest_distance:
                        shortest_path, shortest_distance = path, distance

            # extend the stored path
            if shortest_path:
                total_path.extend(shortest_path if not total_path else shortest_path[1:])
                total_distance += shortest_distance
            else:
                return f"No valid path found between {src_city} and {dst_city}."

        # now format it as [... { city : [ airports to reach this city ] } ...]
        path_dict = {}
        airport_cities = []
        last_idx = 0
        for idx, path in enumerate(total_path):
            found_city = self.city_translator[path]
            if found_city in cities:
                path_dict.update({ found_city : total_path[last_idx:idx] })
                last_idx = idx
                
        return path_dict

    @lru_cache(maxsize=None) # Memoization on the shortest path between any two airports
    def _get_shortest_airport_path(self, src, dst):
        """Returns the shortest path and distance between two airports via memoization."""
        try:
            path = nx.shortest_path(self.graph, source=src, target=dst, weight='distance')
            distance = sum(self.graph[u][v].get('distance', 1) for u, v in zip(path, path[1:]))
            return path, distance
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None, float('inf')
    
    def _create_airport_city_map(self):
        """Directly maps airport IATA codes to their respective city names using preloaded pickle data."""
        return {row["iata"]: row["city"] for _, row in self.airports_df.iterrows()}
        
if __name__ == "__main__":
    # Initialize the SearchFlights
    graph = SearchFlights()

    # Test the multiple cities path finding
    cities = ["New York", "Los Angeles", "Chicago", "Miami", "Delhi", "Tokyo", "Philadelphia"]
    path = graph.find_path_between_multiple_cities(cities)
    print(f"Path between {cities}: {path}")
