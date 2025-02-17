#take the coordinates of the person and find the optimal path for getting the route from point A to point B
#maybe be useful:            https://huggingface.co/model


deliveries={
  "locations": [
    {"name": "Supermarket_A", "latitude": 40.730610, "longitude": -73.935242},
    {"name": "Supermarket_B", "latitude": 40.741895, "longitude": -73.989308},
    {"name": "Supermarket_C", "latitude": 40.748817, "longitude": -73.985428},
    {"name": "User_1", "latitude": 40.755783, "longitude": -73.978797},
    {"name": "User_2", "latitude": 40.764800, "longitude": -73.980000},
    {"name": "User_3", "latitude": 40.758896, "longitude": -73.985130}
  ]
}


import heapq
import math

class Location:
    def __init__(self, name, latitude, longitude):
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
    
    def __repr__(self):
        return f"{self.name} ({self.latitude}, {self.longitude})"
    
    def distance_to(self, other):
        # Euclidean distance between two locations, simplified for the sake of the example
        return math.sqrt((self.latitude - other.latitude)**2 + (self.longitude - other.longitude)**2)

class AStarPathfinding:
    def __init__(self, locations):
        self.locations = {loc.name: loc for loc in locations}
        self.edges = {}  # Graph representation
    
    def add_edge(self, start, end, distance):
        if start not in self.edges:
            self.edges[start] = []
        self.edges[start].append((end, distance))
    
    def a_star(self, start_name, goal_name):
        # Initialize the open list (priority queue) and closed set
        # TODO A* Logic
        
        return None  # No path found
    
    def heuristic(self, loc1, loc2):
        # TODO find nice hueristic
        return None
    def reconstruct_path(self, came_from, current):
        # TODO construct the path.
        return None
       


# # Sample usage

# locations = [
#     Location("Supermarket_A", 40.730610, -73.935242),
#     Location("Supermarket_B", 40.741895, -73.989308),
#     Location("Supermarket_C", 40.748817, -73.985428),
#     Location("User_1", 40.755783, -73.978797),
#     Location("User_2", 40.764800, -73.980000),
#     Location("User_3", 40.758896, -73.985130)
# ]

# # Create the A* pathfinding instance
# pathfinder = AStarPathfinding(locations)

# # Add connections (in real-world, distances can be dynamically fetched based on the road network)
# pathfinder.add_edge("Supermarket_A", "User_1", 2.0)
# pathfinder.add_edge("Supermarket_A", "User_2", 4.5)
# pathfinder.add_edge("Supermarket_B", "User_1", 1.2)
# pathfinder.add_edge("Supermarket_B", "User_3", 1.8)
# pathfinder.add_edge("Supermarket_C", "User_2", 1.5)
# pathfinder.add_edge("Supermarket_C", "User_3", 1.0)

# # Find the best path from Supermarket_B to User_1
# best_path = pathfinder.a_star("Supermarket_B", "User_1")
# print(f"Best path: {best_path}")