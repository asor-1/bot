#python osm_OpenMind.py
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import math, json

def haversine_form(latitude_1, longitude_1,latitude_2, longitude_2): 
    earth_Radius = 3958  # Radius of the Earth in miles
    #just the haversine formula
    d_latitude = math.radians(latitude_2 - latitude_1)
    d_longitude = math.radians(longitude_2 - longitude_1)
    intermediate_1 = math.sin(d_latitude/ 2) ** 2 + math.cos(math.radians(latitude_1)) * math.cos(math.radians(latitude_2)) * math.sin(d_longitude/ 2) ** 2
    intermediate_2 = 2 * math.atan2(math.sqrt(intermediate_1), math.sqrt(1 - intermediate_1))
    #kept this as distance so that for future reference I understand that it is returning the distance and I dont have to go through the code again
    distance = earth_Radius * intermediate_2
    return distance

#add json file
def finding_closest(lat, long):
    with open("locations.json") as j:
        places = json.load(j)

    closest_location = None 

    min = 1000000 #can't be this close in miles

    #there has to be a better algorithm
    for location in places: 
        dist = haversine_form(lat, long, location["latitude"], location["longitude"])
        if dist < min:
            min = dist
            closest_location = location
    #dictionary because I need to access different parts of the return
    closest_location_info = {
        "toString": f"The closest psychiatric facility to you is: {closest_location['name']}, which is {min:.2f} miles away.",
        "latitude": closest_location["latitude"],
        "longitude": closest_location["longitude"],
        "name": closest_location["name"],
        "distance_from_user": min
    }
    return closest_location_info

# I realized I didn't need this function 
def insert(direct, input, positioning_input):
    return direct[:positioning_input] + input+ direct[positioning_input:]

def get_osm_graph(place_name):
    g = ox.graph_from_place(place_name, network_type='drive')
    return g

'''def plot_graph(g):
    ox.plot_graph(g)'''

# I want to write this function without needing ox. It is a variation of Dijkstra's algorithm
# Finds the closest route from the user to the facility
def dijkstra_Path(graph, user, destination):
    u_node = ox.distance.nearest_nodes(graph, user[1], user[0])
    d_node = ox.distance.nearest_nodes(graph, destination[1], destination[0])

    #print(f"Origin node: {o_node}, Destination node: {d_node}")

    path = nx.shortest_path(graph, u_node, d_node, weight='length')
    return path

# The proble with this function is that "d" has repeats in it. 
def finding_Directions(g, route):
    d = []
    street_names = []
    turning_Directions = []
    for i in range(len(route) - 1):
        a = route[i]
        b = route[i+1]
        
        if i > 1: 
            y = route[i-1]
            z = route[i]

            prev_e = g.get_edge_data(y, z)[0]

            #finding where the user should be heading
            previously_heading = prev_e.get("heading", 0)
            e_data_1 = g.get_edge_data(a, b)[0]
            currently_heading = e_data_1.get("heading", 0)

            # there is something wrong with the turning logic.
            difference = currently_heading - previously_heading
            if difference > 30:
                turn = "Turn right onto"
            elif difference < -30:
                turn = "Turn left onto"
            else:
                turn = "Continue on"

            turning_Directions.append(turn)
        
        #Getting edge data so we can find the street
        e_data_2 = g.get_edge_data(a, b)[0]
        s_name = e_data_2.get('name', 'Unnamed Road')
        street_names.append(s_name)
        l = e_data_2["length"]
        # Convert meters to feet
        l = l * 3.28 
        direction = f" {s_name} for {l:.2f} feet."
        d.append(direction)

        d_new= []
        for turning, direct in zip(turning_Directions, d):
            new_direction = insert(direct, turning, 0)
            d_new.append(new_direction)  
    return d_new

#place_name = "Chicago, Illinois, USA"
#graph = get_osm_graph(place_name)
#plot_graph(graph)

# Example locations (latitude, longitude)
origin = (41.8789, -87.6359)  # Sears Tower
destination = (41.898773, -87.622925)  # Navy Pier

# the main method of this file
def final_Directions(user_latitude, user_longitude, place_name = 'Chicago, Illinois, USA'):
    closest_Location = finding_closest(user_latitude, user_longitude)
    closest_Location_lat = closest_Location["latitude"]
    closest_Location_long = closest_Location["longitude"]
    closest_Location_toString = closest_Location["toString"]

    # I want to change the graph model
    graph = get_osm_graph(place_name)

    try:
        route = dijkstra_Path(graph, (user_latitude, user_longitude), (closest_Location_lat, closest_Location_long))
        
        directions = finding_Directions(graph, route)
        for d in directions:
            print(d)
        
        print(closest_Location_toString)
    
    # Chatgpted Exceptions, I have no idea how they work
    except nx.NetworkXNoPath:
        print(f"No path between the specified nodes.")
    except Exception as e:
        print(f"An error occurred: {e}")


# testing purposes. 
#final_Directions(41.8789, -87.6359) #coordinates for Sears Tower in Chicago