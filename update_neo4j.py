import pandas as pd
import ast
from py2neo import Graph

# -------------------------------------------
# 1. Connect to your Neo4j Instance
# -------------------------------------------
graph = Graph("bolt://localhost:7687", auth=("neo4j", "EasyP@ss71"))

# -------------------------------------------
# 2. Remove the Existing Graph Model
# -------------------------------------------
print("Clearing existing graph data...")
graph.run("MATCH (n) DETACH DELETE n")
print("Graph data cleared.")

# -------------------------------------------
# 3. Load CSV Data from File
# -------------------------------------------
csv_file_path = "data/processed_aviation_data.csv"  # Update this with your CSV filepath
df = pd.read_csv(csv_file_path)

# -------------------------------------------
# 4. Process Each Row and Build the Advanced Model
# -------------------------------------------
for idx, row in df.iterrows():
    # --- Create or Merge the Accident Node ---
    accident_query = """
    MERGE (a:Accident {Oid: $oid})
      SET a.NtsbNumber    = $ntsbNumber,
          a.EventDate     = datetime($eventDate),
          a.HighestInjury = $highestInjury,
          a.ProbableCause = $probableCause
    RETURN a
    """
    accident_params = {
        "oid": row["Oid"],
        "ntsbNumber": row["NtsbNumber"],
        "eventDate": row["EventDate"],
        "highestInjury": row["HighestInjury"],
        "probableCause": row["ProbableCause"],
    }
    graph.run(accident_query, accident_params)

    # --- Create or Merge the Topic Node (Probable Cause Label) ---
    topic_query = """
    MATCH (a:Accident {Oid: $oid})
    MERGE (t:Topic {TopicID: $topicId})
      SET t.TopicName = $topicName
    MERGE (a)-[:HAS_PROBABLE_CAUSE]->(t)
    """
    topic_params = {
        "oid": row["Oid"],
        "topicId": row["TopicID"],
        "topicName": row["TopicName"]
    }
    graph.run(topic_query, topic_params)

    # --- Create or Merge the Location Node ---
    location_query = """
    MATCH (a:Accident {Oid: $oid})
    MERGE (l:Location {City: $city, State: $state, Country: $country})
      SET l.Latitude  = toFloat($latitude),
          l.Longitude = toFloat($longitude)
    MERGE (a)-[:OCCURRED_AT]->(l)
    """
    location_params = {
        "oid": row["Oid"],
        "city": row["City"],
        "state": row["State"],
        "country": row["Country"],
        "latitude": row["Latitude"],
        "longitude": row["Longitude"]
    }
    graph.run(location_query, location_params)

    # --- Parse and Create/Merge Vehicle Nodes ---
    # The Vehicles field is a string that looks like:
    # "[{'VehicleNumber': 1, 'DamageLevel': 'Substantial', ...}]"
    try:
        vehicles_list = ast.literal_eval(row["Vehicles"])
    except Exception as e:
        print(f"Error parsing Vehicles for accident Oid {row['Oid']}: {e}")
        vehicles_list = []
    
    for vehicle in vehicles_list:
        vehicle_query = """
        MATCH (a:Accident {Oid: $oid})
        MERGE (v:Vehicle {SerialNumber: $serialNumber})
          SET v.VehicleNumber    = $vehicleNumber,
              v.Make             = $make,
              v.Model            = $model,
              v.AircraftCategory = $aircraftCategory,
              v.OperatorName     = $operatorName
        MERGE (a)-[:INVOLVED_VEHICLE]->(v)
        """
        vehicle_params = {
            "oid": row["Oid"],
            "vehicleNumber": vehicle.get("VehicleNumber"),
            "make": vehicle.get("Make"),
            "model": vehicle.get("Model"),
            "serialNumber": vehicle.get("SerialNumber"),
            "aircraftCategory": vehicle.get("AircraftCategory"),
            "operatorName": vehicle.get("OperatorName")
        }
        graph.run(vehicle_query, vehicle_params)

print("Advanced NTSB data loaded into Neo4j successfully.")
