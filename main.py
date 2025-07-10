from datetime import datetime

from pathlib import Path
import uuid
import networkx as nx
import pm4py
from pandas.core.interchange.dataframe_protocol import DataFrame
import xmlplot
from collections import defaultdict
from pm4py import OCEL

from totemAlgorithm import mine_totem

TIEBREAKER_LIST = list()

def main():
    ocel = read_ocel()
    if ocel is None:
        return

    xmlplot.generate_object_ids(ocel)
    set_tiebreaker(ocel)
    object_lifecycles = generate_lifecycles(ocel)
    totem = mine_totem(ocel, 1)
    fragments = apply_fragmentation(ocel, totem)
    fragment_logs = flatten_save_fragments(ocel, fragments)
    xmlplot.finish_bpmn(ocel, fragments, totem, object_lifecycles)
    return

def flatten_save_fragments(ocel, fragments):
    directory = Path('generated_data/fragments')
    for file in directory.iterdir():
        if file.is_file():
            file.unlink()

    ocel_object_types = ocel.objects[ocel.object_type_column].values.unique()
    flattened_fragment_logs = defaultdict(DataFrame)

    # Create a lookup for event ID -> activity type
    eid_to_activity = {}
    for index, row in ocel.events.iterrows():
        eid_to_activity[row['ocel:eid']] = row['ocel:activity']

    # Flatten the log and save as bpmn
    for object_type in ocel_object_types:
        if object_type not in fragments.keys():
            print(f"No Actvities belong to the '{object_type}'.")
            continue

        # Filter the fragments log for the activities
        flattened_ocel = pm4py.ocel_flattening(ocel, object_type)

        # Mask that returns true if the activity with instance ocel:eid belongs to the fragment of the current object_type
        fragment_mask = flattened_ocel['ocel:eid'].map(eid_to_activity).isin(fragments[object_type])

        print(f"'{object_type}' fragment initial length: {len(flattened_ocel.index)}")
        flattened_ocel = flattened_ocel[fragment_mask]
        print(f"'{object_type}' fragment length after fragmentation filtering: {len(flattened_ocel.index)}")

        flattened_fragment_logs[object_type] = flattened_ocel
        bpmn = pm4py.discover_bpmn_inductive(flattened_ocel)

        # removes start and end nodes
        end_nodes = set(filter(filter_end, bpmn.get_nodes()))
        for end_node in end_nodes:
            bpmn.remove_node(end_node)
        start_nodes = set(filter(filter_start, bpmn.get_nodes()))
        for start_node in start_nodes:
            bpmn.remove_node(start_node)

        pm4py.write_bpmn(bpmn, f"generated_data/fragments/bpmn_{object_type}.bpmn")
        pm4py.save_vis_bpmn(bpmn, f"generated_data/fragments/bpmn_{object_type}.png")

    return flattened_fragment_logs

def filter_end(node):
    if str(node).__contains__('@end'):
        return True
    return False

def filter_start(node):
    if str(node).__contains__('@start'):
        return True
    return False

# find the timestamp for the first occurrence of each oid
def first_occurrence(ocel):
    first_object_occurrence : dict[str, datetime] = dict()

    for index, row in ocel.object_changes.iterrows():
        if row['ocel:oid'] not in first_object_occurrence:
            first_object_occurrence[row['ocel:oid']] = row['ocel:timestamp'].tz_localize(None)
            continue

        if first_object_occurrence[row['ocel:oid']] > row['ocel:timestamp'].tz_localize(None):
            first_object_occurrence[row['ocel:oid']] = row['ocel:timestamp'].tz_localize(None)

    return first_object_occurrence

def get_sorted_oid_paths(ocel):
    object_paths_new : dict[str, list[tuple[str, datetime]]] = dict()
    object_paths : dict[str, list[str]] = dict()

    for index, row in ocel.relations.iterrows():
        if row['ocel:oid'] not in object_paths_new:
            object_paths_new[row['ocel:oid']] = [(row['ocel:activity'], row['ocel:timestamp'].tz_localize(None))]
            continue
        object_paths_new[row['ocel:oid']].append((row['ocel:activity'], row['ocel:timestamp'].tz_localize(None)))

    for key in object_paths_new:
        object_paths_new[key].sort(key=lambda x: x[1])
        object_paths[key] = [path for path, _ in object_paths_new[key]]
        object_paths[key].insert(0, 'new')

    return object_paths

def build_graph(paths):
    graph = defaultdict(set)
    for path in paths:
        for i in range(len(path) - 1):
            graph[path[i]].add(path[i + 1])
    return {node: list(neighbors) for node, neighbors in graph.items()}

def create_nx_graph(graph_dict):
    G = nx.DiGraph()
    for node, neighbors in graph_dict.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)
    return G

def generate_lifecycles(ocel):
    object_id_to_type : dict[str, str] = dict()
    object_paths = get_sorted_oid_paths(ocel)

    for index, row in ocel.objects.iterrows():
        object_id_to_type[row['ocel:oid']] = row['ocel:type']

    type_to_paths = defaultdict(list)
    for obj_id, path in object_paths.items():
        obj_type = object_id_to_type[obj_id]
        type_to_paths[obj_type].append(path)

    # transforms the different state paths to graph edge pairs
    type_to_graph = {obj_type: build_graph(paths) for obj_type, paths in type_to_paths.items()}
    # builds the graphs from the edges
    nx_graphs = {obj_type: create_nx_graph(graph) for obj_type, graph in type_to_graph.items()}

    return nx_graphs

def get_all_activities_without_objects(ocel: OCEL):
    activities = set(ocel.events['ocel:activity'])
    for index, row in ocel.relations.iterrows():
        if row['ocel:activity'] in activities:
            activities.remove(row['ocel:activity'])
    return activities

def get_convergent(ocel):
    # Mapping from event type to set of convergent object types
    event_type_to_convergent = defaultdict(set)
    # Temporary mapping from event ID to counts of each object type in that event
    event_obj_counts = defaultdict(lambda: defaultdict(int))

    # Count object occurrences per event
    for index, row in ocel.relations.iterrows():
        eid = row['ocel:eid']
        obj_type = row['ocel:type']
        event_obj_counts[eid][obj_type] += 1

    # Create a lookup for event ID -> activity type
    eid_to_activity = {}
    for index, row in ocel.events.iterrows():
        eid_to_activity[row['ocel:eid']] = row['ocel:activity']

    # Identify convergence: event types with object types that appear multiple times in an event
    for eid, type_counts in event_obj_counts.items():
        activity_type = eid_to_activity.get(eid)
        if activity_type is None:
            continue
        for obj_type, count in type_counts.items():
            if count > 1:
                # This event had multiple objects of this type -> convergence detected
                event_type_to_convergent[activity_type].add(obj_type)

    return event_type_to_convergent

def get_divergent(ocel):
    # Mapping from object type to set of divergent activity types
    object_type_to_divergent = defaultdict(set)
    # Temporary mapping from object ID to a set of event IDs for each activity type
    object_activity_events = defaultdict(lambda: defaultdict(set))

    # Gather events per object per activity type
    for index, row in ocel.relations.iterrows():
        oid = row['ocel:oid']
        activity = row['ocel:activity']
        eid = row['ocel:eid']
        object_activity_events[oid][activity].add(eid)

    # Lookup for object ID -> object type
    oid_to_type = {}
    for index, row in ocel.objects.iterrows():
        oid_to_type[row['ocel:oid']] = row['ocel:type']

    # Identify divergence: object types with activity types that appear multiple times for the same object
    for oid, activities in object_activity_events.items():
        obj_type = oid_to_type.get(oid)
        if obj_type is None:
            continue
        for activity_type, event_ids in activities.items():
            if len(event_ids) > 1:
                # Object `oid` is involved in multiple distinct events of type `activity_type`
                object_type_to_divergent[activity_type].add(obj_type)

    return object_type_to_divergent

def get_activity_objects(ocel):
    # Mapping from event type to set of object types
    activity_objects = defaultdict(set)

    # Iterate through every row and add the occurring object types to the activity type
    for index, row in ocel.relations.iterrows():
        activity = row['ocel:activity']
        obj_type = row['ocel:type']
        activity_objects[activity].add(obj_type)

    return activity_objects

def get_unique_activities(activity_objects):
    # Identify activities with 1 object type
    unique_activities = defaultdict(str)
    for activity_type, objects in activity_objects.items():
        if len(objects) == 1:
            unique_activities[activity_type] = (next(iter(objects)))

    return unique_activities

def apply_fragmentation(ocel, totem):
    activities_without_objects = get_all_activities_without_objects(ocel)
    convergent_activities = get_convergent(ocel)
    divergent_objects = get_divergent(ocel)
    activity_objects = get_activity_objects(ocel)
    unique_activities = get_unique_activities(activity_objects)
    all_activities = set(ocel.events['ocel:activity'])

    # Map each object type (fragment) to all occurring activities within the fragment
    fragment_activities = defaultdict(set)

    # Add activities that interact with only 1 object type
    for activity, object_type in unique_activities.items():
        print(f"Activity '{activity}' was added to {object_type} fragment because it is the only object type interaction.")
        fragment_activities[object_type].add(activity)

    for activity in all_activities:
        # Check if activity interacts with an object
        if activity in activities_without_objects:
            print(f"Activity '{activity}' doesn't interact with any objects.")
            continue

        # This case is already handled above
        if activity in unique_activities.keys():
            continue

        # All object types that are connected to the activity
        connected_object_types = activity_objects[activity]

        # Safe convergent and divergent state of each object type as boolean tuple (is_convergent, is_divergent)
        con_div_check = defaultdict(tuple)

        # Counter how many tuples are (false, false)
        not_conv_not_div_counter = 0
        # Counter how many tuples are (false, true)
        only_divergent_counter = 0

        # Populate con_div_check and counter
        for object_type in connected_object_types:
            is_convergent = convergent_activities[activity].__contains__(object_type)
            is_divergent = divergent_objects[activity].__contains__(object_type)
            con_div_check[object_type] = (is_convergent, is_divergent)
            print(f"Object type '{object_type}' is '({is_convergent, is_divergent})'.")
            # Increment (false, false) counter
            if not is_convergent and not is_divergent:
                not_conv_not_div_counter += 1
            # Increment (false, true) counter
            if not is_convergent and is_divergent:
                only_divergent_counter += 1

        if not_conv_not_div_counter == 1:
            # Find the (false, false) tuple
            for object_type, checks in con_div_check.items():
                if checks[0] or checks[1]:
                    # Not the (false, false) tuple
                    continue

                # Found the object type that isn't convergent/ divergent
                fragment_activities[object_type].add(activity)
                print(f"Activity '{activity}' was added to the '{object_type}' fragment by 'not_conv_not_div_counter == 1'.")
                # Stop the search
                break
            # Continue with the next activity
            continue

        if not_conv_not_div_counter > 1:
            tiebreaker_fragment = apply_tiebreaker(connected_object_types)
            fragment_activities[tiebreaker_fragment].add(activity)
            print(f"Activity '{activity}' was added to the '{tiebreaker_fragment}' fragment by 'not_conv_not_div_counter > 1' tiebreaker.")
            continue

        if only_divergent_counter == 1:
            # Find the (false, true) tuple
            for object_type, checks in con_div_check.items():
                if checks[0] or not checks[1]:
                    # Not the (false, true) tuple
                    continue

                # Found the object type that is only divergent
                fragment_activities[object_type].add(activity)
                print(f"Activity '{activity}' was added to the '{object_type}' fragment by 'only_divergent_counter == 1'.")
                break
            continue

        if only_divergent_counter == 2:
            object_type_1 = None
            object_type_2 = None
            for object_type, checks in con_div_check.items():
                if checks[0] or not checks[1]:
                    # Not the (false, true) tuple
                    continue

                if object_type_1 is None:
                    object_type_1 = object_type
                    continue

                object_type_2 = object_type
                break

            # Check the UML Diagram for relationship between both object types
            found_relationship_flag = False

            # Check for one-to-one relationship
            one_to_one = get_one_to_one_mapping(totem)
            for relationship in one_to_one:
                # Didn't find the one-to-one tuple yet
                if not (relationship.__contains__(object_type_1) and relationship.__contains__(object_type_2)):
                    continue

                # Found a one-to-one relationship
                tiebreaker_fragment = apply_tiebreaker(connected_object_types)
                fragment_activities[tiebreaker_fragment].add(activity)
                print(f"Activity '{activity}' was added to the '{tiebreaker_fragment}' fragment by 'one_to_one' tiebreaker.")
                found_relationship_flag = True

            if found_relationship_flag:
                continue

            # Check for one-to-many relationship
            one_to_many = get_one_to_many_mapping(totem)
            for relationship in one_to_many:
                # Didn't find the one-to-many tuple yet
                if not (relationship.__contains__(object_type_1) and relationship.__contains__(object_type_2)):
                    continue

                # Found a one-to-many relationship
                # Activity belongs to the many side of the relationship
                fragment_activities[relationship[1]].add(activity)
                print(f"Activity '{activity}' was added to the '{relationship[1]}' fragment by 'one_to_many'.")
                found_relationship_flag = True

            if found_relationship_flag:
                continue

            # Check for many-to-many relationship
            many_to_many = get_many_to_many_mapping(totem)
            for relationship in many_to_many:
                # Didn't find the many-to-many tuple yet
                if not (relationship.__contains__(object_type_1) and relationship.__contains__(object_type_2)):
                    continue
                # (object_type1, object_type2, (cardinalities))
                # Found a many-to-many relationship
                # Check if both have 0 as lower boundary
                if relationship[2][0].__contains__('0') and relationship[2][1].__contains__('0'):
                    # Tiebreaker will be performed later
                    continue

                # Find the single lower bound
                if relationship[2][0].__contains__('0'):
                    fragment_activities[relationship[0]].add(activity)
                    print(f"Activity '{activity}' was added to the '{relationship[0]}' fragment by 'many-to-many'.")
                    found_relationship_flag = True
                    continue

                # Find the single lower bound
                if relationship[2][1].__contains__('0'):
                    fragment_activities[relationship[1]].add(activity)
                    print(f"Activity '{activity}' was added to the '{relationship[1]}' fragment by 'many-to-many'.")
                    found_relationship_flag = True
                    continue

            if found_relationship_flag:
                continue


        # if only_divergent_counter > 2:
        # The rest is tiebreaker
        tiebreaker_fragment = apply_tiebreaker(connected_object_types)
        fragment_activities[tiebreaker_fragment].add(activity)
        print(f"Activity '{activity}' was added to the '{tiebreaker_fragment}' fragment by 'final' tiebreaker.")

    return fragment_activities


# Returns tuples in which the first key is the '1' cardinality
def get_one_to_many_mapping(totem: dict[str, dict[str, tuple[str, str]]]):
    result = set()
    for outer_key, inner_dict in totem.items():
        for inner_key, value in inner_dict.items():
            if value in {('1', '0..*'), ('1', '1..*')}:
                result.add((outer_key, inner_key))
            elif value in {('0..*', '1'), ('1..*', '1')}:
                result.add((inner_key, outer_key))

    return result

def get_one_to_one_mapping(totem: dict[str, dict[str, tuple[str, str]]]):
    result = {
        (outer_key, inner_key)
        for outer_key, inner_dict in totem.items()
        for inner_key, value in inner_dict.items()
        if value in {("1", "1"), ("1", "0..1"), ("0..1", "1")}
    }
    return result

# returns outer_key, inner_key, tuple_relationship
def get_many_to_many_mapping(totem: dict[str, dict[str, tuple[str, str]]]):
    result = {
        (outer_key, inner_key, value)
        for outer_key, inner_dict in totem.items()
        for inner_key, value in inner_dict.items()
        if value in {('0..*', '0..*'), ('1..*', '1..*'), ('0..1', '0..1'),
                     ('0..*', '1..*'), ('0..*', '0..1'), ('1..*', '0..1'),
                     ('1..*', '0..*'), ('0..1', '0..*'), ('0..1', '1..*')}
    }
    return result

# Uses the tiebreaker list
# Takes a list of all objects connected to an activity
# Returns fragment as object_type
def apply_tiebreaker(activity_objects):
    for object_type in TIEBREAKER_LIST:
        if activity_objects.__contains__(object_type):
            return object_type

def set_tiebreaker(ocel):
    global TIEBREAKER_LIST
    TIEBREAKER_LIST = list(ocel.objects['ocel:type'].values.unique())
    print("Sometimes there isn't a single right way to assign activities to fragments.")
    print("In these cases this code uses a tiebreaker (priority list) to which fragments an activity shall be assigned.")
    print(f"The current tiebreaker list is:")
    for index, object_type in enumerate(TIEBREAKER_LIST):
        print(f"{index}: {object_type}")
    print("If you want to keep this list, enter 'Y'.")
    print("If you want to change this list, enter the new sequence of index numbers, seperated by commas.")
    while True:
        user_input = input("Your action: ")
        user_input = user_input.strip()
        if user_input == 'Y' or user_input == 'y':
            return
        try:
            number_list = [int(num.strip()) for num in user_input.split(',')]

            # Check if provided list is correct length
            if set(number_list) != set(range(len(TIEBREAKER_LIST))):
                raise ValueError(f"Indices must be a permutation of the provided indices.")

            reordered = [None] * len(TIEBREAKER_LIST)
            for new_index, old_index in enumerate(number_list):
                reordered[new_index] = TIEBREAKER_LIST[old_index]

            TIEBREAKER_LIST = reordered
            print("Your selected tiebreaker list is:", TIEBREAKER_LIST)
            return
        except ValueError:
            # Reset tiebreaker if something went wrong
            TIEBREAKER_LIST = list(ocel.objects['ocel:type'].values.unique())
            print("Invalid input. Please enter only numbers separated by commas or 'Y'.")

def read_ocel():
    print("Enter the path to the OCEL event log")
    while True:
        user_input = input("Path: ")
        user_input = user_input.strip()

        try:
            ocel = pm4py.read_ocel2(user_input)
        except Exception:
            try:
                ocel = pm4py.read_ocel(user_input)
            except Exception:
                print("Invalid input, exciting program")
                return None
        return ocel

def generate_object_ids(ocel):
    object_name_to_id = dict()
    object_types = ocel.objects[ocel.object_type_column].values.unique()

    for object_type in object_types:
        object_name_to_id[object_type] = f"Object_{uuid.uuid4().hex[:24]}"

    return object_name_to_id

if __name__ == '__main__':
    main()