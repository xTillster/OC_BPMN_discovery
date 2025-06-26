import math

import networkx as nx
from lxml import etree as ET
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from pm4py import OCEL

import os
import zipfile

state_name_to_ids = defaultdict(lambda: defaultdict(str))
object_name_to_ids = dict()

def finish_bpmn(ocel: OCEL, connected_object_activity_graph: defaultdict[str, defaultdict[str, set]], fragments: defaultdict[str, set]):
    '''
    # Generate the output data objects first so that they may be reused for input again
    # Remove start and end events
    for activity, values in connected_object_activity_graph.items():
        for object_type, states in values.items():
            add_data_object_to_bpmn(f'./generated_data/flattened/bpmn_{object_type}.bpmn', activity, f'{object_type}\n[{activity}]', input=False)
            remove_start_and_end_events(f'./generated_data/flattened/bpmn_{object_type}.bpmn')

    # Generate input data objects
    for activity, values in connected_object_activity_graph.items():
        for object_type, states in values.items():
            states_name = ""
            if len(states) == 0:
                states_name = "[new shouldn't be]"
            else:
                states_name = "[" + " | ".join(states) + "]"

            add_data_object_to_bpmn(f'./generated_data/flattened/bpmn_{object_type}.bpmn', activity, f'{object_type}\n{states_name}', input=True)
    '''
    input_count = defaultdict(int)
    output_count = defaultdict(int)

    #reverse key-value of fragment list to match activity to object type
    reversed_fragments = defaultdict(str)
    for key, values in fragments.items():
        for value in values:
            reversed_fragments[value] = key
    print(reversed_fragments)
    # Generate the output data objects first so that they may be reused for input again
    # Remove start and end events
    for activity, values in connected_object_activity_graph.items():
        for object_type, states in values.items():
            if len(reversed_fragments[activity]) == 0:
                print("can this even happen?")
                continue
            output_count[activity] += 1
            add_data_object_to_bpmn(f'./generated_data/flattened/bpmn_{reversed_fragments[activity]}.bpmn',
                                    activity,
                                    f'{object_type}\n[{activity}]',
                                    output_count[activity],
                                    dataclass_name=object_type,
                                    states_list=states,
                                    input=False)
            # not every object type has his own fragment
            if object_type in fragments.keys():
                remove_start_and_end_events(f'./generated_data/flattened/bpmn_{object_type}.bpmn')

    # Generate input data objects
    for activity, values in connected_object_activity_graph.items():
        for object_type, states in values.items():
            states_name = ""
            if len(states) == 0:
                states_name = "[new shouldn't be]"
            else:
                states_name = "[" + " | ".join(states) + "]"
            input_count[activity] += 1
            add_data_object_to_bpmn(f'./generated_data/flattened/bpmn_{reversed_fragments[activity]}.bpmn',
                                    activity,
                                    f'{object_type}\n{states_name}',
                                    input_count[activity],
                                    dataclass_name=object_type,
                                    states_list=states,
                                    input=True)


def add_data_object_to_bpmn_old(file_path, activity_name, data_object_name, in_out_counter, input=True, output_path=None):
    parser = ET.XMLParser(remove_blank_text=True)
    tree = ET.parse(file_path, parser)
    root = tree.getroot()
    nsmap = root.nsmap
    bpmn_ns = nsmap.get('bpmn')
    bpmndi_ns = nsmap.get('bpmndi')
    dc_ns = nsmap.get('omgdc')

    if not (bpmn_ns and bpmndi_ns and dc_ns):
        raise ValueError("Missing required BPMN/DI namespaces. Ensure 'bpmn', 'bpmndi', and 'omgdc' are defined.")

    # Find process
    process = root.find(f'.//{{{bpmn_ns}}}process')
    if process is None:
        raise ValueError("No <bpmn:process> element found.")

    # Find task
    task = None
    for t in process.findall(f'.//{{{bpmn_ns}}}task'):
        if t.get('name') == activity_name:
            task = t
            break
    if task is None:
        raise ValueError(f"No task with name '{activity_name}' found.")
    task_id = task.get('id')

    # Try to find existing dataObjectReference by name
    data_obj_ref = None
    for existing_ref in process.findall(f'.//{{{bpmn_ns}}}dataObjectReference'):
        if existing_ref.get('name') == data_object_name:
            data_obj_ref = existing_ref
            break

    # Reuse or create new one
    is_new_object = data_obj_ref is None

    if is_new_object:
        data_obj_id = f"DataObject_{uuid.uuid4().hex[:8]}"
        data_obj_ref_id = f"{data_obj_id}_ref"
        data_object = ET.Element(f"{{{bpmn_ns}}}dataObject", id=data_obj_id)
        data_obj_ref = ET.Element(
            f"{{{bpmn_ns}}}dataObjectReference",
            id=data_obj_ref_id,
            dataObjectRef=data_obj_id,
            name=data_object_name
        )
        process.append(data_object)
        process.append(data_obj_ref)
    else:
        data_obj_ref_id = data_obj_ref.get('id')
        #data_obj_id = data_obj_ref.get('dataObjectRef')

    data_assoc_id = f"DataAssoc_{uuid.uuid4().hex[:8]}"
    shape_id = f"{data_obj_ref_id}_di"

    # Create data association
    assoc_tag = 'dataInputAssociation' if input else 'dataOutputAssociation'
    data_assoc = ET.Element(f"{{{bpmn_ns}}}{assoc_tag}", id=data_assoc_id)

    if input:
        source_ref = ET.Element(f"{{{bpmn_ns}}}sourceRef")
        source_ref.text = data_obj_ref_id
        target_ref = ET.Element(f"{{{bpmn_ns}}}targetRef")
        target_ref.text = task_id
    else:
        source_ref = ET.Element(f"{{{bpmn_ns}}}sourceRef")
        source_ref.text = task_id
        target_ref = ET.Element(f"{{{bpmn_ns}}}targetRef")
        target_ref.text = data_obj_ref_id

    data_assoc.extend([source_ref, target_ref])
    task.append(data_assoc)

    # Add BPMNShape in diagram
    plane = root.find(f'.//{{{bpmndi_ns}}}BPMNPlane')
    if plane is None:
        raise ValueError("No <bpmndi:BPMNPlane> found.")

    # Find shape of the task
    task_shape = None
    for shape in plane.findall(f'.//{{{bpmndi_ns}}}BPMNShape'):
        if shape.get('bpmnElement') == task_id:
            task_shape = shape
            break
    if task_shape is None:
        raise ValueError(f"No BPMNShape found for task '{activity_name}'.")

    bounds = task_shape.find(f'.//{{{dc_ns}}}Bounds')
    if bounds is None:
        raise ValueError("No <dc:Bounds> in task shape.")

    x = float(bounds.get('x'))
    y = float(bounds.get('y'))
    width = float(bounds.get('width'))
    #height = float(bounds.get('height'))

    # Determine shape position for data object
    if is_new_object:
        # Place data object left or right above task
        '''
        if input:
            data_x = x + (width - 36) / 2 - 50
            data_y = y - 100
        else:
            data_x = x + (width - 36) / 2 + 50
            data_y = y - 100 
        '''
        if input:
            if in_out_counter % 2 == 1:
                data_x = x + (width - 36) / 2 - 50 * in_out_counter
                data_y = y - 100
            else:
                data_x = x + (width - 36) / 2 - 50 * (in_out_counter - 1)
                data_y = y + 100
        else:
            if in_out_counter % 2 == 1:
                data_x = x + (width - 36) / 2 + 50 * in_out_counter
                data_y = y - 100
            else:
                data_x = x + (width - 36) / 2 + 50 * (in_out_counter - 1)
                data_y = y + 100

        # Create new BPMNShape
        data_shape = ET.Element(f"{{{bpmndi_ns}}}BPMNShape", id=shape_id, bpmnElement=data_obj_ref_id)
        bounds_elem = ET.Element(f"{{{dc_ns}}}Bounds", x=str(data_x), y=str(data_y), width="36", height="50")
        label_elem = ET.Element(f"{{{bpmndi_ns}}}BPMNLabel")
        label_bounds = ET.Element(f"{{{dc_ns}}}Bounds", x=str(data_x + 6), y=str(data_y + 57), width="25", height="14")
        label_elem.append(label_bounds)
        data_shape.append(bounds_elem)
        data_shape.append(label_elem)
        plane.append(data_shape)
    else:
        # Reuse existing shape
        existing_shape = None
        for shape in plane.findall(f'.//{{{bpmndi_ns}}}BPMNShape'):
            if shape.get('bpmnElement') == data_obj_ref_id:
                existing_shape = shape
                break
        if existing_shape is None:
            raise ValueError(f"No BPMNShape found for reused data object '{data_object_name}'.")
        existing_bounds = existing_shape.find(f'.//{{{dc_ns}}}Bounds')
        if existing_bounds is None:
            raise ValueError("No <dc:Bounds> in reused data object shape.")
        data_x = float(existing_bounds.get('x'))
        data_y = float(existing_bounds.get('y'))

    # Add BPMNEdge
    di_ns = nsmap.get('omgdi') or 'http://www.omg.org/spec/DD/20100524/DI'
    edge_id = f"{data_assoc_id}_edge"
    bpmn_edge = ET.Element(f"{{{bpmndi_ns}}}BPMNEdge", id=edge_id, bpmnElement=data_assoc_id)

    '''
    # From data object (bottom center) to task (top center) or vice versa
    if input:
        wp1 = ET.Element(f"{{{di_ns}}}waypoint", x=str(data_x + 18), y=str(data_y + 50))  # bottom left of data object
        wp2 = ET.Element(f"{{{di_ns}}}waypoint", x=str(x + width / 2 - 10), y=str(y))  # top left of task
    else:
        wp1 = ET.Element(f"{{{di_ns}}}waypoint", x=str(x + width / 2 + 10), y=str(y))  # top right of task
        wp2 = ET.Element(f"{{{di_ns}}}waypoint", x=str(data_x + 18), y=str(data_y + 50))  # bottom right of data object
    '''
    if input:
        if in_out_counter % 2 == 1:
            wp1 = ET.Element(f"{{{di_ns}}}waypoint", x=str(data_x + 18), y=str(data_y + 50))  # bottom left of data object
            wp2 = ET.Element(f"{{{di_ns}}}waypoint", x=str(x + width / 2 - 10), y=str(y))  # top left of task
        else:
            wp1 = ET.Element(f"{{{di_ns}}}waypoint", x=str(data_x + 18), y=str(data_y))  # top left of data object
            wp2 = ET.Element(f"{{{di_ns}}}waypoint", x=str(x + width / 2 - 10), y=str(y+36))  # top left of task
    else:
        if in_out_counter % 2 == 1:
            wp1 = ET.Element(f"{{{di_ns}}}waypoint", x=str(x + width / 2 + 10), y=str(y))  # top right of task
            wp2 = ET.Element(f"{{{di_ns}}}waypoint", x=str(data_x + 18), y=str(data_y + 50))  # bottom right of data object
        else:
            wp1 = ET.Element(f"{{{di_ns}}}waypoint", x=str(x + width / 2 + 10), y=str(y + 36))  # top right of task
            wp2 = ET.Element(f"{{{di_ns}}}waypoint", x=str(data_x + 18), y=str(data_y))  # bottom right of data object

    bpmn_edge.extend([wp1, wp2])
    plane.append(bpmn_edge)

    # Save file
    output_path = output_path or file_path
    tree.write(output_path, pretty_print=True, xml_declaration=True, encoding='UTF-8')
    print(f"Data object '{data_object_name}' added above task '{activity_name}'.")



def add_data_object_to_bpmn(file_path, activity_name, data_object_name, in_out_counter,
                            dataclass_name=None, states_list=None, input=True, output_path=None):

    parser = ET.XMLParser(remove_blank_text=True)
    tree = ET.parse(file_path, parser)
    root = tree.getroot()

    # Get namespaces from the root element
    # ET.Element.nsmap is an lxml specific feature
    nsmap = root.nsmap if hasattr(root, 'nsmap') else {}

    bpmn_ns = nsmap.get('bpmn', 'http://www.omg.org/spec/BPMN/20100524/MODEL')
    bpmndi_ns = nsmap.get('bpmndi', 'http://www.omg.org/spec/BPMN/20100524/DI')
    dc_ns = nsmap.get('dc', 'http://www.omg.org/spec/DD/20100524/DC')
    di_ns = nsmap.get('di', 'http://www.omg.org/spec/DD/20100524/DI')  # Explicitly get di namespace

    # Define and register FCM namespace
    # IMPORTANT: If your BPMN tool uses a different URI for fcm, update this.
    fcm_ns_uri = "http://www.fcm.org/bpmn/extensions"
    ET.register_namespace('fcm', fcm_ns_uri)
    nsmap[
        'fcm'] = fcm_ns_uri  # Add to the dictionary for consistency, though register_namespace handles actual serialization

    if not (bpmn_ns and bpmndi_ns and dc_ns):
        raise ValueError("Missing required BPMN/DI namespaces. Ensure 'bpmn', 'bpmndi', and 'dc' are defined.")

    # Find process
    # Use the full namespace URI for finding elements
    process = root.find(f'.//{{{bpmn_ns}}}process')
    if process is None:
        raise ValueError("No <bpmn:process> element found.")

    # Find task
    task = None
    for t in process.findall(f'.//{{{bpmn_ns}}}task'):
        if t.get('name') == activity_name:
            task = t
            break
    if task is None:
        raise ValueError(f"No task with name '{activity_name}' found.")
    task_id = task.get('id')

    # Try to find existing dataObjectReference by name
    data_obj_ref = None
    # Use the full namespace URI for finding elements
    for existing_ref in process.findall(f'.//{{{bpmn_ns}}}dataObjectReference'):
        if existing_ref.get('name') == data_object_name:
            data_obj_ref = existing_ref
            break

    # Reuse or create new one
    is_new_object = data_obj_ref is None

    print(f"my states are {states_list}")
    states_id_list = list()
    for state in states_list:
        states_id_list.append(state_name_to_ids[dataclass_name][state])

    if is_new_object:
        data_obj_id = f"DataObject_{uuid.uuid4().hex[:8]}"
        data_obj_ref_id = f"{data_obj_id}_ref"

        # Create dataObject element
        data_object = ET.Element(f"{{{bpmn_ns}}}dataObject", id=data_obj_id)

        # Create dataObjectReference element
        data_obj_ref = ET.Element(
            f"{{{bpmn_ns}}}dataObjectReference",
            id=data_obj_ref_id,
            dataObjectRef=data_obj_id,
            name=data_object_name
        )

        # Add FCM attributes if provided
        if dataclass_name:
            data_obj_ref.set(f"{{{fcm_ns_uri}}}dataclass", object_name_to_ids[dataclass_name])
        if states_list is not None:  # Check for None explicitly, as an empty list is valid
            states_str = " ".join(states_id_list)
            data_obj_ref.set(f"{{{fcm_ns_uri}}}states", states_str)

        process.append(data_object)
        process.append(data_obj_ref)
    else:
        # Reuse existing dataObjectReference
        data_obj_ref_id = data_obj_ref.get('id')

        # Update FCM attributes on existing dataObjectReference
        if dataclass_name:
            data_obj_ref.set(f"{{{fcm_ns_uri}}}dataclass", object_name_to_ids[dataclass_name])
        if states_list is not None:  # Check for None explicitly
            states_str = " ".join(states_id_list)
            data_obj_ref.set(f"{{{fcm_ns_uri}}}states", states_str)

    data_assoc_id = f"DataAssoc_{uuid.uuid4().hex[:8]}"
    shape_id = f"{data_obj_ref_id}_di"

    # Create data association
    assoc_tag = 'dataInputAssociation' if input else 'dataOutputAssociation'
    data_assoc = ET.Element(f"{{{bpmn_ns}}}{assoc_tag}", id=data_assoc_id)

    if input:
        source_ref = ET.Element(f"{{{bpmn_ns}}}sourceRef")
        source_ref.text = data_obj_ref_id
        target_ref = ET.Element(f"{{{bpmn_ns}}}targetRef")
        target_ref.text = task_id
    else:
        source_ref = ET.Element(f"{{{bpmn_ns}}}sourceRef")
        source_ref.text = task_id
        target_ref = ET.Element(f"{{{bpmn_ns}}}targetRef")
        target_ref.text = data_obj_ref_id

    data_assoc.extend([source_ref, target_ref])
    task.append(data_assoc)

    # Add BPMNShape in diagram
    plane = root.find(f'.//{{{bpmndi_ns}}}BPMNPlane')
    if plane is None:
        raise ValueError("No <bpmndi:BPMNPlane> found.")

    # Find shape of the task
    task_shape = None
    for shape in plane.findall(f'.//{{{bpmndi_ns}}}BPMNShape'):
        if shape.get('bpmnElement') == task_id:
            task_shape = shape
            break
    if task_shape is None:
        raise ValueError(f"No BPMNShape found for task '{activity_name}'.")

    bounds = task_shape.find(f'.//{{{dc_ns}}}Bounds')
    if bounds is None:
        raise ValueError("No <dc:Bounds> in task shape.")

    x = float(bounds.get('x'))
    y = float(bounds.get('y'))
    width = float(bounds.get('width'))

    # Determine shape position for data object
    if is_new_object:
        # Place data object left or right above task
        if input:
            if in_out_counter % 2 == 1:
                data_x = x + (width - 36) / 2 - 50 * in_out_counter
                data_y = y - 100
            else:
                data_x = x + (width - 36) / 2 - 50 * (in_out_counter - 1)
                data_y = y + 100
        else:
            if in_out_counter % 2 == 1:
                data_x = x + (width - 36) / 2 + 50 * in_out_counter
                data_y = y - 100
            else:
                data_x = x + (width - 36) / 2 + 50 * (in_out_counter - 1)
                data_y = y + 100

        # Create new BPMNShape
        data_shape = ET.Element(f"{{{bpmndi_ns}}}BPMNShape", id=shape_id, bpmnElement=data_obj_ref_id)
        bounds_elem = ET.Element(f"{{{dc_ns}}}Bounds", x=str(data_x), y=str(data_y), width="36", height="50")
        label_elem = ET.Element(f"{{{bpmndi_ns}}}BPMNLabel")
        label_bounds = ET.Element(f"{{{dc_ns}}}Bounds", x=str(data_x + 6), y=str(data_y + 57), width="25",
                                  height="14")
        label_elem.append(label_bounds)
        data_shape.append(bounds_elem)
        data_shape.append(label_elem)
        plane.append(data_shape)
    else:
        # Reuse existing shape
        existing_shape = None
        for shape in plane.findall(f'.//{{{bpmndi_ns}}}BPMNShape'):
            if shape.get('bpmnElement') == data_obj_ref_id:
                existing_shape = shape
                break
        if existing_shape is None:
            raise ValueError(f"No BPMNShape found for reused data object '{data_object_name}'.")
        existing_bounds = existing_shape.find(f'.//{{{dc_ns}}}Bounds')
        if existing_bounds is None:
            raise ValueError("No <dc:Bounds> in reused data object shape.")
        data_x = float(existing_bounds.get('x'))
        data_y = float(existing_bounds.get('y'))

    # Add BPMNEdge
    edge_id = f"{data_assoc_id}_edge"
    bpmn_edge = ET.Element(f"{{{bpmndi_ns}}}BPMNEdge", id=edge_id, bpmnElement=data_assoc_id)

    if input:
        if in_out_counter % 2 == 1:
            wp1 = ET.Element(f"{{{di_ns}}}waypoint", x=str(data_x + 18),
                             y=str(data_y + 50))  # bottom left of data object
            wp2 = ET.Element(f"{{{di_ns}}}waypoint", x=str(x + width / 2 - 10), y=str(y))  # top left of task
        else:
            wp1 = ET.Element(f"{{{di_ns}}}waypoint", x=str(data_x + 18), y=str(data_y))  # top left of data object
            wp2 = ET.Element(f"{{{di_ns}}}waypoint", x=str(x + width / 2 - 10), y=str(y + 36))  # top left of task
    else:
        if in_out_counter % 2 == 1:
            wp1 = ET.Element(f"{{{di_ns}}}waypoint", x=str(x + width / 2 + 10), y=str(y))  # top right of task
            wp2 = ET.Element(f"{{{di_ns}}}waypoint", x=str(data_x + 18),
                             y=str(data_y + 50))  # bottom right of data object
        else:
            wp1 = ET.Element(f"{{{di_ns}}}waypoint", x=str(x + width / 2 + 10), y=str(y + 36))  # top right of task
            wp2 = ET.Element(f"{{{di_ns}}}waypoint", x=str(data_x + 18),
                             y=str(data_y))  # bottom right of data object

    bpmn_edge.extend([wp1, wp2])
    plane.append(bpmn_edge)

    # Save file
    output_path = output_path or file_path
    # Serialize with pretty print and XML declaration using lxml
    tree.write(output_path, pretty_print=True, xml_declaration=True, encoding='UTF-8')
    print(f"Data object '{data_object_name}' added/updated for task '{activity_name}'.")


def remove_start_and_end_events(file_path, output_path=None):
    parser = ET.XMLParser(remove_blank_text=True)
    tree = ET.parse(file_path, parser)
    root = tree.getroot()
    nsmap = root.nsmap

    bpmn_ns = nsmap.get('bpmn')
    bpmndi_ns = nsmap.get('bpmndi')

    if not (bpmn_ns and bpmndi_ns):
        raise ValueError("Missing required namespaces: bpmn, bpmndi.")

    process = root.find(f'.//{{{bpmn_ns}}}process')
    plane = root.find(f'.//{{{bpmndi_ns}}}BPMNPlane')

    if process is None:
        raise ValueError("No <bpmn:process> element found.")

    # Remove start/end events and track their IDs
    event_ids = []
    for tag in ['startEvent', 'endEvent']:
        for elem in process.findall(f'{{{bpmn_ns}}}{tag}'):
            event_ids.append(elem.get('id'))
            process.remove(elem)

    # Remove related sequenceFlows and track their IDs
    removed_sequenceflow_ids = []
    for seq_flow in list(process.findall(f'{{{bpmn_ns}}}sequenceFlow')):
        if seq_flow.get('sourceRef') in event_ids or seq_flow.get('targetRef') in event_ids:
            removed_sequenceflow_ids.append(seq_flow.get('id'))
            process.remove(seq_flow)

    # Clean up incoming/outgoing references
    for elem in process.iter():
        for tag in ['incoming', 'outgoing']:
            for ref in list(elem.findall(f'{{{bpmn_ns}}}{tag}')):
                if ref.text in removed_sequenceflow_ids:
                    elem.remove(ref)

    # Remove related diagram shapes and edges
    if plane is not None:
        for element in list(plane):
            bpmn_element_id = element.get('bpmnElement')
            if (
                bpmn_element_id in event_ids or
                bpmn_element_id in removed_sequenceflow_ids
            ):
                plane.remove(element)

    # Save output
    output_path = output_path or file_path
    tree.write(output_path, pretty_print=True, xml_declaration=True, encoding='UTF-8')

def generate_unique_id(prefix):
        return f"{prefix}_{uuid.uuid4().hex[:24]}"

def create_lifecycle_xml(type_to_paths,object_name_to_id, filename = "olcs.xml"):
    global state_name_to_ids
    olcs = list(dict())

    SCALE_FACTOR = 250

    type_to_graph = {obj_type: build_graph(paths) for obj_type, paths in type_to_paths.items()}
    nx_graphs = {obj_type: create_nx_graph(graph) for obj_type, graph in type_to_graph.items()}

    #generates uuids for each state as well
    for obj_type, G in nx_graphs.items():
        obj_positions = get_lifecycle_positions(G)

        states = list()
        for key, value in obj_positions.items():
            states.append({"name": key, "x":int(value[0]*SCALE_FACTOR), "y":int(value[1]*SCALE_FACTOR)})
            state_name_to_ids[obj_type][key] = f"State_{uuid.uuid4().hex[:24]}"
        transitions = list()
        for transition in G.edges:
            transitions.append({"source": transition[0], "target": transition[1]})
        obj_dict = {"name": obj_type, "states": states, "transitions": transitions}
        olcs.append(obj_dict)

    ET.register_namespace('olc', "http://bptlab/schema/olc")
    ET.register_namespace('olcDi', "http://bptlab/schema/olcDi")

    root = ET.Element("{http://bptlab/schema/olc}definitions")

    for olc in olcs:
        olc_id = generate_unique_id("Olc")
        class_ref = object_name_to_id.get(olc['name'])

        olc_elem = ET.SubElement(root, "{http://bptlab/schema/olc}olc", {
            "name": olc['name'],
            "id": olc_id,
            "classRef": class_ref
        })

        state_id_map = {}
        for state in olc.get('states', []):
            state_id = state_name_to_ids[olc['name']][state['name']]

            state_id_map[state['name']] = state_id
            ET.SubElement(olc_elem, "{http://bptlab/schema/olc}state", {
                "id": state_id,
                "name": state['name'],
                "type": "olc:State",
                "x": str(state['x']),
                "y": str(state['y'])
            })

        for trans in olc.get('transitions', []):
            ET.SubElement(olc_elem, "{http://bptlab/schema/olc}transition", {
                "id": generate_unique_id("Transition"),
                "sourceState": state_id_map[trans['source']],
                "targetState": state_id_map[trans['target']],
                "type": "olc:Transition"
            })

    tree = ET.ElementTree(root)
    tree.write(filename, encoding="utf-8", xml_declaration=True)

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

def get_lifecycle_positions(G):
    plt.figure(figsize=(8, 6))
    #pos = nx.spring_layout(G)  # Position nodes for visualization
    pos = nx.kamada_kawai_layout(G)
    print(f"position is: {pos}")

    for key, value in pos.items():
        print(f"{key}: {int(value[0]*100)}, {int(value[1]*100)}")
    print()

    return pos

def get_uml_positions(lc_data: dict[str, dict[str, tuple[str, str]]]):
    G = nx.DiGraph()
    for source, targets in lc_data.items():
        for target, _ in targets.items():
            G.add_edge(source, target)
    pos = nx.kamada_kawai_layout(G)
    return pos


def create_uml_xml(lc_data, object_name_to_id, output_filename="dataModel_new.xml"):
    positions = get_uml_positions(lc_data)

    # --- XML Namespace Definitions ---
    # Define XML namespaces for the document
    od_ns = "http://tk/schema/od"
    odDi_ns = "http://tk/schema/odDi"
    dc_ns = "http://www.omg.org/spec/DD/20100524/DC"

    # Define a namespace map to ensure all desired namespaces are declared on the root
    nsmap = {
        'od': od_ns,
        'odDi': odDi_ns,
        'dc': dc_ns
    }

    # --- Root Element ---
    # Create the root <od:definitions> element with all desired namespaces using nsmap
    definitions = ET.Element(
        f"{{{od_ns}}}definitions",
        nsmap=nsmap # Pass the nsmap to declare namespaces on this element
    )

    # --- odBoard and odDiRootBoard ---
    # Create the <od:odBoard> element which will contain class and association definitions
    od_board = ET.SubElement(definitions, f"{{{od_ns}}}odBoard", id="Board")
    # Create the <odDi:odRootBoard> element which will contain diagram interchange information
    od_di_root_board = ET.SubElement(definitions, f"{{{odDi_ns}}}odRootBoard", id="RootBoard")
    # Create the <odDi:odPlane> element inside <odDi:odRootBoard>
    od_di_plane = ET.SubElement(od_di_root_board, f"{{{odDi_ns}}}odPlane", id="Plane", boardElement="Board")

    # --- Data Structures for Processing ---
    class_name_to_id = {}  # Maps class names (e.g., "Container") to their generated/provided IDs (e.g., "Object_123")
    association_details = []  # Stores (source_class_name, target_class_name, source_card, target_card, association_id)

    # Counter for generating unique IDs
    obj_id_counter = 0
    assoc_id_counter = 0

    # --- Scaling and Offset for Positions ---
    # NetworkX layout positions are typically between -1 and 1.
    # We need to scale and offset them to get reasonable pixel coordinates.
    SCALE_FACTOR = 500  # Multiplier for coordinates
    X_OFFSET = 400      # Offset to center the diagram horizontally
    Y_OFFSET = 400     # Offset to center the diagram vertically
    CLASS_WIDTH = 160   # Fixed width for class shapes
    CLASS_HEIGHT = 110  # Fixed height for class shapes

    def get_unique_object_id(class_name):
        nonlocal obj_id_counter
        if class_name in object_name_to_id:
            return object_name_to_id[class_name]
        new_id = f"Object_{obj_id_counter}"
        obj_id_counter += 1
        return new_id

    def get_unique_association_id():
        nonlocal assoc_id_counter
        new_id = f"Association_{assoc_id_counter:06x}" # Use hex for shorter, unique IDs
        assoc_id_counter += 1
        return new_id

    # --- Step 1: Collect all unique class names and assign IDs ---
    all_class_names = set()
    for source_class, targets in lc_data.items():
        all_class_names.add(source_class)
        for target_class in targets:
            all_class_names.add(target_class)

    for class_name in all_class_names:
        if class_name not in class_name_to_id: # Avoid re-assigning if already in object_name_to_id
            class_name_to_id[class_name] = get_unique_object_id(class_name)

    # --- Step 2: Create <od:class> elements and prepare association data ---
    # A dict to store lists of associations for each source class
    class_associations_refs = {cls_name: [] for cls_name in class_name_to_id}

    for source_class_name, target_map in lc_data.items():
        source_class_id = class_name_to_id[source_class_name]
        for target_class_name, cardinalities in target_map.items():
            target_class_id = class_name_to_id[target_class_name]
            source_cardinality, target_cardinality = cardinalities

            # Generate a unique ID for this association
            association_id = get_unique_association_id()
            association_details.append(
                (source_class_name, target_class_name, source_cardinality,
                 target_cardinality, source_class_id, target_class_id, association_id)
            )
            # Add the association reference to the source class
            class_associations_refs[source_class_name].append(association_id)

    # Now, add the class elements to od_board
    for class_name in sorted(class_name_to_id.keys()): # Sort for consistent output order
        class_id = class_name_to_id[class_name]
        class_elem = ET.SubElement(
            od_board,
            f"{{{od_ns}}}class",
            name=class_name,
            id=class_id,
            attributeValues="",
            type="od:Class"
        )
        # Add association references if this class is a source for any association
        for assoc_ref_id in class_associations_refs[class_name]:
            ET.SubElement(class_elem, f"{{{od_ns}}}associations").text = assoc_ref_id

    # --- Step 3: Create <od:association> elements ---
    for source_class_name, target_class_name, source_card, target_card, \
        source_class_id, target_class_id, association_id in association_details:
        ET.SubElement(
            od_board,
            f"{{{od_ns}}}association",
            id=association_id,
            type="od:Association",
            sourceCardinality=source_card,
            targetCardinality=target_card,
            sourceRef=source_class_id,
            targetRef=target_class_id
        )

    # --- Step 4: Create <odDi:odShape> (Class Shapes) and <odDi:association> (Waypoints) elements ---
    for class_name in sorted(class_name_to_id.keys()): # Sort for consistent output order
        class_id = class_name_to_id[class_name]

        # Get positions and scale them
        raw_x, raw_y = positions.get(class_name, (0, 0)) # Default to (0,0) if no position
        print(f"raw_x: {raw_x}, raw_y: {raw_y}")
        # Convert networkx-like positions [-1, 1] to pixel coordinates
        # Adjust for scaling and centering; (0,0) becomes (X_OFFSET, Y_OFFSET)
        # Assuming raw_x and raw_y are in [-1, 1] range approximately
        x_pos = round(raw_x * SCALE_FACTOR + X_OFFSET)
        y_pos = round(raw_y * SCALE_FACTOR + Y_OFFSET)

        # Create odDi:odShape for the class
        od_di_shape = ET.SubElement(
            od_di_plane,
            f"{{{odDi_ns}}}odShape",
            id=f"{class_id}_di",
            boardElement=class_id
        )
        # Add dc:Bounds for position and size
        ET.SubElement(
            od_di_shape,
            f"{{{dc_ns}}}Bounds",
            x=str(x_pos),
            y=str(y_pos),
            width=str(CLASS_WIDTH),
            height=str(CLASS_HEIGHT)
        )

    # Create <odDi:association> waypoints
    for source_class_name, target_class_name, _, _, \
        source_class_id, target_class_id, association_id in association_details:

        # Get the coordinates for the association waypoints.
        # As per your request, the waypoints are derived from the class shape's top-left corner.
        # This means the line for the association will start and end at these points.

        # Fetch source class position
        source_raw_x, source_raw_y = positions.get(source_class_name, (0, 0))
        source_x_pos = round(source_raw_x * SCALE_FACTOR + X_OFFSET)
        source_y_pos = round(source_raw_y * SCALE_FACTOR + Y_OFFSET)

        # Fetch target class position
        target_raw_x, target_raw_y = positions.get(target_class_name, (0, 0))
        target_x_pos = round(target_raw_x * SCALE_FACTOR + X_OFFSET)
        target_y_pos = round(target_raw_y * SCALE_FACTOR + Y_OFFSET)


        od_di_association = ET.SubElement(
            od_di_plane,
            f"{{{odDi_ns}}}association",
            id=f"{association_id}_di",
            boardElement=association_id
        )
        # Add two waypoints: one for source, one for target
        ET.SubElement(
            od_di_association,
            f"{{{odDi_ns}}}waypoint",
            x=str(source_x_pos),
            y=str(source_y_pos)
        )
        ET.SubElement(
            od_di_association,
            f"{{{odDi_ns}}}waypoint",
            x=str(target_x_pos),
            y=str(target_y_pos)
        )

        # Add two labels: one for source, one for target
        od_source_label = ET.SubElement(
            od_di_association,
            f"{{{odDi_ns}}}odSourceLabel"
        )
        position = offset_initial_waypoint(source_x_pos,target_x_pos,source_y_pos,target_y_pos)
        ET.SubElement(
            od_source_label,
            f"{{{dc_ns}}}Bounds",
            x=str(position[0]),
            y=str(position[1]),
            width="26",
            height="18"
        )
        od_target_label = ET.SubElement(
            od_di_association,
            f"{{{odDi_ns}}}odTargetLabel"
        )
        position = offset_initial_waypoint(target_x_pos,source_x_pos,target_y_pos,source_y_pos)
        ET.SubElement(
            od_target_label,
            f"{{{dc_ns}}}Bounds",
            x=str(position[0]),
            y=str(position[1]),
            width="26",
            height="18"
        )

    # --- Write XML to File ---
    # Pretty print the XML using lxml's tostring with pretty_print=True
    # and ensure UTF-8 encoding with XML declaration.
    final_xml_bytes = ET.tostring(
        definitions,
        pretty_print=True,
        xml_declaration=True,
        encoding='UTF-8'
    )

    with open(output_filename, "wb") as f: # Use 'wb' for writing bytes
        f.write(final_xml_bytes)

    print(f"UML diagram XML saved to {output_filename}")

def generate_object_ids(ocel):
    global object_name_to_ids
    object_types = ocel.objects[ocel.object_type_column].values.unique()

    for object_type in object_types:
        object_name_to_ids[object_type] = f"Object_{uuid.uuid4().hex[:24]}"

    return object_name_to_ids


def calculate_y_coordinate_range(file_path):
    try:
        parser = ET.XMLParser(remove_blank_text=True)
        tree = ET.parse(file_path, parser)
        root = tree.getroot()
    except Exception as e:
        raise ValueError(f"Error parsing XML file '{file_path}': {e}")

    # Define namespaces. It's robust to get them from the root if present,
    # but also provide defaults for common BPMN namespaces.
    # IMPORTANT: The keys in this dictionary are the prefixes used in XPath.
    # The values are the full namespace URIs.
    nsmap_from_root = root.nsmap if hasattr(root, 'nsmap') else {}

    # Ensure all relevant namespaces are available, providing common defaults if not found in nsmap
    # These prefixes (bpmn, bpmndi, dc, di) are conventionally used.
    # Make sure to include all namespaces that might appear in the XML and you want to query.
    namespaces = {
        'bpmn': nsmap_from_root.get('bpmn', 'http://www.omg.org/spec/BPMN/20100524/MODEL'),
        'bpmndi': nsmap_from_root.get('bpmndi', 'http://www.omg.org/spec/BPMN/20100524/DI'),
        'dc': nsmap_from_root.get('dc', 'http://www.omg.org/spec/DD/20100524/DC'),
        'di': nsmap_from_root.get('di', 'http://www.omg.org/spec/DD/20100524/DI')
    }

    # Collect all 'y' coordinates
    y_coordinates = []

    # Find y coordinates in <omgdi:waypoint> elements
    waypoint_elements = root.xpath(f"//di:waypoint", namespaces=namespaces)
    for wp in waypoint_elements:
        if 'y' in wp.attrib:  # Changed 'x' to 'y'
            try:
                y_coordinates.append(float(wp.get('y')))  # Changed 'x' to 'y'
            except ValueError:
                print(
                    f"Warning: Could not convert waypoint y-coordinate '{wp.get('y')}' to float.")  # Changed 'x' to 'y'

    # Find y coordinates in <omgdc:Bounds> elements
    bounds_elements = root.xpath(f"//dc:Bounds", namespaces=namespaces)
    for bounds in bounds_elements:
        if 'y' in bounds.attrib:  # Changed 'x' to 'y'
            try:
                y_coordinates.append(float(bounds.get('y')))  # Changed 'x' to 'y'
            except ValueError:
                print(
                    f"Warning: Could not convert Bounds y-coordinate '{bounds.get('y')}' to float.")  # Changed 'x' to 'y'

    if not y_coordinates:  # Changed 'x' to 'y'
        print("No 'y' coordinates found in the specified elements.")  # Changed 'x' to 'y'
        return 0.0
    else:
        min_y = min(y_coordinates)  # Changed 'x' to 'y'
        max_y = max(y_coordinates)  # Changed 'x' to 'y'
        return max_y - min_y  # Changed 'x' to 'y'


def offset_y_coordinates_in_bpmn(file_path, y_offset):  # Changed 'x_offset' to 'y_offset'
    try:
        parser = ET.XMLParser(remove_blank_text=True)
        tree = ET.parse(file_path, parser)
        root = tree.getroot()
    except Exception as e:
        raise ValueError(f"Error parsing XML file '{file_path}': {e}")

    nsmap_from_root = root.nsmap if hasattr(root, 'nsmap') else {}
    namespaces = {
        'bpmn': nsmap_from_root.get('bpmn', 'http://www.omg.org/spec/BPMN/20100524/MODEL'),
        'bpmndi': nsmap_from_root.get('bpmndi', 'http://www.omg.org/spec/BPMN/20100524/DI'),
        'dc': nsmap_from_root.get('dc', 'http://www.omg.org/spec/DD/20100524/DC'),
        'di': nsmap_from_root.get('di', 'http://www.omg.org/spec/DD/20100524/DI')
    }

    modified_count = 0

    # Find and update y coordinates in <omgdi:waypoint> elements
    waypoint_elements = root.xpath(f"//di:waypoint", namespaces=namespaces)
    for wp in waypoint_elements:
        if 'y' in wp.attrib:  # Changed 'x' to 'y'
            try:
                original_y = float(wp.get('y'))  # Changed 'x' to 'y'
                new_y = original_y + y_offset  # Changed 'x_offset' to 'y_offset' and 'original_x' to 'original_y'
                wp.set('y', str(new_y))  # Changed 'x' to 'y'
                modified_count += 1
            except ValueError:
                print(
                    f"Warning: Could not convert waypoint y-coordinate '{wp.get('y')}' to float. Skipping.")  # Changed 'x' to 'y'

    # Find and update y coordinates in <omgdc:Bounds> elements
    bounds_elements = root.xpath(f"//dc:Bounds", namespaces=namespaces)
    for bounds in bounds_elements:
        if 'y' in bounds.attrib:  # Changed 'x' to 'y'
            try:
                original_y = float(bounds.get('y'))  # Changed 'x' to 'y'
                new_y = original_y + y_offset  # Changed 'x_offset' to 'y_offset' and 'original_x' to 'original_y'
                bounds.set('y', str(new_y))  # Changed 'x' to 'y'
                modified_count += 1
            except ValueError:
                print(
                    f"Warning: Could not convert Bounds y-coordinate '{bounds.get('y')}' to float. Skipping.")  # Changed 'x' to 'y'

    if modified_count > 0:
        # Save the modified file to the new output_file_path
        tree.write(file_path, pretty_print=True, xml_declaration=True, encoding='UTF-8')
        print(
            f"Successfully offset {modified_count} y-coordinates from '{file_path}' and saved to '{file_path}' by {y_offset}.")  # Changed 'x' to 'y'
    else:
        print(f"No y-coordinates found or modified in '{file_path}'. No new file was written.")  # Changed 'x' to 'y'


def merge_two_bpmn_files_old(input_file_path, target_file_path, is_empty=False):
    parser = ET.XMLParser(remove_blank_text=True)

    # Define common BPMN namespaces for consistent use
    bpmn_ns = "http://www.omg.org/spec/BPMN/20100524/MODEL"
    bpmndi_ns = "http://www.omg.org/spec/BPMN/20100524/DI"
    # Register namespaces for lxml's findall/xpath to work correctly if not in root already
    ET.register_namespace('bpmn', bpmn_ns)
    ET.register_namespace('bpmndi', bpmndi_ns)

    # --- Handle is_empty = True case (direct copy) ---
    if is_empty:
        try:
            input_tree = ET.parse(input_file_path, parser)
            input_tree.write(target_file_path, pretty_print=True, xml_declaration=True, encoding='UTF-8')
            print(f"Successfully copied '{input_file_path}' to '{target_file_path}' (target was empty).")
            return
        except Exception as e:
            raise ValueError(f"Error copying '{input_file_path}' to '{target_file_path}': {e}")

    # --- Handle is_empty = False case (granular BPMN merge) ---
    input_tree = None
    input_root = None
    input_definitions = None
    try:
        input_tree = ET.parse(input_file_path, parser)
        input_root = input_tree.getroot()
        # Check if the root itself is the bpmn:definitions element
        if input_root.tag == f"{{{bpmn_ns}}}definitions":
            input_definitions = input_root
        else:
            # If not, try to find definitions as a direct child
            input_definitions = input_root.find(f"{{{bpmn_ns}}}definitions")
            if input_definitions is None:
                print(
                    f"Warning: Root element of input file '{input_file_path}' is not <bpmn:definitions> and no <bpmn:definitions> child found. Cannot merge content.")
                input_root = None  # Mark as unmergeable
    except Exception as e:
        print(
            f"Warning: Could not parse input BPMN file '{input_file_path}'. This input will be skipped for merge: {e}")
        input_root = None  # Mark input as unmergeable

    target_tree = None
    target_root = None
    target_definitions = None
    target_process = None
    target_diagram = None
    try:
        target_tree = ET.parse(target_file_path, parser)
        target_root = target_tree.getroot()
        # Check if the root itself is the bpmn:definitions element
        if target_root.tag == f"{{{bpmn_ns}}}definitions":
            target_definitions = target_root
        else:
            # If not, try to find definitions as a direct child
            target_definitions = target_root.find(f"{{{bpmn_ns}}}definitions")
            if target_definitions is None:
                raise ValueError(
                    f"No <bpmn:definitions> element found in target file '{target_file_path}' (neither as root nor child). Cannot perform granular merge.")

        target_process = target_definitions.find(f"{{{bpmn_ns}}}process")
        target_diagram = target_definitions.find(f"{{{bpmndi_ns}}}BPMNDiagram")

        if target_process is None:
            raise ValueError(
                f"No <bpmn:process> element found in target file '{target_file_path}' within definitions. Cannot merge processes.")
        if target_diagram is None:
            print(
                f"Warning: No <bpmndi:BPMNDiagram> element found in target file '{target_file_path}' within definitions. Diagrams will not be merged.")

    except Exception as e:
        # If target parsing or definition finding fails, and input also has no valid definitions,
        # then we cannot proceed with a merge.
        if input_root is None:
            raise ValueError(
                f"Neither input file '{input_file_path}' nor target file '{target_file_path}' "
                "contains valid BPMN definitions for granular merging. "
                "If target is empty, consider using 'is_empty=True'."
            ) from e  # Chain exception for better debugging
        else:
            # If target is problematic but input is fine (has definitions),
            # effectively treat as 'copy input definitions to target file' as a fallback.
            print(
                f"Warning: Target file '{target_file_path}' is invalid or missing essential BPMN elements for granular merge. "
                f"Proceeding by copying all definitions content from '{input_file_path}' into '{target_file_path}'.")

            input_tree_fresh = ET.parse(input_file_path, parser)  # Re-parse to ensure a clean copy
            input_tree_fresh.write(target_file_path, pretty_print=True, xml_declaration=True, encoding='UTF-8')
            print(f"Content from '{input_file_path}' (definitions) copied to '{target_file_path}'.")
            return  # Exit after copying

    # If we reach here, target_definitions is valid and target_root exists.
    # input_root might be None if input file was unparseable or had no definitions.
    if input_root is None:
        print(f"Input file '{input_file_path}' has no valid content to merge. "
              f"Target file '{target_file_path}' remains unchanged.")
        return

    # Now, perform the granular merge
    merged_elements_count = 0

    # Merge processes
    input_processes = input_definitions.findall(f"{{{bpmn_ns}}}process")
    if target_process is not None:
        for p in input_processes:
            # Append children of input process to target's first process
            for child_of_p in p:
                target_process.append(child_of_p)
                merged_elements_count += 1
        print(
            f"Merged contents of {len(input_processes)} input process(es) into '{target_file_path}'s first process.")
    else:
        print(f"Skipping process merge: No <bpmn:process> found in '{target_file_path}' to merge into.")

    # Merge diagrams
    input_diagrams = input_definitions.findall(f"{{{bpmndi_ns}}}BPMNDiagram")
    if target_diagram is not None:
        for d in input_diagrams:
            # Append children of input diagram to target's first diagram
            for child_of_d in d:
                target_diagram.append(child_of_d)
                merged_elements_count += 1
        print(
            f"Merged contents of {len(input_diagrams)} input diagram(s) into '{target_file_path}'s first diagram.")
    else:
        print(f"Skipping diagram merge: No <bpmndi:BPMNDiagram> found in '{target_file_path}' to merge into.")

    if merged_elements_count > 0:
        # Save the modified target tree back to the target_file_path
        target_tree.write(target_file_path, pretty_print=True, xml_declaration=True, encoding='UTF-8')
        print(
            f"Successfully completed granular merge into '{target_file_path}'. Total elements added: {merged_elements_count}.")
        print(
            "WARNING: This type of merge does NOT handle ID conflicts. The resulting BPMN file might be invalid if IDs overlap.")
    else:
        print(f"No specific BPMN elements found or merged. '{target_file_path}' remains unchanged.")


def merge_two_bpmn_files(input_file_path, target_file_path, is_empty=False):
    BPMN_NS = "http://www.omg.org/spec/BPMN/20100524/MODEL"
    BPMNDI_NS = "http://www.omg.org/spec/BPMN/20100524/DI"

    ET.register_namespace('bpmn', BPMN_NS)
    ET.register_namespace('bpmndi', BPMNDI_NS)
    parser = ET.XMLParser(remove_blank_text=True)

    # --- Handle `is_empty=True` case (direct copy) ---
    if is_empty:
        try:
            input_tree = ET.parse(input_file_path, parser)
            input_tree.write(target_file_path, pretty_print=True, xml_declaration=True, encoding='UTF-8')
            print(f"Successfully copied '{input_file_path}' to '{target_file_path}' (target was empty).")
            return
        except Exception as e:
            raise IOError(f"Error copying '{input_file_path}' to '{target_file_path}': {e}")

    # --- Begin granular BPMN merge ---

    # Step 1: Parse the input file and find its root <definitions> element.
    try:
        input_tree = ET.parse(input_file_path, parser)
        input_definitions = input_tree.getroot()
        if input_definitions.tag != f"{{{BPMN_NS}}}definitions":
            input_definitions = input_definitions.find(f"{{{BPMN_NS}}}definitions")
        if input_definitions is None:
            raise ValueError("No <bpmn:definitions> element found.")
    except Exception as e:

        return

    # Step 2: Parse the target file and find its root <definitions> element.
    try:
        target_tree = ET.parse(target_file_path, parser)
        target_definitions = target_tree.getroot()
        if target_definitions.tag != f"{{{BPMN_NS}}}definitions":
            target_definitions = target_definitions.find(f"{{{BPMN_NS}}}definitions")
        if target_definitions is None:
            raise ValueError("No <bpmn:definitions> element found.")
    except (FileNotFoundError, ET.ParseError) as e:
        print(
            f"Warning: Target file '{target_file_path}' not found or is invalid ({e}). Copying input file to target path.")
        input_tree.write(target_file_path, pretty_print=True, xml_declaration=True, encoding='UTF-8')
        return
    except Exception as e:
        print(
            f"Error: Could not parse or find definitions in target file '{target_file_path}'. Merge aborted. Details: {e}")
        return

    # --- Step 3: Find the primary merge points in the target file ---
    target_process = target_definitions.find(f"{{{BPMN_NS}}}process")
    target_diagram = target_definitions.find(f"{{{BPMNDI_NS}}}BPMNDiagram")

    if target_process is None:
        print(f"Error: No <bpmn:process> element found in target file '{target_file_path}'. Cannot merge process logic.")
        return
    if target_diagram is None:
        print(f"Error: No <bpmndi:BPMNDiagram> element found in target file '{target_file_path}'. Cannot merge diagram layout.")
        return

    target_plane = target_diagram.find(f"{{{BPMNDI_NS}}}BPMNPlane")
    if target_plane is None:
        return

    # --- Step 4: Collect all elements to be merged from the input file ---
    process_children_to_merge = []
    plane_children_to_merge = []
    other_definitions_to_merge = []

    for element in input_definitions:
        if element.tag == f"{{{BPMN_NS}}}process":
            process_children_to_merge.extend(list(element))
        elif element.tag == f"{{{BPMNDI_NS}}}BPMNDiagram":
            input_plane = element.find(f"{{{BPMNDI_NS}}}BPMNPlane")
            if input_plane is not None:
                plane_children_to_merge.extend(list(input_plane))
        else:
            # Collect other definitions like collaboration, message, etc.
            other_definitions_to_merge.append(element)

    # --- Step 5: Perform the merge by appending the collected elements ---
    total_merged_count = 0
    if process_children_to_merge:
        print(f"Merging {len(process_children_to_merge)} child element(s) into the target <bpmn:process>.")
        for child in process_children_to_merge:
            target_process.append(child)
        total_merged_count += len(process_children_to_merge)

    if plane_children_to_merge:
        print(f"Merging {len(plane_children_to_merge)} diagram element(s) into the target <bpmndi:BPMNPlane>.")
        for child in plane_children_to_merge:
            target_plane.append(child)
        total_merged_count += len(plane_children_to_merge)

    if other_definitions_to_merge:
        print(f"Merging {len(other_definitions_to_merge)} other top-level definition(s).")
        for definition in other_definitions_to_merge:
            target_definitions.append(definition)
        total_merged_count += len(other_definitions_to_merge)

    # --- Step 6: Save the result ---
    if total_merged_count > 0:
        target_tree.write(target_file_path, pretty_print=True, xml_declaration=True, encoding='UTF-8')
        print(f"Successfully completed granular merge into '{target_file_path}'.")
        print(f"Total elements added: {total_merged_count}.")
        print(
            "IMPORTANT: This merge does NOT handle ID conflicts. The resulting BPMN file might be invalid if element IDs from the source file already exist in the target.")
    else:
        print(f"No new elements were merged. '{target_file_path}' remains unchanged.")


def merge_bpmn_files_to_fragment():
    folder_path = Path('generated_data/flattened')
    file_paths = list(folder_path.glob('*.bpmn'))

    PROCESS_OFFSET = 120

    with open('fragments.bpmn', 'w') as f:
        pass

    for index, file_path in enumerate(file_paths):
        print(f"Processing '{file_path}'")
        if index == 0:
            merge_two_bpmn_files(file_path, 'fragments.bpmn', True)
        else:
            offset = calculate_y_coordinate_range('fragments.bpmn') + PROCESS_OFFSET
            offset_y_coordinates_in_bpmn(file_path, offset)
            merge_two_bpmn_files(file_path, 'fragments.bpmn', False)
            print(file_path)

def create_fcm_zip():
    file_paths = [
        'dataModel.xml',
        'goalState.xml',
        'fragments.bpmn',
        'olcs.xml'
    ]
    zip_name = 'fcm_js.zip'

    try:
        with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in file_paths:
                if os.path.exists(file_path):
                    zipf.write(file_path, os.path.basename(file_path))
        print(f"Successfully created {zip_name}")
        return os.path.abspath(zip_name)
    except Exception as e:
        print(f"An error occurred while zipping files: {e}")
        return None

# Calculates offset for an association label with source and target waypoints
def offset_initial_waypoint(source_x, target_x, source_y, target_y):
    offset = 42
    dx = target_x - source_x
    dy = target_y - source_y
    distance = math.hypot(dx, dy)
    scale = offset / distance
    x = source_x + dx * scale
    y = source_y + dy * scale
    return x, y