from lxml import etree as ET
import uuid
from collections import defaultdict

from pm4py import OCEL

def finish_bpmn(ocel: OCEL, connected_object_activity_graph: defaultdict[str, defaultdict[str, set]]):
    # Generate the output data objects first so that they may be reused for input again
    for activity, values in connected_object_activity_graph.items():
        for object_type, states in values.items():
            add_data_object_to_bpmn(f'./generated_data/flattened/bpmn_{object_type}.bpmn', activity, f'{object_type}\n[{activity}]', input=False)

    # Generate input data objects
    for activity, values in connected_object_activity_graph.items():
        for object_type, states in values.items():
            states_name = ""
            if len(states) == 0:
                states_name = "[new]"
            else:
                states_name = "[" + " | ".join(states) + "]"

            add_data_object_to_bpmn(f'./generated_data/flattened/bpmn_{object_type}.bpmn', activity, f'{object_type}\n{states_name}', input=True)


def add_data_object_to_bpmn(file_path, activity_name, data_object_name, input=True, output_path=None):
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

    # If found, reuse it; otherwise, create new one
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
        data_obj_id = data_obj_ref.get('dataObjectRef')

    data_assoc_id = f"DataAssoc_{uuid.uuid4().hex[:8]}"
    shape_id = f"{data_obj_ref_id}_di"

    # Create data association and add to task
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

    # Add BPMNShape in diagram (above task)
    plane = root.find(f'.//{{{bpmndi_ns}}}BPMNPlane')
    if plane is None:
        raise ValueError("No <bpmndi:BPMNPlane> found.")

    # Find the shape of the task to position data object
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
    height = float(bounds.get('height'))

    # Determine shape position for data object
    if is_new_object:
        # Place data object left or right above task
        if input:
            data_x = x + (width - 36) / 2 - 50
            data_y = y - 100
        else:
            data_x = x + (width - 36) / 2 + 50
            data_y = y - 100

        # Create BPMNShape
        data_shape = ET.Element(f"{{{bpmndi_ns}}}BPMNShape", id=shape_id, bpmnElement=data_obj_ref_id)
        bounds_elem = ET.Element(f"{{{dc_ns}}}Bounds", x=str(data_x), y=str(data_y), width="36", height="50")
        label_elem = ET.Element(f"{{{bpmndi_ns}}}BPMNLabel")
        label_bounds = ET.Element(f"{{{dc_ns}}}Bounds", x=str(data_x + 6), y=str(data_y + 57), width="25", height="14")
        label_elem.append(label_bounds)
        data_shape.append(bounds_elem)
        data_shape.append(label_elem)
        plane.append(data_shape)
    else:
        # Reuse coordinates from existing shape
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

    # Add BPMNEdge to show the data association visually
    di_ns = nsmap.get('omgdi') or 'http://www.omg.org/spec/DD/20100524/DI'
    edge_id = f"{data_assoc_id}_edge"
    bpmn_edge = ET.Element(f"{{{bpmndi_ns}}}BPMNEdge", id=edge_id, bpmnElement=data_assoc_id)

    # From data object (bottom center) to task (top center) or vice versa
    if input:
        # Input: data → task
        wp1 = ET.Element(f"{{{di_ns}}}waypoint", x=str(data_x + 18), y=str(data_y + 50))  # bottom left of data object
        wp2 = ET.Element(f"{{{di_ns}}}waypoint", x=str(x + width / 2 - 10), y=str(y))  # top left of task
    else:
        # Output: task → data
        wp1 = ET.Element(f"{{{di_ns}}}waypoint", x=str(x + width / 2 + 10), y=str(y))  # top right of task
        wp2 = ET.Element(f"{{{di_ns}}}waypoint", x=str(data_x + 18), y=str(data_y + 50))  # bottom right of data object

    bpmn_edge.extend([wp1, wp2])
    plane.append(bpmn_edge)

    # Save file
    output_path = output_path or file_path
    tree.write(output_path, pretty_print=True, xml_declaration=True, encoding='UTF-8')
    print(f"Data object '{data_object_name}' added above task '{activity_name}'.")