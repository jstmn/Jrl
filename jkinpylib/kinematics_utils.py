from typing import List, Tuple, Dict
from dataclasses import dataclass
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element

# See http://wiki.ros.org/urdf/XML/joint
# All types: 'revolute', 'continuous', 'prismatic', 'fixed', 'floating', 'planar'
UNHANDLED_JOINT_TYPES = ["prismatic", "floating", "planar"]


# TODO: Consider using an existing urdf parser (like https://github.com/ros/urdf_parser_py/tree/melodic-devel/src)
@dataclass
class Link:
    name: str

    def __str__(self):
        return f"<Link(), '{self.name}'>\n"


@dataclass
class Joint:
    name: str
    # parent, child are the name of the links that this joint connects to
    parent: str
    child: str
    origin_rpy: Tuple[float, float, float]
    origin_xyz: Tuple[float, float, float]
    axis_xyz: Tuple[float, float, float]
    joint_type: str
    limits: Tuple[float, float]

    @property
    def is_actuated(self) -> bool:
        return self.joint_type != "fixed"

    def __post_init__(self):
        assert len(self.origin_rpy) == 3
        assert len(self.origin_xyz) == 3
        assert (
            len(self.limits) == 2
        ), f"limits should be length 2, currently {len(self.limits)} (self.limits={self.limits})"
        assert self.joint_type not in UNHANDLED_JOINT_TYPES

        # Note: 'fixed' joints have been observed to have non zero limits, for example (0, 0.04) - see 'panda.urdf'. Not
        # sure what's up with that. Ignoring this for now.
        if self.is_actuated:
            assert (
                self.limits[0] <= self.limits[1]
            ), f"lower limit should be less or equal than upper limit, currently {self.limits[0]} <= {self.limits[1]}"

        # If joint_type is 'fixed' we can ignore `axis_xyz`
        if not self.joint_type == "fixed":
            assert len(self.axis_xyz) == 3

    def __str__(self):
        ret = f"\n<Joint(), '{self.name}'>\n"
        ret += f"  joint_type: {self.joint_type}\n"
        ret += f"  parent:     {self.parent}\n"
        ret += f"  child:      {self.child}\n"
        ret += f"  origin_rpy: {self.origin_rpy}\n"
        ret += f"  origin_xyz: {self.origin_xyz}\n"
        ret += f"  axis_xyz:   {self.axis_xyz}\n"
        ret += f"  limits:     {self.limits}"
        return ret


def _len3_tuple_from_str(s) -> Tuple[float, float, float]:
    """Return a length 3 tuple of floats from a string.

    Example input:
        '0 0 0.333'
        '0 0 1'
    """
    s = s.strip()
    s = s.replace("[", "")
    s = s.replace("]", "")
    s = s.replace("  ", " ")
    space_split = s.split(" ")
    out = tuple(float(dig) for dig in space_split)
    return out


def parse_urdf(urdf_filepath: str) -> Tuple[Dict[str, Joint], Dict[str, Link]]:
    """Return all joints and links in a urdf, represented as Joints and Links.

    NOTE: The limits of each joint are [0, 0] if none are present"""
    links = {}
    joints = {}
    with open(urdf_filepath, "r") as urdf_file:
        root = ET.fromstring(urdf_file.read())
        for child in root:
            child: Element
            if child.tag == "link":
                link_name = child.attrib["name"]
                links[link_name] = Link(link_name)

            elif child.tag == "joint":
                joint_type = child.attrib["type"]
                joint_name = child.attrib["name"]
                limits = [0, 0]

                origin_rpy, origin_xyz, axis_xyz = None, None, None
                parent, joint_child = None, None

                for subelem in child:
                    subelem: Element
                    if subelem.tag == "origin":
                        try:
                            origin_rpy = _len3_tuple_from_str(subelem.attrib["rpy"])
                        except KeyError:
                            # Per (http://library.isr.ist.utl.pt/docs/roswiki/urdf(2f)XML(2f)Joint.html):
                            #       "rpy (optional: defaults 'to zero vector 'if not specified)"
                            origin_rpy = [0, 0, 0]
                        try:
                            origin_xyz = _len3_tuple_from_str(subelem.attrib["xyz"])
                        except RuntimeError:
                            raise ValueError(
                                f"Error: joint <joint name='{child.get('name')}'> has no xyz attribute, or it's"
                                " illformed"
                            )
                    elif subelem.tag == "axis":
                        axis_xyz = _len3_tuple_from_str(subelem.attrib["xyz"])
                    elif subelem.tag == "parent":
                        parent = subelem.attrib["link"]
                    elif subelem.tag == "child":
                        joint_child = subelem.attrib["link"]
                    elif subelem.tag == "limit":
                        limits[0] = float(subelem.attrib["lower"])
                        limits[1] = float(subelem.attrib["upper"])

                joint = Joint(
                    name=joint_name,
                    parent=parent,
                    child=joint_child,
                    origin_rpy=origin_rpy,
                    origin_xyz=origin_xyz,
                    axis_xyz=axis_xyz,
                    joint_type=joint_type,
                    limits=tuple(limits),
                )
                joints[joint_name] = joint

    return joints, links


def get_joint_chain(urdf_filepath: str, active_joints: List[str], end_effector_name: str) -> List[Joint]:
    """Return the 'joint chain', which is the list of joints that form the kinematic chain that is being
    represented. This list is in a sense a linked-list, where the child of each joint is the parent of the next
    joint.

    Args:
        active_joints (List[str]): The joints in the kinematic chain (specified by the user).
    """
    all_joints, all_links = parse_urdf(urdf_filepath)

    # Check that all joints in `joints` are in the urdf
    for joint in active_joints:
        assert (
            joint in all_joints
        ), f"joint '{joint}' not in the urdf (present active_joints: {list(x for x in all_links)})"

    # Check that `end_effector_name` is in the urdf
    assert (
        end_effector_name in all_links
    ), f"link '{end_effector_name}' not found in the urdf (present links: {list(x for x in all_links)})"

    # Check that `active_joints` forms a linked list
    for i in range(len(active_joints) - 1):
        joint = all_joints[active_joints[i]]
        next_joint = all_joints[active_joints[i + 1]]
        assert (
            joint.child == next_joint.parent
        ), f"Error: joint('{joint.name}').child != joint_('{next_joint.name}').parent"

    assert (
        all_joints[active_joints[-1]].child == end_effector_name
    ), f"Error: the final joint's child != end_effector_link_name ('{self.end_effector_link_name}')"

    return [all_joints[joint_name] for joint_name in active_joints]
