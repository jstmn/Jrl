import os
from typing import List, Tuple, Dict
from dataclasses import dataclass
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, ElementTree

import numpy as np

from jkinpylib.config import URDF_DOWNLOAD_DIR
from jkinpylib.utils import get_filepath, safe_mkdir

# See http://wiki.ros.org/urdf/XML/joint
# All types: 'revolute', 'continuous', 'prismatic', 'fixed', 'floating', 'planar'
UNHANDLED_JOINT_TYPES = ["floating", "planar"]


# TODO: Consider using an existing urdf parser (like https://github.com/ros/urdf_parser_py/tree/melodic-devel/src)
@dataclass
class Link:
    name: str

    # TODO: link Links and Joints
    # parent_joints: List["Joint"] = None
    # child_joints: List["Joint"] = None

    def __str__(self):
        return f"<Link(), '{self.name}'>\n"

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, __o: object) -> bool:
        return self.name == __o.name


@dataclass
class Joint:
    name: str
    # parent, child are the name of the links that this joint connects to
    parent: str
    child: str
    # TODO: link Links and Joints
    # parent_link: Link
    # child_link: Link
    origin_rpy: Tuple[float, float, float]
    origin_xyz: Tuple[float, float, float]
    axis_xyz: Tuple[float, float, float]
    joint_type: str
    limits: Tuple[float, float]

    @property
    def is_actuated(self) -> bool:
        return self.joint_type != "fixed"

    def set_to_fixed(self):
        self.joint_type = "fixed"
        self.limits = (0, 0)
        self.__post_init__()

    def __post_init__(self):
        assert len(self.origin_rpy) == 3
        assert len(self.origin_xyz) == 3
        assert (
            len(self.limits) == 2
        ), f"limits should be length 2, currently {len(self.limits)} (self.limits={self.limits})"

        # Note: 'fixed' joints have been observed to have non zero limits, for example (0, 0.04) - see 'panda.urdf'. Not
        # sure what's up with that. Ignoring this for now.
        if self.is_actuated:
            assert (
                self.limits[0] <= self.limits[1]
            ), f"lower limit should be less or equal than upper limit, currently {self.limits[0]} <= {self.limits[1]}"

        # If joint_type is 'fixed' we can ignore `axis_xyz`
        if self.joint_type != "fixed":
            assert len(self.axis_xyz) == 3

        assert isinstance(self.limits[0], (int, float))
        assert isinstance(self.limits[1], (int, float))

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


def _get_link_by_name(link_name: str, all_links: List[Link]) -> Link:
    """Returns the link with the given name"""
    matching = None
    for link in all_links:
        if link.name == link_name:
            assert matching is None, f"Multiple links with name '{link_name}' found"
            matching = link
    assert matching is not None, f"link '{link_name}' not found (all known={[link.name for link in all_links]})"
    assert isinstance(matching, Link)
    return matching


def _get_joint_by_name(joint_name: str, all_joints: List[Joint]) -> Joint:
    """Returns the link with the given name"""
    matching = None
    for joint in all_joints:
        if joint.name == joint_name:
            assert matching is None, f"Multiple joints with name '{joint_name}' found"
            matching = joint
    assert matching is not None, f"joint '{joint_name}' not found (all known={[joint.name for joint in all_joints]})"
    assert isinstance(matching, Joint)
    return matching


def _len3_tuple_from_str(s) -> Tuple[float, float, float]:
    """Return a length 3 tuple of floats from a string.

    Example input:
        '0 0 0.333'
        '0 0 1'
    """
    s = s.strip()
    s = s.replace("[", "")
    s = s.replace("]", "")
    while "  " in s:
        s = s.replace("   ", " ")
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
                        except ValueError as exc:
                            raise ValueError(
                                f"Error: _len3_tuple_from_str() returned ValueError for joint '{child.get('name')}'."
                                f" 'rpy': {subelem.attrib['rpy']}"
                            ) from exc

                        try:
                            origin_xyz = _len3_tuple_from_str(subelem.attrib["xyz"])
                        except RuntimeError as exc:
                            raise ValueError(
                                f"Error: joint <joint name='{child.get('name')}'> has no xyz attribute, or it's"
                                " illformed"
                            ) from exc
                    elif subelem.tag == "axis":
                        axis_xyz = _len3_tuple_from_str(subelem.attrib["xyz"])
                    elif subelem.tag == "parent":
                        parent = subelem.attrib["link"]
                    elif subelem.tag == "child":
                        joint_child = subelem.attrib["link"]
                    elif subelem.tag == "limit":
                        if child.tag == "joint" and child.attrib["type"] == "continuous":
                            limits[0] = -np.pi
                            limits[1] = np.pi

                            print(
                                "Heads up: Setting joint limits to [-pi, pi] for continuous joint"
                                f" '{child.attrib['name']}'"
                            )

                        else:
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


def get_urdf_filepath_w_filenames_updated(original_filepath: str, download_dir: str = URDF_DOWNLOAD_DIR) -> str:
    """Save a copy of the urdf filepath, but with the mesh filepaths updated to absolute paths."""
    _, filename = os.path.split(original_filepath)
    filename = filename.replace(".urdf", "_link_filepaths_absolute.urdf")
    safe_mkdir(download_dir)
    output_filepath = os.path.join(download_dir, filename)

    with open(original_filepath, "r") as urdf_file:
        root = ET.fromstring(urdf_file.read())
        for mesh_element in root.iter("mesh"):
            if "filename" in mesh_element.attrib:
                mesh_element.attrib["filename"] = get_filepath(mesh_element.attrib["filename"])

    with open(output_filepath, "wb") as f:
        tree = ElementTree(root)
        tree.write(f)

    return output_filepath


class DFSSearcher:
    def __init__(self, all_joints: List[Joint], all_links: List[Link], base_link: Link, end_effector_link: Link):
        self._all_joints = all_joints
        self._all_links = all_links
        self._base_link = base_link
        self._end_effector_link = end_effector_link
        assert isinstance(self._base_link, Link)
        assert isinstance(self._end_effector_link, Link)

    def _get_child_links(self, link_name: str) -> List[Link]:
        """Returns the names of all links that are children of the given link"""
        return [
            _get_link_by_name(joint.child, self._all_links) for joint in self._all_joints if joint.parent == link_name
        ]

    def _path_from_start(self, node: Tuple, child_to_parent_map: Dict):
        path = [node]
        child = node
        while child in child_to_parent_map:
            parent = child_to_parent_map[child]
            path.append(parent)
            child = parent
        path.reverse()
        return path

    def dfs(self) -> List[Tuple]:
        """Searches the graph using depth-first search

        Returns:
            List[Tuple]: The path found by the search
        """

        stack = [self._base_link]
        visited = set()
        child_to_parent_map = {}

        def _is_goal_state(node):
            return node == self._end_effector_link  # _eq_ is overloaded for Link

        while len(stack) > 0:
            parent = stack.pop(0)
            if parent in visited:
                continue

            if _is_goal_state(parent):
                return self._path_from_start(parent, child_to_parent_map)

            visited.add(parent)
            for child in self._get_child_links(parent.name):
                child_to_parent_map[child] = parent
                stack.insert(0, child)

        raise RuntimeError("No path found")


def joint_path_from_link_path(link_path: List[Link], all_joints: List[Joint]) -> List[Joint]:
    """Returns the joint path corresponding to the given link path"""
    path = []

    for i in range(len(link_path) - 1):
        joint_found = False
        for joint in all_joints:
            if link_path[i].name == joint.parent and link_path[i + 1].name == joint.child:
                assert joint_found is False, "Found multiple joints between two links"
                joint_found = True
                path.append(joint)

    return path


def get_joint_chain(
    urdf_filepath: str, active_joints: List[str], base_link_name: str, end_effector_name: str
) -> List[Joint]:
    """Returns a list of joints from the base link to the end effector. Runs DFS to find the path, and checks that the
    path is valid before returning it.

    Args:
        active_joints (List[str]): The joints in the kinematic chain (specified by the user).
    """
    all_joints, all_links = parse_urdf(urdf_filepath)  # Dicts
    all_joints = list(all_joints.values())
    all_joint_names = tuple([j.name for j in all_joints])
    all_links = list(all_links.values())
    all_link_names = tuple([j.name for j in all_links])

    base_link = _get_link_by_name(base_link_name, all_links)
    end_effector_link = _get_link_by_name(end_effector_name, all_links)

    # Run DFS to find the path from the base link to the end effector
    dfs_searcher = DFSSearcher(all_joints, all_links, base_link, end_effector_link)
    link_path = dfs_searcher.dfs()
    joint_path = joint_path_from_link_path(link_path, all_joints)
    assert len(joint_path) >= len(active_joints), (
        f"Expected as many joints in the joint_path as active_joints (expected {len(active_joints)} but got"
        f" {len(joint_path)}"
    )

    # Set all joints not in `active_joints` to fixed
    for joint in all_joints:
        if joint.name not in active_joints:
            joint.set_to_fixed()

    # Check that all joints in `active_joints` are in the urdf
    for active_joint_name in active_joints:
        assert (
            active_joint_name in all_joint_names
        ), f"active joint '{active_joint_name}' not found in the urdf (all present: {all_joint_names})"
        matching_joint = _get_joint_by_name(active_joint_name, all_joints)
        assert matching_joint.joint_type != "fixed", f"active joint '{active_joint_name}' is fixed"

    for joint in all_joints:
        if joint.name in active_joints:
            assert joint.joint_type != "fixed", f"joint '{joint.name}' should not be fixed, because it's active"
        else:
            assert joint.joint_type == "fixed", f"joint '{joint.name}' should be fixed, because it's in active_joints"

    # Check that `end_effector_name` and `base_link_name` are in the urdf and in the link path
    for link in (base_link, end_effector_link):
        assert link in all_links, f"link '{link}' not found in the urdf (all links: {all_link_names})"
        assert link in link_path, f"link '{link}' not found in the link_path (link_path: {link_path})"

    return joint_path
