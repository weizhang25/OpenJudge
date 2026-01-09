"""This module provides a utility function for collecting the core information of all Grader classes."""

import ast
import json
import re
import time
from pathlib import Path
from queue import Queue
from typing import Any, Dict, List


class _GraderInfo:
    """A class that stores core information of a Grader class"""

    def __init__(
        self,
        file_path: str = "",
        class_name: str = "",
        parent_class_names: list = None,
        init_method: str = "",
        aevaluate_method: str = "",
    ):
        self.file_path = file_path
        self.class_name = class_name
        self.parent_class_names = parent_class_names
        self.init_method = init_method
        self.aevaluate_method = aevaluate_method

    def __str__(self):
        return json.dumps(self.__dict__, ensure_ascii=False)


def get_all_grader_info() -> List[Dict[str, Any]]:
    """Collect the information of all graders defined under the openjudge/graders folder."""
    t0 = time.time_ns()
    current_file_abs_path = Path(__file__).resolve()
    # print(f'current __file__:{__file__}')
    # print(f'current file abs path:{current_file_abs_path}')

    defs_of_classes_having_parent = {}
    unprocessed_folders = Queue()
    unprocessed_folders.put(Path(current_file_abs_path.parent.parent, "graders"))
    while not unprocessed_folders.empty():
        folder = unprocessed_folders.get()
        for item in folder.iterdir():
            if item.is_dir():
                unprocessed_folders.put(item)
            elif item.stem == "__init__" or item.suffix != ".py":
                # use heuristics to reduce candidate count
                continue
            else:
                _get_defs_of_classes_having_parent(item, defs_of_classes_having_parent)

    t1 = time.time_ns()
    print("------------------------------")
    print(f"defs_of_classes_having_parent:{len(defs_of_classes_having_parent)}")
    print(f"{(t1-t0)/1000000}ms")

    all_grader_class_defs = _get_grader_class_def(defs_of_classes_having_parent)

    t2 = time.time_ns()
    all_grader_info = []
    for _, (class_def, source_code) in all_grader_class_defs.items():
        grader_info = _parse_grader_class_def(class_def, source_code)
        all_grader_info.append(grader_info.__dict__)

    t3 = time.time_ns()
    print("------------------------------")
    print(f"all_grader_info:{len(all_grader_info)}")
    print(f"{(t3-t2)/1000000}ms")

    return all_grader_info


def _get_defs_of_classes_having_parent(py_file: Path, defs_of_classes_having_parent: Dict[ast.ClassDef, str]):
    """Get the definitions and source codes of all classe that have parent class."""
    # print(f'-----------------------\nparse py file:{py_file}')
    # t0 = time.time_ns()
    with open(py_file, "r", encoding="utf-8") as file:
        source_code = file.read()

    # Parse the source code into an AST node
    # t1 = time.time_ns()
    tree = ast.parse(source_code)
    # t2 = time.time_ns()
    for node in ast.walk(tree):
        # print(f'---------------\nnode:{node}\n---------------')
        if isinstance(node, ast.ClassDef) and node.bases:
            parent_count = len(node.bases)
            if parent_count > 1 or node.bases[0].id != "ABC":
                # print(f'--------\nClassDef node with non-ABC parent:{node}\n---------')
                defs_of_classes_having_parent[node] = source_code

    # t3 = time.time_ns()
    # print(f'----------')
    # print(f'file:{py_file}')
    # print(f'load: {(t1-t0)/100000}ms')
    # print(f'ast.parse: {(t2-t1)/100000}ms')
    # print(f'find sub classes: {(t3-t2)/100000}ms')


def _get_grader_class_def(defs_of_classes_having_parent: Dict[ast.ClassDef, str]):
    """Get the definitions of Grader classes"""
    # get grader class def nodes
    t1 = time.time_ns()
    all_grader_class_defs = {"BaseGrader": None}
    found_grader = True
    while found_grader:
        found_grader = False

        known_defs = []
        new_defs = []

        for class_def, source_code in defs_of_classes_having_parent.items():
            # print(f'checking {class_def.name}, parents:{[parent.id for parent in class_def.bases]}')
            # Skip whose already processed
            if class_def.name in all_grader_class_defs:
                known_defs.append(class_def)
                continue

            for parent in class_def.bases:
                if parent.id in all_grader_class_defs:
                    all_grader_class_defs[class_def.name] = (class_def, source_code)
                    new_defs.append(class_def)
                    found_grader = True
                break

        # print(f'known_defs:{known_defs}')
        # print(f'new_defs:{new_defs}')

        # optimize data for the next round, by removing whose already processed in this round
        for n in known_defs:
            defs_of_classes_having_parent.pop(n)
        for n in new_defs:
            defs_of_classes_having_parent.pop(n)

    all_grader_class_defs.pop("BaseGrader")

    t2 = time.time_ns()
    print("------------------------------")
    print(f"all_grader_class_defs:{len(all_grader_class_defs)}")
    print(f"{(t2-t1)/1000000}ms")
    return all_grader_class_defs


_INIT_METHOD = "__init__"
_AEVALUATE_METHOD = "aevaluate"
_TARGET_METHODS = set()
_TARGET_METHODS.add(_INIT_METHOD)
_TARGET_METHODS.add(_AEVALUATE_METHOD)

_NEWLINE_OR_MULTI_SPACE_PATTERN = re.compile(r"(\n|\s\s+)")


def _parse_grader_class_def(class_def: ast.ClassDef, source_code: str) -> _GraderInfo:
    """Use ast util to extract core information from a Grader class,
    and put the information into a _GraderInfo object."""

    init_method = ""
    aeval_method = ""
    # Find target methods within the class body
    for sub_node in class_def.body:
        if isinstance(sub_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            method_name = sub_node.name
            if method_name not in _TARGET_METHODS:
                continue

            # print(f'---------------\nFunctionDef node:{sub_node}\n---------------')
            # print(f'method:{method_name}')
            # print(f'method docstring:{ast.get_docstring(sub_node, clean=True)}')
            # t0 = time.time_ns()
            segment = ast.get_source_segment(source_code, sub_node)
            # t1 = time.time_ns()
            # print(f'method segment:||||{segment}||||')
            segment = _NEWLINE_OR_MULTI_SPACE_PATTERN.sub(" ", segment.strip())
            # t2 = time.time_ns()
            segment = segment.replace(") :", "):").replace(") ->", ")->").replace(" :", ":")
            # t3 = time.time_ns()
            # print(f'----------')
            # print(f'grader {class_def.name}')
            # print(f'ast.get_source_segment: {(t1-t0)/100000}ms')
            # print(f'segment re.sub: {(t2-t1)/100000}ms')
            # print(f'segment replace: {(t3-t2)/100000}ms')

            # Method head ends in two ways, w/ or w/o return type annocation.
            # Figure it out.
            idx0 = segment.find("):")
            idx1 = segment.find(")->")
            if idx1 > 0:
                idx1 = segment.find(":", idx1)

            if idx0 > 0 and idx1 > 0:
                if idx0 < idx1:
                    idx = idx0 + 2
                else:
                    idx = idx1 + 1
            elif idx0 > 0:
                idx = idx0 + 2
            elif idx1 > 0:
                idx = idx1 + 1
            else:
                idx = -1

            # def foo(...) -> type:
            # def foo(...):
            if idx > 0:
                signature = segment[0:idx]
            else:
                signature = "SIGNATURE_NOT_FOUND"

            if method_name == _INIT_METHOD:
                init_method = signature
            elif method_name == _AEVALUATE_METHOD:
                aeval_method = signature
            # print(f'method signature:||||{signature}||||\n')

    # t0 = time.time_ns()
    grader_info_obj = _GraderInfo(
        class_name=class_def.name,
        parent_class_names=[p.id for p in class_def.bases],
        init_method=init_method,
        aevaluate_method=aeval_method,
    )
    # t1 = time.time_ns()
    # print(f'to _GraderInfo obj: {(t1-t0)/100000}ms')
    return grader_info_obj


if __name__ == "__main__":
    graders = get_all_grader_info()
    print("-------------")
    print(f"{len(graders)} graders")
    for g_i in graders:
        print("-------------")
        print(g_i)
