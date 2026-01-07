"""This module provides a utility function for collecting the core information of all Grader classes."""

import ast
import json
import re
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
    current_file_abs_path = Path(__file__).resolve()
    # print(f'current __file__:{__file__}')
    # print(f'current file abs path:{current_file_abs_path}')

    # List[_GraderInfo]
    graders_info = []
    unprocessed_folders = Queue()
    unprocessed_folders.put(Path(current_file_abs_path.parent.parent, "graders"))

    while not unprocessed_folders.empty():
        folder = unprocessed_folders.get()
        for item in folder.iterdir():
            if item.is_dir():
                unprocessed_folders.put(item)
            elif item.suffix == ".py":
                graders_info.extend(_parse_grader(item))

    # print(f'{len(graders)} graders')
    # for gi in graders:
    #     print(str(gi))
    return [gi.__dict__ for gi in graders_info]


_INIT_METHOD = "__init__"
_AEVALUATE_METHOD = "aevaluate"
_TARGET_METHODS = set()
_TARGET_METHODS.add(_INIT_METHOD)
_TARGET_METHODS.add(_AEVALUATE_METHOD)


def _parse_grader(py_file: Path) -> List[_GraderInfo]:
    """Use ast util to extract core information from a Grader class,
    and put the information into a _GraderInfo object."""
    # print(f'-----------------------\nparse py file:{py_file}')
    graders = []
    with open(py_file, "r", encoding="utf-8") as file:
        source_code = file.read()

    # Parse the source code into an AST node
    tree = ast.parse(source_code)
    for node in ast.walk(tree):
        # print(f'---------------\nnode:{node}\n---------------')
        if not isinstance(node, ast.ClassDef):
            # print(f'not a class, skipped')
            continue

        # print(f'---------------\nClassDef node:{node}\n---------------')
        is_grader = False
        for parent in node.bases:
            if parent.id and parent.id.endswith("Grader"):
                is_grader = True
                break

        if not is_grader:
            # print(f'--------\nnot a grader class, skipped----------')
            continue

        # print(f'Grader Class:{node.name}')

        parents = []
        for parent in node.bases:
            parents.append(parent.id)
        # if parents:
        #     if len(parents) == 1:
        #         print(f'parent:{parents[0]}')
        #     else:
        #         print(f'parents:{parents}')
        # else:
        #     print(f'parent not found in bases:{node.bases}')

        init_method = ""
        aeval_method = ""
        # Find target methods within the class body
        for sub_node in node.body:
            if isinstance(sub_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_name = sub_node.name
                if method_name not in _TARGET_METHODS:
                    continue

                # print(f'---------------\nFunctionDef node:{sub_node}\n---------------')
                # print(f'method:{method_name}')
                # for arg in sub_node.args.args:
                #     print(f'args.arg:{arg.arg}')
                # for arg in sub_node.args.posonlyargs:
                #     print(f'args.posonlyarg:{arg.arg}')
                # for arg in sub_node.args.kwonlyargs:
                #     print(f'args.kwonlyarg:{arg.arg}')
                # print(f'args.vararg:{sub_node.args.vararg.arg if sub_node.args.vararg else None}')
                # print(f'args.kwarg:{sub_node.args.kwarg.arg if sub_node.args.kwarg else None}')
                # print(f'method docstring:{ast.get_docstring(sub_node, clean=True)}')
                segment = ast.get_source_segment(source_code, sub_node)
                # print(f'method segment:||||{segment}||||')
                segment = re.sub(r"(\n|\s\s+)", " ", segment.strip())
                segment = segment.replace(") :", "):").replace(") ->", ")->").replace(" :", ":")
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

        graders.append(
            _GraderInfo(
                file_path=str(py_file),
                class_name=node.name,
                parent_class_names=parents,
                init_method=init_method,
                aevaluate_method=aeval_method,
            )
        )

    return graders


if __name__ == "__main__":
    all_grader_info = get_all_grader_info()
    for grader_info in all_grader_info:
        print("-------------")
        print(grader_info)
