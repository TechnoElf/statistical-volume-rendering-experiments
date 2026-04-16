"""Reconstruct Python source from a .pyc file using bytecode analysis."""

import dis
import io
import marshal
import sys
import types

pyc_path = sys.argv[1]

with open(pyc_path, "rb") as f:
    f.read(16)  # skip header
    code = marshal.load(f)

print(f"# Reconstructed from: {code.co_filename}")
print(f"# Original source size noted in header")
print()


def get_all_code_objects(co, path=""):
    name = f"{path}.{co.co_name}" if path else co.co_name
    result = [(name, co)]
    for c in co.co_consts:
        if isinstance(c, types.CodeType):
            result.extend(get_all_code_objects(c, name))
    return result


for name, co in get_all_code_objects(code):
    print(f"{'=' * 60}")
    print(f"# {name}  (line {co.co_firstlineno})")
    print(f"# args: {co.co_varnames[: co.co_argcount]}")
for name, co in get_all_code_objects(code):
    print(f"{'=' * 60}")
    print(f"# {name}  (line {co.co_firstlineno})")
    print(f"# args: {co.co_varnames[: co.co_argcount]}")
    print(f"# locals: {co.co_varnames}")
    consts = [c for c in co.co_consts if not isinstance(c, types.CodeType)]
    print(f"# consts: {consts}")
    print(f"# names: {co.co_names}")
    print(f"# freevars: {co.co_freevars}")
    print(f"# cellvars: {co.co_cellvars}")
    print()
    dis.dis(co)
    print()
