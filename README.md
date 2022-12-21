# Jkinpylib

'Jeremy's kinematics python library'. This library runs forward and inverse kinematics in parrallel on the gpu/cpu using pytorch. It also can call single value FK/IK solvers from Klamp't

Note: This project uses the `w,x,y,z` format for quaternions.

## Installation

Recommended: clone the repo and install with pip
```
git clone https://github.com/jstmn/jkinpylib.git && cd jkinpylib/
pip install -e .

```
or:
```
pip install -e ".[dev]"
```

Second option: Install from pypi (not recomended - the pypi version will likely be out of date until this project hardens)
``` bash
pip install jkinpylib
```


## Todos
- [x] Remove `fix_urdf.py` hackery - don't change joint types. Maintain the original urdf, save a formatted one with minor tweaks
- [ ] Add additional robots (from the ikflow repo)
- [ ] batched IK optimization steps