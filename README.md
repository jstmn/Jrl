# Jkinpylib

'Jeremy's kinematics python library'. This library runs forward and inverse kinematics in parrallel on the gpu/cpu using pytorch. It also can call single value FK/IK solvers from Klamp't

## Installation

Install base dependencies
```
sudo apt install python3-pip
python3.8 -m pip install --user virtualenv
```

Create virtual environment
```
python3.8 -m venv venv/ && source venv/bin/activate
pip install -e .
```

## Todos
- [ ] Remove `fix_urdf.py` hackery - don't change joint types. Maintain the original urdf, save an additional one for klampt (with problematic elements removed)
- [ ] Add additional robots (from the ikflow repo)
- [ ] batched IK optimization steps