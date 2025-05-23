# Refactor of my codebase into a user-friendly reflection classification library. See demo.py for example usage. 

(preliminary): train_setfit and train_fastfit use NVIDIA CUDA (device="cuda") by default, and requirements.txt attempts to install torch 2.4.1+cu124. If you're on a Mac and using Apple silicon/mps, pass the parameter device="mps" to calls to train_setfit and train_fastfit to change to Pytorch's mps backend.

All functions have docstrings that explain correct usage; use them for guidance and demo.py for examples.

Unfortunately, due to shortcomings with FastFit and SetFit's interfaces (and a backwards compatibility bug with transformers), this code requires specially altered versions of those libraries' source codes to run. Provided in altered_dependency_files are the files that must be replaced in the necessary libraries (after installing them in requirements.txt). 