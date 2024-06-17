# Safety Arithmetic: A Framework for Test-time Safety Alignment of Language Models by Steering Parameters and Activations

:point_right: Dataset coming soon!

## Table of Contents

- [Installation](#installation)
- [Experiments](#experiments)
- [FileStructure](#filestructure)
- [Citation](#citation)

## Installation

```
pip install -r requirement.txt
```

## Experiments 

<ol>
  <li>Safety Arithmetic</li>
  <li>Harm Direction Removal (HDR): TIES, Task Vector</li>
  <li>ICV</li>
</ol>

## FileStructure

# Safety Arithmetic
```
Run Safety_Arithmetic_Base_and_SFT.ipynb file for BASE and SFT models.
Run Safety_Arithmetic_Edited.ipynb file for EDITED models.
```
# Harm Direction Removal (HDR) (w/ TIES)
```
Run HDR/HDR_TIES_BASE_AND_SFT.ipynb for SFT models and BASE models
Run HDR/HDR_TIES_EDITED.ipynb for EDITED model.
```
# Harm Direction Removal (HDR) (w/ Task Vector)
```
Run HDR/HDR_Task_Vector_BASE.ipynb for BASE models
Run HDR/HDR_Task_Vector_SFT.ipynb for SFT models
Run HDR/HDR_Task_Vector_EDITED.ipynb for EDITED models.
```
# Only ICV
```
Run Safety_Arithmetic_Base_and_SFT.ipynb file by passing direct base/sft (without HDR).
Run Safety_Arithmetic_Edited.ipynb file by passing direct edited (without HDR).
```

## Citation
If you find this useful in your research, please consider citing:
