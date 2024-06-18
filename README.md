# Safety Arithmetic: A Framework for Test-time Safety Alignment of Language Models by Steering Parameters and Activations

:point_right: Dataset coming soon!

👉 [Read the paper](https://arxiv.org/abs/2406.11801)

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

### Safety Arithmetic
```
Run Safety_Arithmetic_Base_and_SFT.ipynb file for BASE and SFT models.
Run Safety_Arithmetic_Edited.ipynb file for EDITED models.
```
### Harm Direction Removal (HDR) (w/ TIES)
```
Run HDR/HDR_TIES_BASE_AND_SFT.ipynb for SFT models and BASE models
Run HDR/HDR_TIES_EDITED.ipynb for EDITED model.
```
### Harm Direction Removal (HDR) (w/ Task Vector)
```
Run HDR/HDR_Task_Vector_BASE.ipynb for BASE models
Run HDR/HDR_Task_Vector_SFT.ipynb for SFT models
Run HDR/HDR_Task_Vector_EDITED.ipynb for EDITED models.
```
### Only ICV
```
Run Safety_Arithmetic_Base_and_SFT.ipynb file by passing direct base/sft (without HDR).
Run Safety_Arithmetic_Edited.ipynb file by passing direct edited (without HDR).
```

## Citation
If you find this useful in your research, please consider citing:

```
@misc{hazra2024safety,
      title={Safety Arithmetic: A Framework for Test-time Safety Alignment of Language Models by Steering Parameters and Activations}, 
      author={Rima Hazra and Sayan Layek and Somnath Banerjee and Soujanya Poria},
      year={2024},
      eprint={2406.11801},
      archivePrefix={arXiv},
      primaryClass={id='cs.CL' full_name='Computation and Language' is_active=True alt_name='cmp-lg' in_archive='cs' is_general=False description='Covers natural language processing. Roughly includes material in ACM Subject Class I.2.7. Note that work on artificial languages (programming languages, logics, formal systems) that does not explicitly address natural-language issues broadly construed (natural-language processing, computational linguistics, speech, text retrieval, etc.) is not appropriate for this area.'}
}
```
