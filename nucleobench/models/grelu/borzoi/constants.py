"""Constants for gRelu's Borzoi model.


To test some parts of this:
```zsh
python -m nucleobench.models.grelu.borzoi.constants
```
"""

import os

import pandas as pd

BORZOI_REPO_ID = "Genentech/borzoi-model"
BORZOI_FILENAME = "human_rep0.ckpt"
BORZOI_TRAIN_LEN_ = 524_288

# In gRelu, this list is determined from:
# ```python
# model.data_params['tasks']['description']
# ```
this_dir = os.path.dirname(os.path.abspath(__file__))
BORZOI_TASKS_ = pd.read_csv(
    os.path.join(this_dir, "borzoi_tasks.csv")
).description.values.tolist()

activate_muscle_tracks = """
'DNASE:psoas muscle male adult (27 years) and male adult (35 years)'

'DNASE:skeletal muscle myoblast'

'DNASE:myotube originated from skeletal muscle myoblast'

'DNASE:myocyte originated from LHCN-M2'

'DNASE:skeletal muscle cell'

'DNASE:muscle of back female embryo (113 days)'

'DNASE:muscle of leg male embryo (97 days)'

'DNASE:muscle of back female embryo (105 days)'

'DNASE:muscle of leg male embryo (96 days)'

'DNASE:muscle of arm female embryo (115 days)'

'DNASE:muscle of back female embryo (98 days)'

'DNASE:muscle of back female embryo (115 days)'

'DNASE:muscle of back male embryo (104 days)'

'DNASE:forelimb muscle female embryo (108 days)'

'DNASE:muscle of leg male embryo (101 day)'

'DNASE:muscle of back male embryo (101 day)'

'DNASE:muscle of back female embryo (85 days)'

'DNASE:muscle of leg female embryo (85 days)'

'DNASE:muscle of trunk female embryo (120 days)'

'DNASE:muscle of arm male embryo (113 days)'

'DNASE:muscle of leg female embryo (113 days)'

'DNASE:muscle of leg female embryo (105 days)'

'DNASE:muscle of leg male embryo (97 days)'

'DNASE:muscle of back male embryo (97 days)'

'DNASE:muscle of leg male embryo (96 days)'

'DNASE:muscle of arm male embryo (96 days)'

'DNASE:muscle of arm female embryo (85 days)'

'DNASE:muscle of back male embryo (91 day)'

'DNASE:muscle of arm male embryo (101 day)'

'DNASE:muscle of arm female embryo (120 days)'

'DNASE:muscle of back male embryo (105 days)'

'DNASE:hindlimb muscle male embryo (120 days)'

'DNASE:muscle of trunk female embryo (121 day)'

'DNASE:psoas muscle male child (3 years)'

'DNASE:muscle of back male embryo (96 days)'

'DNASE:muscle of trunk female embryo (113 days)'

'DNASE:muscle of arm male embryo (120 days)'

'DNASE:muscle of back male embryo (108 days)'

'DNASE:muscle of leg male embryo (105 days)'

'DNASE:muscle of arm female embryo (105 days)'

'DNASE:muscle of back female embryo (105 days)'

'DNASE:muscle of arm male embryo (97 days)'

'DNASE:muscle of leg male embryo (127 days)'

'DNASE:muscle of arm male embryo (97 days)'

'DNASE:muscle of back male embryo (96 days)'

'DNASE:muscle of arm female embryo (98 days)'

'DNASE:muscle of back male embryo (127 days)'

'DNASE:muscle of arm embryo (101 day)'

'DNASE:muscle of leg male embryo (104 days)'

'DNASE:muscle of arm male embryo (115 days)'

'DNASE:muscle of arm male embryo (96 days)'

'DNASE:muscle of leg male embryo (115 days)'

'DNASE:muscle of leg female embryo (115 days)'

'DNASE:muscle of arm male embryo (105 days)'

'DNASE:muscle of arm male embryo (104 days)'

'CHIP:H3K27ac:skeletal muscle myoblast male adult (22 years)'

'CHIP:H3K27ac:muscle layer of colon female adult (56 years)'

'CHIP:H3K27ac:psoas muscle female adult (30 years)'

'CHIP:H3K27ac:skeletal muscle tissue female adult (72 years)'

'CHIP:H3K27ac:psoas muscle male child (3 years)'

'CHIP:H3K27ac:muscle layer of duodenum male adult (59 years)'

'CHIP:H3K27ac:muscle of trunk female embryo (115 days)'

'CHIP:H3K27ac:psoas muscle male adult (34 years)'

'CAGE:skeletal muscle, adult, pool1'

'CAGE:Skeletal muscle cells differentiated into Myotubes - multinucleated,'

'CAGE:skeletal muscle, fetal,'

'CAGE:eye - muscle superior,'

'CAGE:eye - muscle lateral,'

'CAGE:eye - muscle medial,'

'CAGE:eye - muscle inferior rectus,'

'CAGE:skeletal muscle - soleus muscle,'
"""
activate_muscle_tracks = [  # type: ignore[assignment]
    x.strip("'") for x in activate_muscle_tracks.split("\n") if len(x) > 0
]
# print(activate_muscle_tracks)
assert len(activate_muscle_tracks) == 71


deactivate_liver_tracks = """
'CHIP:H3K27me3:liver male adult (31 year)'

'CHIP:H3K27me3:liver male adult (78 years)'

'CHIP:H3K9me3:liver male adult (31 year)'

'CHIP:H3K9me2:hepatocyte originated from H9'

'CHIP:H3K9me3:liver male adult (78 years)'

'CHIP:H3K9me3:liver male adult (32 years)'

'CHIP:H3K9me3:right lobe of liver female adult (53 years)'

'CHIP:H3K27me3:hepatocyte originated from H9'

'CHIP:H3K27me3:liver male adult (32 years)'

'CHIP:H3K9me3:hepatocyte originated from H9'

'CHIP:H3K9me3:liver female adult (25 years)'

'CHIP:H3K27me3:liver female adult (25 years)'"""
deactivate_liver_tracks = [  # type: ignore[assignment]
    x.strip("'") for x in deactivate_liver_tracks.split("\n") if len(x) > 0
]
# print(activate_muscle_tracks)
assert len(deactivate_liver_tracks) == 12

activate_liver_tracks = """
'DNASE:hepatocyte'

'DNASE:liver embryo (59 days) and embryo (80 days)'

'DNASE:hepatocyte originated from H9'

'DNASE:right lobe of liver female adult (53 years)'

'DNASE:liver female embryo (101 day) and female embryo (113 days)'

'CHIP:H3K27ac:liver male adult (31 year)'

'CHIP:H3K27ac:hepatocyte originated from H9'

'CHIP:H3K27ac:liver female adult (25 years)'

'CHIP:H3K27ac:right lobe of liver female adult (53 years)'

'CAGE:liver, adult, pool1'

'CAGE:liver, fetal, pool1'"""
activate_liver_tracks = [  # type: ignore[assignment]
    x.strip("'") for x in activate_liver_tracks.split("\n") if len(x) > 0
]
assert len(activate_liver_tracks) == 11

deactivate_muscle_tracks = """
'CHIP:H3K27me3:skeletal muscle myoblast male adult (22 years)'

'CHIP:H3K9me3:skeletal muscle myoblast male adult (22 years)'

'CHIP:H3K9me3:muscle layer of duodenum male adult (59 years)'

'CHIP:H3K9me3:muscle layer of duodenum male adult (73 years)'

'CHIP:H3K9me3:muscle of trunk female embryo (115 days)'

'CHIP:H3K27me3:skeletal muscle satellite cell female adult originated from mesodermal cell'

'CHIP:H3K9me3:muscle layer of colon female adult (77 years)'

'CHIP:H3K27me3:skeletal muscle tissue'

'CHIP:H3K27me3:skeletal muscle tissue male adult (54 years)'

'CHIP:H3K27me3:muscle layer of colon female adult (77 years)'

'CHIP:H3K9me3:psoas muscle male child (3 years)'

'CHIP:H3K9me3:skeletal muscle tissue'

'CHIP:H3K27me3:skeletal muscle tissue female adult (72 years)'

'CHIP:H3K9me3:skeletal muscle tissue female adult (72 years)'

'CHIP:H3K9me3:skeletal muscle satellite cell female adult originated from mesodermal cell'

'CHIP:H3K9me3:muscle layer of colon female adult (56 years)'

'CHIP:H3K9me3:skeletal muscle tissue male adult (54 years)'

'CHIP:H3K27me3:muscle of trunk female embryo (115 days)'

'CHIP:H3K27me3:psoas muscle male adult (34 years)'

'CHIP:H3K27me3:muscle layer of duodenum male adult (73 years)'

'CHIP:H3K9me3:muscle of leg female embryo (110 days)'

'CHIP:H3K27me3:psoas muscle male child (3 years)'

'CHIP:H3K27me3:muscle layer of colon female adult (56 years)'

'CHIP:H3K27me3:muscle layer of duodenum male adult (59 years)'

'CHIP:H3K27me3:muscle of leg female embryo (110 days)'"""
deactivate_muscle_tracks = [  # type: ignore[assignment]
    x.strip("'") for x in deactivate_muscle_tracks.split("\n") if len(x) > 0
]
assert len(deactivate_muscle_tracks) == 25

for track in (
    activate_muscle_tracks
    + deactivate_liver_tracks
    + activate_liver_tracks
    + deactivate_muscle_tracks
):
    assert track in BORZOI_TASKS_


def activate_muscle_idx():
    return [BORZOI_TASKS_.index(t) for t in activate_muscle_tracks]


def deactivate_liver_idx():
    return [BORZOI_TASKS_.index(t) for t in deactivate_liver_tracks]


def activate_liver_idx():
    return [BORZOI_TASKS_.index(t) for t in activate_liver_tracks]


def deactivate_muscle_idx():
    return [BORZOI_TASKS_.index(t) for t in deactivate_muscle_tracks]


def idxs_by_name(aggregation_type: str) -> tuple[list[int], list[int]]:
    if aggregation_type == "muscle_not_liver":
        positive_idxs = activate_muscle_idx() + deactivate_liver_idx()
        negative_idxs = activate_liver_idx() + deactivate_muscle_idx()
    else:
        raise ValueError(f"Unknown aggregation type: {aggregation_type}")
    return positive_idxs, negative_idxs


if __name__ == "__main__":
    positive_idxs, negative_idxs = idxs_by_name("muscle_not_liver")
    print(f"Positive idxs: {positive_idxs}")
    print(f"Negative idxs: {negative_idxs}")
    assert len(positive_idxs) == 83, len(positive_idxs)
    assert len(negative_idxs) == 36, len(negative_idxs)

    with open("borzoi_tasks.txt", "w") as f:
        for task in set(BORZOI_TASKS_):
            f.write(task + "\n")
