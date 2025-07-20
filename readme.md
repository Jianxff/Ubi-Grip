# Ubi-Grip

Code repository for paper **Ubi Grip: Ubiquitous Grip-Based Tangible Object Utilization in Augmented Reality**


### Installation
Clone the repository
```bash
git clone ...
cd ...
```

Install requirements
```
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Build and Install SRT3D
```bash
cd src/srt3d
pip install .
```

### Evaluation
Download the pre-trained [HaMeR](https://github.com/geopavlakos/hamer) model from [here](https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz), and put only `hamer.ckpt` in `data/ckpt`.

Prepare the dataset
1. [HO3Dv3](https://github.com/shreyashampali/ho3d)
2. [H2O](https://github.com/taeinkwon/h2odataset)

If something went wrong with template-views generation by [SRT3D](https://github.com/DLR-RM/3DObjectTracking/blob/master/SRT3D/readme.md), you can download pre-generated templates from [here](#coming-soon).

**Modified** `eval.py` to run evaluation.
```bash
## Please modify the file before running evaluation on different datasets.
python eval.py
```

### Demo
*Coming Soon.*

