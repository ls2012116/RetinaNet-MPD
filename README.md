# RetinaNet-MPD

## Getting Started

The codes build Rotated RetinaNet with the proposed RetinaNet-MPD method for rotation object detection. The supported datasets include: DOTA, HRSC2016, ICDAR2013, ICDAR2015, UCAS-AOD, NWPU VHR-10, VOC. 

### Installation
Insatll requirements:
```
pip install -r requirements.txt
pip install git+git://github.com/lehduong/torch-warmup-lr.git
```
Build the Cython  and CUDA modules:
```
cd $ROOT/utils
sh make.sh
cd $ROOT/utils/overlaps_cuda
python setup.py build_ext --inplace
```
Installation for DOTA_devkit:
```
cd $ROOT/datasets/DOTA_devkit
sudo apt-get install swig
swig -c++ -python polyiou.i
python setup.py build_ext --inplace
```
### Inference
You can use the following command to test a dataset. Note that `weight`, `img_dir`, `dataset`,`hyp` should be modified as appropriate.
```
python demo.py
```

### Train
1. Move the dataset to the `$ROOT` directory.
2. Generate imageset files for daatset division via:
```
cd $ROOT/datasets
python generate_imageset.py
```
3. Modify the configuration file `hyp.py` and arguments  in `train.py`, then start training:
```
python train.py
```
### Evaluation

Different datasets use different test methods. For UCAS-AOD/HRSC2016/VOC/NWPU VHR-10, you need to prepare labels in the appropriate format in advance. Take evaluation on HRSC2016 for example:
```
cd $ROOT/datasets/evaluate
python hrsc2gt.py
```
then you can conduct evaluation:
```
python eval.py
```
Note that :

- the script  needs to be executed **only once**, but testing on different datasets needs to be executed again.
- the imageset file used in `hrsc2gt.py` is generated from `generate_imageset.py`.

因为数据和代码涉及到后面的工作，完整的代码暂时不能发布出来，敬请原谅。

