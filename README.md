OpenFX plugin for "Robust Video Matting"
========================================

This repo contains an OpenFX plugin using the AI models from :

https://peterl1n.github.io/RobustVideoMatting/#/
https://github.com/PeterL1n/RobustVideoMatting

This allows for automatic background removal using a TorchScript model
in any compatible OpenFX host. Currently only tested with Davinci Resolve 18.1


Build
-----

* Make sure to clone this repo using `--recursive` or to init/update
  submodules afterwards since the openFX support libraries are required
  for the build.

* Install the LibTorch binaries from https://pytorch.org/ for your target.
  Make sure to use the non-cxx11 version ! (At least required for use with
  Davinci Resolve).

* Then build using cmake the usual way.


Install
-------

Currently there is no install target and you need to copy files manually.
On linux the layout should end up looking like :

```
/usr/OFX/Plugins/rvmofx.ofx.bundle
/usr/OFX/Plugins/rvmofx.ofx.bundle/Contents
/usr/OFX/Plugins/rvmofx.ofx.bundle/Contents/Linux-x86-64
/usr/OFX/Plugins/rvmofx.ofx.bundle/Contents/Linux-x86-64/rvmofx.ofx
/usr/OFX/Plugins/rvmofx.ofx.bundle/Contents/Resources
/usr/OFX/Plugins/rvmofx.ofx.bundle/Contents/Resources/rvm_mobilenetv3_fp16.torchscript
/usr/OFX/Plugins/rvmofx.ofx.bundle/Contents/Resources/rvm_resnet50_fp16.torchscript
/usr/OFX/Plugins/rvmofx.ofx.bundle/Contents/Resources/rvm_resnet50_fp32.torchscript
/usr/OFX/Plugins/rvmofx.ofx.bundle/Contents/Resources/rvm_mobilenetv3_fp32.torchscript
```

(you need to download and install the pre-trained models from the original repo)
