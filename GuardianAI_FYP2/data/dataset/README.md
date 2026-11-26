# 📁 GuardianAI Dataset Directory

Place your real-world fall detection dataset video clips in these subfolders:

- alls/ : Video clips of human falls (URFD fall clips, Le2i falls, or custom fall recordings).
- dl/   : Activities of Daily Living / Normal activity clips (walking, sitting, standing, bending, lying down on bed).
- warning/ (optional) : Pre-fall instability, stumbling, tripping clips.

### Supported formats
.mp4, .avi, .mov, .mkv, .wmv

### How to Train
Run from the GuardianAI_Merged root folder:
`powershell
.venv\Scripts\python train.py --benchmark generic --data-dir  data/dataset --arch cnn_lstm --epochs 50 --loss focal --augment --export-onnx
`
