python train.py -source scannet -target modelnet -scaler 1 -weight 1 -tb_log_dir ./logs/sa_ss2m
python train.py -source scannet -target shapenet -scaler 1 -weight 0.5 -tb_log_dir ./logs/sa_ss2s
python train.py -source modelnet -target scannet  -scaler 1 -weight 0.5 -tb_log_dir ./logs/sa_m2ss
python train.py -source modelnet -target shapenet -scaler 1 -weight 1 -tb_log_dir ./logs/sa_m2s
python train.py -source shapenet -target modelnet -scaler 1 -weight 1 -tb_log_dir ./logs/sa_s2m
python train.py -source shapenet -target scannet -scaler 1 -weight 0.5 -tb_log_dir ./logs/sa_s2ss
