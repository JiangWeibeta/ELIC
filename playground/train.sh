work_path=$(dirname $0)
export PYTHONPATH=..:$PYTHONPATH
nohup python train.py --metrics mse --exp open_elic_0800 --lambda 0.08 --gpu_id 0 -lr 1e-4 -c /data2/jiangwei/work_space/ELIC/playground/experiments/open_elic_0800/checkpoints/checkpoint_057.pth.tar --clip_max_norm 1.0 --seed 1984 --batch-size 8 & > 0800v2.txt
