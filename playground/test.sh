work_path=$(dirname $0)
export PYTHONPATH=..:$PYTHONPATH
python -X faulthandler test.py --gpu_id 0 -c /amax/npr/work_space/ELIC_STAR/checkpoint_110.pth.tar
