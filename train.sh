CFG_PATH="cfg/clotho/base.yaml"
accelerate launch --multi_gpu --main_process_port=1200 train.py $CFG_PATH