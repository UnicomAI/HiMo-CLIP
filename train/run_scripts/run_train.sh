
export CUDA_VISIBLE_DEVICES=$1
JOB_NAME=$2
master_port=$3 # 2519
gpu_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F ',' '{print NF}')

echo use_gpu: ${CUDA_VISIBLE_DEVICES}
echo gpu_num=${gpu_num}

## 加入clip库的路径
export PYTHONPATH=/home/jovyan/LMM/wrj/workspace/project/CLIP:$PYTHONPATH

WORKDIR=/home/jovyan/LMM/wrj/workspace/project/LongClip/HiMo-CLIP
cd ${WORKDIR}
export PYTHONPATH=$PWD:$PYTHONPATH
cd train/

LOG_DIR=output/logs
mkdir -p ${LOG_DIR}
START_TIME=`date "+%Y%m%d_%H%M%S"`
LOG_FILE=${LOG_DIR}/train-log-$JOB_NAME-$START_TIME
torchrun -m \
    --nnodes=1 \
    --nproc_per_node=${gpu_num} \
    --master_port=${master_port} \
    train \
    --base_model '/home/jovyan/LMM/wrj/workspace/project/LongClip/Yuanjing-CLIP/train/weights/ViT-L-14.pt' \
    --jobname ${JOB_NAME}-${START_TIME} \
    --batch-size 128 \
    --accum_steps 1 \
    --epochs 10 \
    --pca_ratio 0.9 \
    2>&1 | tee $LOG_FILE > /dev/null &
