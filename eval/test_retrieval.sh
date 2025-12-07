model_path=$1
gpu=$2

WORKDIR="YOUR_PATH/HiMo-CLIP"
cd ${WORKDIR}
export PYTHONPATH=${WORKDIR}:$PYTHONPATH

echo model_path: ${model_path}
cd ${WORKDIR}/eval/retrieval/
echo "~~~~~~~~retrieval~~~~~~~~"
echo "==========en urban-1k========="
export CUDA_VISIBLE_DEVICES=$gpu && python Urban1k.py ${model_path} en

echo "==========en docci========="
export CUDA_VISIBLE_DEVICES=$gpu && python docci.py ${model_path} en

echo "==========en dci========="
export CUDA_VISIBLE_DEVICES=$gpu && python dci_long.py ${model_path} en

echo "==========en flickr30k========="
export CUDA_VISIBLE_DEVICES=$gpu && python flickr30k.py ${model_path} en

echo "==========en coco========="
export CUDA_VISIBLE_DEVICES=$gpu && python coco.py ${model_path} en

