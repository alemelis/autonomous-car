module load apps/python/anaconda3-4.2.0
module load libs/cudnn/5.1/binary-cuda-8.0.44
source activate tf
cd /fastdata/me1ame/car/
singularity exec /usr/local/packages/singularity/images/tensorflow/gpu.img python main.py
