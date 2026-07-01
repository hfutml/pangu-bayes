export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_ALGO=Tree
export NCCL_PROTO=Simple
torchrun --nproc_per_node=8 --master_port=12354 pretrain.py