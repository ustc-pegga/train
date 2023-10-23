    # 'mate30e_cpu':[5, 7, 5, 7, 11, 5, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    # 'mate30e_gpu':[5, 9, 5, 9, 5, 5, 5, 7, 5, 5, 7, 5, 5, 5, 5, 3, 3],
    # 'mate30e_npu':[5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5],
    # 'mi11_cpu':[5, 7, 5, 7, 5, 5, 7, 5, 5, 5, 5, 5, 5, 9, 5, 5, 5],
    # 'mi11_gpu':[5, 9, 5, 3, 11, 5, 7, 5, 3, 3, 3, 5, 5, 5, 3, 5, 3],
CUDA_VISIBLE_DEVICES=0 \
python train.py \
    --model mobilenetv2 \
    --dataset cifar100 \
    --batch_size 256 \
    --n_worker 36 \
    --n_epoch 200 \
    --seed 2024 \
    --data_root /home/hujie/datasets/cifar100 \
    --name 'mate30_gpu' \
    --kernel_list "3, 5, 7, 9, 3, 7, 3, 9, 5, 9, 7, 3, 9, 9, 3, 3, 3" \
    --ckpt_path /home/hujie/experiment/train/checkpoint/mobilenetv2_cifar100_200_resize224_kernel=3-run120/ckpt.best.pth.tar \
    --lr 0.1 \
    --tran True

# CUDA_VISIBLE_DEVICES=0 \
# python train.py \
#     --model mobilenetv2 \
#     --dataset imagenet-100 \
#     --batch_size 256 \
#     --n_worker 36 \
#     --n_epoch 200 \
#     --seed 2024 \
#     --data_root /home/hujie/datasets/imagenet100 \
#     --name 'mate30e_cpu_pretrain' \
#     --kernel_list "5, 7, 5, 7, 11, 5, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5" \
#     --ckpt_path /home/hujie/experiment/train/checkpoint_1/mobilenetv2_imagenet-100_300_origin-run1/ckpt.best.pth.tar \
#     --lr 0.1 \
#     --trans True

# CUDA_VISIBLE_DEVICES=0 \
# python train.py \
#     --model mobilenetv2 \
#     --dataset imagenet-100 \
#     --batch_size 256 \
#     --n_worker 36 \
#     --n_epoch 200 \
#     --seed 2024 \
#     --data_root /home/hujie/datasets/imagenet100 \
#     --name 'mate30e_gpu_pretrain_L1_0.35' \
#     --kernel_list "3, 5, 7, 9, 3, 7, 3, 9, 5, 9, 7, 3, 9, 9, 3, 3, 3" \
#     --ckpt_path /home/hujie/experiment/train/checkpoint_1/mobilenetv2_imagenet-100_200_mate30e_gpu_pretrain-run9/ckpt.best.pth.tar \
#     --lr 0.1 \
#     --width_rate 0.35

# CUDA_VISIBLE_DEVICES=0 \
# python train.py \
#     --model mobilenetv2 \
#     --dataset imagenet-100 \
#     --batch_size 256 \
#     --n_worker 36 \
#     --n_epoch 200 \
#     --seed 2024 \
#     --data_root /home/hujie/datasets/imagenet100 \
#     --name 'origin_L1_0.25' \
#     --ckpt_path /home/hujie/experiment/train/checkpoint_1/mobilenetv2_imagenet-100_300_origin-run8/ckpt.best.pth.tar \
#     --lr 0.1 \
#     --width_rate 0.25


# CUDA_VISIBLE_DEVICES=1 \
# python train.py \
#     --model mobilenetv2 \
#     --dataset imagenet-100 \
#     --batch_size 256 \
#     --n_worker 36 \
#     --n_epoch 300 \
#     --seed 2024 \
#     --data_root /home/hujie/datasets/imagenet100 \
#     --name 'origin' 
#     --kernel_list "5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5" \
#     # --lr 0.1 
#     # --trans True

# CUDA_VISIBLE_DEVICES=1 \
# python train.py \
#     --model mobilenetv2 \
#     --dataset imagenet-100 \
#     --batch_size 256 \
#     --n_worker 36 \
#     --n_epoch 200 \
#     --seed 2024 \
#     --data_root /home/hujie/datasets/imagenet100 \
#     --name 'mi11_cpu_pretrain' \
#     --kernel_list "5, 7, 5, 7, 5, 5, 7, 5, 5, 5, 5, 5, 5, 9, 5, 5, 5" \
#     --ckpt_path /home/hujie/experiment/train/checkpoint_1/mobilenetv2_imagenet-100_300_origin-run1/ckpt.best.pth.tar \
#     --lr 0.1 \
#     --trans True

# CUDA_VISIBLE_DEVICES=1 \
# python train.py \
#     --model mobilenetv2 \
#     --dataset imagenet-100 \
#     --batch_size 256 \
#     --n_worker 36 \
#     --n_epoch 200 \
#     --seed 2024 \
#     --data_root /home/hujie/datasets/imagenet100 \
#     --name 'mi11_cpu_pretrain' \
#     --kernel_list "5, 9, 5, 3, 11, 5, 7, 5, 3, 3, 3, 5, 5, 5, 3, 5, 3" \
#     --ckpt_path /home/hujie/experiment/train/checkpoint_1/mobilenetv2_imagenet-100_300_origin-run1/ckpt.best.pth.tar \
#     --lr 0.1 \
#     --trans True