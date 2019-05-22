script = """
python -m text.run --batch-size 32 --z-dim 50 --device 0 --model proto \
       --dataset stackoverflow --n-proto 64 \
       --ood-class wordpress \
       --output-dir ./data/05_17/stackoverflow_ood_wordpress
"""

python -m text.run --batch-size 32 --z-dim 50 --device 0 --model proto \
       --dataset stackoverflow --n-proto 64 \
       --ood-class wordpress \
       --output-dir ./data/05_17/stackoverflow_ood_wordpress
