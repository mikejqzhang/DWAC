python -m text.run --model proto --dataset imdb --n-proto 1 --output-dir ./data/imdb_proto_1
python -m text.run --model proto --dataset imdb --n-proto 4 --output-dir ./data/imdb_proto_4
python -m text.run --model proto --dataset imdb --n-proto 16 --output-dir ./data/imdb_proto_16
python -m text.run --model proto --dataset imdb --n-proto 64 --output-dir ./data/imdb_proto_64
python -m text.run --model proto --dataset imdb --n-proto 256 --output-dir ./data/imdb_proto_256
python -m text.run --model proto --dataset imdb --n-proto 1024 --output-dir ./data/imdb_proto_1024

python -m text.run --model proto --dataset stackoverflow --n-proto 1 --output-dir ./data/stackoverflow_proto_1
python -m text.run --model proto --dataset stackoverflow --n-proto 4 --output-dir ./data/stackoverflow_proto_4
python -m text.run --model proto --dataset stackoverflow --n-proto 16 --output-dir ./data/stackoverflow_proto_16
python -m text.run --model proto --dataset stackoverflow --n-proto 64 --output-dir ./data/stackoverflow_proto_64
python -m text.run --model proto --dataset stackoverflow --n-proto 256 --output-dir ./data/stackoverflow_proto_256
python -m text.run --model proto --dataset stackoverflow --n-proto 1024 --output-dir ./data/stackoverflow_proto_1024
