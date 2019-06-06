python run.py --model dwac --ood-class stack_overflow --max-epochs 5
tar -cvzf data/dwac_stack_overflow data/temp
python run.py --model proto ood-class stack_overflow --max-epochs 5
tar -cvzf data/proto_stack_overflow data/temp
