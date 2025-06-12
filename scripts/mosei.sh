# unaligned & ood
nohup python -u main.py --dataset mosei --ood --task binary --n_trials 300 >mosei_ua_ood_binary.txt 2>&1 &
nohup python -u main.py --dataset mosei --ood --task seven --n_trials 300 >mosei_ua_ood_seven.txt 2>&1 &

# unaligned & iid
nohup python -u main.py --dataset mosei --task binary --n_trials 300 >mosei_ua_iid_binary.txt 2>&1 &
nohup python -u main.py --dataset mosei --task seven --n_trials 300 >mosei_ua_iid_seven.txt 2>&1 &

# aligned & iid
nohup python -u main.py --dataset mosei --aligned --task binary --n_trials 300 >mosei_a_iid_binary.txt 2>&1 &
nohup python -u main.py --dataset mosei --aligned --task seven --n_trials 300 >mosei_a_iid_seven.txt 2>&1 &

# aligned & cross_dataset
nohup python -u main.py --dataset mosei --aligned --cross_dataset --task binary --n_trials 300 >[cross_dataset]mosei_a_iid_binary.txt 2>&1 &
nohup python -u main.py --dataset mosei --aligned --cross_dataset --task seven --n_trials 300 >[cross_dataset]mosei_a_iid_seven.txt 2>&1 &