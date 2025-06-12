# unaligned & ood
nohup python -u main.py --dataset mosi --ood --task binary --n_trials 300 >mosi_ua_ood_binary.txt 2>&1 &
nohup python -u main.py --dataset mosi --ood --task seven --n_trials 300 >mosi_ua_ood_seven.txt 2>&1 &

# unaligned & iid
nohup python -u main.py --dataset mosi --task binary --n_trials 300 >mosi_ua_iid_binary.txt 2>&1 &
nohup python -u main.py --dataset mosi --task seven --n_trials 300 >mosi_ua_iid_seven.txt 2>&1 &

# aligned & iid
nohup python -u main.py --dataset mosi --aligned --task binary --n_trials 300 >mosi_a_iid_binary.txt 2>&1 &
nohup python -u main.py --dataset mosi --aligned --task seven --n_trials 300 >mosi_a_iid_seven.txt 2>&1 &

# aligned & cross_dataset
nohup python -u main.py --dataset mosi --aligned --cross_dataset --task binary --n_trials 300 >[cross_dataset]mosi_a_iid_binary.txt 2>&1 &
nohup python -u main.py --dataset mosi --aligned --cross_dataset --task seven --n_trials 300 >[cross_dataset]mosi_a_iid_seven.txt 2>&1 &