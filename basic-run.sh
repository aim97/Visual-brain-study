# python eeg_signal_classification.py -mt AttnSleep -ed .\\data\\block\\eeg_signals_raw_with_mean_std.pth
# -ed .\\data\\block\\eeg_55_95_std.pth

# AttnSleep
## Raw
# python eeg_signal_classification.py -mt AttnSleep -ed "./data/block/eeg_signals_raw_with_mean_std.pth" 

# LSTM 4 layers
## Raw
# python eeg_signal_classification.py -mt lstm -ed "./data/block/eeg_signals_raw_with_mean_std.pth"  -mp lstm_layers=4

# --pretrained_net "./lstm__subject0_epoch_193.pth"

# -lrde 100 -e 100 
# --pretrained_net ".\stored models\EEGChannelNet 55-95 Training\EEGChannelNet__subject0_epoch_16.pth"

# EEGChannelNet
## 55-95 std
python eeg_signal_classification.py -mt EEGChannelNet -ed "./data/block/eeg_55_95_std.pth"

python eeg_signal_classification.py -mt SleepingPower -mt core_model=SimpleEEGCNN spec_type=stft -ed "./data/block/eeg_55_95_std.pth"