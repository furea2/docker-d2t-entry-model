FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime
# FROM pytorch/pytorch:latest # 新しすぎた...

# <-- Start setup.pyを走らせる準備 -->
RUN apt-get update && apt-get -y install libasound2-plugins libsox-fmt-all libsox-dev sox
#RUN apt-get install libasound2-python


# OpenNMTのclone
RUN cd /workspace && git clone https://github.com/OpenNMT/OpenNMT-py.git && cd OpenNMT-py && pip install -r requirements.opt.txt && python setup.py install


# data2text-entity-pyのclone
# torchtextのversionは2.1または3.0
RUN cd /workspace && git clone https://github.com/ratishsp/data2text-entity-py.git && cd data2text-entity-py && pip install -r requirements.txt && pip install torchtext==0.2.1

# <-- End setup.pyを走らせる準備 -->


# <-- Start 特徴量抽出～学習～評価 -->

# Preprocessingの準備
RUN cd /workspace/data2text-entity-py && mkdir /root/boxscore-data
ADD ./boxscore-data /root/boxscore-data

# RUN cd /workspace/data2text-entity-py && BASE=/root/boxscore-data && mkdir $BASE/entity_preprocess && python preprocess.py -train_src $BASE/rotowire/src_train.txt -train_tgt $BASE/rotowire/tgt_train.txt -valid_src $BASE/rotowire/src_valid.txt -valid_tgt $BASE/rotowire/tgt_valid.txt -save_data $BASE/entity_preprocess/roto -src_seq_length 1000 -tgt_seq_length 1000 -dynamic_dict


# Training (and Downloading Trained Models)
# RUN cd /workspace/data2text-entity-py && BASE=/root/boxscore-data && IDENTIFIER=cc && GPUID=0 && python train.py -data $BASE/entity_preprocess/roto -save_model $BASE/gen_model/$IDENTIFIER/roto -encoder_type mean -input_feed 1 -layers 2 -batch_size 5 -feat_merge mlp -seed 1234 -report_every 100 -gpuid $GPUID -start_checkpoint_at 4 -epochs 25 -copy_attn -truncated_decoder 100 -feat_vec_size 600 -word_vec_size 600 -rnn_size 600 -optim adagrad -learning_rate 0.15 -adagrad_accumulator_init 0.1 -reuse_copy_attn -start_decay_at 4 -learning_rate_decay 0.97 -entity_memory_size 300 -valid_batch_size 5

# Generation

# Automatic evaluation using IE metrics

# Evaluation using BLEU script

# IE models

# <-- End 特徴量抽出～学習～評価 -->

