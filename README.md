# Speech2Slot

### Step 0. Preparation

Git clone the code from the repository.

```
git clone http://...
```

Download 4 opensource ASR datasets: THCHS-30, Aishell, Free ST Chinese Mandarin Corpus, and Primewords Chinese Corpus Set4. These datasets can be download from http://www.openslr.org/resources.php.

Download Voice Navigation Dataset in Chinese(VNDC) from http://www.speech2slot.com/dataset.

Move these datasets to the data directory.

```shell
mv dataset_files Speech2Slot/AM/data/
```

### Step 1. Train and test acoustic model

It could be implemented by yourself as long as it provides proper phoneme posterior with comparable accuracy.

```shell
# go to AM directory
cd  Speech2Slot/AM/

# train acoustic model
python train_am.py \
  --am_model_name=AM_model.h5

# test on vndc human test set
python test_am.py \
	--vndc_testing_file=VNDC/vndc_test_human.txt \
  --am_model_name=AM_model.h5
  
# test on vndc tts test set
python test_am.py \
	--vndc_testing_file=VNDC/vndc_test_tts.txt \
  --am_model_name=AM_model.h5
```

### Step 2. Make train and test data for speech2slot model.

It produces phoneme sequence as input for speech2slot model.

```shell
# make training data for speech2slot
python make_train_data.py \
	--data_dir=../train_data_phoneme/ \
  --vndc_train_file=VNDC/vndc_train.txt \
  --am_model_name=AM_model.h5 \
  --data_length=820000
  
# make test data
python make_test_data.py \
	--data_dir=../test_data_phoneme/ \
  --output_file=vndc_test_human_phoneme.pkl\
  --vndc_test_file=VNDC/vndc_test_human.txt \
  --am_model_name=AM_model.h5
  
# make test data
python make_test_data.py \
  --data_dir=../test_data_phoneme/ \
  --output_file=vndc_test_tts_phoneme.pkl \
  --vndc_test_file=VNDC/vndc_test_tts.txt \
  --am_model_name=AM_model.h5
  
# back to Speech2Slot directory
cd ..
```

### Step 3. Train speech2slot

```shell
# start training
python train.py \
    --config_file=./config.json \
    --train_sheet_file_list=tf_list.txt \
    --train_batch_size=64 \
    --num_train_epochs=40.0 \
    --ckpt_dir=./ckpt/speech2slot_0920 \
    --num_train_sample=820000 \
    --maxlen1=40 \
    --maxlen2=10 \
    --vocab_size=1296 \
    --encoder_masked_size=4 \
    --learning_rate=1.5e-5 \
    --is_standalone=True \
    --buckets=./logs
```

### Step 4. Test speech2slot on human and TTS test set

```shell
# test on human test set
python3 test.py \
        --config_file=config.json \
        --source_max_length=40 \
        --dest_max_length=10 \
        --test_batch_size=64 \
        --ckpt=ckpt/speech2slot_0816 \
        --num_beam=10 \
        --place_whole_name_dict=AM/data/VNDC/slot_dict.txt \
        --target_vocab=AM/voc/voc_target_pinyin.txt \
        --source_vocab=AM/voc/voc_source_pinyin.txt \
        --test_dir=result/speech2slot_0816 \
        --beam_search_res=beam_search_res_speech2slot_0816.txt \
        --forward_step=10 \
        --testing_file=test_data_phoneme/vndc_test_human_phoneme.pkl

# test on tts test set
python3 test.py \
        --config_file=config.json \
        --source_max_length=40 \
        --dest_max_length=10 \
        --test_batch_size=64 \
        --ckpt=ckpt/speech2slot_0816 \
        --num_beam=10 \
        --place_whole_name_dict=AM/data/VNDC/slot_dict.txt \
        --target_vocab=AM/voc/voc_target_pinyin.txt \
        --source_vocab=AM/voc/voc_source_pinyin.txt \
        --test_dir=result/speech2slot_0816 \
        --beam_search_res=beam_search_res_speech2slot_0816.txt \
        --forward_step=10 \
        --testing_file=test_data_phoneme/vndc_test_tts_phoneme.pkl
```


