import argparse
import configparser
import os

import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.utils import multi_gpu_model
from keras.utils.np_utils import to_categorical


from model import model as sst_model

train_specInput_root_path = None
train_tempInput_root_path = None
train_label_root_path = None

test_specInput_root_path = None
test_tempInput_root_path = None
test_label_root_path = None

result_path = None
model_save_path = None


input_width = None
specInput_length = None
temInput_length = None

depth_spec = None
depth_tem = None
gr_spec = None
gr_tem = None
nb_dense_block = None
nb_class = None

nbEpoch = None
batch_size = None
lr = None


def read_config(config_path):
    conf = configparser.ConfigParser()
    conf.read(config_path)

    global train_specInput_root_path, train_tempInput_root_path, train_label_root_path, test_specInput_root_path, test_tempInput_root_path, test_label_root_path
    train_specInput_root_path = conf['path']['train_specInput_root_path']
    train_tempInput_root_path = conf['path']['train_tempInput_root_path']
    train_label_root_path = conf['path']['train_label_root_path']
    test_specInput_root_path = conf['path']['test_specInput_root_path']
    test_tempInput_root_path = conf['path']['test_tempInput_root_path']
    test_label_root_path = conf['path']['test_label_root_path']

    global result_path, model_save_path
    result_path = conf['path']['result_path']
    model_save_path = conf['path']['model_save_path']

    if not os.path.exists(result_path):
        os.mkdir(result_path)
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    global input_width, specInput_length, temInput_length, depth_spec, depth_tem, gr_spec, gr_tem, nb_dense_block, nb_class
    input_width = int(conf['model']['input_width'])
    specInput_length = int(conf['model']['specInput_length'])
    temInput_length = int(conf['model']['temInput_length'])
    depth_spec = int(conf['model']['depth_spec'])
    depth_tem = int(conf['model']['depth_tem'])
    gr_spec = int(conf['model']['gr_spec'])
    gr_tem = int(conf['model']['gr_tem'])
    nb_dense_block = int(conf['model']['nb_dense_block'])
    nb_class = int(conf['model']['nb_class'])

    global nbEpoch, batch_size, lr
    nbEpoch = int(conf['training']['nbEpoch'])
    batch_size = int(conf['training']['batch_size'])
    lr = float(conf['training']['lr'])


def run():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

    K.set_image_data_format('channels_last')
    #K.set_learning_phase(1)

    all_result_file = open(os.path.join(result_path, 'all_meta_result.txt'), "w")
    all_result_file.close()

    # Meta-Learning Hyperparameters
    meta_iterations = 300
    meta_lr = 0.1
    inner_epochs = 3

    for test_subject in range(1):  # SMOKE TEST: Subject 0 only — restore to range(15) for full LOSO
        print(f"\n{'='*50}\n--- LOSO Evaluation: Test Subject {test_subject} ---\n{'='*50}")
        
        # Build new model
        model = sst_model.sst_emotionnet(input_width=input_width, specInput_length=specInput_length, temInput_length=temInput_length,
                                        depth_spec=depth_spec, depth_tem=depth_tem, gr_spec=gr_spec, gr_tem=gr_tem, nb_dense_block=nb_dense_block, nb_class=nb_class)
        adam = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8, clipnorm=1.0)
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        
        meta_weights = model.get_weights()
        train_subjects = [i for i in range(15) if i != test_subject]
        
        # 1. Meta-Training (Reptile)
        print("Starting Reptile Meta-Training...")
        for iteration in range(meta_iterations):
            # Sample a random task
            task_subject = np.random.choice(train_subjects)
            task_session = np.random.choice(3)
            
            # Load task data
            task_train_spec = np.load(os.path.join(train_specInput_root_path, f"subject_{task_subject}/section_{task_session}_data.npy"))
            task_train_temp = np.load(os.path.join(train_tempInput_root_path, f"subject_{task_subject}/section_{task_session}_data.npy"))
            task_train_label = np.load(os.path.join(train_label_root_path, f"subject_{task_subject}/section_{task_session}_label.npy"))
            task_train_label = to_categorical(task_train_label, num_classes=nb_class)
            
            # Set to current meta weights
            model.set_weights(meta_weights)
            
            # Inner update
            model.fit([task_train_spec, task_train_temp], task_train_label, epochs=inner_epochs, batch_size=batch_size, verbose=0)
            
            # Update meta weights
            task_weights = model.get_weights()
            meta_weights = [mw + meta_lr * (tw - mw) for mw, tw in zip(meta_weights, task_weights)]
            if (iteration + 1) % 10 == 0:
                print(f"  Meta-Iteration {iteration+1}/{meta_iterations} complete.")
        
        # 2. Fine-Tuning
        print(f"Fine-Tuning on Subject {test_subject}...")
        model.set_weights(meta_weights)
        
        # Load all 3 sections of the test subject's train data
        ft_spec_list, ft_temp_list, ft_label_list = [], [], []
        for j in range(3):
            ft_spec_list.append(np.load(os.path.join(train_specInput_root_path, f"subject_{test_subject}/section_{j}_data.npy")))
            ft_temp_list.append(np.load(os.path.join(train_tempInput_root_path, f"subject_{test_subject}/section_{j}_data.npy")))
            ft_lbl = np.load(os.path.join(train_label_root_path, f"subject_{test_subject}/section_{j}_label.npy"))
            ft_label_list.append(to_categorical(ft_lbl, num_classes=nb_class))
            
        ft_spec = np.concatenate(ft_spec_list, axis=0)
        ft_temp = np.concatenate(ft_temp_list, axis=0)
        ft_label = np.concatenate(ft_label_list, axis=0)
        
        idx = np.arange(ft_spec.shape[0])
        np.random.shuffle(idx)
        ft_spec, ft_temp, ft_label = ft_spec[idx], ft_temp[idx], ft_label[idx]
        
        # Load all 3 sections of the test subject's test data
        test_spec_list, test_temp_list, test_label_list = [], [], []
        for j in range(3):
            test_spec_list.append(np.load(os.path.join(test_specInput_root_path, f"subject_{test_subject}/section_{j}_data.npy")))
            test_temp_list.append(np.load(os.path.join(test_tempInput_root_path, f"subject_{test_subject}/section_{j}_data.npy")))
            t_lbl = np.load(os.path.join(test_label_root_path, f"subject_{test_subject}/section_{j}_label.npy"))
            test_label_list.append(to_categorical(t_lbl, num_classes=nb_class))
            
        eval_spec = np.concatenate(test_spec_list, axis=0)
        eval_temp = np.concatenate(test_temp_list, axis=0)
        eval_label = np.concatenate(test_label_list, axis=0)
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1)
        save_model = ModelCheckpoint(filepath=os.path.join(model_save_path, f'Sub_{test_subject}_Meta.h5'), monitor='val_loss', save_best_only=True, mode='min')
        
        history = model.fit([ft_spec, ft_temp], ft_label, epochs=nbEpoch, batch_size=batch_size,
                            validation_data=([eval_spec, eval_temp], eval_label), callbacks=[early_stopping, save_model], verbose=1)
        
        import json
        with open(os.path.join(result_path, f'Sub_{test_subject}_history.json'), 'w') as f:
            hist_dict = {k: [float(val) for val in v] for k, v in history.history.items()}
            json.dump(hist_dict, f)
            
        model = load_model(os.path.join(model_save_path, f'Sub_{test_subject}_Meta.h5'))
        #K.set_learning_phase(0)
        loss, accuracy = model.evaluate([eval_spec, eval_temp], eval_label)
        print(f'\nSubject {test_subject} test loss: {loss}, accuracy: {accuracy}')
        
        all_result_file = open(os.path.join(result_path, 'all_meta_result.txt'), "a")
        print(f'Subject {test_subject}: {accuracy}', file=all_result_file)
        all_result_file.close()
        keras.backend.clear_session()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argument of running SST-EmotionNet.')
    parser.add_argument('-c', type=str, help='Config file path.', required=True)
    args = parser.parse_args()
    read_config(args.c)
    run()
