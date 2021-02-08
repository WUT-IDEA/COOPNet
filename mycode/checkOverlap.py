import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import warnings
warnings.filterwarnings('ignore')

import argparse
import numpy as np
import pandas as pd

from mycode.text_plus_fig import buildModel_text_fig_regu_finetune_readweights_attention
from mycode.text_plus_fig import read_data
from mycode.Generator import generator_fig_text_multioutput_batch

from mycode.text_plus_fig import test_batch_size, VALID_RATE, fig_resize_shape# 参数




def main():
    test_steps = int(1900/test_batch_size)


    # mode = _args.mode
    # corp = _args.corp

    # train_label, valid_label, test_label, generator_train, generator_valid, generator_test = read_data(corp)

    # train = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', 'output', 'train_rmpattern_seq_shuffle.pd'))
    test = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', 'output', 'test_rmpattern_seq.pd'))


    # 将训练集分成训练集和验证集
    # train_size = int(len(train)*(1-VALID_RATE))
    # valid = train[train_size:]
    # train = train[:train_size]

    # 取出pd中的label
    test_label = test['label'].apply(lambda gender: [1] if gender=='male' else [0])


    test_fig_path_list = test['fig_path_list']
    test_text_seq_list = test['seq']

    generator_test = generator_fig_text_multioutput_batch(test_text_seq_list, test_fig_path_list, test_label,
                                                          test_batch_size, fig_resize_shape)

    model = buildModel_text_fig_regu_finetune_readweights_attention()

    model.load_weights(os.path.join(os.path.dirname(__file__), '..', 'ckp',
                                    'text_fig_norm_atten_finetune_readweights_drop_multi_ckp',
                                    'model-epoch_08.hdf5'))

    prediction = model.predict_generator(generator_test, steps=2)

    prediction_arr = np.asarray(prediction)
    print(prediction_arr)
    print(prediction_arr.shape)






if __name__ == '__main__':
    # _argparser = argparse.ArgumentParser(
    #     description='run model...',
    #     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # _argparser.add_argument('--epochs', type=int, required=True, metavar='INTEGER')
    # _argparser.add_argument('--mode', type=str, required=True)
    # _argparser.add_argument('--corp', type=str, required=True)
    # _args = _argparser.parse_args()
    main()