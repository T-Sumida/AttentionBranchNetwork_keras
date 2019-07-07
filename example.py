# coding:utf-8

import attention_branch_network

if __name__ == "__main__":
    input_shape = (200, 200, 3)
    output_num = 3
    model = attention_branch_network.build_model(input_shape, output_num, feature_extractor="dense121")
