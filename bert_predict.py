# -*- coding: utf-8 -*-
#!/usr/bin/python3

import os
import pandas as pd

from termcolor import colored

from helper import import_tf, set_logger
import tensorflow as tf

__all__ = ['BertSim']

class BertSim(object):
    def __init__(self, gpu_no, log_dir, bert_sim_dir, verbose=False):
        self.bert_sim_dir = bert_sim_dir
        self.logger = set_logger(colored('BS', 'cyan'), log_dir, verbose)

        self.tf = import_tf(gpu_no, verbose)

        # add tokenizer
        import tokenization
        self.tokenizer = tokenization.FullTokenizer(os.path.join('chinese_L-12_H-768_A-12', 'vocab.txt'))
        # add placeholder
        self.input_ids = self.tf.placeholder(self.tf.int32, (None, 50), 'input_ids')
        self.input_mask = self.tf.placeholder(self.tf.int32, (None, 50), 'input_mask')
        self.input_type_ids = self.tf.placeholder(self.tf.int32, (None, 50), 'input_type_ids')
        # init graph
        self._init_graph()

    def _init_graph(self):
        """
        init bert graph
        """
        try:
            import modeling
            bert_config = modeling.BertConfig.from_json_file(os.path.join('chinese_L-12_H-768_A-12', 'bert_config.json'))
            self.model = modeling.BertModel(config=bert_config,
                                            is_training=False,
                                            input_ids=self.input_ids,
                                            input_mask=self.input_mask,
                                            token_type_ids=self.input_type_ids,
                                            use_one_hot_embeddings=False)

            # get output weights and output bias
            ckpt = self.tf.train.get_checkpoint_state(self.bert_sim_dir).all_model_checkpoint_paths[-1]
            reader = self.tf.train.NewCheckpointReader(ckpt)
            output_weights = reader.get_tensor('output_weights')
            output_bias = reader.get_tensor('output_bias')

            # get result op
            output_layer = self.model.get_pooled_output()
            logits = self.tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = self.tf.nn.bias_add(logits, output_bias)
            self.probabilities = self.tf.nn.softmax(logits, axis=-1)

            sess_config = self.tf.ConfigProto(allow_soft_placement=True)
            sess_config.gpu_options.allow_growth = True

            graph = self.probabilities.graph
            saver = self.tf.train.Saver()
            self.sess = self.tf.Session(config=sess_config, graph=graph)
            self.sess.run(self.tf.global_variables_initializer())
            self.tf.reset_default_graph()
            saver.restore(self.sess, ckpt)

        except Exception as e:
            self.logger.error(e)

    def predict(self, request_list):
        """
        bert model predict
        :return: label, similarity
        :param request_list: request list, each element is text_a and text_b
        """
        # with self.sess.as_default():
        # if len(request_list) != 2:
        #     raise ValueError('输入的pair的长度必须为2')

        input_ids = []
        input_masks = []
        segment_ids = []

        for d in request_list:
            text_a = d[0]
            text_b = d[1]

            input_id, input_mask, segment_id = self._convert_single_example(text_a, text_b)
            input_ids.append(input_id)
            input_masks.append(input_mask)
            segment_ids.append(segment_id)

        predict_result = None
        try:
            predict_result = self.sess.run(self.probabilities, feed_dict={self.input_ids: input_ids,
                                                                          self.input_mask: input_masks,
                                                                          self.input_type_ids: segment_ids})
        except Exception as e:
            self.logger.error(e)
        finally:
            return predict_result

    def _convert_single_example(self, text_a, text_b):
        """
        convert text a and text b to id, padding [CLS] [SEP]
        :param text_a: text a
        :param text_b: text b
        :return: input ids, input mask, segment ids
        """
        tokens = []
        input_ids = []
        segment_ids = []
        input_mask = []
        try:
            text_a = self.tokenizer.tokenize(text_a)
            if text_b:
                text_b = self.tokenizer.tokenize(text_b)
            self._truncate_seq_pair(text_a, text_b)

            tokens.append("[CLS]")
            segment_ids.append(0)

            for token in text_a:
                tokens.append(token)
                segment_ids.append(0)
            segment_ids.append(0)
            tokens.append("[SEP]")

            if text_b:
                for token in text_b:
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append('[SEP]')
                segment_ids.append(1)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            input_mask = [1] * len(input_ids)

            while len(input_ids) < 50:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

        except:
            self.logger.error()

        finally:
            return input_ids, input_mask, segment_ids

    def _truncate_seq_pair(self, tokens_a, tokens_b):
        """
        Truncates a sequence pair in place to the maximum length.
        :param tokens_a: text a
        :param tokens_b: text b
        """
        try:
            while True:
                if tokens_b:
                    total_length = len(tokens_a) + len(tokens_b)
                else:
                    total_length = len(tokens_a)

                if total_length <= 45 - 3:
                    break
                if tokens_b == None:
                    break
                if len(tokens_a) > len(tokens_b):
                    tokens_a.pop()
                else:
                    tokens_b.pop()
        except:
            self.logger.error()

def Compare_from_txt(classifier, path = './data/message.txt'):
    messages = []
    data = pd.DataFrame()
    with open(path, encoding='utf8')  as file:
        for line in file.readlines():
            line = line.strip()
            messages.append(line)
    messages = messages[:-1]
    length = len(messages)
    print(messages[-2], messages[-1])
    print('len(message.txt):', length)

    all_loop = length * (length - 1) // 2
    count = 0
    for i in range(length):
        text_a = []
        text_b = []
        problity = []
        for j in range(i + 1, length):
            pairs = [messages[i], messages[j]]
            text_a.append(messages[i])
            text_b.append(messages[j])
            predict_problity = round(classifier.predict([pairs])[0][1],4)
            problity.append(predict_problity)
            count += 1
            print('{}/{}'.format(count, all_loop))
        temp = pd.DataFrame(data={'text_a': text_a, 'text_b': text_b, 'problity': problity},
                            columns=['text_a', 'text_b', 'problity'])
        temp = temp.sort_values(by=['problity'], ascending=False)[:10]
        data = pd.concat([data, temp], axis=0)
        print(temp)
        data.to_csv('./text_pair.csv', index=None, encoding='utf8')

def Compare_from_df(path='./data/message_gaiqian_cluster.csv', classifier=None):

    data_cluster = pd.read_csv(path, encoding='utf8')
    cluster_index = data_cluster.cluster.unique()
    data = pd.DataFrame()
    for index in cluster_index:
        messages = list(data_cluster[data_cluster.cluster == index].meassage)

        length = len(messages)
        print('cluster:{}, length:{}'.format(index,length))

        all_loop = length * (length - 1) // 2
        count = 0
        for i in range(length):
            text_a = []
            text_b = []
            problity = []
            cluster = []
            for j in range(i + 1, length):
                pairs = [messages[i], messages[j]]
                text_a.append(messages[i])
                text_b.append(messages[j])
                predict_problity = round(classifier.predict([pairs])[0][1],4)
                problity.append(predict_problity)
                cluster.append(index)
                count += 1
                if count%50==0:
                    print('index:{} \t\t{}/{}'.format(index,count, all_loop))
            temp = pd.DataFrame(data={'text_a': text_a, 'text_b': text_b, 'problity': problity, 'cluster':cluster},
                                columns=['text_a', 'text_b', 'problity', 'cluster'])
            temp = temp.sort_values(by=['problity'], ascending=False)[:5]
            data = pd.concat([data, temp], axis=0)
            print(temp)
            data.to_csv('./cluster_text_pair.csv', index=None, encoding='utf8')

def Compare_from_df_label(path='./data/message_gaiqian_cluster.csv', label_path = './data/gaiqian_label.csv',classifier=None):
    data_cluster = pd.read_csv(path, encoding='utf8')
    data_label = pd.read_csv(label_path, encoding='utf8')
    cluster_index = data_label.cluster.unique()
    data = pd.DataFrame()

    for index in cluster_index:
        messages = list(data_cluster[data_cluster.cluster == index].meassage)
        data_label_sub = data_label[data_label.cluster == index]
        all_loop = len(messages)* data_label_sub.shape[0]
        count = 0
        for line in data_label_sub.iterrows():
            text_a = line[1][0]
            cluster = line[1][1]
            label = line[1][2]
            text_as = []
            text_bs = []
            clusters = []
            labels = []
            problitys = []
            for text_b in messages:
                pairs = [text_a, text_b]
                predict_problity = round(classifier.predict([pairs])[0][1], 4)
                text_as.append(text_a)
                text_bs.append(text_b)
                clusters.append(cluster)
                labels.append(label)
                problitys.append(predict_problity)
                count += 1
                if count%50==0:
                    print('index:{} \t\t{}/{}'.format(index, count, all_loop))
            temp = pd.DataFrame(data={'text_a': text_as, 'text_b': text_bs, 'problity': problitys,'label':labels, 'cluster': clusters},
                                columns=['text_a', 'text_b', 'problity','label', 'cluster'])
            temp = temp.sort_values(by=['problity'], ascending=False)[:5]
            data = pd.concat([data, temp], axis=0)
            print(temp)
            data.to_csv('./cluster_label_text_pair.csv', index=None, encoding='utf8')


def predict_classifier(classifier , path):
    message = ''
    count = 0
    with open(path, mode='r', encoding='utf8') as file:
        for line in file.readlines():
            try:
                mes = line.strip().split()
                sentense = mes[0]
                label = mes[1]
                pair = [sentense, None]
                ans = classifier.predict([pair])[0]
                print(count,sentense, label,ans)
                ans = [str(i) for i in ans]
                ans = ' '.join(ans)
                ans += '\n'
                message += ans
            except:
                print('error :', line)
                continue
            count += 1
    with open('./text_class_data/test_ans_tmp.txt', mode='w', encoding='utf8') as flie:
        flie.write(message)





if __name__ == "__main__":
    bs_classfier = BertSim(gpu_no=0, log_dir='log', bert_sim_dir='./model/classifier_model', verbose=True)
    print(bs_classfier.predict([['帮我退票',None],
                                ['钱退的太少了', None]]))
    print(bs_classfier.predict([['我要取消退票', None]]))

    bs_similarity = BertSim(gpu_no=0, log_dir='log', bert_sim_dir='./model/sim_23w_model', verbose=True)
    print(bs_similarity.predict([['帮我退票', '请你帮我把票退了']]))
    #predict_classifier(bs, './text_class_data/sim_test.txt')
    # Compare_from_txt(path='./data/message.txt', classifier=bs)
    # Compare_from_df(path='./data/message_gaiqian_cluster.csv', classifier=bs)
    # Compare_from_df_label(path='./sim_data/message_gaiqian_cluster.csv',label_path = './data/gaiqian_label.csv' ,classifier=bs)



