import torch
import torch.nn as nn
from transformers import BertModel, AlbertModel, DistilBertModel, RobertaModel, XLNetModel, LongformerModel

model_ckpts = {'bert': "bert-base-uncased",
               'albert': "albert-base-v2",
               'roberta': "roberta-base",
               'xlnet': "xlnet-base-cased",
               'distilbert': "distilbert-base-uncased",
               'longformer': "allenai/longformer-base-4096"}

class MultiTaskNet(nn.Module):
    def __init__(self, task_configs=[],
                 device='cpu',
                 finetuning=True,
                 lm='bert',
                 bert_pt=None,
                 bert_path=None):
        super().__init__()

        assert len(task_configs) > 0

        # load the model or model checkpoint
        if bert_path == None:
            if lm == 'bert':
                self.bert = BertModel.from_pretrained(model_ckpts[lm])
            elif lm == 'distilbert':
                self.bert = DistilBertModel.from_pretrained(model_ckpts[lm])
            elif lm == 'albert':
                self.bert = AlbertModel.from_pretrained(model_ckpts[lm])
            elif lm == 'xlnet':
                self.bert = XLNetModel.from_pretrained(model_ckpts[lm])
            elif lm == 'roberta':
                self.bert = RobertaModel.from_pretrained(model_ckpts[lm])
            elif lm == 'longformer':
                self.bert = LongformerModel.from_pretrained(model_ckpts[lm])
        else:
            output_model_file = bert_path
            model_state_dict = torch.load(output_model_file,
                                          map_location=lambda storage, loc: storage)
            if lm == 'bert':
                self.bert = BertModel.from_pretrained(model_ckpts[lm],
                        state_dict=model_state_dict)
            elif lm == 'distilbert':
                self.bert = DistilBertModel.from_pretrained(model_ckpts[lm],
                        state_dict=model_state_dict)
            elif lm == 'albert':
                self.bert = AlbertModel.from_pretrained(model_ckpts[lm],
                        state_dict=model_state_dict)
            elif lm == 'xlnet':
                self.bert = XLNetModel.from_pretrained(model_ckpts[lm],
                        state_dict=model_state_dict)
            elif lm == 'roberta':
                self.bert = RobertaModel.from_pretrained(model_ckpts[lm],
                        state_dict=model_state_dict)

        self.device = device
        self.finetuning = finetuning
        self.task_configs = task_configs
        self.module_dict = nn.ModuleDict({})
        self.lm = lm

        # hard corded for now
        hidden_size = 768
        hidden_dropout_prob = 0.1

        for config in task_configs:
            name = config['name']
            task_type = config['task_type']
            vocab = config['vocab']

            if task_type == 'tagging':
                # for tagging
                vocab_size = len(vocab) # 'O' and '<PAD>'
                if 'O' not in vocab:
                    vocab_size += 1
                if '<PAD>' not in vocab:
                    vocab_size += 1
            else:
                # for pairing and classification
                vocab_size = len(vocab)

            self.module_dict['%s_dropout' % name] = nn.Dropout(hidden_dropout_prob)
            self.module_dict['%s_fc' % name] = nn.Linear(hidden_size, vocab_size)

            # Lokesh: Adding attention based classifier layers
            self.module_dict['variational_fc'] = nn.Linear(hidden_size, 1)
            self.module_dict['fc_cls_var'] = nn.Linear(hidden_size*2, vocab_size)


    def forward(self, x, y, y_self_sup=None,
                augment_batch=None,
                aug_enc=None,
                second_batch=None,
                x_enc=None,
                task='hotel_tagging',
                get_enc=False):
        """Forward function of the BERT models for classification/tagging.

        Args:
            x (Tensor):
            y (Tensor):
            augment_batch (tuple of Tensor, optional):
            aug_enc (Tensor, optional):
            second_batch (Tensor, optional):
            task (string, optional):
            get_enc (boolean, optional):

        Returns:
            Tensor: logits
            Tensor: y
            Tensor: yhat
            Tensor (optional): enc"""

        # move input to GPU
        x = x.to(self.device)
        y = y.to(self.device)

        # Lokesh: Push y_self_sup also to device
        if y_self_sup is not None:
            y_self_sup = y_self_sup.to(self.device)

        if second_batch != None:
            index, lam = second_batch
            lam = torch.tensor(lam).to(self.device)
        if augment_batch != None:
            aug_x, aug_lam = augment_batch
            aug_x = aug_x.to(self.device)
            aug_lam = torch.tensor(aug_lam).to(self.device)

        dropout = self.module_dict[task + '_dropout']
        fc = self.module_dict[task + '_fc']

        # Lokesh: Fetching the classifier and attention output layers
        var_fc = self.module_dict["variational_fc"]
        fc_cls_var = self.module_dict['fc_cls_var']

        if 'tagging' in task: # TODO: this needs to be changed later
            if self.training and self.finetuning:
                self.bert.train()
                if x_enc is None:
                    enc = self.bert(x)[0]
                else:
                    enc = x_enc
                # Dropout
                enc = dropout(enc)
            else:
                self.bert.eval()
                with torch.no_grad():
                    enc = self.bert(x)[0]

            if augment_batch != None:
                if aug_enc is None:
                    aug_enc = self.bert(aug_x)[0]
                enc[:aug_x.shape[0]] *= aug_lam
                enc[:aug_x.shape[0]] += aug_enc * (1 - aug_lam)

            if second_batch != None:
                enc = enc * lam + enc[index] * (1 - lam)
                enc = dropout(enc)

            logits = fc(enc)
            y_hat = logits.argmax(-1)
            if get_enc:
                return logits, y, y_hat, enc
            else:
                return logits, y, y_hat
        else:
            if self.training and self.finetuning:
                self.bert.train()
                if x_enc is None:
                    output = self.bert(x)
                    pooled_output = output[0][:, 0, :]
                    pooled_output = dropout(pooled_output)
                else:
                    pooled_output = x_enc
            else:
                self.bert.eval()
                with torch.no_grad():
                    output = self.bert(x)
                    pooled_output = output[0][:, 0, :]
                    pooled_output = dropout(pooled_output)

            if augment_batch != None:
                if aug_enc is None:
                    output_aug = self.bert(aug_x)
                    pooled_output_aug = output_aug[0][:, 0, :]
                    pooled_output_aug = dropout(pooled_output_aug)
                pooled_output[:aug_x.shape[0]] *= aug_lam
                pooled_output[:aug_x.shape[0]] += pooled_output_aug * (1 - aug_lam)

            if second_batch != None:
                pooled_output = pooled_output * lam + pooled_output[index] * (1 - lam)

            logits = fc(pooled_output)


            # This is for the case study where we aggregate only based on y_self_sup
            # print(f"output: {output[0].shape}, output_aug: {output_aug[0].shape}")
            all_enc = output[0]
            if augment_batch != None:
                all_enc = torch.cat([output[0], output_aug[0]], axis=0)
            # y_self_sup_temp = y_self_sup.unsqueeze(dim=1)
            # print(f"all_enc {all_enc.shape}, y_self_sup: {y_self_sup.shape}")
            # variational_emb = torch.matmul(y_self_sup_temp, all_enc)
            variational_emb = torch.einsum('ijk,ij->ik', all_enc, y_self_sup)
            # variational_emb = variational_emb.squeeze(dim=1)
            variational_emb = variational_emb / torch.norm(variational_emb, dim=0)

            # Lokesh: Combine the variational_emb to match the original and augment batches
            if augment_batch != None:
                variational_emb = (aug_lam * variational_emb[:aug_x.shape[0]]) + \
                                    ((1-aug_lam) * variational_emb[aug_x.shape[0]:]) 

            # print(f"Final experiment variational_emb shape is : {variational_emb.shape}")

            # Lokesh: For this experiment variational_attn is meaningless
            variational_attn = torch.rand(2)
            variational_attn_aug = torch.rand(2)


            # # Lokesh: Compute the logits with attention aggregated embeddings
            # variational_attn = var_fc(output[0])
            # variational_attn = torch.sigmoid(variational_attn)
            # variational_attn = variational_attn.reshape((variational_attn.shape[0], 
            #                                 variational_attn.shape[2], variational_attn.shape[1]))
            # variational_emb = torch.matmul(variational_attn, output[0])
            # variational_emb = variational_emb.squeeze(dim=1)
            # variational_emb = variational_emb / torch.norm(variational_emb, dim=0)


            # # Lokesh: Compute variational embedding also as a convex combination
            # if augment_batch != None:
            #     variational_attn_aug = var_fc(output_aug[0])
            #     variational_attn_aug = torch.sigmoid(variational_attn_aug)
            #     variational_attn_aug = variational_attn_aug.reshape((variational_attn_aug.shape[0], 
            #                                     variational_attn_aug.shape[2], variational_attn_aug.shape[1]))
            #     variational_emb_aug = torch.matmul(variational_attn_aug, output_aug[0])
            #     variational_emb_aug = variational_emb_aug.squeeze(dim=1)
            #     variational_emb_aug = variational_emb_aug / torch.norm(variational_emb_aug, dim=0)

            #     variational_emb = (variational_emb * aug_lam) + ((1-aug_lam) * variational_emb_aug) 


            variational_emb = torch.cat([pooled_output, variational_emb], dim=1)

            # Lokesh: Get the variational Logits using variational embedding
            logits_variational = fc_cls_var(variational_emb)


            # Return variational_attn for self supervision
            # Pass the variational attn of the augmented batch also
            if augment_batch != None:
                variational_attn = torch.cat([variational_attn, variational_attn_aug], dim=0)

            if 'sts-b' in task:
                y_hat = logits
                y_hat_var = logits_variational
            else:
                y_hat = logits.argmax(-1)
                y_hat_var = logits_variational.argmax(-1)
            if get_enc:
                return logits_variational, y, y_hat_var, pooled_output
            else:
                return logits_variational, y, y_hat_var, y_self_sup, variational_attn
