import torch
from utils import length2mask
import torch.nn as nn
from torch.nn import functional as nnf
from model import ClipCaptionPrefix, ClipCaptionModel, MappingType
from param import args
from transformers import GPT2Tokenizer

class Translator(nn.Module):
    def __init__(self):
        super(Translator, self).__init__()
        self.mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[args.mapping_type]
        self.prefix_dim = 768
        self.prefix_length = args.prefix_length
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        self.reasoner = ClipCaptionModel(self.prefix_length, clip_length=args.prefix_length_clip, prefix_size=self.prefix_dim,
                                  num_layers=args.num_layers, mapping_type=self.mapping_type)


    def infer_batch(self, can_feats, lengths, text_prompt, prompt_mask):
        """

        :param sampling: if not, use argmax. else use softmax_multinomial
        :param train: Whether in the train mode
        :return: if sampling: return insts(np, [batch, max_len]),
                                        log_probs(torch, requires_grad, [batch,max_len])
                                        hiddens(torch, requires_grad, [batch, max_len, dim})
                        And if train: the log_probs and hiddens are detached
                    if not sampling: returns insts(np, [batch, max_len])
        """
        self.reasoner.eval()
        batch_size = can_feats.shape[0]
        img_mask = torch.tensor(~(length2mask(lengths))).float()
        img_num = max(lengths)
        prefix, img_mask = can_feats, img_mask
        prefix_embed = (self.reasoner.clip_project(prefix)*img_mask.unsqueeze(-1)).view(batch_size, img_num*self.prefix_length, -1)
       
        text_enc = self.reasoner.gpt.transformer.wte(text_prompt)
        output_text = []
        for id, each_text in enumerate(text_prompt):
            each_prompt = torch.cat([prefix_embed[id], text_enc[id][:prompt_mask[id]]], dim=0).unsqueeze(0)
            output_text.append(self.generate2(embed=each_prompt))
            #outputs = self.reasoner.gpt(inputs_embeds=each_prompt, output_hidden_states=True)
        return output_text
        # outputs = self.reasoner.gpt(inputs_embeds=prefix_embed, output_hidden_states=True)
        # self.generate2(embed=prefix_embed)
        # return outputs.hidden_states[12]



    def generate2(self, tokens=None, prompt=None, embed=None, entry_count=1, entry_length=80,  
                    top_p=0.8,
                    temperature=1.0,
                    stop_token: str = '....',
                    ):
            self.reasoner.eval()
            generated_num = 0
            generated_list = []
            stop_token_index = self.tokenizer.encode(stop_token)[0]
            filter_value = -float("Inf")
            device = next(self.reasoner.parameters()).device

            with torch.no_grad():

                for entry_idx in range(entry_count):
                    if embed is not None:
                        generated = embed[entry_idx].unsqueeze(0)
                    else:
                        if tokens is None:
                            tokens = torch.tensor(self.tokenizer.encode(prompt))
                            tokens = tokens.unsqueeze(0).to(device)

                        generated = self.reasoner.gpt.transformer.wte(tokens)

                    tokens = None
                    for i in range(entry_length):
                        
                        outputs = self.reasoner.gpt(inputs_embeds=generated)
                        logits = outputs.logits
                        logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(
                            nnf.softmax(sorted_logits, dim=-1), dim=-1
                        )
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                            ..., :-1
                        ].clone()
                        sorted_indices_to_remove[..., 0] = 0

                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        logits[:, indices_to_remove] = filter_value
                        next_token = torch.argmax(logits, -1).unsqueeze(0)
                        next_token_embed = self.reasoner.gpt.transformer.wte(next_token)
                        if tokens is None:
                            tokens = next_token
                        else:
                            tokens = torch.cat((tokens, next_token), dim=1)
                        generated = torch.cat((generated, next_token_embed), dim=1)
                        if stop_token_index == next_token.item():
                            break
                    try:
                        output_list = list(tokens.squeeze().cpu().numpy()) # joslin:delete tolist()
                    except TypeError:
                        output_list = [tokens.squeeze().cpu().numpy()]
                    output_text = self.tokenizer.decode(output_list)
                    generated_list.append(output_text)

            return generated_list
    
    def load(self, path):
        ''' Loads parameters (but not training state) '''
        print("Load the speaker's state dict from %s" % path)
        pretrained_dict = torch.load(path)
        model_dict = self.reasoner.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        self.reasoner.load_state_dict(model_dict)
    

    def clipcap(self, prefix, lengths, seq_tensor, seq_mask, target_tensor=None, train=True):
        if train:
            self.reasoner.train()
        else:
            self.reasoner.eval()

        batch_size, total_prefix = len(lengths), max(lengths)
        img_mask = torch.tensor(~(length2mask(lengths))).float()
        mask = torch.cat([torch.ones(batch_size,total_prefix*args.prefix_length).cuda(), seq_mask], dim=1)

        outputs = self.reasoner(seq_tensor, prefix, mask, img_mask)
        logits = outputs.logits[:, (total_prefix*args.prefix_length) - 1: -1]
        
        loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), target_tensor.flatten(), ignore_index=0)

        return loss, outputs.hidden_states[-1][:, -1, :].unsqueeze(1)