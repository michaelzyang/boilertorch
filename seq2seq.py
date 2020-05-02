"""
Copyright 2020 Michael Yang

The MIT License(MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from gadgets import TorchGadget
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime

class Seq2SeqGadget(TorchGadget):
    def __init__(self, *args, **kwargs):
        super(Seq2SeqGadget, self).__init__(*args, **kwargs)
        from Levenshtein import distance as levenshtein_distance
        self.levenshtein_distance = levenshtein_distance

    def get_outputs(self, batch, sample_prob=0.0):
        """
        CUSTOMIZABLE FUNCTION
        Takes a batch and runs it through the model's forward method to get outputs.
        Overload this function appropriately with your Dataset class's output and model forward function signature
        """
        # Unpack batch
        x, y, xlens, ylens = batch
        x, y, xlens, ylens = x.to(self.device), y.to(self.device), xlens.to(self.device), ylens.to(self.device)

        # Compute outputs
        logits, srclens, attns = self.model(x, xlens, y[:-1, :], sample_prob)  # truncate final input time step

        # Clean up
        del x
        # no need to del batch (deleted in the calling function)
        torch.cuda.empty_cache()

        return logits, y[1:, :], ylens - 1, srclens, attns  # shift targets left

    def get_predictions(self, batch, max_len, beam_width=1):
        """
        CUSTOMIZABLE FUNCTION
        Takes a batch and generates predictions from the model e.g. class labels for classification models
        Overload this function appropriately with your Dataset class's output and model generation function signature
        """
        # Unpack batch
        x, y, xlens, ylens = batch
        x, y, xlens, ylens = x.to(self.device), y.to(self.device), xlens.to(self.device), ylens.to(self.device)

        # Generate predictions
        padded_seqs, seq_lens, srclens, attns = self.model.predict(x, xlens, max_len, beam_width)

        # Clean up
        del x, xlens
        # no need to del batch (deleted in the calling function)
        torch.cuda.empty_cache()

        return padded_seqs, y, seq_lens, ylens, srclens, attns

    def compute_loss(self, batch, criterion, sample_prob=0.0):
        """
        CUSTOMIZABLE FUNCTION
        Computes the average loss over a batch of data for a given criterion (loss function).
        Overload this function appropriately with your Dataset class's output and model forward function signature
        """
        logits, target, target_lens, srclens, attns = self.get_outputs(batch, sample_prob)

        vocab_size = logits.shape[2]
        total_loss = criterion(logits.view(-1, vocab_size), target.view(-1))  # flatten all time steps across batches
        total_tokens = target_lens.sum()
        loss = total_loss / total_tokens

        # Clean up
        del logits, target, target_lens, srclens, attns
        # no need to del batch (deleted in the calling function)
        torch.cuda.empty_cache()

        return loss

    def compute_metric(self, batch, max_len, index2token, beam_width=1):
        """
        CUSTOMIZABLE FUNCTION
        Computes the average evaluation metric over a batch of data for a given criterion (loss function).
        Overload this function appropriately with your Dataset class's output and model forward or inference function
        signature
        """
        padded_seqs, y, seq_lens, ylens, srclens, attns = self.get_predictions(batch, max_len, beam_width)
        metric = self.padded_levenshtein_distance(padded_seqs, y, seq_lens, ylens, index2token)

        # Clean up
        del padded_seqs, seq_lens, y, ylens, srclens, attns
        # no need to del batch (deleted in the calling function)
        torch.cuda.empty_cache()

        return metric

    def padded_levenshtein_distance(self, seqs, y, seq_lens, y_lens, index2token, batch_first=False):
        """
        :param seqs: (max_len, N) Batch input of generated text INCLUDING <sos> and <eos> tokens
        :param seq_lens: (N, ) Generated seq lengths INCLUDING <sos> and <eos> tokens
        :param y: (max_ylen, N) Batch input of target text INCLUDING <sos> and <eos> tokens
        :param y_lens: (N, ) Target text lengths INCLUDING <sos> and <eos> tokens
        :param index2token: the dictionary mapping indices to string tokens
        :param batch_first: Bool
        """
        N = len(y_lens)
        distances = []
        pred_seq_strings = self.idx2string(seqs, seq_lens, index2token, batch_first=batch_first)
        true_seq_strings = self.idx2string(y, y_lens, index2token, batch_first=batch_first)
        for i in range(N):
            if i == 0:  # print one prediction TODO DELETE
                print('hyp:', pred_seq_strings[i])
                print('ref:', true_seq_strings[i])

            distance = self.levenshtein_distance(pred_seq_strings[i], true_seq_strings[i])
            distances.append(distance)
        print(distances)
        return sum(distances) / N

    def idx2string(self, seqs, lens, index2token, batch_first=False):
        """
        Converts a (batch) of token indices in a tensor to a list of output strings
        :param seqs: (max_len, N) Batch input of target text INCLUDING <sos> and <eos> tokens
        :param lens: (N, ) generated seq lengths INCLUDING <sos> and <eos> tokens
        :param index2token: the dictionary mapping indices to string tokens
        :param batch_first: Bool
        """
        N = len(lens)
        seq_strings = []
        for i in range(N):
            # Strip <pad> tokens
            seq_tokens = seqs[i, :lens[i]] if batch_first else seqs[:lens[i], i]

            # Strip <sos> and <eos> tokens and convert tensors of indices into strings
            seq_str = ''.join([index2token[x.item()] for x in seq_tokens[1:-1]])
            seq_strings.append(seq_str)
        return seq_strings

    def visualize_attention(self, batch, sample_prob=1.0, num_visualizations=-1):
        """
        Visualizes the attention for a batch of data.
        :param sample_prob: scheduled sampling probability (set to 0.0 to teacher force; 1.0 for greedy search)
        """
        _, _, target_lens, srclens, attns = self.get_outputs(batch, sample_prob)
        target_lens = target_lens.cpu().detach().numpy()  # (N, )
        srclens = srclens.cpu().detach().numpy()  # (N, )
        attns = attns.cpu().detach().numpy()  # (T, N, S)
        T, N, S = attns.shape
        num_visualizations = min(num_visualizations, N) if num_visualizations > -1 else N
        for i in range(num_visualizations):
            fig, ax = plt.subplots(figsize=(srclens[i] * 0.1, target_lens[i] * 0.1))
            ax.imshow(attns[:target_lens[i], i, :srclens[i]])
            ax.set_xticks(np.arange(0, srclens[i], 4))
            ax.set_yticks(np.arange(0, target_lens[i], 2))
            ax.set_xlabel('Input time steps')
            ax.set_ylabel('Output time steps')
            # ax.set_xticklabels(list(data[i][0]))
            # ax.set_yticklabels(data[i][1].split() + ['</s>'])
            # ax.set_ylim(target_lens[i] - 0.5, -0.5)
            plt.show()

    def train_epoch(self, train_loader, criterion, epoch_num=1, batch_group_size=0):
        """
        TODO
        """
        sample_prob = min((epoch_num / 100) * 2, 0.5)
        print(f'Epoch {epoch_num}: using scheduled sampling probability of {sample_prob}.')

        avg_loss = 0.0  # Accumulate loss over subsets of batches for reporting
        for i, batch in enumerate(train_loader):
            batch_num = i + 1
            self.optimizer.zero_grad()

            loss = self.compute_loss(batch, criterion, sample_prob)  # EDITED HERE
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()

            # Accumulate loss for reporting
            avg_loss += loss.item()
            if batch_group_size and (batch_num) % batch_group_size == 0:
                print(
                    f'Epoch {epoch_num}, Batch {batch_num}\tTrain loss: {avg_loss / batch_group_size:.4f}\t{datetime.now()}')
                avg_loss = 0.0

            # Cleanup
            del batch
            torch.cuda.empty_cache()

        for batch in train_loader:
            self.visualize_attention(batch, sample_prob=1.0, num_visualizations=1)
            break
        return avg_loss

    def print_epoch_log(self, epoch_num, loss=None, metric=None, dataset='dev'):
        epoch_log = f'Epoch {epoch_num} complete.'
        if loss is not None and metric is not None:
            epoch_log += f'\t{dataset} loss: {loss:.4f}\t{dataset} ppl: {np.exp(loss.cpu().detach().numpy()):.4f}\t{dataset} metric: {metric:.4f}'
        print(f'{epoch_log}\t{datetime.now()}')

    def predict_set(self, data_loader, max_len, index2token, beam_width=1):
        """
        Returns the a list of strings generated by get_predictions
        :param data_loader: A dataloader for the data over which to evaluate
        :param max_len: maximum output time steps
        :param index2token: the dictionary mapping indices to string tokens
        :return: Concatenated predictions of the same type as returned by self.get_predictions
        """
        self.model.eval()
        predictions_set = []

        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                padded_seqs, _, seq_lens, _, _, _ = self.get_predictions(batch, max_len, beam_width)
                # predictions_batch = outputs.detach().to('cpu')
                predictions_set.extend(self.idx2string(padded_seqs, seq_lens, index2token))

                # Clean up
                del batch
                torch.cuda.empty_cache()

        self.model.train()

        del padded_seqs, seq_lens
        torch.cuda.empty_cache()

        return predictions_set


class BeamSearch():
    def __init__(self, max_len, pad_idx, sos_idx, eos_idx, batch_size, beam_width):
        """
        TODO
        """
        self.T = max_len
        self.N = batch_size
        self.K = beam_width
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.rnn_hidden_states = None

        # sum of log probability scores
        self.scores = None  # (N, K)
        # input token indices (i.e. <sos> is token 0)
        self.seqs = None  # (T, N, K)
        # sequence lengths
        self.lens = None  # (N, K)
        # mask of which beam candidates have not finished decoding
        self.active_mask = None  # (N, K)

    def step(self, logits, step_num=0):
        """
        Beam search step
        :param logits: (N * K, V)
        :param step_num: int
        """
        lprob = torch.nn.functional.log_softmax(logits, dim=-1)  # step == 0 ? (N, V) : (N * K, V)

        if step_num == 0:
            # Initialize beams
            self.N = len(logits)
            self.seqs = torch.full((self.T, self.N, self.K), fill_value=self.pad_idx, dtype=torch.long,
                                   device=logits.device)
            self.lens = torch.zeros((self.N, self.K), dtype=torch.long, device=logits.device)
            self.scores, self.seqs[step_num + 1] = lprob.topk(self.K, sorted=False)  # (N, K), (N, K)

            # Check if any candidates reached <eos> and update active_mask and lens
            self.active_mask = (self.seqs[step_num + 1] != self.eos_idx).clone().detach().bool()  # (N, K)
            self.lens[~self.active_mask] = step_num + 2
        else:
            # Add lprobs to scores
            lprob = lprob.reshape(self.N, self.K, -1)  # (N, K, V)
            all_cand_scores = self.scores.unsqueeze(-1) + lprob  # (N, K, V) broadcast scores over V

            # Loop over each of the N beams
            for i in range(self.N):
                k = self.active_mask[i].sum()  # active beam width
                if k > 0:
                    # Extend each seq in the beam by k candidates
                    cand_scores, tokens = all_cand_scores[i].topk(k, sorted=False)  # (K, k), (K, k)
                    cand_scores[~self.active_mask[i]] = float('-inf')  # ensure inactive candidates are not selected

                    # Prune down to top k beams for each instance in the batch
                    pruned_scores, flat_idxs = torch.flatten(cand_scores).topk(k, sorted=False)  # (k, ), (k, )
                    pruned_row_idxs = torch.floor_divide(flat_idxs, self.K)  # the cands to extend
                    pruned_col_idxs = torch.fmod(flat_idxs, k)  # the tokens to extend them with
                    pruned_tokens = tokens[pruned_row_idxs, pruned_col_idxs]

                    # Update scores and seqs (note the use of nonzero().squeeze(1), which turns masks into indices)
                    active_idxs = self.active_mask[i].nonzero().squeeze(1)
                    self.scores[i, active_idxs] = pruned_scores
                    self.seqs[:step_num + 1, i, active_idxs] = self.seqs[:step_num + 1, i, pruned_row_idxs]  # base seqs
                    self.seqs[
                        step_num + 1, i, active_idxs] = pruned_tokens  # extend the base seqs with the selected tokens

            # Check if any candidates reached <eos> and update active_mask and lens
            eos_mask = (self.seqs[step_num + 1] == self.eos_idx).clone().detach().bool()
            self.active_mask[eos_mask] = 0  # deactivate the newly ended cands
            self.lens[eos_mask] = step_num + 2  # deactivate the newly ended cands

        # Short circuit beam search if highest scoring candidate has ended
        top_cands = torch.argmax(self.scores, dim=1)  # (N, )
        finished_beam_mask = self.active_mask[
                                 torch.arange(self.N), top_cands] == 0  # (N, )  beam finished if top cand inactive
        self.active_mask[finished_beam_mask.nonzero().squeeze(1)] = 0  # set all cands in finished beams to inactive

    # def generate(self, keys, values, src_key_lens=None, max_len=250, beam_width=1):
    #     """
    #     :param key :(S, N, key_size) Output of the Encoder Key projection layer
    #     :param values: (S, N, value_size) Output of the Encoder Value projection layer
    #     :params src_key_lens: (N, ) lengths of source sequences in the batch
    #     :params max_len: maximum sequence length to generate
    #     :params beam_width: beam width for the beam search algorithm
    #     :return padded_seqs: (max_len, N) output sequences (with <sos> and <eos> tokens)
    #     :return seq_lens: (N, ) includes <sos> and <eos> tokens
    #     :return attns: (max_len, N, S)
    #     """
    #     S, N = keys.shape[:2]
    #     if src_key_lens is None:
    #         src_key_lens = torch.full((N,), fill_value=S, device=values.device)  # default full length
    #
    #     search = BeamSearch(max_len, self.pad_idx, self.sos_idx, self.eos_idx, N, beam_width)
    #     search.seqs = torch.full((max_len, N, beam_width), fill_value=self.pad_idx, dtype=torch.long, device=values.device)  # (T, N, K)
    #     search.seqs[0, :, :] = self.sos_idx
    #     attns = []
    #
    #     # Generate step 0
    #     tokens = torch.full((N, ), fill_value=self.sos_idx, dtype=torch.long, device=values.device)  # (N, )
    #     context = None
    #     hidden_states = None
    #     step_logits, hidden_states, context, attn = self.forward(keys, values, tokens, src_key_lens, hidden_states, context)
    #     attns.append(attn.repeat_interleave(beam_width, dim=0))
    #     search.step(step_logits, 0)
    #
    #     # Broadcast N inputs from encoder into N * K inputs for subsequent steps
    #     hidden_states = [
    #         [hidden_states[0][0].repeat_interleave(beam_width, dim=0), hidden_states[0][1].repeat_interleave(beam_width, dim=0)],
    #         [hidden_states[1][0].repeat_interleave(beam_width, dim=0), hidden_states[1][1].repeat_interleave(beam_width, dim=0)]
    #     ]
    #     context = context.repeat_interleave(beam_width, dim=0)
    #     keys = keys.repeat_interleave(beam_width, dim=1)
    #     values = values.repeat_interleave(beam_width, dim=1)
    #     src_key_lens = src_key_lens.repeat_interleave(beam_width, dim=0)
    #
    #     # Generate subsequent steps
    #     for step_num in range(1, max_len - 1):  # -1 because final token must be <eos>
    #         if search.active_mask.sum() == 0:
    #             break
    #         # Step
    #         tokens = search.seqs[step_num].flatten()  # (N * K, )
    #         step_logits, hidden_states, context, attn = self.forward(keys, values, tokens, src_key_lens, hidden_states, context)
    #         attns.append(attn)
    #         search.step(step_logits, step_num)
    #     search.seqs[-1, :, :] = self.eos_idx  # ensure all seqs end with <eos> even if not predicted by model
    #
    #     # Return top candidate from each beam
    #     top_cands = torch.argmax(search.scores, dim=1)  # (N, )
    #     padded_seqs = search.seqs[:, torch.arange(search.N), top_cands]  # (T, N)
    #     seq_lens = search.lens[torch.arange(search.N), top_cands]  # (N, )
    #
    #     attns = torch.stack(attns, dim=0)
    #     return padded_seqs, seq_lens, attns