import numpy as np

from Inference.infer_base import BeamGenerateBase
from Inference.huggingface_beamsearch import BeamSearchScorer

class Beam_Generate(BeamGenerateBase):
    def __init__(
        self,
        beam_size: int,
        batch_size: int,
        bos_token_ids: int,
        pad_token_ids: int,
        eos_token_ids: int,
        vocab: dict[str:int],
        rvocab: dict[int:str],
        length_penalty=1.,
        min_len=1,
        max_len=256,
        beam_group=1,
        temperature=1.,
        top_k=0,
        top_p=0.,
        return_num=10,
        remove_finish_batch=True,
    ):
        super(Beam_Generate, self).__init__(
            beam_size=beam_size,
            batch_size=batch_size,
            bos_token_ids=bos_token_ids,
            pad_token_ids=pad_token_ids,
            eos_token_ids=eos_token_ids,
            length_penalty=length_penalty,
            min_len=min_len,
            max_len=max_len,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            return_num=return_num
        )
        self.beam_group = beam_group
        self.remove_finish_batch = remove_finish_batch
        self.vocab = vocab
        self.rvocab = rvocab

        assert self.beam_size % self.beam_group == 0
        self.beam_per_group = self.beam_size // self.beam_group
        self.beam_search = BeamSearchScorer(
            batch_size=self.batch_size,
            num_beams=self.beam_size,
            length_penalty=self.length_penalty,
            do_early_stopping=False,
            num_beam_hyps_to_keep=self.beam_size,
            num_beam_groups=self.beam_group
        )
        self.last_alive = self.unfinish_batch.tolist()
        self.ids_table = np.arange(self.batch_size * self.beam_size, dtype=np.int64).reshape(self.batch_size, self.beam_size)

        self._prepare()

    @property
    def is_done(self) -> bool:
        return (self.beam_search.is_done) or (self.all_token.shape[-1] >= self.max_len)

    @property
    def unfinish_batch(self) -> np.ndarray:
        return np.logical_not(self.beam_search._done)

    @property
    def current_token(self) -> np.ndarray:
        if self.remove_finish_batch:
            cur_token = self.cur_token.reshape(self.batch_size, self.beam_size)[
                self.unfinish_batch].reshape(-1)
        else:
            cur_token = self.cur_token.reshape(-1)
        return cur_token

    @property
    def mem_ids(self) -> np.ndarray:
        if self.remove_finish_batch:
            mem_ids = self.beam_idx.reshape(self.batch_size, self.beam_size)[
                self.unfinish_batch]
            mem_ids = mem_ids.reshape(-1)
            if sum(self.last_alive) < self.batch_size:
                self.ids_table[self.last_alive] = np.arange(sum(self.last_alive) * self.beam_size,
                                                               dtype=np.int64).reshape(-1, self.beam_size)
                mem_ids = self.ids_table.reshape(-1)[mem_ids]
            self.last_alive = self.unfinish_batch.tolist()
        else:
            mem_ids = self.beam_idx.reshape(-1)
        return mem_ids

    def _prepare(
        self
    ):
        self.scores = np.array([], dtype=np.float32)
        self.beam_scores = np.full((self.batch_size, self.beam_size), -float('inf'), dtype=np.float32)
        self.beam_scores[:, ::self.beam_per_group] = 0.
        self.beam_scores = self.beam_scores.reshape(self.batch_size * self.beam_size)
        self.cur_token = np.full((self.batch_size * self.beam_size,), self.bos_token_ids, dtype=np.int64)
        self.all_token = np.full((self.batch_size * self.beam_size,), self.bos_token_ids, dtype=np.int64).reshape(-1, 1)

        self.batchid_list = [i // self.beam_size for i in range(self.batch_size * self.beam_size)]
        self.groupid_list = []
        for i in range(self.batch_size):
            self.groupid_list.extend([j // self.beam_per_group for j in range(self.beam_size)])
        self.beam_idx = np.arange(self.batch_size * self.beam_size, dtype=np.int64)

    def _scores_process(
        self,
        # each token's logits score, size(batch * beam, vocab_size)
        scores: np.ndarray
    ):
        cur_len = self.all_token.shape[-1]
        if cur_len < self.min_len:
            scores[:, self.eos_token_ids] = -float('inf')
        return scores

    def _finished_batch_pad(
        self,
        # the decoder output like [unfinish_batch * beam, vocab_size], before log_softmax
        dec_output: np.ndarray
    ):
        if dec_output.shape[0] < self.batch_size * self.beam_size:
            new_dec_output = np.zeros((self.batch_size, self.beam_size * dec_output.shape[-1]),
                dtype=dec_output.dtype)
            new_dec_output[self.unfinish_batch] = dec_output.reshape(
                -1, self.beam_size * dec_output.shape[-1])
            return new_dec_output.reshape(self.batch_size * self.beam_size, -1)
        else:
            return dec_output
    
    def _log_softmax(self, inputs: np.ndarray) -> np.ndarray:
        inputs = np.exp(inputs / self.temperature)

        return np.log(inputs / inputs.sum(-1, keepdims=True))
    
    def _top_k(self, inputs: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        idx = np.argpartition(inputs, -k, -1)[..., -k:]
        colIdx = np.arange(idx.shape[0], dtype=np.int64).reshape(-1, 1)
        res = inputs[colIdx, idx]
        sort_idx = np.argsort(res, -1)[..., ::-1]
        return res[colIdx, sort_idx], idx[colIdx, sort_idx]

    def generate(
        self,
        # the decoder output like [unfinish_batch * beam, 1, vocab_size], before log_softmax
        dec_output: np.ndarray
    ):
        dec_output = dec_output[:, -1, :]
        if self.remove_finish_batch:
            dec_output = self._finished_batch_pad(dec_output)
        # size(batch * beam, vocab_size)
        next_token_logits = self._log_softmax(dec_output)

        next_token_logits = self._scores_process(next_token_logits)
        next_token_logits = next_token_logits + self.beam_scores.reshape(-1, 1)

        vocab_size = next_token_logits.shape[-1]

        if self.beam_group > 1:
            next_token_logits = next_token_logits.reshape(self.batch_size, self.beam_group, self.beam_per_group * vocab_size)
            for group_id in range(self.beam_group):
                group_idx = [group_id == i for i in self.groupid_list]
                group_token = self.all_token[group_idx]
                group_logits = next_token_logits[:, group_id]
                next_token_scores, next_tokens = self._top_k(group_logits, self.beam_per_group * 2)
                next_ids, next_tokens = np.floor_divide(next_tokens, vocab_size), np.fmod(next_tokens, vocab_size)
                beam_output = self.beam_search.process(
                    input_ids=group_token,
                    next_scores=next_token_scores,
                    next_tokens=next_tokens,
                    next_indices=next_ids,
                    pad_token_id=self.pad_token_ids,
                    eos_token_id=self.eos_token_ids
                )
                self.beam_scores[group_idx] = beam_output['next_beam_scores']
                self.all_token[group_idx] = group_token[beam_output['next_beam_indices']]
                self.cur_token = beam_output['next_beam_tokens']
                self.beam_idx[group_idx] = np.floor_divide(beam_output['next_beam_indices'], self.beam_per_group) * self.beam_per_group + group_id * self.beam_per_group + np.fmod(beam_output['next_beam_indices'], self.beam_per_group)
        else:
            next_token_logits = next_token_logits.reshape(self.batch_size, self.beam_per_group * vocab_size)
            next_token_scores, next_tokens = self._top_k(next_token_logits, self.beam_per_group * 2)
            next_ids, next_tokens = np.floor_divide(next_tokens, vocab_size), np.fmod(next_tokens, vocab_size)
            beam_output = self.beam_search.process(
                input_ids=self.all_token,
                next_scores=next_token_scores,
                next_tokens=next_tokens,
                next_indices=next_ids,
                pad_token_id=self.pad_token_ids,
                eos_token_id=self.eos_token_ids
            )
            self.beam_scores = beam_output['next_beam_scores']
            self.all_token = self.all_token[beam_output['next_beam_indices']]
            self.cur_token = beam_output['next_beam_tokens']
            self.beam_idx = beam_output['next_beam_indices']

        self.all_token = np.concatenate([self.all_token, np.expand_dims(self.cur_token, -1)], -1)

    def finish_generate(
        self
    ):
        beam_output = self.beam_search.finalize(
            input_ids=self.all_token,
            final_beam_scores=self.beam_scores,
            return_num=self.return_num
        )
        return beam_output['sequences'], beam_output['sequence_scores']
