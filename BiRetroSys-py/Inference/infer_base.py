import numpy as np

class BeamGenerateBase():
    def __init__(
        self,
        beam_size: int,
        batch_size: int,
        bos_token_ids: int,
        pad_token_ids: int,
        eos_token_ids: int,
        length_penalty=1.,
        min_len=1,
        max_len=256,
        temperature=1.,
        top_k=0,
        top_p=0.,
        return_num=10
    ):
        self.beam_size = beam_size
        self.batch_size = batch_size
        self.bos_token_ids = bos_token_ids
        self.pad_token_ids = pad_token_ids
        self.eos_token_ids = eos_token_ids
        self.length_penalty = length_penalty
        self.min_len = min_len
        self.max_len = max_len
        self.temperature = temperature
        assert top_k >= 0 and top_p >= 0.
        self.top_k = top_k
        self.top_p = top_p
        assert return_num <= beam_size
        self.return_num = return_num
    
    @property
    def is_done(self) -> bool:
        raise NotImplementedError()
    
    @property
    def current_token(self) -> np.ndarray:
        raise NotImplementedError()
    
    @property
    def mem_ids(self) -> np.ndarray:
        raise NotImplementedError()
    
    def _prepare(self):
        raise NotImplementedError()
    
    def generate(self):
        raise NotImplementedError()
    
    def finish_generate(self):
        raise NotImplementedError()