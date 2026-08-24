from freetoken.core import Batch, Req, SamplingParams


def test_batch_phase():
    req = Req(
        input_ids=[1, 2, 3],
        table_idx=0,
        cached_len=0,
        output_len=16,
        uid=1,
        sampling_params=SamplingParams(),
        cache_handle=None,
    )
    batch = Batch(reqs=[req], phase="decode")
    assert batch.is_decode
    assert not batch.is_prefill
    assert batch.size == 1
    assert req.can_decode
