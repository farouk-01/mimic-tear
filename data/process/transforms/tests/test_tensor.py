import torch
import pytest

from data.process.transforms.tensor import Lag


class TestTemporalTransform:

    @pytest.mark.parametrize("fill", ["first", "zero"])
    class TestLag:

        def test_lag_complete_and_single_is_same_result(self, fill):
            data = torch.arange(10)

            lag1 = Lag(input="in", output="out", fill=fill, periods=3)
            lag2 = Lag(input="in", output="out", fill=fill, periods=3)

            complete = lag1(data)
            single = torch.cat([lag2(v.unsqueeze(0)) for v in data])

            torch.testing.assert_close(complete, single)

        def test_lag_preserves_lookback_across_batches(self, fill):
            data = torch.arange(20)

            lag1 = Lag(input="in", output="out", fill=fill, periods=3)
            lag2 = Lag(input="in", output="out", fill=fill, periods=3)

            complete = lag1(data)

            batched_data = torch.chunk(data, 4)
            batched = torch.cat([lag2(batch) for batch in batched_data])

            torch.testing.assert_close(complete, batched)