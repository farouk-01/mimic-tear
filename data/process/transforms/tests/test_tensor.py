import torch
import pytest

from data.process.transforms.tensor import Lag, ConsecutiveCount


class TestTemporalTransform:

    @pytest.mark.parametrize("fill", ["first", "zero"])
    class TestLag:

        def test_lag_complete_and_single_is_same_result(self, fill):
            data = torch.arange(10)

            lag1 = Lag(input="in", output="out", fill=fill, periods=3)
            lag2 = Lag(input="in", output="out", fill=fill, periods=3)

            complete = lag1(data)
            
            # process the data in batch of 1
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

    class TestConsecutiveCount:

        def test_resets_on_zero(self):
            data = torch.tensor([0, 1, 1, 0, 1, 1, 1, 0])

            held_duration = ConsecutiveCount(input="in", output="out")

            result = held_duration(data)
            expected = torch.tensor([0, 1, 2, 0, 1, 2, 3, 0])

            torch.testing.assert_close(result, expected)

        def test_preserves_lookback_count_across_batches(self):
            data1 = torch.tensor([0, 1, 1, 1])
            data2 = torch.tensor([1, 0, 1, 1])

            held_duration = ConsecutiveCount(input="in", output="out")

            _ = held_duration(data1) # [0, 1, 2, 3]

            result2 = held_duration(data2)
            expected2 = torch.tensor([4, 0, 1, 2])

            torch.testing.assert_close(result2, expected2)

        def test_if_no_reset_then_count_added_everywhere(self):
            data1 = torch.tensor([1, 1, 1])
            data2 = torch.tensor([1, 1, 1])

            held_duration = ConsecutiveCount(input="in", output="out")

            _ = held_duration(data1) # [1, 2, 3]

            result2 = held_duration(data2)
            expected2 = torch.tensor([4, 5, 6])

            torch.testing.assert_close(result2, expected2)

        def test_max_count_clamps_max_counted_value(self):
            data = torch.tensor([1, 1, 1, 1, 1])

            held_duration = ConsecutiveCount(input="in", output="out", max_count=3)

            result = held_duration(data)
            expected = torch.tensor([1, 2, 3, 3, 3])

            torch.testing.assert_close(result, expected)

        def test_shift_shifts_count(self):
            data = torch.tensor([1, 1, 1, 1, 1])

            held_duration = ConsecutiveCount(input="in", output="out", shift=1)

            result = held_duration(data)
            expected = torch.tensor([0, 1, 2, 3, 4])

            torch.testing.assert_close(result, expected)

        def test_shift_shifts_lookback_count(self):
            data1 = torch.tensor([1, 1, 1])
            data2 = torch.tensor([1, 1, 1])

            held_duration = ConsecutiveCount(input="in", output="out", shift=1)

            _ = held_duration(data1) # [0, 1, 2]

            result2 = held_duration(data2)
            expected2 = torch.tensor([3, 4, 5])

            torch.testing.assert_close(result2, expected2)