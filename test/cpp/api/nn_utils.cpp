#include <gtest/gtest.h>

#include <torch/torch.h>

#include <test/cpp/api/support.h>

using namespace torch::nn;

struct NNUtilsTest : torch::test::SeedingFixture {};

TEST_F(NNUtilsTest, ClipGradNorm) {
  auto l = Linear(10, 10);
  float max_norm = 2;
  auto compute_norm = [&](float norm_type) -> float {
    float total_norm = 0.0;
    if (norm_type != std::numeric_limits<float>::infinity()) {
      for (const auto& p : l->parameters()) {
        total_norm +=
            p.grad().data().abs().pow(norm_type).sum().item().toFloat();
      }
      return std::pow(total_norm, 1.0 / norm_type);
    } else {
      for (const auto& p : l->parameters()) {
        auto param_max = p.grad().data().abs().max().item().toFloat();
        if (param_max > total_norm) {
          total_norm = param_max;
        }
      }
      return total_norm;
    }
  };
  auto compare_scaling =
      [&](const std::vector<torch::Tensor>& grads) -> torch::Tensor {
    std::vector<torch::Tensor> p_scale;
    for (int i = 0; i < grads.size(); i++) {
      auto param = l->parameters()[i];
      auto grad = grads[i];
      p_scale.push_back(param.grad().data().div(grad).view(-1));
    }
    auto scale = torch::cat(p_scale);
    return scale; // need to assert std is 0.
  };

  std::vector<torch::Tensor> grads = {
      torch::arange(1.0, 101).view({10, 10}),
      torch::ones({10}).div(1000),
  };
  std::vector<float> norm_types = {
      0.5,
      1.5,
      2.0,
      4.0,
      std::numeric_limits<float>::infinity(),
  };
  for (auto norm_type : norm_types) {
    for (int i = 0; i < grads.size(); i++) {
      l->parameters()[i].grad() =
          grads[i].clone().view_as(l->parameters()[i].data());
    }
    auto norm_before = compute_norm(norm_type);
    auto norm = utils::clip_grad_norm_(l->parameters(), max_norm, norm_type);
    auto norm_after = compute_norm(norm_type);
    ASSERT_FLOAT_EQ(norm, norm_before);
    ASSERT_FLOAT_EQ(norm_after, max_norm);
    ASSERT_LE(norm_after, max_norm);
    auto scaled = compare_scaling(grads);
    ASSERT_NEAR(0, scaled.std().item().toFloat(), 1e-7);
  }
  // Small gradients should be left unchanged
  grads = {
      torch::rand({10, 10}).div(10000),
      torch::ones(10).div(500),
  };
  for (auto norm_type : norm_types) {
    for (int i = 0; i < grads.size(); i++) {
      l->parameters()[i].grad().data().copy_(grads[i]);
    }
    auto norm_before = compute_norm(norm_type);
    auto norm = utils::clip_grad_norm_(l->parameters(), max_norm, norm_type);
    auto norm_after = compute_norm(norm_type);
    ASSERT_FLOAT_EQ(norm, norm_before);
    ASSERT_FLOAT_EQ(norm_before, norm_after);
    ASSERT_LE(norm_after, max_norm);
    auto scaled = compare_scaling(grads);
    ASSERT_NEAR(0, scaled.std().item().toFloat(), 1e-7);
    ASSERT_EQ(scaled[0].item().toFloat(), 1);
  }
  // should accept a single tensor as input
  auto p1 = torch::randn({10, 10});
  auto p2 = torch::randn({10, 10});
  auto g = torch::arange(1., 101).view({10, 10});
  p1.grad() = g.clone();
  p2.grad() = g.clone();
  for (const auto norm_type : norm_types) {
    utils::clip_grad_norm_(p1, max_norm, norm_type);
    utils::clip_grad_norm_({p2}, max_norm, norm_type);
    ASSERT_TRUE(torch::allclose(p1.grad(), p2.grad()));
  }
}

TEST_F(NNUtilsTest, ClipGradValue) {
  auto l = Linear(10, 10);
  float clip_value = 2.5;

  torch::Tensor grad_w = torch::arange(-50., 50).view({10, 10}).div_(5);
  torch::Tensor grad_b = torch::ones({10}).mul_(2);
  std::vector<std::vector<torch::Tensor>> grad_lists = {
      {grad_w, grad_b}, {grad_w, torch::Tensor()}};
  for (auto grad_list : grad_lists) {
    for (int i = 0; i < grad_list.size(); i++) {
      auto p = l->parameters()[i];
      auto g = grad_list[i];
      p.grad() = g.defined() ? g.clone().view_as(p.data()) : g;
    }

    utils::clip_grad_value_(l->parameters(), clip_value);
    for (const auto& p : l->parameters()) {
      if (p.grad().defined()) {
        ASSERT_LE(
            p.grad().data().max().item().toFloat(), clip_value);
        ASSERT_GE(
            p.grad().data().min().item().toFloat(), -clip_value);
      }
    }
  }

  // Should accept a single Tensor as input
  auto p1 = torch::randn({10, 10});
  auto p2 = torch::randn({10, 10});
  auto g = torch::arange(-50., 50).view({10, 10}).div_(5);
  p1.grad() = g.clone();
  p2.grad() = g.clone();
  utils::clip_grad_value_(p1, clip_value);
  utils::clip_grad_value_({p2}, clip_value);
  ASSERT_TRUE(torch::allclose(p1.grad(), p2.grad()));
}

TEST_F(NNUtilsTest, ConvertParameters) {
  std::vector<torch::Tensor> parameters{
    torch::arange(9, torch::kFloat32),
    torch::arange(9, torch::kFloat32).view({3, 3}),
    torch::arange(8, torch::kFloat32).view({2, 2, 2})
  };

  auto expected = torch::cat({
    torch::arange(9, torch::kFloat32),
    torch::arange(9, torch::kFloat32).view(-1),
    torch::arange(8, torch::kFloat32).view(-1)
  });
  auto vector = utils::parameters_to_vector(parameters);
  ASSERT_TRUE(vector.allclose(expected));

  std::vector<torch::Tensor> zero_parameters{
    torch::zeros({9}, torch::kFloat32),
    torch::zeros({9}, torch::kFloat32).view({3, 3}),
    torch::zeros({8}, torch::kFloat32).view({2, 2, 2})
  };

  utils::vector_to_parameters(vector, zero_parameters);
  for (int i = 0; i < zero_parameters.size(); ++i) {
    ASSERT_TRUE(zero_parameters[i].allclose(parameters[i]));
  }

  {
    auto conv1 = Conv2d(3, 10, 5);
    auto fc1 = Linear(10, 20);
    auto model = Sequential(conv1, fc1);

    auto vec = utils::parameters_to_vector(model->parameters());
    ASSERT_EQ(vec.size(0), 980);
  }
  {
    auto conv1 = Conv2d(3, 10, 5);
    auto fc1 = Linear(10, 20);
    auto model = Sequential(conv1, fc1);

    auto vec = torch::arange(0., 980);
    utils::vector_to_parameters(vec, model->parameters());

    auto sample = model->parameters()[0][0][0][0];
    ASSERT_TRUE(torch::equal(sample.data(), vec.data().slice(0, 0, 5)));
  }
}

int64_t PackedSequenceTest_batch_size = 5;
int64_t PackedSequenceTest_max_length = 6;

/*
    def _ordered_sequence(self, tensor_type):
        """Create ordered list of random sequences"""
        seqs = [tensor_type(random.randint(1, self.max_length))
                for _ in range(self.batch_size)]
        seqs = [s.random_(-128, 128) for s in seqs]
        ordered = sorted(seqs, key=len, reverse=True)
        return ordered
*/
template <typename Dtype>
std::vector<Tensor> _ordered_sequence() {
  std::vector<Tensor> seqs;
  seqs.reserve(PackedSequenceTest_batch_size);
  for (int64_t i = 0; i < PackedSequenceTest_batch_size, i++) {
    seqs.emplace_back(torch::empty({
      torch::randint(1, PackedSequenceTest_max_length, {1}).item<int64_t>()
    }, Dtype));
  }
  for (auto& s : seqs) {
    s.random_(-128, 128);
  }
  auto compare_tensor_size = [&](cosnt Tensor& t1, cosnt Tensor& t2) {
    return t1.size(0) > t2.size(0);
  }
  sort(seqs.begin(), seqs.end(), compare_tensor_size);
  for (const auto& s : seqs) {
    std::cout << s << std::endl; // yf225 TODO: DEBUG
  }
  return seqs;
}

/*
def _padded_sequence(self, tensor_type):
        """Create Tensor of random padded sequences"""
        ordered = self._ordered_sequence(tensor_type)
        lengths = list(map(len, ordered))
        padded_tensor = rnn_utils.pad_sequence(ordered)
        return padded_tensor, lengths
*/
template <typename Dtype>
std::vector<Tensor> _padded_sequence() {
  // Create Tensor of random padded sequences
  auto ordered = _ordered_sequence<Dtype>();
  std::vector<int64_t> lengths;
  lengths.reserve(ordered.size());
  for (const auto& t : ordered) {
    lengths.emplace_back(t.size(0));
  }
  auto padded_tensor = torch::nn::utils::rnn::pad_sequence(ordered);
  // yf225 TODO: alright, let's merge the pad_sequence PR https://github.com/pytorch/pytorch/pull/32387 first!!!
  // yf225 TODO: also, let's try to make the test cases for the new util functions simple... no need to test too much, let's just do unit test for each function
}
/*
class PackedSequenceTest(TestCase):

    _type_by_name = {
        'torch.DoubleTensor': (torch.DoubleTensor, 'double'),
        'torch.FloatTensor': (torch.FloatTensor, 'float'),
        # We leave out `'torch.HalfTensor': (torch.HalfTensor, 'half'),`
        # because of an error in `pad_packed_sequence`
        # > AttributeError: 'torch.HalfTensor' object has no attribute 'fill_'
        'torch.LongTensor': (torch.LongTensor, 'long'),
        'torch.IntTensor': (torch.IntTensor, 'int'),
        'torch.ShortTensor': (torch.ShortTensor, 'short'),
        'torch.CharTensor': (torch.CharTensor, 'char'),
        'torch.ByteTensor': (torch.ByteTensor, 'byte'),
    }

    def __init__(self, *args, **kwargs):
        super(PackedSequenceTest, self).__init__(*args, **kwargs)
        self.batch_size = 5
        self.max_length = 6

    def test_type_casts(self):
        """Test type casting of `PackedSequence` against type casting of tensor"""
        for _, (input_type, _) in self._type_by_name.items():
            for expected_type_str, (_, cast_str) in self._type_by_name.items():
                for enforce_sorted in [True, False]:
                    padded, lengths = self._padded_sequence(input_type)
                    packed = rnn_utils.pack_padded_sequence(
                        padded, lengths, enforce_sorted=enforce_sorted)
                    # Apply cast to `PackedSequence` instance and unpack
                    masked = getattr(packed, cast_str)()
                    unpacked, lengths_out = rnn_utils.pad_packed_sequence(masked)
                    self.assertEqual(unpacked.type(), expected_type_str)

    def test_wrong_order(self):
        a = torch.ones(25, 300)
        b = torch.ones(22, 300)
        b_a = rnn_utils.pad_sequence([b, a])
        self.assertRaises(
            RuntimeError,
            lambda: rnn_utils.pack_padded_sequence(b_a, [22, 25], enforce_sorted=True))

    def test_total_length(self):
        padded, lengths = self._padded_sequence(torch.FloatTensor)
        max_length = max(lengths)
        packed = rnn_utils.pack_padded_sequence(padded, lengths)
        # test ValueError if total_length < max_length
        for total_length in (-1, 0, max_length - 1):
            for batch_first in (True, False):
                def err_fn():
                    rnn_utils.pad_packed_sequence(packed, batch_first=batch_first,
                                                  total_length=total_length)
            self.assertRaisesRegex(ValueError,
                                   r'Expected total_length to be at least the '
                                   r'length of the longest sequence in input',
                                   err_fn)
        # test that pad_packed_sequence returns results of correct length
        for batch_first in (True, False):
            no_extra_pad, _ = rnn_utils.pad_packed_sequence(packed, batch_first=batch_first)
            for total_length_delta in (0, 1, 8):
                total_length = max_length + total_length_delta
                unpacked, lengths_out = rnn_utils.pad_packed_sequence(packed, batch_first=batch_first,
                                                                      total_length=total_length)
                self.assertEqual(lengths, lengths_out)
                self.assertEqual(unpacked.size(1 if batch_first else 0), total_length)
                if total_length_delta == 0:
                    ref_output = no_extra_pad
                elif batch_first:
                    extra_pad = no_extra_pad.new_zeros(self.batch_size, total_length_delta)
                    ref_output = torch.cat([no_extra_pad, extra_pad], 1)
                else:
                    extra_pad = no_extra_pad.new_zeros(total_length_delta, self.batch_size)
                    ref_output = torch.cat([no_extra_pad, extra_pad], 0)
                self.assertEqual(unpacked, ref_output)

    def test_to(self):
        for enforce_sorted in (True, False):
            padded, lengths = self._padded_sequence(torch.IntTensor)
            a = rnn_utils.pack_padded_sequence(
                padded, lengths, enforce_sorted=enforce_sorted).cpu()

            self.assertIs(a, a.to('cpu'))
            self.assertIs(a, a.cpu())
            self.assertIs(a, a.to('cpu', dtype=torch.int32))
            self.assertEqual(a.long(), a.to(torch.int64))

            if torch.cuda.is_available():
                for cuda in ['cuda', 'cuda:0' if torch.cuda.device_count() == 1 else 'cuda:1']:
                    b = a.cuda(device=cuda)
                    self.assertIs(b, b.to(cuda))
                    self.assertIs(b, b.cuda())
                    self.assertEqual(a, b.to('cpu'))
                    self.assertEqual(b, a.to(cuda))
                    self.assertEqual(a, b.to('cpu', dtype=torch.int32))
                    self.assertIs(b, b.to(dtype=torch.int32))
                    self.assertEqual(b.long(), b.to(dtype=torch.int64))

    def test_to_memory_format(self):
        m = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, bias=True)
        m = m.to(memory_format=torch.channels_last)
        for param in m.parameters():
            if param.dim() == 4:
                self.assertTrue(param.is_contiguous(memory_format=torch.channels_last))
*/

/*
def test_pack_sequence(self):
    def _compatibility_test(sequences, lengths, batch_first, enforce_sorted=False):
        padded = rnn_utils.pad_sequence(sequences, batch_first)
        packed = rnn_utils.pack_sequence(sequences, enforce_sorted)
        unpacked = rnn_utils.pad_packed_sequence(packed, batch_first)
        self.assertEqual(padded, unpacked[0])
        pack_padded = rnn_utils.pack_padded_sequence(
            padded, lengths, batch_first, enforce_sorted)
        self.assertEqual(packed, pack_padded)

    # single dimensional
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5])
    c = torch.tensor([6])
    packed = rnn_utils.pack_sequence([a, b, c], enforce_sorted=False)
    expected = torch.tensor([1, 4, 6, 2, 5, 3])
    self.assertEqual(packed.batch_sizes, [3, 2, 1])
    self.assertEqual(packed.data.data, expected)
    self.assertEqual(packed.sorted_indices, [0, 1, 2])
    self.assertEqual(packed.unsorted_indices, [0, 1, 2])

    packed_unsorted = rnn_utils.pack_sequence([b, c, a], enforce_sorted=False)
    self.assertEqual(packed_unsorted.batch_sizes, [3, 2, 1])
    self.assertEqual(packed_unsorted.data.data, expected)
    self.assertEqual(packed_unsorted.sorted_indices, [2, 0, 1])
    self.assertEqual(packed_unsorted.unsorted_indices, [1, 2, 0])

    # single dimensional, enforce_sorted = True
    packed_enforce_sorted = rnn_utils.pack_sequence([a, b, c], enforce_sorted=True)
    self.assertEqual(packed_enforce_sorted.batch_sizes, [3, 2, 1])
    self.assertEqual(packed_enforce_sorted.data.data, expected)
    self.assertTrue(packed_enforce_sorted.sorted_indices is None)
    self.assertTrue(packed_enforce_sorted.unsorted_indices is None)

    with self.assertRaisesRegex(RuntimeError, 'must be sorted in decreasing order'):
        rnn_utils.pack_sequence([b, c, a], enforce_sorted=True)

    with self.assertRaisesRegex(RuntimeError, 'You can pass `enforce_sorted=False`'):
        rnn_utils.pack_sequence([b, c, a], enforce_sorted=True)

    # more dimensions
    maxlen = 9
    for num_dim in (0, 1, 2, 3):
        sequences = []
        lengths = []
        trailing_dims = [4] * num_dim
        for i in range(maxlen, 0, -1):
            seq_len = i * i
            lengths.append(seq_len)
            sequences.append(torch.rand(seq_len, 5, *trailing_dims))
        unsorted_sequences = [s.clone() for s in sequences]
        random.shuffle(unsorted_sequences)
        unsorted_sequences_lengths = [t.size(0) for t in unsorted_sequences]

        # compatibility with other utilities
        for batch_first in (True, False):
            for enforce_sorted in (True, False):
                _compatibility_test(sequences, lengths, batch_first, enforce_sorted)
            _compatibility_test(unsorted_sequences, unsorted_sequences_lengths,
                                batch_first)

def test_pack_padded_sequence(self):
    def generate_test_case(sorted_lengths, should_shuffle):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        max_length = sorted_lengths[0]
        batch_sizes = [sum(map(bool, filter(lambda x: x >= i, sorted_lengths)))
                       for i in range(1, max_length + 1)]
        offset = 0
        padded = torch.cat([pad(i * 100 + torch.arange(1., 5 * l + 1).view(l, 1, 5), max_length)
                            for i, l in enumerate(sorted_lengths, 1)], 1)
        expected_data = [[torch.arange(1., 6) + (i + 1) * 100 + 5 * n for i in range(batch_size)]
                         for n, batch_size in enumerate(batch_sizes)]
        expected_data = list(itertools.chain.from_iterable(expected_data))
        expected_data = torch.stack(expected_data, dim=0)

        if should_shuffle:
            # Shuffle the padded sequence to create an unsorted sequence
            permutation = list(range(len(sorted_lengths)))
            random.shuffle(permutation)

            unsorted_indices = torch.tensor(permutation)
            padded = padded.index_select(1, unsorted_indices)
            lengths = torch.tensor(sorted_lengths).index_select(0, unsorted_indices)
        else:
            unsorted_indices = None
            lengths = sorted_lengths

        return padded.requires_grad_(), lengths, expected_data, batch_sizes, unsorted_indices

    test_cases = [
        # sorted_lengths, should_shuffle
        [[10, 8, 4, 2, 2, 2, 1], False],
        [[11, 10, 8, 6, 4, 3, 1], False],
        [[11, 10, 8, 6, 4, 3, 1], True],
    ]

    for test_case, batch_first in itertools.product(test_cases, (True, False)):
        sorted_lengths, should_shuffle = test_case
        padded, lengths, expected_data, batch_sizes, unsorted_indices = generate_test_case(
            sorted_lengths, should_shuffle)

        src = padded
        if batch_first:
            src = src.transpose(0, 1)

        # check output
        packed = rnn_utils.pack_padded_sequence(src, lengths, batch_first=batch_first,
                                                enforce_sorted=not should_shuffle)
        self.assertEqual(packed.data.data, expected_data)
        self.assertEqual(packed.batch_sizes, batch_sizes)
        self.assertEqual(packed.unsorted_indices, unsorted_indices)

        # test inverse
        unpacked, unpacked_len = rnn_utils.pad_packed_sequence(packed, batch_first=batch_first)
        self.assertEqual(unpacked, src)
        self.assertEqual(unpacked_len, lengths)

        # check grad
        if padded.grad is not None:
            padded.grad.data.zero_()
        grad_output = unpacked.data.clone().normal_()
        unpacked.backward(grad_output)
        if batch_first:
            grad_output.transpose_(0, 1)
        for i, l in enumerate(lengths):
            self.assertEqual(padded.grad.data[:l, i], grad_output[:l, i])
            if l < 10:
                self.assertEqual(padded.grad.data[l:, i].abs().sum(), 0)

    # test error messages
    with self.assertRaisesRegex(RuntimeError, 'You can pass `enforce_sorted=False`'):
        packed = rnn_utils.pack_padded_sequence(torch.randn(3, 3), [1, 3, 2])
    with self.assertRaisesRegex(RuntimeError, 'empty tensor'):
        packed = rnn_utils.pack_padded_sequence(torch.randn(0, 0), [])
*/
