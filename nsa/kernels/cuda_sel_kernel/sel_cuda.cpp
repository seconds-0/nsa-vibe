// Selection Attention forward (CUDA via ATen ops)
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>

using at::Tensor;

static inline Tensor build_index_from_ranges(const Tensor& ranges_row, int64_t S_kv) {
  // ranges_row: [n,2] int32 on CUDA
  auto n = ranges_row.size(0);
  std::vector<Tensor> pieces;
  pieces.reserve(n);
  for (int64_t i = 0; i < n; ++i) {
    int64_t s = ranges_row[i][0].item<int32_t>();
    int64_t e = ranges_row[i][1].item<int32_t>();
    s = std::max<int64_t>(0, std::min<int64_t>(s, S_kv));
    e = std::max<int64_t>(s, std::min<int64_t>(e, S_kv));
    if (e > s) {
      pieces.push_back(at::arange(s, e, ranges_row.options().dtype(at::kLong)));
    }
  }
  if (pieces.empty()) {
    return at::empty({0}, ranges_row.options().dtype(at::kLong));
  }
  return at::cat(pieces);
}

at::Tensor sel_forward(
    const Tensor& Q,      // [B,S,G,h,Dk]
    const Tensor& K,      // [B,G,S_kv,Dk]
    const Tensor& V,      // [B,G,S_kv,Dv]
    const Tensor& ranges  // [B,S,G,n,2] int32
) {
  TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda() && ranges.is_cuda(), "all tensors must be CUDA");
  TORCH_CHECK(Q.scalar_type() == at::kHalf || Q.scalar_type() == at::kBFloat16,
              "Q must be fp16 or bf16");
  TORCH_CHECK(K.scalar_type() == Q.scalar_type() && V.scalar_type() == Q.scalar_type(), "dtype mismatch");

  auto B = Q.size(0);
  auto S = Q.size(1);
  auto G = Q.size(2);
  auto h = Q.size(3);
  auto Dk = Q.size(4);
  auto S_kv = K.size(2);
  auto Dv = V.size(3);

  auto out = at::empty({B, S, G, h, Dv}, V.options());
  const double scale = 1.0 / std::sqrt(static_cast<double>(Dk));

  for (int64_t b = 0; b < B; ++b) {
    for (int64_t s = 0; s < S; ++s) {
      for (int64_t g = 0; g < G; ++g) {
        Tensor ranges_row = ranges[b][s][g]; // [n,2]
        Tensor idx = build_index_from_ranges(ranges_row, S_kv);
        if (idx.numel() == 0) {
          out[b][s][g].zero_();
          continue;
        }
        Tensor Kbg = K[b][g].index_select(0, idx);   // [L,Dk]
        Tensor Vbg = V[b][g].index_select(0, idx);   // [L,Dv]
        Tensor Qbsg = Q[b][s][g];                    // [h,Dk]

        // scores: [h,L] = Q * K^T
        // TODO: Consider batching across (b,s,g) and heads to reduce kernel launches
        Tensor scores = at::matmul(Qbsg.to(at::kFloat), Kbg.to(at::kFloat).transpose(0,1)) * scale; // [h,L]
        Tensor probs = at::softmax(scores, /*dim=*/-1); // [h,L]
        Tensor O = at::matmul(probs, Vbg.to(at::kFloat)); // [h,Dv]
        out[b][s][g].copy_(O.to(V.dtype()));
      }
    }
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sel_forward", &sel_forward, "Selection attention forward (CUDA via ATen)");
}
