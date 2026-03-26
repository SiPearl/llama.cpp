#include "ggml-backend.h"
#include "ggml-cpu-impl.h"
#include "traits.h"

// GGML internal header

#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
ggml_backend_buffer_type_t ggml_backend_amx_buffer_type(void);
#endif

namespace ggml::cpu::amx {
class tensor_traits : public ggml::cpu::tensor_traits
{public:
 bool work_size(int /* n_threads */, const struct ggml_tensor * op, size_t & size) override;
bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) override;
};

class extra_buffer_type: ggml::cpu::extra_buffer_type
{public:
bool supports_op(ggml_backend_dev_t, const struct ggml_tensor * op) override;
ggml::cpu::tensor_traits * get_tensor_traits(const struct ggml_tensor * op) override;
};
}  // namespace ggml::cpu::amx


