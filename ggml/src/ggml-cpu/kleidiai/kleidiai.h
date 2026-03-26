// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//
#include "traits.h"
#pragma once

#include "ggml-alloc.h"

#ifdef  __cplusplus
extern "C" {
#endif

ggml_backend_buffer_type_t ggml_backend_cpu_kleidiai_buffer_type(void);

#ifdef  __cplusplus
}
#endif
namespace ggml::cpu::kleidiai {

class tensor_traits: public ggml::cpu::tensor_traits
{

public:
bool work_size(int /* n_threads */, const struct ggml_tensor * op, size_t & size) override;
bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * dst) override;
bool compute_forward_qx(struct ggml_compute_params * params, struct ggml_tensor * dst);
bool compute_forward_fp16(ggml_compute_params * params, struct ggml_tensor * dst);
bool compute_forward_f32(ggml_compute_params * params, struct ggml_tensor * dst);
bool compute_forward_get_rows(struct ggml_compute_params * params, struct ggml_tensor * dst);
int repack(struct ggml_tensor * tensor, const void * data, size_t data_size) ;
};

class extra_buffer_type: ggml::cpu::extra_buffer_type
{public:
bool supports_op(ggml_backend_dev_t, const struct ggml_tensor * op) override;
ggml::cpu::tensor_traits * get_tensor_traits(const struct ggml_tensor * op) override;
};
}// namespace ggml::cpu::kleidiai

