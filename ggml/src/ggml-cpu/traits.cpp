#include "traits.h"

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "amx/amx.h"
#include "kleidiai/kleidiai.h"
#include "repack.h"
namespace ggml::cpu {
tensor_traits::~tensor_traits() {}

extra_buffer_type::~extra_buffer_type() {}
}  // namespace ggml::cpu
bool ggml_cpu_extra_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op)
{
    int count=0;
	for (auto extra : ggml_backend_cpu_get_extra_buffer_types()) 
	{
if (extra && extra->context) {
            #if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
	    if (strcmp(extra->iface.get_name(extra), "AMX")!=0 && strcmp(extra->iface.get_name(extra), "CPU_AARCH64")!=0 && strcmp(extra->iface.get_name(extra), "CPU")!=0)
			{ 
	      auto buf_extra_ = (ggml::cpu::amx::extra_buffer_type *)ggml_backend_amx_buffer_type()->context;
	      auto tensor_traits =  buf_extra_->get_tensor_traits(op); //#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
		      if (tensor_traits && tensor_traits->compute_forward(params, op)) {
			      return true;
            }
	    }
	    #endif 
	    auto buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            if (tensor_traits && tensor_traits->compute_forward(params, op)) {
                return true;
            }
        
		} count++;
		if (extra)
		{
			if  (strcmp(extra->iface.get_name(extra), "CPU_HBM")==0 ) 
			{
			#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)      	
 			auto buf_extra_ = (ggml::cpu::amx::extra_buffer_type *)ggml_backend_amx_buffer_type()->context;
                       #elif GGML_USE_CPU_KLEIDIAI
		       auto buf_extra_ =(ggml::cpu::kleidiai::extra_buffer_type *)ggml_backend_cpu_kleidiai_buffer_type()->context;	
                       #elif GGML_USE_CPU_REPACK
                       auto buf_extra_ =(ggml::cpu::repack::extra_buffer_type *) ggml_backend_cpu_repack_buffer_type() ; 
		       #endif
			auto tensor_traits =  buf_extra_->get_tensor_traits(op); 
		       if (tensor_traits && tensor_traits->compute_forward(params, op)) 
			        {
				return true;
			        }
			}
		}
    }
    	return false;
}		 






bool ggml_cpu_extra_work_size(int n_threads, const struct ggml_tensor * op, size_t * size) {
    for (auto extra : ggml_backend_cpu_get_extra_buffer_types()) {
        if (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            if (tensor_traits && tensor_traits->work_size(n_threads, op, *size)) {
                return true;
            }
        }
    }
    return false;
}
