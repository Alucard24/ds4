/* Explicit instantiations for the ported MMQ kernels.
 *
 * Upstream keeps these in mmq.cu: the header declares them `extern template`
 * so that the heavy kernel tree is instantiated once, in its own translation
 * unit.  Only the types ds4 actually executes are listed; adding a type costs
 * compile time and nothing else.
 */
#include "ds4_ggml_stubs.h"
#include "new/mmq.cuh"

ds4n_DECL_MMQ_CASE(GGML_TYPE_Q2_K);
ds4n_DECL_MMQ_CASE(GGML_TYPE_Q4_K);
ds4n_DECL_MMQ_CASE(GGML_TYPE_IQ2_XXS);
ds4n_DECL_MMQ_CASE(GGML_TYPE_IQ2_XS);
ds4n_DECL_MMQ_CASE(GGML_TYPE_IQ2_S);
ds4n_DECL_MMQ_CASE(GGML_TYPE_IQ3_XXS);
ds4n_DECL_MMQ_CASE(GGML_TYPE_IQ3_S);
ds4n_DECL_MMQ_CASE(GGML_TYPE_IQ4_XS);
