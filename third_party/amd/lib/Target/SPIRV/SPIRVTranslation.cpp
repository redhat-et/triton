#include <stdio.h>
#include "amd/include/Target/SPIRV/SPIRVTranslation.h"

#include "LLVMSPIRVLib.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Pass.h"


// NOTE(mrodden): could never get this to link properly,
// the SPIRVTranslate symbol can't be found/linked by ldd, but
// its also unknown if that is the correct path for the future as well
//#if defined(LLVM_SPIRV_BACKEND_TARGET_PRESENT)

//// most of this is taken verbatum from LLIR-SPRIV-Translators sample
//
//namespace llvm {
//
//using namespace llvm;
//using namespace SPIRV;
//
//// LLVM-SPIRV backend provides this function as their API, but it is
//// not marked exported as its a newer API than the SPIRVTranslateModule function
//extern "C" bool SPIRVTranslate(Module *M, std::string &SpirvObj,
//                               std::string &ErrMsg,
//                               const std::vector<std::string> &AllowExtNames,
//                               llvm::CodeGenOptLevel OLevel,
//                               Triple TargetTriple);
//
//bool runSpirvBackend(Module *M, std::string &Result, std::string &ErrMsg,
//                     const SPIRV::TranslatorOpts &TranslatorOpts) {
//  static const std::string DefaultTriple = "spirv64v1.6-unknown-unknown";
//  static const std::vector<std::string> AllowExtNames{"all"};
//
//  // vulkan1.0 corresponds to spirv1.0
//  // vulkan1.1 corresponds to spirv1.3
//  // vulkan1.2 corresponds to spirv1.5
//  // vulkan1.3 corresponds to spirv1.6
//  Triple target("spirv64v1.5-amd-vulkan1.2");
//
//  // Translate the Module into SPIR-V
//  return SPIRVTranslate(M, Result, ErrMsg, AllowExtNames,
//                        CodeGenOptLevel::Aggressive, target);
//}
//
//bool runSpirvBackend(Module *M, std::ostream &OS, std::string &ErrMsg,
//                     const SPIRV::TranslatorOpts &TranslatorOpts) {
//  std::string Result;
//  bool Status = runSpirvBackend(M, Result, ErrMsg, TranslatorOpts);
//  if (Status)
//    OS << Result;
//  return Status;
//}
//
//} // namespace llvm
//  
//#endif // LLVM_SPIRV_BACKEND_TARGET_PRESENT



static SPIRV::TranslatorOpts getSPIRVOpts() {
  SPIRV::TranslatorOpts SPIRVOpts{SPIRV::VersionNumber::SPIRV_1_4};

  static constexpr std::array<SPIRV::ExtensionID, 19> AllowedExtensions{
      SPIRV::ExtensionID::SPV_EXT_shader_atomic_float_add,
      SPIRV::ExtensionID::SPV_INTEL_2d_block_io,
      SPIRV::ExtensionID::SPV_INTEL_arbitrary_precision_integers,
      SPIRV::ExtensionID::SPV_INTEL_arithmetic_fence,
      SPIRV::ExtensionID::SPV_INTEL_bfloat16_conversion,
      SPIRV::ExtensionID::SPV_INTEL_cache_controls,
      SPIRV::ExtensionID::SPV_INTEL_fp_fast_math_mode,
      SPIRV::ExtensionID::SPV_INTEL_inline_assembly,
      SPIRV::ExtensionID::SPV_INTEL_kernel_attributes,
      SPIRV::ExtensionID::SPV_INTEL_memory_access_aliasing,
      SPIRV::ExtensionID::SPV_INTEL_split_barrier,
      SPIRV::ExtensionID::SPV_INTEL_subgroup_matrix_multiply_accumulate,
      SPIRV::ExtensionID::SPV_INTEL_subgroups,
      SPIRV::ExtensionID::SPV_INTEL_tensor_float32_conversion,
      SPIRV::ExtensionID::SPV_INTEL_unstructured_loop_controls,
      SPIRV::ExtensionID::SPV_INTEL_vector_compute,
      SPIRV::ExtensionID::SPV_KHR_bfloat16,
      SPIRV::ExtensionID::SPV_KHR_bit_instructions,
      SPIRV::ExtensionID::SPV_KHR_non_semantic_info};

  SPIRVOpts.setMemToRegEnabled(true);
  SPIRVOpts.setPreserveOCLKernelArgTypeMetadataThroughString(true);
  SPIRVOpts.setPreserveAuxData(false);
  SPIRVOpts.setSPIRVAllowUnknownIntrinsics({"llvm.amdgcn."});

  for (auto &Ext : AllowedExtensions)
    SPIRVOpts.setAllowedToUseExtension(Ext, true);
  return SPIRVOpts;
}

class SmallVectorBuffer : public std::streambuf {
  // All memory management is delegated to llvm::SmallVectorImpl
  llvm::SmallVectorImpl<char> &OS;

  // Since we don't touch any pointer in streambuf(pbase, pptr, epptr) this is
  // the only method we need to override.
  virtual std::streamsize xsputn(const char *s, std::streamsize n) override {
    OS.append(s, s + n);
    return n;
  }

public:
  SmallVectorBuffer() = delete;
  SmallVectorBuffer(const SmallVectorBuffer &) = delete;
  SmallVectorBuffer &operator=(const SmallVectorBuffer &) = delete;
  SmallVectorBuffer(llvm::SmallVectorImpl<char> &O) : OS(O) {}
};


namespace amd {
// Takes in LLVM IR and returns spirv bitcode as a std::string
// This function is specialized to AMDGPUs currently.
std::string translate_llir_to_spirv(llvm::Module &mod) {
  llvm::SmallVector<char, 0> buffer;

  // verify and store llvm
  llvm::legacy::PassManager pm;
  pm.add(llvm::createVerifierPass());
  pm.run(mod);

  if (mod.materializeAll()) {
    llvm::errs() << "SPIRVTranslation: failed to read the LLVM module IR!";
    llvm::errs().flush();
    std::string result(buffer.begin(), buffer.end());
    return result;
  }

  // buffers
  SmallVectorBuffer StreamBuf(buffer);
  std::ostream OS(&StreamBuf);
  std::string Err;

  // spirv opts specific to AMDGPU
  SPIRV::TranslatorOpts SPIRVOpts = getSPIRVOpts();

  // run translation
  //auto success = llvm::runSpirvBackend(&mod, OS, Err, SPIRVOpts);
  auto success = llvm::writeSpirv(&mod, SPIRVOpts, OS, Err);

  if (!success) {
    llvm::errs() << "SPIRVTranslation: SPIRV translation failed with"
                 << Err.c_str();
    llvm::errs().flush();
  }

  std::string result(buffer.begin(), buffer.end());
  return result;
}

} // namespace amd

// vim: sw=2 ts=2 sts=2 et
