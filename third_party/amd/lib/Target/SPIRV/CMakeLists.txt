# SPIRV-LLVM-Translator is required.
find_package(SPIRVToLLVMTranslator)

# Check if there is the LLVM SPIR-V backend.
is_llvm_target_library("SPIRV" spirv_present_result INCLUDED_TARGETS)

add_compile_definitions(LLVM_SPIRV_BACKEND_TARGET_PRESENT)
add_mlir_translation_library(AMDSPIRV
  SPIRVTranslation.cpp

  LINK_COMPONENTS
  Core
  # spirv backend
  #SPIRVCodeGen

  LINK_LIBS PUBLIC
  TritonLLVMIR
  # spirv tools
  LLVMSPIRVLib
)

target_include_directories(AMDSPIRV PRIVATE ${SPIRVToLLVMTranslator_INCLUDE_DIR})
set_property(GLOBAL APPEND PROPERTY TRITON_LIBS AMDSPIRV)
llvm_update_compile_flags(AMDSPIRV)
