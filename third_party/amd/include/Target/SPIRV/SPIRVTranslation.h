#include <string>

namespace llvm {
class Module;
} // namespace llvm

namespace amd {
// roughly same function proto as intel's
std::string translate_llir_to_spirv(llvm::Module &module);
}
