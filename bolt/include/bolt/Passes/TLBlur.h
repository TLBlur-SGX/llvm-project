#ifndef BOLT_PASSES_TLBLUR_H
#define BOLT_PASSES_TLBLUR_H

#include "bolt/Core/BinaryBasicBlock.h"
#include "bolt/Core/BinaryContext.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Core/BinarySection.h"
#include "bolt/Core/FunctionLayout.h"
#include "bolt/Passes/BinaryPasses.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/MC/MCSymbol.h"

namespace llvm {
namespace bolt {

struct InstructionLookupResult {
  uint64_t Addr;
  uint64_t Size;
  BinaryFunction::iterator BFIt;
  BinaryBasicBlock::iterator BBIt;
  BinaryFunction *NextFn;
};

class TLBlurPass : public BinaryFunctionPass {
private:
  // Size caches to speed up recomputation of memory addresses
  DenseMap<BinaryBasicBlock *, uint64_t> BBSizeCache;
  DenseMap<BinaryFunction *, uint64_t> BFSizeCache;

  // Jumps and blocks that are already instrumented, to avoid duplication of instrumentation.
  DenseMap<BinaryBasicBlock *, SmallPtrSet<BinaryBasicBlock *, 4>> InstrumentedJumps;
  SmallPtrSet<BinaryBasicBlock *, 16> InstrumentedBlocks;

  // Addresses of functions and blocks, used to determine if control flow crosses a page boundary.
  DenseMap<BinaryFunction *, uint64_t> FunctionAddresses;
  DenseMap<BinaryBasicBlock *, std::pair<uint64_t, uint64_t>> BlockAddresses;
      
  // Total code size
  uint64_t Size = 0;

  void updateOutputAddresses(BinaryContext &BC);

  std::optional<InstructionLookupResult>
  findInstructionAtAddress(BinaryContext &BC, uint64_t Addr);

  void invalidateSizeCache(BinaryBasicBlock *BB);

public:
  explicit TLBlurPass() : BinaryFunctionPass(false) {}

  const char *getName() const override { return "TLBlur"; }

  bool shouldInstrument(BinaryFunction &BF);
  bool shouldInstrument(StringRef Name, std::optional<StringRef> SectionName);

  /// Pass entry point
  void runOnFunctions(BinaryContext &BC) override;

  void instrumentFallthroughs(BinaryContext &BC);
  bool instrumentJumps(BinaryContext &BC);
  BinaryBasicBlock::iterator insertInstrumentation(BinaryContext &BC, BinaryBasicBlock &BB,
                             const MCSymbol *Target,
                             BinaryBasicBlock::iterator It, bool SaveEflags);
};

} // namespace bolt
} // namespace llvm

#endif
