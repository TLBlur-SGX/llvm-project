#include "bolt/Passes/TLBlur.h"
#include "bolt/Core/BinaryBasicBlock.h"
#include "bolt/Core/BinaryContext.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Core/FunctionLayout.h"
#include "bolt/Core/ParallelUtilities.h"
#include "bolt/Utils/CommandLineOpts.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"

#define DEBUG_TYPE "bolt-tlblur"
#define INSTRUMENTATION_LENGTH 19

namespace opts {
cl::opt<uint64_t> TLBlurPageMask(
    "tlblur-page-mask",
    cl::desc("mask to use when determining the page of an address"), cl::Hidden,
    cl::cat(BoltCategory), cl::init(0xfff));

cl::opt<bool> TLBlurInstrumentJumpTargets(
    "tlblur-instrument-jump-targets",
    cl::desc("instrument jump targets instead of jumps"), cl::Hidden,
    cl::cat(BoltCategory), cl::init(false));

cl::opt<bool>
    TLBlurSaveEflags("tlblur-store-eflags",
                     cl::desc("store eflags before calling instrumentation"),
                     cl::Hidden, cl::cat(BoltCategory), cl::init(true));
} // namespace opts

using namespace llvm;

namespace opts {} // end namespace opts

namespace llvm {
namespace bolt {

/// Gets the page number of a given address.
///
/// An option can be set to configure the page size, which can be used
/// to simulate larger code pages.
static inline uint64_t pageNum(uint64_t Addr) {
  return Addr & ~opts::TLBlurPageMask;
}

/// Finds the instruction at the given address
std::optional<InstructionLookupResult>
TLBlurPass::findInstructionAtAddress(BinaryContext &BC, uint64_t Addr) {
  BinaryFunction *LastBF = nullptr;
  auto Funcs = BC.getAllBinaryFunctions();
  for (size_t I = 0; I < Funcs.size(); I++) {
    auto *BF = Funcs[I];
    if (Addr < FunctionAddresses[BF]) {
      if (!LastBF)
        return std::nullopt;

      // Addr is inside LastBF
      BinaryFunction::iterator BFIt = LastBF->begin();
      auto FAddr = FunctionAddresses[LastBF];
      while (BFIt != LastBF->end()) {
        auto &BB = *BFIt;
        auto Range = BlockAddresses[&BB];
        auto Start = FAddr + Range.first;
        auto End = FAddr + Range.second;
        // LLVM_DEBUG(errs() << llvm::format_hex(Addr, 16) << " in [" <<
        // llvm::format_hex(Start, 16) << " : " << llvm::format_hex(End, 16) <<
        // "]?\n");
        if (Start <= Addr && Addr < End) {
          // Addr is inside BB
          uint64_t Offset = 0;
          BinaryBasicBlock::iterator It = BB.begin();
          while (It != BB.end()) {
            uint64_t Size = BC.computeInstructionSize(*It);
            uint64_t InstAddr = Start + Offset;
            Offset += Size;
            if (Start + Offset > Addr) {
              InstructionLookupResult Res;
              Res.Addr = InstAddr;
              Res.Size = Size;
              Res.BFIt = BFIt;
              Res.BBIt = It;
              if (I + 1 < Funcs.size())
                Res.NextFn = Funcs[I + 1];
              return Res;
            }
            ++It;
          }
        }
        ++BFIt;
      }
    }
    LastBF = BF;
  }

  return std::nullopt;
}

void TLBlurPass::invalidateSizeCache(BinaryBasicBlock *BB) {
  BBSizeCache[BB] = 0;
  BFSizeCache[BB->getParent()] = 0;
}

/// Recomputes the addresses of all basic blocks
///
/// Uses caches for sizes of basic blocks and functions that must be explicitly
/// invalidated using `invalidateSizeCache` before this method should be called.
void TLBlurPass::updateOutputAddresses(BinaryContext &BC) {
  uint64_t Offset = 0x400000; // TODO: Don't hardcode this
  std::vector<BinaryFunction *> Functions = BC.getAllBinaryFunctions();

  for (BinaryFunction *BF : Functions) {
    uint64_t Start = Offset;
    FunctionAddresses[BF] = Offset;

    uint64_t Size = BFSizeCache.getOrInsertDefault(BF);
    if (Size == 0) {
      uint64_t InnerOffset = 0;

      for (BinaryBasicBlock &BB : BF->blocks()) {
        BB.setAlignment(4);
        std::pair<uint64_t, uint64_t> Range;
        Range.first = InnerOffset;
        uint64_t Size = BBSizeCache.getOrInsertDefault(&BB);
        if (Size == 0)
          Size = BC.computeCodeSize(BB.begin(), BB.end());
        BBSizeCache[&BB] = Size;
        InnerOffset += Size;
        Range.second = InnerOffset;
        BlockAddresses[&BB] = Range;
      }

      std::pair<uint64_t, uint64_t> P = BC.calculateEmittedSize(*BF, false);
      Size = P.first + P.second;
    }
    BFSizeCache[BF] = Size;
    Offset = Start + Size;

    uint64_t Alignment = BF->getAlignment();
    uint64_t Remainder = Offset % Alignment;
    Offset += (Remainder == 0) ? 0 : (Alignment - Remainder);
  }

  Size = Offset;
}

bool TLBlurPass::shouldInstrument(BinaryFunction &BF) {
  return shouldInstrument(BF.getDemangledName(), BF.getOriginSectionName());
}

bool TLBlurPass::shouldInstrument(StringRef Name, std::optional<StringRef> SectionName) {
  // HACK
  if (Name == "sgx_spin_lock" || Name == "salsa20" || Name == "rsa_ossl_mod_exp") 
    return false;
      
  return SectionName == ".tlblur.text" &&
         Name != "tlblur_tlb_update";
}

/// Finds the first terminator of a block
static BinaryBasicBlock::iterator getFirstTerminator(BinaryBasicBlock &BB) {
  BinaryContext &BC = BB.getFunction()->getBinaryContext();
  auto Itr = BB.end();
  if (BB.empty())
    return Itr;
  Itr--;
  while (Itr != BB.begin()) {
    // BUG: Not all blocks end with a terminator, could also be CFI
    if (!BC.MIB->isTerminator(*Itr) && !BC.MIB->isPseudo(*Itr) &&
        !BC.MIB->isCFI(*Itr))
      return ++Itr;
    Itr--;
  }
  return Itr;
}

/// Instruments jumps that cross a page boundary. Can be called multiple times
/// to add additional instrumentation after the binary layout has changed.
bool TLBlurPass::instrumentJumps(BinaryContext &BC) {
  LLVM_DEBUG(errs() << "Instrumenting jumps...\n");
  bool Changed = false;
  std::vector<BinaryFunction *> Functions = BC.getAllBinaryFunctions();

  for (BinaryFunction *BF : Functions) {
    if (!shouldInstrument(*BF))
      continue;

    auto FAddr = FunctionAddresses[BF];

    for (BinaryBasicBlock &BB : BF->blocks()) {
      uint64_t Page = pageNum(FAddr + BlockAddresses[&BB].second);
      bool ChangedBlock = false;

      // Direct calls
      BinaryBasicBlock::iterator It = BB.begin();
      while (It != BB.end()) {
        // Only direct calls, not indirect calls
        if (BC.MIB->isCall(*It) && !BC.MIB->isIndirectCall(*It)) {
          const MCSymbol *Target = BC.MIB->getTargetSymbol(*It);
          uint64_t TargetPage = pageNum(Target->getOffset());

          BinaryFunction *BF = BC.getFunctionForSymbol(Target);

          // Only instrument when control flow crosses a page boundary
          if (BF && shouldInstrument(*BF) && TargetPage != Page) {
            LLVM_DEBUG(errs() << "Instrumenting call to " << *Target << "\n");
            // Either instrument each jump/call, or each basic block
            if (!opts::TLBlurInstrumentJumpTargets || !BF ||
                !BF->hasInstructions()) {
              // Use annotations to keep track of which instructions
              // have already been instrumented.
              if (!BC.MIB->hasAnnotation(*It, "instrumented")) {
                Changed = true;
                ChangedBlock = true;
                BC.MIB->addAnnotation(*It, "instrumented", true);
                It = insertInstrumentation(BC, BB, Target, It,
                                           opts::TLBlurSaveEflags);

                if (!BC.MIB->isTailCall(*It)) {
                  BC.MIB->addAnnotation(*It, "instrumented-ret", true);
                  ++It;
                  It = insertInstrumentation(BC, BB, BB.getLabel(), It,
                                             opts::TLBlurSaveEflags);
                }
                // Invalidate the size cache of the basic block.
                // If the size of a block changes, the entire code layout
                // changes.
                invalidateSizeCache(&BB);
              }
            } else {
              BinaryBasicBlock &Succ = *BF->begin();

              // Keep track of which blocks have already been instrumented to
              // ensure we instrument each block only once.
              if (!InstrumentedBlocks.contains(&Succ)) {
                Changed = true;
                ChangedBlock = true;
                InstrumentedBlocks.insert(&Succ);
                insertInstrumentation(BC, Succ, Target, Succ.begin(),
                                      opts::TLBlurSaveEflags);
                // Invalidate the size cache of the basic block.
                // If the size of a block changes, the entire code layout
                // changes.
                invalidateSizeCache(&Succ);
              }
            }
          }
        }
        if (It != BB.end())
          ++It;
      }

      // Direct jumps
      It = getFirstTerminator(BB);
      for (BinaryBasicBlock *Succ : BB.successors()) {
        if (!shouldInstrument(*Succ->getParent()))
          continue;
        auto SuccFAddr = FunctionAddresses[Succ->getParent()];
        if (InstrumentedJumps[&BB].contains(Succ) ||
            InstrumentedBlocks.contains(Succ))
          continue;
        uint64_t SuccPage = pageNum(SuccFAddr + BlockAddresses[Succ].second);
        // Only instrument if the jump target is on a different page
        if (SuccPage != Page) {
          LLVM_DEBUG(errs() << "Instrumenting jump at "
                            << llvm::format_hex(BlockAddresses[&BB].second, 16)
                            << "\n");
          Changed = true;
          ChangedBlock = true;

          // Either instrument at the start or at the end of each basic block
          if (!opts::TLBlurInstrumentJumpTargets) {
            // When instrumenting at the end, there can be instrumentation
            // for each successor of the basic block
            InstrumentedJumps[&BB].insert(Succ);
            It = insertInstrumentation(BC, BB, Succ->getLabel(), It,
                                       opts::TLBlurSaveEflags);
            // Invalidate the size cache of the basic block.
            // If the size of a block changes, the entire code layout changes.
            invalidateSizeCache(&BB);
          } else {
            // When instrumenting at the start, we only need to instrument the
            // target block once, regardless of the predecessor.
            InstrumentedBlocks.insert(Succ);
            insertInstrumentation(BC, *Succ, Succ->getLabel(), Succ->begin(),
                                  opts::TLBlurSaveEflags);
            // Invalidate the size cache of the basic block.
            // If the size of a block changes, the entire code layout changes.
            invalidateSizeCache(Succ);
          }
        }
      }

      // If we changed a block, immediately update the output addresses
      if (ChangedBlock)
        updateOutputAddresses(BC);
    }
  }

  return Changed;
}

/// Instrument basic blocks that straddle accross two pages
void TLBlurPass::instrumentFallthroughs(BinaryContext &BC) {
  // For each page boundary
  for (uint64_t Addr = 0x0; Addr < Size; Addr += (opts::TLBlurPageMask + 1)) {
    // Find the instruction at the boundary
    auto Res = findInstructionAtAddress(BC, Addr - INSTRUMENTATION_LENGTH);

    if (Res.has_value()) {
      auto Info = Res.value();
      BinaryFunction::iterator BB = Info.BFIt;

      if (!shouldInstrument(*BB->getParent()))
        continue;

      LLVM_DEBUG(errs() << "Instrumenting boundary at "
                        << llvm::format_hex(Info.Addr, 16) << "\n");

      // Fix: make sure we don't insert instrumentation in between terminators.
      //
      // Note that skipping instrumentation in this case would not be secure,
      // since it's possible that one of the terminators is cross-page, in which
      // case we do need to log the access to the next page, even when none of
      // the terminators jump to that page. Hence, we insert fallthrough
      // instrumentation anyway.
      BinaryBasicBlock::iterator It = Info.BBIt;
      while (It != BB->begin() && BC.MIB->isTerminator(*std::prev(It)))
        It--;

      // Create a symbol somewhere on the next page
      MCSymbol *Target = BC.getOrCreateGlobalSymbol(Info.Addr + 100, "page");

      // Insert instrumentation
      insertInstrumentation(BC, *BB, Target, It, true);

      invalidateSizeCache(&*BB);
      updateOutputAddresses(BC);
    } else {
      LLVM_DEBUG(errs() << "No instruction found at boundary "
                        << llvm::format_hex(Addr, 16) << "\n");
    }
  }
}

static BinaryBasicBlock::iterator
insertInstructions(InstructionListType &Instrs, BinaryBasicBlock &BB,
                   BinaryBasicBlock::iterator Iter) {
  for (MCInst &NewInst : Instrs) {
    Iter = BB.insertInstruction(Iter, NewInst);
    ++Iter;
  }
  return Iter;
}

BinaryBasicBlock::iterator TLBlurPass::insertInstrumentation(
    BinaryContext &BC, BinaryBasicBlock &BB, const MCSymbol *Target,
    BinaryBasicBlock::iterator It, bool SaveEflags) {
  InstructionListType Instrs = BC.MIB->createTLBlurInstrumentationCall(
      Target, BC.Ctx->getOrCreateSymbol("tlblur_tlb_update"), BC.Ctx.get(),
      SaveEflags);
  return insertInstructions(Instrs, BB, It);
}

void TLBlurPass::runOnFunctions(BinaryContext &BC) {
  BBSizeCache.clear();
  BFSizeCache.clear();
  InstrumentedJumps.clear();
  InstrumentedBlocks.clear();
  FunctionAddresses.clear();
  BlockAddresses.clear();
  Size = 0;

  for (auto &It : BC.getBinaryFunctions()) {
    BinaryFunction &Function = It.second;
    if (!BC.shouldEmit(Function) || !Function.isSimple())
      continue;

    Function.fixBranches();
  }
  updateOutputAddresses(BC);

  // While there are some cross-page jumps left uninstrumented, instrument them
  // This converges to a fixpoint, but we may end up with redundant
  // instrumentation
  while (instrumentJumps(BC))
    ;

  // We have decided which jumps need to be instrumented, now insert
  // instrumentation at page boundaries
  instrumentFallthroughs(BC);
}

} // end namespace bolt
} // end namespace llvm
