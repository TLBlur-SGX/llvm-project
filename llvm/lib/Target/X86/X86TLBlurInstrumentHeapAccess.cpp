#include "MCTargetDesc/X86MCTargetDesc.h"
#include "X86.h"
#include "X86CallingConv.h"
#include "X86FrameLowering.h"
#include "X86InstrBuilder.h"
#include "X86InstrInfo.h"
#include "X86MachineFunctionInfo.h"
#include "X86RegisterInfo.h"
#include "X86Subtarget.h"
#include "X86TargetMachine.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Type.h"
#include "llvm/MC/MCRegister.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "x86-tlblur"

#define DEBUG_FILTER "read_markers"

#define COUNTER_REG X86::R15

// Number of array entries of size 8
#define TLBLUR_VTLB_SIZE 0x10000

static cl::opt<bool>
    TLBlurInline("x86-tlblur-inline",
                 cl::desc("Use inlined TLBlur instrumentation"),
                 cl::init(false), cl::Hidden);

cl::opt<bool>
    TLBlurCounterRegister("x86-tlblur-counter-register",
                          cl::desc("Use dedicated register for global counter"),
                          cl::init(false), cl::Hidden);

namespace {

/// Data access and indirect call instrumentation for TLBlur
class TLBlurInstrumentHeap : public MachineFunctionPass {
  MachineRegisterInfo *MRI = nullptr;
  const X86InstrInfo *TII;
  const X86RegisterInfo *TRI;
  uint64_t Counter = 0;
  bool Debug = false;
  SmallPtrSet<MachineInstr *, 8> RDICopies;
  DenseMap<MachineBasicBlock *, Register> CounterRegs;

  std::optional<X86AddressMode> getAddressMode(MachineInstr &MI);
  Register writeAddrModeToReg(X86AddressMode AM,
                              MachineBasicBlock::iterator &InsertPoint,
                              unsigned int OpCode = X86::LEA64r);
  Register writeRIPToReg(MachineBasicBlock::iterator InsertPoint,
                         MachineBasicBlock &MBB);
  std::optional<Register> writeIndirectCallAddrToReg(MachineInstr &MI);

  MachineBasicBlock::iterator
  insertTLBUpdate(Register AddrReg, MachineBasicBlock::iterator InsertPoint,
                  MachineBasicBlock &MBB, LivePhysRegs &LiveIns);

  unsigned saveEFLAGS(MachineBasicBlock &MBB,
                      MachineBasicBlock::iterator InsertPt,
                      const DebugLoc &Loc);
  void restoreEFLAGS(MachineBasicBlock &MBB,
                     MachineBasicBlock::iterator InsertPt, const DebugLoc &Loc,
                     Register Reg);

  unsigned saveReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator InsertPt,
                   const DebugLoc &Loc, const TargetRegisterClass *RegClass,
                   Register SourceReg);
  void restoreReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator InsertPt,
                  const DebugLoc &Loc, Register SavedReg, Register TargetReg);

  Register loadGlobalCounter(MachineBasicBlock &MBB,
                             MachineBasicBlock::iterator InsertPt,
                             const DebugLoc &Loc);
  void storeGlobalCounter(MachineBasicBlock &MBB,
                          MachineBasicBlock::iterator InsertPt,
                          const DebugLoc &Loc, Register Reg);

public:
  static char ID;

  TLBlurInstrumentHeap() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "X86 TLBlur instrument heap accesses";
  }
};

} // namespace

unsigned TLBlurInstrumentHeap::saveEFLAGS(MachineBasicBlock &MBB,
                                          MachineBasicBlock::iterator InsertPt,
                                          const DebugLoc &Loc) {
  return saveReg(MBB, InsertPt, Loc, &X86::GR32RegClass, X86::EFLAGS);
}

void TLBlurInstrumentHeap::restoreEFLAGS(MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator InsertPt,
                                         const DebugLoc &Loc, Register Reg) {
  restoreReg(MBB, InsertPt, Loc, Reg, X86::EFLAGS);
}

unsigned TLBlurInstrumentHeap::saveReg(MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator InsertPt,
                                       const DebugLoc &Loc,
                                       const TargetRegisterClass *RegClass,
                                       Register SourceReg) {
  Register Reg = MRI->createVirtualRegister(RegClass);
  BuildMI(MBB, InsertPt, Loc, TII->get(X86::COPY), Reg).addReg(SourceReg);
  return Reg;
}

void TLBlurInstrumentHeap::restoreReg(MachineBasicBlock &MBB,
                                      MachineBasicBlock::iterator InsertPt,
                                      const DebugLoc &Loc, Register SavedVReg,
                                      Register TargetReg) {
  BuildMI(MBB, InsertPt, Loc, TII->get(X86::COPY), TargetReg).addReg(SavedVReg);
}

std::optional<X86AddressMode>
TLBlurInstrumentHeap::getAddressMode(MachineInstr &MI) {
  if (!MI.mayLoadOrStore() && !MI.isCall()) {
    // If you can't load or store, you can't access the heap
    return std::nullopt;
  }

  if (MI.getNumOperands() > 0) {
    auto &Last = MI.getOperand(MI.getNumOperands() - 1);
    if (Last.isReg() && Last.getReg().isPhysical() &&
        Last.getReg().asMCReg() == X86::FS) {
      LLVM_DEBUG(
          errs() << "HACK: Skipping Thread Local Storage instrumentation!\n");
      return std::nullopt;
    }

    for (auto Op : MI.explicit_operands()) {
      if (Op.isReg() && Op.getReg().isVirtual()) {
        const auto *RegClass = MRI->getRegClass(Op.getReg());
        if (!RegClass)
          continue;
        auto ID = RegClass->getID();
        if (ID == X86::VR256RegClassID || ID == X86::VR256XRegClassID ||
            ID == X86::VR512RegClassID || ID == X86::VR512_0_15RegClassID) {
          LLVM_DEBUG(errs() << "HACK: Skipping VR register instrumentation!\n");
          return std::nullopt;
        }
      }
    }
  }

  std::optional<X86AddressMode> AM = getAddressFromInstr(MI);

  if (!AM.has_value())
    return std::nullopt;

  if (AM.value().BaseType == X86AddressMode::FrameIndexBase) {
    // Is definitely a stack access
    return std::nullopt;
  }

  auto &MRI = MI.getParent()->getParent()->getRegInfo();
  if (Register::isVirtualRegister(AM.value().Base.Reg)) {
    MachineInstr *Def = MRI.getVRegDef(AM.value().Base.Reg);
    if (Def->getOpcode() == X86::LEA64r) {
      // The base register is an address loaded from an address mode with LEA
      std::optional<X86AddressMode> AM = getAddressFromInstr(*Def);
      if (!AM.has_value() ||
          AM.value().BaseType == X86AddressMode::FrameIndexBase) {
        // We load the effective address from a frame index, so it's a store to
        // the stack
        return std::nullopt;
      }
    }
  }

  // Assume it might be a heap access in all other cases
  return AM;
}

Register TLBlurInstrumentHeap::writeAddrModeToReg(
    X86AddressMode AM, MachineBasicBlock::iterator &InsertPoint,
    unsigned int OpCode) {
  auto Result =
      MRI->createVirtualRegister(TRI->getRegClass(X86::GR64RegClassID));
  auto &MBB = *InsertPoint->getParent();
  auto &MF = *MBB.getParent();

  MachineInstr *NewMI = nullptr;
  // Make sure we insert instrumentation before the call sequence start!
  for (MachineInstr &PrevMI :
       llvm::reverse(llvm::make_range(MBB.begin(), InsertPoint))) {

    // If we encounter a call frame destroy, we are not inside a call sequence
    if (PrevMI.getOpcode() == TII->getCallFrameDestroyOpcode())
      break;

    // If we find a call frame setup, insert instrumentation before this
    // instruction
    if (PrevMI.getOpcode() == TII->getCallFrameSetupOpcode()) {
      NewMI = &PrevMI;
      break;
    }
  }

  if (NewMI) {
    LLVM_DEBUG(
        errs() << "Inside call sequence, trying to fix the address mode\n");
    for (MachineInstr &MI : llvm::make_range(
             MachineBasicBlock::iterator(NewMI->getIterator()), InsertPoint)) {
      if (MI.definesRegister(AM.IndexReg)) {
        if (MI.isCopyLike()) {
          AM.IndexReg = MI.getOperand(1).getReg();
        } else {
          LLVM_DEBUG(errs() << "Skipping instrumentation in call sequence!\n");
          return MCRegister::NoRegister;
        }
      }

      if (AM.BaseType == X86AddressMode::RegBase &&
          MI.definesRegister(AM.Base.Reg)) {
        if (MI.isCopyLike()) {
          AM.Base.Reg = MI.getOperand(1).getReg();
        } else {
          LLVM_DEBUG(errs() << "Skipping instrumentation in call sequence!\n");
          return MCRegister::NoRegister;
        }
      }
    }

    InsertPoint = NewMI->getIterator();
  }

  auto &LEA =
      addFullAddress(BuildMI(MF, DebugLoc(), TII->get(OpCode), Result), AM);
  MBB.insert(InsertPoint, LEA);
  return Result;
}

Register
TLBlurInstrumentHeap::writeRIPToReg(MachineBasicBlock::iterator InsertPoint,
                                    MachineBasicBlock &MBB) {
  auto Result =
      MRI->createVirtualRegister(TRI->getRegClass(X86::GR64RegClassID));
  auto &MF = *MBB.getParent();
  auto &Copy = BuildMI(MF, DebugLoc(), TII->get(TargetOpcode::COPY), Result)
                   .addReg(X86::RIP);
  MBB.insert(InsertPoint, Copy);
  return Result;
}

static X86AddressMode getGlobalCounterAM(Module *M) {
  X86AddressMode CounterAM;
  CounterAM.GV = M->getGlobalVariable("__tlblur_global_counter");
  CounterAM.Base.Reg = X86::RIP;
  return CounterAM;
}

Register
TLBlurInstrumentHeap::loadGlobalCounter(MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator InsertPoint,
                                        const DebugLoc &Loc) {
  auto &MF = *MBB.getParent();
  Module *M = MF.getFunction().getParent();
  auto CounterAM = getGlobalCounterAM(M);
  auto CounterReg =
      MRI->createVirtualRegister(TRI->getRegClass(X86::GR64RegClassID));
  addFullAddress(
      BuildMI(MBB, InsertPoint, DebugLoc(), TII->get(X86::MOV64rm), CounterReg),
      CounterAM);
  return CounterReg;
}

void TLBlurInstrumentHeap::storeGlobalCounter(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator InsertPoint,
    const DebugLoc &Loc, Register Reg) {
  auto &MF = *MBB.getParent();
  Module *M = MF.getFunction().getParent();
  auto CounterAM = getGlobalCounterAM(M);
  addFullAddress(BuildMI(MBB, InsertPoint, DebugLoc(), TII->get(X86::MOV64mr)),
                 CounterAM)
      .addReg(Reg);
}

MachineBasicBlock::iterator TLBlurInstrumentHeap::insertTLBUpdate(
    Register AddrReg, MachineBasicBlock::iterator InsertPoint,
    MachineBasicBlock &MBB, LivePhysRegs &LiveIns) {
  auto &MF = *MBB.getParent();
  Module *M = MF.getFunction().getParent();

  if (TLBlurInline) {
    // Insert code inline

    // Save eflags if it is live
    recomputeLiveIns(MBB);
    bool IsEFLAGSLive = LiveIns.contains(X86::EFLAGS);
    Register FlagsReg = 0;
    if (IsEFLAGSLive)
      FlagsReg = saveEFLAGS(MBB, InsertPoint, DebugLoc());

    auto ImageBaseReg =
        MRI->createVirtualRegister(TRI->getRegClass(X86::GR64RegClassID));
    auto LEA = BuildMI(MBB, InsertPoint, DebugLoc(), TII->get(X86::LEA64r),
                       ImageBaseReg);
    X86AddressMode ImageBaseAM;
    ImageBaseAM.GV = M->getGlobalVariable("__ImageBase");
    ImageBaseAM.Base.Reg = X86::RIP;
    addFullAddress(LEA, ImageBaseAM);

    auto PageRegTemp =
        MRI->createVirtualRegister(TRI->getRegClass(X86::GR64RegClassID));
    BuildMI(MBB, InsertPoint, DebugLoc(), TII->get(X86::SUB64rr), PageRegTemp)
        .addReg(AddrReg)
        .addReg(ImageBaseReg);
    auto PageRegTemp2 =
        MRI->createVirtualRegister(TRI->getRegClass(X86::GR64_NOSPRegClassID));
    BuildMI(MBB, InsertPoint, DebugLoc(), TII->get(X86::SAR64ri), PageRegTemp2)
        .addReg(PageRegTemp)
        .addImm(12);
    auto PageRegTemp3 =
        MRI->createVirtualRegister(TRI->getRegClass(X86::GR64_NOSPRegClassID));
    BuildMI(MBB, InsertPoint, DebugLoc(), TII->get(X86::AND64ri32),
            PageRegTemp3)
        .addReg(PageRegTemp2)
        .addImm(TLBLUR_VTLB_SIZE - 1);
    auto PageReg =
        MRI->createVirtualRegister(TRI->getRegClass(X86::GR64_NOSPRegClassID));
    BuildMI(MBB, InsertPoint, DebugLoc(), TII->get(X86::SHL64ri), PageReg)
        .addReg(PageRegTemp3)
        .addImm(3);

    Register Counter = 0;
    if (!TLBlurCounterRegister) {
      auto CounterReg = loadGlobalCounter(MBB, InsertPoint, DebugLoc());
      auto CounterIncReg =
          MRI->createVirtualRegister(TRI->getRegClass(X86::GR64RegClassID));
      BuildMI(MBB, InsertPoint, DebugLoc(), TII->get(X86::INC64r),
              CounterIncReg)
          .addReg(CounterReg);
      Counter = CounterIncReg;
    } else {
      BuildMI(MBB, InsertPoint, DebugLoc(), TII->get(X86::INC64r), COUNTER_REG)
          .addReg(COUNTER_REG);
      Counter = COUNTER_REG;
    }

    X86AddressMode VTLBAM;
    VTLBAM.GV = M->getGlobalVariable("__tlblur_shadow_pt");
    VTLBAM.Base.Reg = X86::RIP;
    auto VTLBReg =
        MRI->createVirtualRegister(TRI->getRegClass(X86::GR64RegClassID));
    addFullAddress(
        BuildMI(MBB, InsertPoint, DebugLoc(), TII->get(X86::LEA64r), VTLBReg),
        VTLBAM);

    auto TargetAddrReg =
        MRI->createVirtualRegister(TRI->getRegClass(X86::GR64RegClassID));
    BuildMI(MBB, InsertPoint, DebugLoc(), TII->get(X86::ADD64rr), TargetAddrReg)
        .addReg(VTLBReg)
        .addReg(PageReg);

    X86AddressMode TargetAM;
    TargetAM.Base.Reg = TargetAddrReg;
    auto Store = BuildMI(MBB, InsertPoint, DebugLoc(), TII->get(X86::MOV64mr));
    addFullAddress(Store, TargetAM).addReg(Counter);

    if (!TLBlurCounterRegister)
      storeGlobalCounter(MBB, InsertPoint, DebugLoc(), Counter);

    // Restore eflags if necessary
    if (FlagsReg)
      restoreEFLAGS(MBB, InsertPoint, DebugLoc(), FlagsReg);
  } else {
    // Insert a function call to update the software TLB

    recomputeLiveIns(MBB);
    bool IsRDILive = LiveIns.contains(X86::RDI);
    bool IsEDILive = LiveIns.contains(X86::EDI);
    bool IsRAXLive = LiveIns.contains(X86::RAX);
    bool IsEAXLive = LiveIns.contains(X86::EAX);
    bool IsEFLAGSLive = LiveIns.contains(X86::EFLAGS);

    Register FlagsReg = 0;
    if (IsEFLAGSLive)
      FlagsReg = saveEFLAGS(MBB, InsertPoint, DebugLoc());
    Register RDI = 0;
    if (IsRDILive || IsEDILive)
      RDI = saveReg(MBB, InsertPoint, DebugLoc(), &X86::GR64RegClass, X86::RDI);
    Register RAX = 0;
    if (IsRAXLive || IsEAXLive)
      RAX = saveReg(MBB, InsertPoint, DebugLoc(), &X86::GR64RegClass, X86::RAX);

    MBB.insert(InsertPoint,
               BuildMI(MF, DebugLoc(), TII->get(TII->getCallFrameSetupOpcode()))
                   .addImm(0)
                   .addImm(0)
                   .addImm(0));

    // Copy address to %rdi
    MachineInstr *RDICopy =
        BuildMI(MF, DebugLoc(), TII->get(TargetOpcode::COPY), X86::RDI)
            .addReg(AddrReg);
    MBB.insert(InsertPoint, RDICopy);

    // Call instrumentation
    MBB.insert(InsertPoint,
               BuildMI(MF, DebugLoc(), TII->get(X86::CALL64pcrel32))
                   .addGlobalAddress(M->getNamedValue("tlblur_tlb_update"))
                   .addRegMask(TRI->getCalleeSavedRegsTLBlurMask())
                   .addUse(X86::RDI, IsRDILive ? RegState::Implicit
                                               : RegState::ImplicitKill));
    MBB.insert(InsertPoint, BuildMI(MF, DebugLoc(),
                                    TII->get(TII->getCallFrameDestroyOpcode()))
                                .addImm(0)
                                .addImm(0));

    if (RAX)
      restoreReg(MBB, InsertPoint, DebugLoc(), RAX, X86::RAX);
    if (RDI)
      restoreReg(MBB, InsertPoint, DebugLoc(), RDI, X86::RDI);
    if (FlagsReg)
      restoreEFLAGS(MBB, InsertPoint, DebugLoc(), FlagsReg);
  }

  return InsertPoint;
}

char TLBlurInstrumentHeap::ID = 0;

FunctionPass *llvm::createX86TLBlurInstrumentHeapPass() {
  return new TLBlurInstrumentHeap();
}

bool TLBlurInstrumentHeap::runOnMachineFunction(MachineFunction &MF) {
  Counter = 0;
  RDICopies.clear();
  CounterRegs.clear();
  const X86Subtarget &ST = MF.getSubtarget<X86Subtarget>();
  TII = ST.getInstrInfo();
  TRI = ST.getRegisterInfo();
  MRI = &MF.getRegInfo();
  Debug = false;

#ifdef DEBUG_FILTER
  if (MF.getFunction().getName() == DEBUG_FILTER) {
    Debug = true;
    LLVM_DEBUG(errs() << "--------------------------------\n");

    LLVM_DEBUG(MF.dump());

    LLVM_DEBUG(errs() << "--------------------------------\n");
  }
#endif

  // Declare externally defined globals
  Module *M = MF.getFunction().getParent();
  M->getOrInsertFunction("tlblur_tlb_update", Type::getVoidTy(M->getContext()),
                         PointerType::getUnqual(M->getContext()));
  M->getOrInsertGlobal("__ImageBase", PointerType::getUnqual(M->getContext()));
  M->getOrInsertGlobal("__tlblur_global_counter",
                       PointerType::getUnqual(M->getContext()));
  M->getOrInsertGlobal("__tlblur_shadow_pt",
                       PointerType::getUnqual(M->getContext()));

  SmallVector<std::tuple<MachineInstr *, X86AddressMode>> ToInstrument;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      // Get the address mode of the instruction
      std::optional<X86AddressMode> AM = getAddressMode(MI);

      // If the instruction has an address mode, insert instrumentation
      if (AM.has_value()) {
        MachineBasicBlock::iterator InsertPoint = MI.getIterator();

        // Try to insert code that writes the address mode to a register
        //
        // Note: indirect calls load from the address to get the function address
        Register AddrReg = writeAddrModeToReg(
            *AM, InsertPoint, MI.isCall() ? X86::MOV64rm : X86::LEA64r);

        if (AddrReg.isValid()) {
          // Figure out which physical registers are currently live
          LivePhysRegs LiveIns(*TRI);
          LiveIns.addLiveOuts(MBB);
          for (MachineInstr &MI :
               llvm::reverse(llvm::make_range(InsertPoint, MBB.end()))) {
            LiveIns.stepBackward(MI);
          }

          // Insert code to update the software TLB
          insertTLBUpdate(AddrReg, InsertPoint, MBB, LiveIns);
          Counter++;
        }
      }
    }
  }

#ifdef DEBUG_FILTER
  if (MF.getFunction().getName() == DEBUG_FILTER) {
    LLVM_DEBUG(errs() << "--------------------------------\n");

    LLVM_DEBUG(errs() << "Result:\n");

    LLVM_DEBUG(errs() << "--------------------------------\n");

    LLVM_DEBUG(MF.dump());

    LLVM_DEBUG(errs() << "--------------------------------\n");
  }
#endif

  // LLVM_DEBUG(errs() << "TLBlur: " << MF.getFunction().getName() << " ("
  //                   << Counter << " instrumentation calls)"
  //                   << "\n");
  errs() << "TLBlur: " << MF.getFunction().getName() << " ("
                    << Counter << " instrumentation calls)"
                    << "\n";

  return true;
}
