//! # `kadmium_core::platform` — Silicon & Operating System Platform Abstraction Layer
//!
//! Decouples Kadmium (`Cd-48`) hardware support from specific machines, board SKUs, or hardcoded
//! hostnames by operating at the **Platform Abstraction Layer**:
//!
//! 1. **Same Silicon $\to$ Supported by Kadmium (`SiliconFamily` + `CpuIsa`):**
//!    Any machine built on a supported silicon architecture inherits the full calibrated capability
//!    profile, memory topology (`UMA` vs. discrete `PCIe VRAM`), SIMD/Tensor kernel target, and
//!    `FreeToken` $q^*$ execution split—regardless of OEM form factor (e.g., an Apple M4 in a
//!    Mac mini, MacBook Air, MacBook Pro, iMac, Mac Studio, or iPad Pro; a Broadcom `BCM2711`/`BCM2712`
//!    in a Seeed reTerminal, Raspberry Pi 4/5, CM4 IO Board, Turing Pi 2 blade, or uConsole; an
//!    NVIDIA `T210`/`T214`/`T234` Tegra in a Jetson Nano/Orin, Shield TV Pro, or modded Switch; a
//!    Canaan `K230` or SpacemiT `K1` across any RISC-V RVV 1.0 carrier board; or any NVIDIA/AMD/Intel/Qualcomm
//!    desktop, laptop, workstation, or server).
//!
//! 2. **Same Operating System $\to$ Supported by Kadmium (`OperatingSystemFamily`):**
//!    Any machine running a supported OS family (`Windows`, `macOS / Darwin`, `Linux`, `Android`,
//!    `iOS / iPadOS`, `FreeBSD / TrueNAS`, `RT-Smart / RTOS`, `Horizon OS / L4T`) automatically shares
//!    unified hardware discovery probes, zero-copy `mmap` `.ktb` weight bank paging, USB/LAN transport
//!    bridges, and `< 200 ms` polite swapchain/thermal/battery auto-yield hooks (`OsHookBackend`).

use crate::{ComputeBackend, DeviceKind, HardwareNode, OsHookBackend};
use serde::{Deserialize, Serialize};

/// Instruction Set Architecture (ISA) family of a compute node's host processor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CpuIsa {
    /// 64-bit x86 (`x86_64` / `amd64`) with SSE4.2, AVX2, AVX-512, AVX-VNNI, or Intel AMX.
    X86_64,
    /// 64-bit ARM (`aarch64` / `arm64` / `arm64e`) with NEON, FP16, DotProd (`SDOT`/`UDOT`), SVE2, SME, or Apple AMX.
    Aarch64,
    /// 64-bit RISC-V (`riscv64gc` / `RVA20` / `RVA22`) with RVV 0.7.1 or RVV 1.0 Vector extensions.
    RiscV64,
}

impl CpuIsa {
    /// Detects the compiled target ISA of the running host process.
    pub fn detect_host_isa() -> Self {
        match std::env::consts::ARCH {
            "x86_64" | "x86" => Self::X86_64,
            "riscv64" => Self::RiscV64,
            _ => Self::Aarch64,
        }
    }

    /// Infers the CPU ISA from hardware model / architecture strings.
    pub fn from_signatures(cpu_name: &str, arch_hint: &str) -> Self {
        let combined = format!(
            "{} {}",
            cpu_name.to_ascii_lowercase(),
            arch_hint.to_ascii_lowercase()
        );
        if combined.contains("riscv")
            || combined.contains("rv64")
            || combined.contains("rvv")
            || combined.contains("xuantie")
            || combined.contains("c908")
            || combined.contains("c910")
            || combined.contains("c920")
            || combined.contains("k230")
            || combined.contains("spacemit")
            || combined.contains("jh7110")
            || combined.contains("sifive")
        {
            Self::RiscV64
        } else if combined.contains("aarch64")
            || combined.contains("arm64")
            || combined.contains("armv8")
            || combined.contains("armv9")
            || combined.contains("cortex-")
            || combined.contains("apple m")
            || combined.contains("apple a1")
            || combined.contains("snapdragon")
            || combined.contains("kryo")
            || combined.contains("oryon")
            || combined.contains("bcm271")
            || combined.contains("rk35")
            || combined.contains("rockchip")
            || combined.contains("tegra")
            || combined.contains("jetson")
            || combined.contains("neoverse")
            || combined.contains("graviton")
            || combined.contains("altra")
            || combined.contains("dimensity")
            || combined.contains("tensor g")
        {
            Self::Aarch64
        } else if combined.contains("x86_64")
            || combined.contains("amd64")
            || combined.contains("intel")
            || combined.contains("core i")
            || combined.contains("core ultra")
            || combined.contains("xeon")
            || combined.contains("ryzen")
            || combined.contains("epyc")
            || combined.contains("threadripper")
            || combined.contains("avx")
        {
            Self::X86_64
        } else {
            Self::detect_host_isa()
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            Self::X86_64 => "x86_64 (AVX2 / AVX-512 / VNNI / AMX)",
            Self::Aarch64 => "AArch64 (ARMv8.2–v9.2 NEON / DotProd / SVE2 / AMX)",
            Self::RiscV64 => "RISC-V 64 (RV64GCV / RVA22 / RVV 1.0 Vector)",
        }
    }
}

/// Operating System family abstraction (`Same Operating System -> Supported by Kadmium`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OperatingSystemFamily {
    /// Microsoft Windows 10 / 11 / Server (`x86_64` and `ARM64`):
    /// DXGI 1.6 swapchain governance, DirectML/MCDM NPU enumeration, CUDA/NVML, HIP, Vulkan, Win32 memory mapping.
    Windows,
    /// Apple macOS / Darwin XNU (`arm64e` Apple Silicon and `x86_64`):
    /// Metal 3/4 `MTLDevice`, Accelerate `AMX`/`BNNS`, CoreML ANE, `IOKit` power/thermal hooks, POSIX zero-copy `mmap`.
    MacOsDarwin,
    /// Linux kernel (`x86_64`, `aarch64`, `riscv64` across Ubuntu, Debian, Fedora, Arch, RHEL, Raspberry Pi OS,
    /// Armbian, NVIDIA L4T/JetPack, Asahi Linux, Yocto):
    /// `DRM/KMS` + `/sys/class/drm`, `/sys/class/accel`, `/proc/device-tree`, CUDA/ROCm/Vulkan, `io_uring`/zero-copy `mmap`.
    Linux,
    /// Android / LineageOS / Termux / AOSP (`aarch64` & `x86_64` across smartphones, tablets, and 20-Slot Phone Farm Boxes):
    /// Adreno/Mali/Immortalis Vulkan 1.3 Subgroup, Qualcomm QNN Hexagon / NNAPI, USB-ADB multiplexed TCP & Pocket-Polite governor.
    Android,
    /// Apple iOS / iPadOS (`arm64e` across M-series iPads and A-series iPhones/iPads):
    /// Metal 3/4 compute shaders, CoreML 16-Core ANE, `NSProcessInfo.thermalState` & charger-aware background compute.
    IosIpadOs,
    /// FreeBSD / TrueNAS CORE (`x86_64` & `aarch64`):
    /// ZFS Adaptive Replacement Cache (`ARC`) zero-copy `.ktb` weight vault, `sysctl` telemetry, and CPU/Vulkan compute.
    FreeBsd,
    /// RT-Smart / NuttX / FreeRTOS / Bare-Metal Co-Processor OS (`riscv64` & `aarch64`, e.g., Kendryte K230 C908 + KPU):
    /// Deterministic zero-jitter SRAM/LPDDR tensor execution over USB CDC-ACM / Bulk or shared-memory IPC.
    RtSmartRtos,
    /// Nintendo Switch Horizon OS (`Atmosphere deko3d sysmodule`) or Switchroot L4T Tegra edge runtime.
    HorizonOsL4t,
}

impl OperatingSystemFamily {
    /// Detects the `OperatingSystemFamily` of the current host process.
    pub fn detect_host_os() -> Self {
        if cfg!(target_os = "windows") {
            Self::Windows
        } else if cfg!(target_os = "macos") {
            Self::MacOsDarwin
        } else if cfg!(target_os = "ios") {
            Self::IosIpadOs
        } else if cfg!(target_os = "android") {
            Self::Android
        } else if cfg!(target_os = "freebsd") {
            Self::FreeBsd
        } else {
            // Check if running inside Android Termux on a Linux kernel
            if std::env::var("TERMUX_VERSION").is_ok()
                || std::path::Path::new("/system/build.prop").exists()
            {
                Self::Android
            } else {
                Self::Linux
            }
        }
    }

    /// Classifies an operating system family from a node or OS version signature string.
    pub fn from_signature(sig: &str) -> Self {
        let lower = sig.to_ascii_lowercase();
        if lower.contains("windows") || lower.contains("win32") || lower.contains("win11") {
            Self::Windows
        } else if lower.contains("ipados") || lower.contains("ios") || lower.contains("ipad") {
            Self::IosIpadOs
        } else if lower.contains("macos") || lower.contains("darwin") || lower.contains("os x") {
            Self::MacOsDarwin
        } else if lower.contains("rt-smart")
            || lower.contains("rtsmart")
            || lower.contains("canmv")
            || lower.contains("freertos")
            || lower.contains("nuttx")
        {
            Self::RtSmartRtos
        } else if lower.contains("horizon") || lower.contains("atmosphere") || lower.contains("deko3d") {
            Self::HorizonOsL4t
        } else if lower.contains("android")
            || lower.contains("termux")
            || lower.contains("lineage")
            || lower.contains("phonefarm")
            || lower.contains("pfb-slot")
            || lower.contains("galaxy")
        {
            Self::Android
        } else if lower.contains("freebsd") || lower.contains("truenas") {
            Self::FreeBsd
        } else if lower.contains("linux")
            || lower.contains("ubuntu")
            || lower.contains("debian")
            || lower.contains("raspbian")
            || lower.contains("bookworm")
            || lower.contains("armbian")
            || lower.contains("l4t")
            || lower.contains("jetpack")
            || lower.contains("asahi")
            || lower.contains("fedora")
            || lower.contains("arch")
        {
            Self::Linux
        } else {
            Self::detect_host_os()
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            Self::Windows => "Windows 10/11/Server (x86_64 & ARM64)",
            Self::MacOsDarwin => "macOS / Darwin XNU (arm64e & x86_64)",
            Self::Linux => "Linux (x86_64 / AArch64 / RISC-V64)",
            Self::Android => "Android / Termux / AOSP (AArch64)",
            Self::IosIpadOs => "iPadOS / iOS (Apple Silicon arm64e)",
            Self::FreeBsd => "FreeBSD / TrueNAS ZFS Vault",
            Self::RtSmartRtos => "RT-Smart / Embedded RTOS (RISC-V64 / ARM)",
            Self::HorizonOsL4t => "Horizon OS / L4T Switchroot (Tegra AArch64)",
        }
    }

    pub fn supported_isas(self) -> &'static [CpuIsa] {
        match self {
            Self::Windows => &[CpuIsa::X86_64, CpuIsa::Aarch64],
            Self::MacOsDarwin => &[CpuIsa::Aarch64, CpuIsa::X86_64],
            Self::Linux => &[CpuIsa::X86_64, CpuIsa::Aarch64, CpuIsa::RiscV64],
            Self::Android => &[CpuIsa::Aarch64, CpuIsa::X86_64, CpuIsa::RiscV64],
            Self::IosIpadOs => &[CpuIsa::Aarch64],
            Self::FreeBsd => &[CpuIsa::X86_64, CpuIsa::Aarch64, CpuIsa::RiscV64],
            Self::RtSmartRtos => &[CpuIsa::RiscV64, CpuIsa::Aarch64],
            Self::HorizonOsL4t => &[CpuIsa::Aarch64],
        }
    }

    /// Returns the OS-native `< 200 ms` swapchain/thermal/battery polite hook backend.
    pub fn os_hook_backend(self) -> OsHookBackend {
        match self {
            Self::Windows => OsHookBackend::WindowsDxgiNvml,
            Self::MacOsDarwin => OsHookBackend::MacOsIoKitMetal,
            Self::Linux => OsHookBackend::LinuxDrmSysfs,
            Self::Android => OsHookBackend::AndroidTermuxAdb,
            Self::IosIpadOs => OsHookBackend::IosIpadOsMetal,
            Self::FreeBsd => OsHookBackend::FreeBsdZfsSysctl,
            Self::RtSmartRtos | Self::HorizonOsL4t => OsHookBackend::EmbeddedRtosWatchdog,
        }
    }

    /// Native hardware discovery subsystems used by Kadmium on this OS family.
    pub fn discovery_probes(self) -> &'static [&'static str] {
        match self {
            Self::Windows => &[
                "DXGI 1.6 / Win32_VideoController CIM",
                "Windows MCDM ComputeAccelerator Registry ({f01a9d53-...})",
                "nvidia-smi / NVML",
                "SetupAPI / Get-PnpDevice USB CDC-ACM",
            ],
            Self::MacOsDarwin => &[
                "sysctl hw.* & machdep.cpu.*",
                "Metal MTLCreateSystemDefaultDevice()",
                "IOKit IOPlatformExpertDevice & IOUSB",
                "Apple Neural Engine (ANE) & AMX/SME",
            ],
            Self::Linux => &[
                "/proc/cpuinfo & /proc/meminfo",
                "/proc/device-tree/model & compatible",
                "/sys/class/drm/card*/device & /sys/class/accel/accel*",
                "nvidia-smi / rocm-smi / /sys/bus/usb/devices",
            ],
            Self::Android => &[
                "getprop ro.soc.model & ro.board.platform",
                "/proc/meminfo & /sys/class/kgsl/kgsl-3d0",
                "Vulkan 1.3 Subgroup & Qualcomm QNN Hexagon HTP",
                "USB-ADB Multiplexed Bridge & dumpsys battery",
            ],
            Self::IosIpadOs => &[
                "MTLCreateSystemDefaultDevice() (Apple Family 9)",
                "CoreML MLComputeUnitsAll (16-Core ANE)",
                "NSProcessInfo.processInfo.physicalMemory & thermalState",
            ],
            Self::FreeBsd => &[
                "sysctl hw.model, hw.physmem & hw.ncpu",
                "kstat.zfs.misc.arcstats (ZFS ARC Zero-Copy KTB Cache)",
                "DRM / Vulkan / CPU AVX-512 Worker",
            ],
            Self::RtSmartRtos => &[
                "USB BootROM / U-Boot VID:PID Enumeration",
                "USB CDC-ACM Raw REPL / KDMF Binary Frame Channel",
                "nncase v2.11 KPU DMA & RVV 1.0 Vector Registers",
            ],
            Self::HorizonOsL4t => &[
                "/etc/nv_tegra_release & /proc/device-tree/compatible (nvidia,tegra210*)",
                "nvgpu CUDA sm_53 / deko3d Maxwell Command Buffers",
                "BQ24193 Dock Power & EMC LPDDR4X Clock Governor",
            ],
        }
    }

    pub fn supports_zero_copy_mmap(self) -> bool {
        !matches!(self, Self::RtSmartRtos)
    }

    pub fn supports_usb_serial_bridge(self) -> bool {
        matches!(
            self,
            Self::Windows | Self::Linux | Self::MacOsDarwin | Self::Android | Self::FreeBsd
        )
    }
}

/// Silicon Designer / Vendor across CPU, GPU, SoC, and NPU architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SiliconVendor {
    Apple,
    Nvidia,
    Amd,
    Intel,
    Qualcomm,
    Broadcom,
    Rockchip,
    CanaanKendryte,
    SpacemiT,
    THeadXuantie,
    StarFive,
    MediaTek,
    GoogleTensor,
    AmpereArm,
    Generic,
}

/// Canonical Silicon Architecture Family (`Same Silicon -> Supported by Kadmium`).
///
/// Any two machines containing the same `SiliconFamily` share identical instruction-set capabilities,
/// memory topology (`UMA` vs. discrete `PCIe`), compute backend priority, and kernel dispatch targets,
/// regardless of which OEM chassis or board houses the chip.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SiliconFamily {
    /// **Apple M-Series (`M1` / `M2` / `M3` / `M4` — Base, Pro, Max, Ultra):**
    /// ARMv8.5–v9.2-A + Metal 3/4 SIMD-group Matrix + Hardware AMX/SME + 16/32-Core ANE + 100–819 GB/s UMA.
    /// Supported across Mac mini, MacBook Air, MacBook Pro, iMac, Mac Studio, Mac Pro, and iPad Pro.
    AppleSiliconMSeries,
    /// **Apple A-Series (`A15` / `A16` / `A17 Pro` / `A18` / `A18 Pro`):**
    /// ARMv8.6–v9.2-A + Metal 3/4 GPU + 16-Core ANE + LPDDR5/5X UMA across iPhone & iPad mini/Air.
    AppleSiliconASeries,
    /// **NVIDIA Blackwell (`SM100` / `SM120` — RTX 5090/5080/5070 Ti/5070/5060, RTX PRO, B100/B200/GB200):**
    /// 5th-Gen Tensor Cores with native FP4/FP8/BF16 + GDDR7 / HBM3e across desktops, laptops, and servers.
    NvidiaBlackwell,
    /// **NVIDIA Hopper & Ada Lovelace (`SM89` / `SM90` — RTX 4090/4080/4070/4060, L4, L40S, H100, H200, GH200):**
    /// 4th-Gen Tensor Cores with FP8/BF16/FP16 + GDDR6X / HBM3 across workstations, laptops, and cloud nodes.
    NvidiaHopperAda,
    /// **NVIDIA Ampere & Turing (`SM75` / `SM80` / `SM86` — RTX 30-series, RTX 20-series, GTX 16-series, A100, A40, A6000):**
    /// 2nd/3rd-Gen Tensor Cores with FP16/BF16/INT8/INT4 across desktops, gaming laptops, and servers.
    NvidiaAmpereTuring,
    /// **NVIDIA Volta, Pascal & Maxwell Discrete (`SM50`–`SM70` — GTX 10-series, GTX 900/750M, Titan X/V, V100, P100, P40):**
    /// CUDA FP16/FP32 + DP4A INT8 across legacy desktops, laptops, and lab servers.
    NvidiaPascalVoltaMaxwell,
    /// **NVIDIA Tegra Orin, Xavier & TX2 (`T234` / `T194` / `T186` — Jetson AGX Orin, Orin NX, Orin Nano, Xavier, TX2):**
    /// Ampere/Volta/Pascal CUDA Tensor Cores + NVDLA + Unified LPDDR5/LPDDR4X across robotics & edge modules.
    NvidiaTegraOrinXavier,
    /// **NVIDIA Tegra X1 & X1+ (`T210` 20nm & `T210B01` / `T214` 16nm — 256-Core Maxwell `sm_53` UMA):**
    /// Supported identically across NVIDIA Jetson Nano, NVIDIA Shield TV Pro, and modded Nintendo Switch (Erista/Mariko).
    NvidiaTegraX1,
    /// **AMD RDNA 2 / RDNA 3 / RDNA 4 Discrete GPUs (`Radeon RX 6000 / 7000 / 9000` & `Radeon PRO`):**
    /// ROCm HIP `WMMA` / Vulkan CooperativeMatrix across desktop and mobile discrete GPUs.
    AmdRdnaDiscrete,
    /// **AMD CDNA / CDNA 2 / CDNA 3 Datacenter Accelerators (`Instinct MI100 / MI210 / MI250X / MI300X / MI325X`):**
    /// Matrix Core `MFMA` FP8/BF16/FP16 + high-bandwidth HBM2e/HBM3.
    AmdCdnaInstinct,
    /// **AMD Ryzen AI Unified Memory APU (`Strix Halo` Ryzen AI Max+ 395 128GB UMA, `Strix Point` Ryzen AI 300, `Phoenix`/`Hawk Point`):**
    /// Zen 4/5 AVX-512 + RDNA 3/3.5 integrated GPU (`Radeon 8060S / 890M / 780M`) + 16–50 TOPS XDNA/XDNA2 NPU sharing LPDDR5X UMA.
    AmdRyzenAiUmaApu,
    /// **AMD Zen CPU Family (`Ryzen 3000–9000`, `Threadripper`, `EPYC Rome/Milan/Genoa/Turin`):**
    /// High-core-count x86_64 AVX2 / AVX-512 / VNNI execution across desktops, NAS vaults, and bare-metal servers.
    AmdZenCpuServerDesktop,
    /// **Intel Arc Xe / Xe2 Discrete GPUs (`Alchemist A770/A750/A380`, `Battlemage B580/B570`, `Data Center GPU Flex/Max`):**
    /// Xe Matrix Extensions (`XMX`) INT4/INT8/FP16 via Vulkan Subgroup / SYCL / OneAPI.
    IntelArcXeDiscrete,
    /// **Intel Core Ultra with AI Boost NPU (`Meteor Lake` Series 1, `Arrow Lake` 200S & `Lunar Lake` 200V Series 2):**
    /// Hybrid P+E cores + Arc Xe-LPG/Xe2 GPU + dedicated 11.5–48 TOPS Intel AI Boost NPU.
    IntelCoreUltraAiBoost,
    /// **Intel Core & Xeon Scalable x86_64 (`Core i3/i5/i7/i9` 6th–14th Gen, `Xeon Sapphire/Emerald/Granite Rapids`):**
    /// AVX2, AVX-512, AVX-VNNI, and Intel AMX matrix tile extensions.
    IntelCoreXeonAvx,
    /// **Qualcomm Snapdragon X Elite / X Plus (`X1E` / `X1P` Oryon 10–12C + Adreno + 45 TOPS Hexagon NPU):**
    /// ARMv8.7-A NEON/DotProd + 45 TOPS Hexagon HTP + 135 GB/s LPDDR5X UMA across Windows ARM64 & Linux laptops/mini-PCs.
    QualcommSnapdragonX,
    /// **Qualcomm Snapdragon 8-Series Mobile & Embedded (`SD 845 / 865 / 888 / 8 Gen 1–3 / 8 Elite`):**
    /// Kryo/Oryon ARM64 + Adreno Vulkan Subgroup + Hexagon HVX/HTP NPU across smartphones, tablets, 20-Slot Phone Farm Boxes, and DragonBoard/RB5 SBCs.
    QualcommSnapdragonMobile,
    /// **Broadcom VideoCore ARM64 SoC (`BCM2711` Cortex-A72 & `BCM2712` Cortex-A76):**
    /// ARMv8-A / ARMv8.2-A NEON/DotProd + VideoCore VI/VII V3D across Raspberry Pi 4B/5, Pi 400/500, CM4/CM5,
    /// Seeed reTerminal, Turing Pi 2 blades, Waveshare industrial gateways, and ClockworkPi uConsole.
    BroadcomVideoCoreArm,
    /// **Rockchip ARM64 + NPU SoC (`RK3588` / `RK3588S` / `RK3576` / `RK3568`):**
    /// Cortex-A76/A72/A55 + ARM Mali-G610/G52 + 6.0 TOPS RKNPU2 across Orange Pi 5, Radxa Rock 5B, Khadas Edge2, and Turing RK1.
    RockchipArmNpu,
    /// **MediaTek Dimensity & Genio ARM64 (`Dimensity 8000 / 9000 / 9300 / 9400`, `Genio 1200`):**
    /// Cortex-X/A7xx ARMv9 + Immortalis/Mali GPU + MediaTek APU across Android phones and IoT SBCs.
    MediaTekDimensityArm,
    /// **Google Tensor ARM64 (`Tensor G1 / G2 / G3 / G4`):**
    /// Cortex-X/A7xx + Mali GPU + Google EdgeTPU across Pixel phones and tablets.
    GoogleTensorArm,
    /// **Ampere / Neoverse Cloud-Native ARM64 Server (`Ampere Altra / AmpereOne`, `AWS Graviton`, `NVIDIA Grace CPU`):**
    /// 64–192 ARMv8.2+/v9 Neoverse cores with BF16/INT8 DotProd/SVE2 and high-channel DDR4/DDR5/LPDDR5X.
    AmpereNeoverseArmServer,
    /// **Canaan Kendryte RISC-V + KPU (`K230` / `K230D` / `K510` — Dual XuanTie C908 RVV 1.0 + 6 TOPS 3rd-Gen KPU):**
    /// Supported across 01Studio CanMV K230, Canaan Official K230, CanMV-K230D Zero, Banana Pi BPI-CanMV, and Yahboom K230.
    CanaanKendryteRiscV,
    /// **SpacemiT RISC-V (`SpacemiT K1` / `M1` — 8-Core X60 RVA22 + 256-bit RVV 1.0 + 2.0 TOPS NPU):**
    /// Supported across Milk-V Jupiter, Banana Pi BPI-F3, Sipeed LicheePi 3A, Muse Pi, and DC-ROMA RISC-V laptops.
    SpacemiTRiscV,
    /// **T-Head XuanTie High-Performance RISC-V (`TH1520` / `SG2042` — XuanTie C910 / C920 / C908 RV64GCV):**
    /// Supported across Sipeed LicheePi 4A, BeagleV-Ahead, and 64-core Milk-V Pioneer workstations.
    THeadXuantieRiscV,
    /// **StarFive SiFive RISC-V (`JH7110` / `JH8100` — Quad SiFive U74 RV64GC):**
    /// Supported across StarFive VisionFive 2, Milk-V Mars, and Pine64 Star64.
    StarFiveRiscV,
    /// **Generic Multi-Core Processor (`x86_64` / `AArch64` / `RISC-V64`):**
    /// Portable SIMD vector + Vulkan compute fallback for any standards-compliant CPU/GPU.
    GenericMultiCore,
}

impl SiliconFamily {
    pub fn vendor(self) -> SiliconVendor {
        match self {
            Self::AppleSiliconMSeries | Self::AppleSiliconASeries => SiliconVendor::Apple,
            Self::NvidiaBlackwell
            | Self::NvidiaHopperAda
            | Self::NvidiaAmpereTuring
            | Self::NvidiaPascalVoltaMaxwell
            | Self::NvidiaTegraOrinXavier
            | Self::NvidiaTegraX1 => SiliconVendor::Nvidia,
            Self::AmdRdnaDiscrete
            | Self::AmdCdnaInstinct
            | Self::AmdRyzenAiUmaApu
            | Self::AmdZenCpuServerDesktop => SiliconVendor::Amd,
            Self::IntelArcXeDiscrete | Self::IntelCoreUltraAiBoost | Self::IntelCoreXeonAvx => {
                SiliconVendor::Intel
            }
            Self::QualcommSnapdragonX | Self::QualcommSnapdragonMobile => SiliconVendor::Qualcomm,
            Self::BroadcomVideoCoreArm => SiliconVendor::Broadcom,
            Self::RockchipArmNpu => SiliconVendor::Rockchip,
            Self::MediaTekDimensityArm => SiliconVendor::MediaTek,
            Self::GoogleTensorArm => SiliconVendor::GoogleTensor,
            Self::AmpereNeoverseArmServer => SiliconVendor::AmpereArm,
            Self::CanaanKendryteRiscV => SiliconVendor::CanaanKendryte,
            Self::SpacemiTRiscV => SiliconVendor::SpacemiT,
            Self::THeadXuantieRiscV => SiliconVendor::THeadXuantie,
            Self::StarFiveRiscV => SiliconVendor::StarFive,
            Self::GenericMultiCore => SiliconVendor::Generic,
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            Self::AppleSiliconMSeries => "Apple Silicon M-Series (M1/M2/M3/M4 Base/Pro/Max/Ultra)",
            Self::AppleSiliconASeries => "Apple Silicon A-Series (A15–A18 Pro Metal + ANE)",
            Self::NvidiaBlackwell => "NVIDIA Blackwell (RTX 50-Series / B200 / GB200)",
            Self::NvidiaHopperAda => "NVIDIA Hopper & Ada Lovelace (RTX 40-Series / L40S / H100 / H200)",
            Self::NvidiaAmpereTuring => "NVIDIA Ampere & Turing (RTX 30/20-Series / A100 / A6000)",
            Self::NvidiaPascalVoltaMaxwell => "NVIDIA Volta / Pascal / Maxwell (V100 / GTX 10/900-Series)",
            Self::NvidiaTegraOrinXavier => "NVIDIA Jetson Tegra Orin / Xavier / TX2 (T234 / T194 / T186 UMA)",
            Self::NvidiaTegraX1 => "NVIDIA Tegra X1 / X1+ (T210 / T214 Maxwell sm_53 UMA)",
            Self::AmdRdnaDiscrete => "AMD Radeon RDNA 2 / 3 / 4 Discrete GPU (ROCm HIP / Vulkan)",
            Self::AmdCdnaInstinct => "AMD Instinct CDNA / CDNA 2 / CDNA 3 (MI100 / MI250X / MI300X)",
            Self::AmdRyzenAiUmaApu => "AMD Ryzen AI APU (Strix Halo / Strix Point / Phoenix UMA + XDNA NPU)",
            Self::AmdZenCpuServerDesktop => "AMD Ryzen / Threadripper / EPYC (Zen AVX2 / AVX-512)",
            Self::IntelArcXeDiscrete => "Intel Arc Alchemist / Battlemage / Data Center Max (Xe XMX)",
            Self::IntelCoreUltraAiBoost => "Intel Core Ultra (Series 1 & 2) + AI Boost NPU",
            Self::IntelCoreXeonAvx => "Intel Core / Xeon Scalable (AVX2 / AVX-512 / AMX)",
            Self::QualcommSnapdragonX => "Qualcomm Snapdragon X Elite / Plus (Oryon + 45 TOPS Hexagon)",
            Self::QualcommSnapdragonMobile => "Qualcomm Snapdragon 8-Series (Kryo/Oryon + Adreno + Hexagon)",
            Self::BroadcomVideoCoreArm => "Broadcom BCM2711 / BCM2712 (Cortex-A72/A76 NEON + VideoCore VI/VII)",
            Self::RockchipArmNpu => "Rockchip RK3588 / RK3576 / RK3568 (Cortex-A76/A55 + 6 TOPS RKNPU2)",
            Self::MediaTekDimensityArm => "MediaTek Dimensity / Genio (ARMv9 + Immortalis + APU)",
            Self::GoogleTensorArm => "Google Tensor G1–G4 (ARMv9 + Mali + EdgeTPU)",
            Self::AmpereNeoverseArmServer => "ARM Neoverse Server (Ampere Altra / AWS Graviton / Grace)",
            Self::CanaanKendryteRiscV => "Canaan Kendryte K230 / K230D / K510 (XuanTie C908 RVV 1.0 + KPU)",
            Self::SpacemiTRiscV => "SpacemiT K1 / M1 Octa-Core RISC-V (RVA22 + RVV 1.0 + 2 TOPS NPU)",
            Self::THeadXuantieRiscV => "T-Head XuanTie C910 / C920 RISC-V (TH1520 / SG2042 RV64GCV)",
            Self::StarFiveRiscV => "StarFive JH7110 / JH8100 RISC-V (SiFive U74 RV64GC)",
            Self::GenericMultiCore => "Generic Multi-Core CPU / Vulkan Accelerator",
        }
    }

    /// Primary compute backend for this silicon architecture.
    pub fn primary_backend(self) -> ComputeBackend {
        match self {
            Self::AppleSiliconMSeries | Self::AppleSiliconASeries => ComputeBackend::MetalUnified,
            Self::NvidiaBlackwell
            | Self::NvidiaHopperAda
            | Self::NvidiaAmpereTuring
            | Self::NvidiaPascalVoltaMaxwell
            | Self::NvidiaTegraOrinXavier
            | Self::NvidiaTegraX1 => ComputeBackend::Cuda,
            Self::AmdRdnaDiscrete | Self::AmdCdnaInstinct | Self::AmdRyzenAiUmaApu => {
                ComputeBackend::Rocm
            }
            Self::IntelArcXeDiscrete => ComputeBackend::VulkanWgpu,
            Self::AmdZenCpuServerDesktop
            | Self::IntelCoreUltraAiBoost
            | Self::IntelCoreXeonAvx => ComputeBackend::CpuAvx512OrAvx2,
            Self::QualcommSnapdragonX
            | Self::QualcommSnapdragonMobile
            | Self::BroadcomVideoCoreArm
            | Self::RockchipArmNpu
            | Self::MediaTekDimensityArm
            | Self::GoogleTensorArm
            | Self::AmpereNeoverseArmServer
            | Self::CanaanKendryteRiscV
            | Self::SpacemiTRiscV
            | Self::THeadXuantieRiscV
            | Self::StarFiveRiscV => ComputeBackend::CpuArmNeonNpu,
            Self::GenericMultiCore => ComputeBackend::CpuAvx512OrAvx2,
        }
    }

    /// Fallback compute backends supported on this silicon family.
    pub fn fallback_backends(self) -> Vec<ComputeBackend> {
        match self {
            Self::AppleSiliconMSeries | Self::AppleSiliconASeries => {
                vec![ComputeBackend::CpuArmNeonNpu, ComputeBackend::VulkanWgpu]
            }
            Self::NvidiaBlackwell
            | Self::NvidiaHopperAda
            | Self::NvidiaAmpereTuring
            | Self::NvidiaPascalVoltaMaxwell => {
                vec![ComputeBackend::VulkanWgpu, ComputeBackend::CpuAvx512OrAvx2]
            }
            Self::NvidiaTegraOrinXavier | Self::NvidiaTegraX1 => {
                vec![ComputeBackend::VulkanWgpu, ComputeBackend::CpuArmNeonNpu]
            }
            Self::AmdRdnaDiscrete | Self::AmdCdnaInstinct | Self::AmdRyzenAiUmaApu => {
                vec![ComputeBackend::VulkanWgpu, ComputeBackend::CpuAvx512OrAvx2]
            }
            Self::IntelArcXeDiscrete => vec![ComputeBackend::CpuAvx512OrAvx2],
            Self::QualcommSnapdragonX
            | Self::QualcommSnapdragonMobile
            | Self::BroadcomVideoCoreArm
            | Self::RockchipArmNpu
            | Self::MediaTekDimensityArm
            | Self::GoogleTensorArm => vec![ComputeBackend::VulkanWgpu],
            _ => vec![ComputeBackend::VulkanWgpu],
        }
    }

    /// Memory architecture topology of this silicon family.
    pub fn memory_topology(self) -> MemoryTopologyKind {
        match self {
            Self::AppleSiliconMSeries
            | Self::AppleSiliconASeries
            | Self::NvidiaTegraOrinXavier
            | Self::NvidiaTegraX1
            | Self::AmdRyzenAiUmaApu
            | Self::QualcommSnapdragonX
            | Self::QualcommSnapdragonMobile
            | Self::BroadcomVideoCoreArm
            | Self::RockchipArmNpu
            | Self::MediaTekDimensityArm
            | Self::GoogleTensorArm
            | Self::CanaanKendryteRiscV
            | Self::SpacemiTRiscV
            | Self::THeadXuantieRiscV
            | Self::StarFiveRiscV => MemoryTopologyKind::UnifiedMemoryArchitecture,
            Self::NvidiaBlackwell
            | Self::NvidiaHopperAda
            | Self::NvidiaAmpereTuring
            | Self::NvidiaPascalVoltaMaxwell
            | Self::AmdRdnaDiscrete
            | Self::AmdCdnaInstinct
            | Self::IntelArcXeDiscrete => MemoryTopologyKind::DiscretePcieVramPlusHostRam,
            Self::AmdZenCpuServerDesktop
            | Self::IntelCoreUltraAiBoost
            | Self::IntelCoreXeonAvx
            | Self::AmpereNeoverseArmServer
            | Self::GenericMultiCore => MemoryTopologyKind::HostRamOnly,
        }
    }

    /// Example OEM machines / form factors that house this exact silicon family and are supported
    /// without any machine-specific code changes.
    pub fn compatible_form_factors(self) -> &'static [&'static str] {
        match self {
            Self::AppleSiliconMSeries => &[
                "Apple Mac mini (M1 / M2 / M4 / M4 Pro)",
                "Apple MacBook Air 13\" & 15\" (M1 / M2 / M3 / M4)",
                "Apple MacBook Pro 14\" & 16\" (M1–M4 Pro / Max)",
                "Apple Mac Studio (M1/M2/M3/M4 Max & Ultra)",
                "Apple iMac 24\" & Mac Pro",
                "Apple iPad Pro (M1 / M2 / M4)",
            ],
            Self::AppleSiliconASeries => &[
                "iPhone 15 Pro / 16 Pro (A17 Pro / A18 Pro)",
                "iPad mini (A17 Pro) & iPad Air",
            ],
            Self::NvidiaBlackwell => &[
                "Custom Desktop / Workstation (RTX 5090 / 5080 / 5070 Ti / 5070 / 5060)",
                "Gaming Laptops (RTX 5090 / 5080 / 5070 Laptop GPU)",
                "Datacenter HGX / DGX B200 & GB200 NVL72",
            ],
            Self::NvidiaHopperAda => &[
                "Desktops & Workstations (RTX 4090 / 4080 / 4070 / 4060 / RTX 6000 Ada)",
                "Gaming & Creator Laptops (RTX 4050–4090 Laptop GPU)",
                "Cloud & Datacenter (NVIDIA H100 / H200 / GH200 / L40S / L4)",
            ],
            Self::NvidiaAmpereTuring => &[
                "Desktops & Workstations (RTX 3090 / 3080 / 3070 / 3060 / RTX 2080 Ti / GTX 1660)",
                "Gaming Laptops (RTX 3060–3080 Ti / RTX 2060–2080 Mobile)",
                "Datacenter (NVIDIA A100 / A40 / A10G / RTX A6000)",
            ],
            Self::NvidiaPascalVoltaMaxwell => &[
                "Legacy Desktops & Laptops (GTX 1080 Ti / 1070 / 1060 / GTX 980 / 750M)",
                "Lab Servers (Tesla V100 / P100 / P40 / Titan V)",
            ],
            Self::NvidiaTegraOrinXavier => &[
                "NVIDIA Jetson AGX Orin (32GB / 64GB)",
                "NVIDIA Jetson Orin NX & Orin Nano Super (8GB / 16GB)",
                "NVIDIA Jetson AGX Xavier / Xavier NX / TX2",
            ],
            Self::NvidiaTegraX1 => &[
                "NVIDIA Jetson Nano Developer Kit (B01 4GB / 2GB T210)",
                "NVIDIA Shield TV Pro (T210B01 / T214)",
                "Modded Nintendo Switch 1 (Mariko T210B01 / Erista T210 / 8GB RAM-Mod)",
                "Salvaged Tegra X1 / X1+ Headless Carrier Blades",
            ],
            Self::AmdRdnaDiscrete => &[
                "Desktops & Workstations (Radeon RX 7900 XTX / 7800 XT / RX 9070 XT / RX 6800 XT / Radeon PRO W7900)",
                "Gaming Laptops (Radeon RX 7900M / 7700S / 6800M)",
            ],
            Self::AmdCdnaInstinct => &[
                "Datacenter & Cloud Nodes (AMD Instinct MI300X / MI325X / MI250X / MI210)",
            ],
            Self::AmdRyzenAiUmaApu => &[
                "Strix Halo Mini-PCs & Workstations (Ryzen AI Max+ 395 w/ 64–128GB UMA + Radeon 8060S)",
                "Strix Point / Phoenix Laptops & Mini-PCs (Ryzen AI 9 HX 370 / 7840U / 8845HS)",
            ],
            Self::AmdZenCpuServerDesktop => &[
                "Desktops (Ryzen 5/7/9 3000/5000/7000/9000)",
                "Workstations & NAS Vaults (Threadripper / EPYC 4004 / 7003 / 9004 / 9005)",
            ],
            Self::IntelArcXeDiscrete => &[
                "Desktops & Laptops (Intel Arc B580 / B570 / A770 16GB / A750 / A380)",
                "Servers (Intel Data Center GPU Max / Flex)",
            ],
            Self::IntelCoreUltraAiBoost => &[
                "Desktops (Intel Core Ultra 9 285K / Ultra 7 265F / Ultra 5 245K Arrow Lake)",
                "Laptops & Mini-PCs (Core Ultra 200V Lunar Lake / Series 1 Meteor Lake)",
            ],
            Self::IntelCoreXeonAvx => &[
                "Desktops & Legacy Laptops (Intel Core i3/i5/i7/i9 6th–14th Gen)",
                "Servers (Intel Xeon Scalable Sapphire / Emerald / Granite Rapids w/ AMX)",
            ],
            Self::QualcommSnapdragonX => &[
                "Windows ARM64 & Linux Laptops (Surface Laptop 7, ThinkPad T14s Gen 6, Yoga Slim 7x, EliteBook Ultra)",
                "Snapdragon Dev Kit & Oryon Mini-PCs",
            ],
            Self::QualcommSnapdragonMobile => &[
                "20-Slot Phone Farm Box Chassis (20x Snapdragon 845 / 865 / 8 Gen 2 stripped mainboards)",
                "Android Smartphones & Tablets (Galaxy S9–S25 Ultra, OnePlus, Xiaomi, RedMagic)",
                "Qualcomm Robotics RB3 / RB5 / DragonBoard SBCs",
            ],
            Self::BroadcomVideoCoreArm => &[
                "Seeed Studio reTerminal / reTerminal DM (CM4104032)",
                "Raspberry Pi 4 Model B / Raspberry Pi 5 / Pi 400 / Pi 500",
                "Raspberry Pi Compute Module 4 & 5 (Official IO Board, Waveshare, Turing Pi 2 Blade, ClockworkPi uConsole)",
            ],
            Self::RockchipArmNpu => &[
                "Orange Pi 5 / 5 Plus / 5 Max (RK3588 / RK3588S)",
                "Radxa ROCK 5B / 5C & Khadas Edge2",
                "Turing RK1 Cluster Blade & FriendlyElec NanoPC-T6",
            ],
            Self::MediaTekDimensityArm => &[
                "Dimensity 9300/9400 Android Phones & MediaTek Genio 1200 SBCs",
            ],
            Self::GoogleTensorArm => &[
                "Google Pixel 6 / 7 / 8 / 9 Pro Smartphones & Pixel Tablet",
            ],
            Self::AmpereNeoverseArmServer => &[
                "Ampere Altra / AmpereOne Rack Servers & Workstation Towers",
                "AWS Graviton3 / Graviton4 & NVIDIA Grace CPU Nodes",
            ],
            Self::CanaanKendryteRiscV => &[
                "01Studio CanMV K230",
                "Canaan Official CanMV-K230 V1.0 / V1.1",
                "CanMV-K230D Zero & Banana Pi BPI-CanMV-K230D",
                "Yahboom K230 2GB Vision AI Module & DshanPi-CanMV",
            ],
            Self::SpacemiTRiscV => &[
                "Milk-V Jupiter (Mini-ITX SpacemiT K1/M1)",
                "Banana Pi BPI-F3 & Sipeed LicheePi 3A",
                "DC-ROMA RISC-V Laptop II & Muse Book / Muse Pi",
            ],
            Self::THeadXuantieRiscV => &[
                "Sipeed LicheePi 4A & BeagleV-Ahead (TH1520 Quad C910)",
                "Milk-V Pioneer 64-Core SG2042 RISC-V Workstation",
            ],
            Self::StarFiveRiscV => &[
                "StarFive VisionFive 2, Milk-V Mars & Pine64 Star64 (JH7110)",
            ],
            Self::GenericMultiCore => &[
                "Any x86_64, AArch64, or RISC-V64 Desktop, Laptop, Server, or Virtual Machine",
            ],
        }
    }

    /// Operating system families supported on this silicon family.
    pub fn supported_os_families(self) -> &'static [OperatingSystemFamily] {
        match self {
            Self::AppleSiliconMSeries => &[
                OperatingSystemFamily::MacOsDarwin,
                OperatingSystemFamily::IosIpadOs,
                OperatingSystemFamily::Linux, // Asahi Linux
            ],
            Self::AppleSiliconASeries => &[OperatingSystemFamily::IosIpadOs],
            Self::NvidiaBlackwell
            | Self::NvidiaHopperAda
            | Self::NvidiaAmpereTuring
            | Self::NvidiaPascalVoltaMaxwell => &[
                OperatingSystemFamily::Windows,
                OperatingSystemFamily::Linux,
                OperatingSystemFamily::FreeBsd,
            ],
            Self::NvidiaTegraOrinXavier => &[OperatingSystemFamily::Linux],
            Self::NvidiaTegraX1 => &[
                OperatingSystemFamily::Linux,
                OperatingSystemFamily::HorizonOsL4t,
                OperatingSystemFamily::Android,
            ],
            Self::AmdRdnaDiscrete
            | Self::AmdCdnaInstinct
            | Self::AmdRyzenAiUmaApu
            | Self::AmdZenCpuServerDesktop
            | Self::IntelArcXeDiscrete
            | Self::IntelCoreUltraAiBoost => &[
                OperatingSystemFamily::Windows,
                OperatingSystemFamily::Linux,
                OperatingSystemFamily::FreeBsd,
            ],
            Self::IntelCoreXeonAvx => &[
                OperatingSystemFamily::Windows,
                OperatingSystemFamily::Linux,
                OperatingSystemFamily::FreeBsd,
                OperatingSystemFamily::MacOsDarwin,
            ],
            Self::QualcommSnapdragonX => &[
                OperatingSystemFamily::Windows,
                OperatingSystemFamily::Linux,
            ],
            Self::QualcommSnapdragonMobile
            | Self::MediaTekDimensityArm
            | Self::GoogleTensorArm => &[
                OperatingSystemFamily::Android,
                OperatingSystemFamily::Linux,
            ],
            Self::BroadcomVideoCoreArm | Self::RockchipArmNpu => &[
                OperatingSystemFamily::Linux,
                OperatingSystemFamily::FreeBsd,
                OperatingSystemFamily::Android,
            ],
            Self::AmpereNeoverseArmServer => &[
                OperatingSystemFamily::Linux,
                OperatingSystemFamily::FreeBsd,
                OperatingSystemFamily::Windows,
            ],
            Self::CanaanKendryteRiscV => &[
                OperatingSystemFamily::RtSmartRtos,
                OperatingSystemFamily::Linux,
            ],
            Self::SpacemiTRiscV | Self::THeadXuantieRiscV | Self::StarFiveRiscV => &[
                OperatingSystemFamily::Linux,
                OperatingSystemFamily::FreeBsd,
                OperatingSystemFamily::RtSmartRtos,
            ],
            Self::GenericMultiCore => &[
                OperatingSystemFamily::Windows,
                OperatingSystemFamily::MacOsDarwin,
                OperatingSystemFamily::Linux,
                OperatingSystemFamily::Android,
                OperatingSystemFamily::IosIpadOs,
                OperatingSystemFamily::FreeBsd,
                OperatingSystemFamily::RtSmartRtos,
                OperatingSystemFamily::HorizonOsL4t,
            ],
        }
    }
}

/// Memory hierarchy topology classification for a silicon platform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryTopologyKind {
    /// CPU, GPU, and NPU share a single coherent physical memory pool (zero PCIe copy overhead; $q^* \approx 0.96$).
    UnifiedMemoryArchitecture,
    /// Discrete GPU VRAM (GDDR6/GDDR7/HBM) connected over PCIe/NVLink to Host DDR4/DDR5 RAM ($q^*$ balances GPU vs. host/PCIe stream).
    DiscretePcieVramPlusHostRam,
    /// Host RAM only (CPU SIMD + optional NPU sharing host DDR4/DDR5/LPDDR5).
    HostRamOnly,
}

/// Complete machine-agnostic platform profile for a node or host.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformProfile {
    pub silicon_vendor: SiliconVendor,
    pub silicon_family: SiliconFamily,
    pub silicon_display_name: String,
    pub cpu_isa: CpuIsa,
    pub os_family: OperatingSystemFamily,
    pub os_display_name: String,
    pub memory_topology: MemoryTopologyKind,
    pub primary_backend: ComputeBackend,
    pub fallback_backends: Vec<ComputeBackend>,
    pub os_hook_backend: OsHookBackend,
    /// True because capabilities are derived from `(SiliconFamily, OperatingSystemFamily)` rather than a hardcoded machine.
    pub machine_agnostic_abstraction: bool,
    pub compatible_form_factors: Vec<String>,
    pub supported_os_families: Vec<OperatingSystemFamily>,
}

/// Entry in the global `(SiliconFamily, OperatingSystemFamily)` compatibility matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformMatrixEntry {
    pub silicon_vendor: SiliconVendor,
    pub silicon_family: SiliconFamily,
    pub silicon_display_name: String,
    pub cpu_isa: CpuIsa,
    pub primary_backend: ComputeBackend,
    pub memory_topology: MemoryTopologyKind,
    pub supported_os_families: Vec<OperatingSystemFamily>,
    pub compatible_form_factors: Vec<String>,
}

/// Platform-level detector and classifier.
pub struct PlatformDetector;

impl PlatformDetector {
    /// Classifies any hardware signature tuple into its canonical `SiliconFamily`.
    pub fn classify_silicon_family(
        cpu_name: &str,
        gpu_name: &str,
        npu_name: Option<&str>,
        board_model: Option<&str>,
    ) -> SiliconFamily {
        let combined = format!(
            "{} {} {} {}",
            cpu_name.to_ascii_lowercase(),
            gpu_name.to_ascii_lowercase(),
            npu_name.unwrap_or("").to_ascii_lowercase(),
            board_model.unwrap_or("").to_ascii_lowercase()
        );

        // 1. RISC-V families
        if combined.contains("k230") || combined.contains("k510") || combined.contains("kendryte") {
            return SiliconFamily::CanaanKendryteRiscV;
        }
        if combined.contains("spacemit") || combined.contains("k1") && combined.contains("riscv") {
            return SiliconFamily::SpacemiTRiscV;
        }
        if combined.contains("th1520")
            || combined.contains("sg2042")
            || combined.contains("c910")
            || combined.contains("c920")
            || combined.contains("xuantie")
        {
            return SiliconFamily::THeadXuantieRiscV;
        }
        if combined.contains("jh7110") || combined.contains("jh8100") || combined.contains("visionfive") || combined.contains("starfive") {
            return SiliconFamily::StarFiveRiscV;
        }

        // 2. Apple Silicon (M-series and A-series)
        if combined.contains("apple m")
            || combined.contains("mac16,")
            || combined.contains("mac15,")
            || combined.contains("mac14,")
            || combined.contains("macbook") && combined.contains("metal")
        {
            return SiliconFamily::AppleSiliconMSeries;
        }
        if combined.contains("apple a1") || combined.contains("iphone") {
            return SiliconFamily::AppleSiliconASeries;
        }

        // 3. NVIDIA Tegra (X1/X1+ vs Orin/Xavier/TX2) & Discrete GPUs
        if combined.contains("tegra x1")
            || combined.contains("t210")
            || combined.contains("t214")
            || combined.contains("mariko")
            || combined.contains("erista")
            || combined.contains("jetson nano")
            || combined.contains("shield tv")
        {
            return SiliconFamily::NvidiaTegraX1;
        }
        if combined.contains("orin")
            || combined.contains("xavier")
            || combined.contains("t234")
            || combined.contains("t194")
            || combined.contains("t186")
            || combined.contains("jetson agx")
            || combined.contains("jetson tx2")
        {
            return SiliconFamily::NvidiaTegraOrinXavier;
        }
        if combined.contains("rtx 50")
            || combined.contains("gb20")
            || combined.contains("b200")
            || combined.contains("b100")
            || combined.contains("gb10")
            || combined.contains("blackwell")
        {
            return SiliconFamily::NvidiaBlackwell;
        }
        if combined.contains("rtx 40")
            || combined.contains("ad10")
            || combined.contains("h100")
            || combined.contains("h200")
            || combined.contains("gh200")
            || combined.contains("l40")
            || combined.contains("rtx 6000 ada")
        {
            return SiliconFamily::NvidiaHopperAda;
        }
        if combined.contains("rtx 30")
            || combined.contains("rtx 20")
            || combined.contains("gtx 16")
            || combined.contains("a100")
            || combined.contains("a6000")
            || combined.contains("a40")
            || combined.contains("a10g")
            || combined.contains("tesla t4")
            || (combined.contains("nvidia") && (combined.contains("ampere") || combined.contains("turing")))
        {
            return SiliconFamily::NvidiaAmpereTuring;
        }
        if combined.contains("gtx 10")
            || combined.contains("gtx 9")
            || combined.contains("gtx 7")
            || combined.contains("v100")
            || combined.contains("p100")
            || combined.contains("p40")
            || combined.contains("titan")
            || combined.contains("geforce")
            || combined.contains("quadro")
        {
            return SiliconFamily::NvidiaPascalVoltaMaxwell;
        }

        // 4. AMD Discrete, Instinct, Ryzen AI APU, and Zen CPUs
        if combined.contains("mi300")
            || combined.contains("mi325")
            || combined.contains("mi250")
            || combined.contains("mi210")
            || combined.contains("mi100")
            || combined.contains("instinct")
        {
            return SiliconFamily::AmdCdnaInstinct;
        }
        if combined.contains("rx 7")
            || combined.contains("rx 6")
            || combined.contains("rx 9")
            || combined.contains("radeon pro w")
            || combined.contains("rdna")
            || combined.contains("radeon rocm")
        {
            return SiliconFamily::AmdRdnaDiscrete;
        }
        if combined.contains("ryzen ai")
            || combined.contains("strix halo")
            || combined.contains("strix point")
            || combined.contains("radeon 8060s")
            || combined.contains("radeon 890m")
            || combined.contains("radeon 780m")
            || combined.contains("xdna")
        {
            return SiliconFamily::AmdRyzenAiUmaApu;
        }

        // 5. Intel Arc Discrete, Core Ultra, and Xeon/Core
        if combined.contains("intel arc")
            || combined.contains("arc a7")
            || combined.contains("arc a5")
            || combined.contains("arc a3")
            || combined.contains("arc b5")
            || combined.contains("battlemage")
            || combined.contains("alchemist")
        {
            return SiliconFamily::IntelArcXeDiscrete;
        }

        // 6. ARM64 SBCs, Mobile & Server SoCs
        if combined.contains("bcm2711")
            || combined.contains("bcm2712")
            || combined.contains("videocore")
            || combined.contains("raspberry pi")
            || combined.contains("compute module 4")
            || combined.contains("compute module 5")
            || combined.contains("reterminal")
        {
            return SiliconFamily::BroadcomVideoCoreArm;
        }
        if combined.contains("rk3588")
            || combined.contains("rk3576")
            || combined.contains("rk3568")
            || combined.contains("rockchip")
            || combined.contains("mali-g610")
            || combined.contains("rknpu")
        {
            return SiliconFamily::RockchipArmNpu;
        }
        if combined.contains("snapdragon x")
            || combined.contains("oryon")
            || combined.contains("x1e")
            || combined.contains("x1p")
        {
            return SiliconFamily::QualcommSnapdragonX;
        }
        if combined.contains("snapdragon")
            || combined.contains("sdm845")
            || combined.contains("sm8250")
            || combined.contains("sm8550")
            || combined.contains("adreno")
            || combined.contains("kryo")
            || combined.contains("hexagon")
        {
            return SiliconFamily::QualcommSnapdragonMobile;
        }
        if combined.contains("dimensity") || combined.contains("mediatek") || combined.contains("genio") {
            return SiliconFamily::MediaTekDimensityArm;
        }
        if combined.contains("tensor g") || combined.contains("edgetpu") || combined.contains("pixel") {
            return SiliconFamily::GoogleTensorArm;
        }
        if combined.contains("ampere altra")
            || combined.contains("ampereone")
            || combined.contains("graviton")
            || combined.contains("neoverse")
            || combined.contains("grace cpu")
        {
            return SiliconFamily::AmpereNeoverseArmServer;
        }

        // 7. Host CPU-primary Intel / AMD families (checked after discrete GPUs so a PC with Ryzen + RTX 4090 is classified by its primary accelerator or CPU)
        if combined.contains("core ultra") || combined.contains("ai boost") {
            return SiliconFamily::IntelCoreUltraAiBoost;
        }
        if combined.contains("ryzen") || combined.contains("epyc") || combined.contains("threadripper") {
            return SiliconFamily::AmdZenCpuServerDesktop;
        }
        if combined.contains("intel") || combined.contains("core i") || combined.contains("xeon") {
            return SiliconFamily::IntelCoreXeonAvx;
        }

        SiliconFamily::GenericMultiCore
    }

    /// Infers the `OperatingSystemFamily` of a `HardwareNode` from its device kind, backend, and signatures.
    pub fn classify_node_os(node: &HardwareNode, os_hint: Option<&str>) -> OperatingSystemFamily {
        if let Some(hint) = os_hint
            && !hint.trim().is_empty()
        {
            return OperatingSystemFamily::from_signature(hint);
        }

        let combined = format!(
            "{} {} {} {}",
            node.id.0.to_ascii_lowercase(),
            node.hostname.to_ascii_lowercase(),
            node.cpu_name.to_ascii_lowercase(),
            node.gpu_name.to_ascii_lowercase()
        );

        if combined.contains("ipad") || combined.contains("iphone") || combined.contains("ios") {
            return OperatingSystemFamily::IosIpadOs;
        }
        if node.device_kind == DeviceKind::AppleSiliconMac
            || node.backend == ComputeBackend::MetalUnified
        {
            return OperatingSystemFamily::MacOsDarwin;
        }
        if node.device_kind == DeviceKind::MobilePhone
            || node.device_kind == DeviceKind::PhoneFarmChassis
            || combined.contains("phonefarm")
            || combined.contains("pfb-slot")
            || combined.contains("galaxy")
            || combined.contains("android")
        {
            return OperatingSystemFamily::Android;
        }
        if combined.contains("k230") || combined.contains("canmv") || combined.contains("rt-smart") {
            return OperatingSystemFamily::RtSmartRtos;
        }
        if combined.contains("switch") && combined.contains("horizon") {
            return OperatingSystemFamily::HorizonOsL4t;
        }
        if combined.contains("truenas") || combined.contains("freebsd") {
            return OperatingSystemFamily::FreeBsd;
        }
        if node.device_kind == DeviceKind::LinuxSbc
            || combined.contains("linux")
            || combined.contains("reterminal")
            || combined.contains("cm4")
            || combined.contains("rk3588")
            || combined.contains("jetson")
            || combined.contains("switch")
            || combined.contains("h100")
            || combined.contains("burst")
        {
            return OperatingSystemFamily::Linux;
        }

        OperatingSystemFamily::detect_host_os()
    }

    /// Builds the complete `PlatformProfile` for any `HardwareNode`.
    pub fn classify_node(node: &HardwareNode) -> PlatformProfile {
        Self::classify_node_with_os_hint(node, None)
    }

    /// Builds the complete `PlatformProfile` for a `HardwareNode` with an optional OS signature hint.
    pub fn classify_node_with_os_hint(
        node: &HardwareNode,
        os_hint: Option<&str>,
    ) -> PlatformProfile {
        let silicon_family = Self::classify_silicon_family(
            &node.cpu_name,
            &node.gpu_name,
            node.npu_name.as_deref(),
            Some(&node.hostname),
        );
        let os_family = Self::classify_node_os(node, os_hint);
        let cpu_isa = CpuIsa::from_signatures(&node.cpu_name, &node.hostname);

        PlatformProfile {
            silicon_vendor: silicon_family.vendor(),
            silicon_family,
            silicon_display_name: silicon_family.display_name().to_string(),
            cpu_isa,
            os_family,
            os_display_name: os_family.display_name().to_string(),
            memory_topology: silicon_family.memory_topology(),
            primary_backend: node.backend,
            fallback_backends: silicon_family.fallback_backends(),
            os_hook_backend: os_family.os_hook_backend(),
            machine_agnostic_abstraction: true,
            compatible_form_factors: silicon_family
                .compatible_form_factors()
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            supported_os_families: silicon_family.supported_os_families().to_vec(),
        }
    }

    /// Returns the full compatibility matrix of all silicon families and operating systems
    /// supported by the Kadmium (`Cd-48`) platform abstraction layer.
    pub fn compatibility_matrix() -> Vec<PlatformMatrixEntry> {
        let families = [
            (SiliconFamily::AppleSiliconMSeries, CpuIsa::Aarch64),
            (SiliconFamily::AppleSiliconASeries, CpuIsa::Aarch64),
            (SiliconFamily::NvidiaBlackwell, CpuIsa::X86_64),
            (SiliconFamily::NvidiaHopperAda, CpuIsa::X86_64),
            (SiliconFamily::NvidiaAmpereTuring, CpuIsa::X86_64),
            (SiliconFamily::NvidiaPascalVoltaMaxwell, CpuIsa::X86_64),
            (SiliconFamily::NvidiaTegraOrinXavier, CpuIsa::Aarch64),
            (SiliconFamily::NvidiaTegraX1, CpuIsa::Aarch64),
            (SiliconFamily::AmdRdnaDiscrete, CpuIsa::X86_64),
            (SiliconFamily::AmdCdnaInstinct, CpuIsa::X86_64),
            (SiliconFamily::AmdRyzenAiUmaApu, CpuIsa::X86_64),
            (SiliconFamily::AmdZenCpuServerDesktop, CpuIsa::X86_64),
            (SiliconFamily::IntelArcXeDiscrete, CpuIsa::X86_64),
            (SiliconFamily::IntelCoreUltraAiBoost, CpuIsa::X86_64),
            (SiliconFamily::IntelCoreXeonAvx, CpuIsa::X86_64),
            (SiliconFamily::QualcommSnapdragonX, CpuIsa::Aarch64),
            (SiliconFamily::QualcommSnapdragonMobile, CpuIsa::Aarch64),
            (SiliconFamily::BroadcomVideoCoreArm, CpuIsa::Aarch64),
            (SiliconFamily::RockchipArmNpu, CpuIsa::Aarch64),
            (SiliconFamily::MediaTekDimensityArm, CpuIsa::Aarch64),
            (SiliconFamily::GoogleTensorArm, CpuIsa::Aarch64),
            (SiliconFamily::AmpereNeoverseArmServer, CpuIsa::Aarch64),
            (SiliconFamily::CanaanKendryteRiscV, CpuIsa::RiscV64),
            (SiliconFamily::SpacemiTRiscV, CpuIsa::RiscV64),
            (SiliconFamily::THeadXuantieRiscV, CpuIsa::RiscV64),
            (SiliconFamily::StarFiveRiscV, CpuIsa::RiscV64),
        ];

        families
            .into_iter()
            .map(|(family, isa)| PlatformMatrixEntry {
                silicon_vendor: family.vendor(),
                silicon_family: family,
                silicon_display_name: family.display_name().to_string(),
                cpu_isa: isa,
                primary_backend: family.primary_backend(),
                memory_topology: family.memory_topology(),
                supported_os_families: family.supported_os_families().to_vec(),
                compatible_form_factors: family
                    .compatible_form_factors()
                    .iter()
                    .map(|s| (*s).to_string())
                    .collect(),
            })
            .collect()
    }
}

impl HardwareNode {
    /// Returns the machine-agnostic `PlatformProfile` (`SiliconFamily` + `OperatingSystemFamily` + `CpuIsa`)
    /// for this node.
    pub fn platform_profile(&self) -> PlatformProfile {
        PlatformDetector::classify_node(self)
    }

    /// Returns `true` if `self` and `other` share the exact same `SiliconFamily`
    /// (`Same Silicon -> Supported by Kadmium`), regardless of OEM chassis or form factor.
    pub fn has_same_silicon_family(&self, other: &HardwareNode) -> bool {
        self.platform_profile().silicon_family == other.platform_profile().silicon_family
    }

    /// Returns `true` if `self` and `other` run the same `OperatingSystemFamily`
    /// (`Same Operating System -> Supported by Kadmium`).
    pub fn has_same_os_family(&self, other: &HardwareNode) -> bool {
        self.platform_profile().os_family == other.platform_profile().os_family
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AppleM4PlatformBridge, KendryteK230PlatformBridge, RpiCm4PlatformBridge,
        TegraX1PlatformController,
    };

    #[test]
    fn test_same_silicon_across_different_machines_maps_to_identical_platform_family() {
        // 1. Apple M4 across Mac mini, MacBook Air, MacBook Pro, iMac, Mac Studio, and iPad Pro
        let m4_products = AppleM4PlatformBridge::catalog_all_m4_products();
        assert!(m4_products.len() >= 6);
        let first_m4_profile = m4_products[0].2.platform_profile();
        assert_eq!(
            first_m4_profile.silicon_family,
            SiliconFamily::AppleSiliconMSeries
        );
        for (_, _, node) in &m4_products {
            assert!(first_m4_profile.silicon_family == node.platform_profile().silicon_family);
            assert!(m4_products[0].2.has_same_silicon_family(node));
        }

        // 2. Broadcom BCM2711 across Seeed reTerminal, Official CM4 IO Board, Turing Pi 2 Blade, Waveshare, uConsole
        let cm4_boards = RpiCm4PlatformBridge::catalog_all_cm4_platforms();
        assert!(cm4_boards.len() >= 5);
        for (_, node) in &cm4_boards {
            let prof = node.platform_profile();
            assert_eq!(prof.silicon_family, SiliconFamily::BroadcomVideoCoreArm);
            assert_eq!(prof.cpu_isa, CpuIsa::Aarch64);
            assert_eq!(prof.os_family, OperatingSystemFamily::Linux);
        }

        // 3. NVIDIA Tegra X1 / X1+ across Modded Switch (Mariko/Erista), Jetson Nano B01, and Shield TV Pro
        let tegra_boards = TegraX1PlatformController::catalog_all_tegra_x1_platforms();
        assert!(tegra_boards.len() >= 5);
        for (_, node) in &tegra_boards {
            let prof = node.platform_profile();
            assert_eq!(prof.silicon_family, SiliconFamily::NvidiaTegraX1);
            assert_eq!(prof.cpu_isa, CpuIsa::Aarch64);
        }

        // 4. Canaan Kendryte K230 across 01Studio, Canaan Official, K230D Zero, Banana Pi, Yahboom
        let k230_boards = KendryteK230PlatformBridge::catalog_all_k230_platforms();
        assert!(k230_boards.len() >= 5);
        for (_, node) in &k230_boards {
            let prof = node.platform_profile();
            assert_eq!(prof.silicon_family, SiliconFamily::CanaanKendryteRiscV);
            assert_eq!(prof.cpu_isa, CpuIsa::RiscV64);
        }
    }

    #[test]
    fn test_same_os_family_shares_unified_hooks_and_discovery() {
        let matrix = PlatformDetector::compatibility_matrix();
        assert!(matrix.len() >= 25);

        for os in [
            OperatingSystemFamily::Windows,
            OperatingSystemFamily::MacOsDarwin,
            OperatingSystemFamily::Linux,
            OperatingSystemFamily::Android,
            OperatingSystemFamily::IosIpadOs,
            OperatingSystemFamily::FreeBsd,
            OperatingSystemFamily::RtSmartRtos,
            OperatingSystemFamily::HorizonOsL4t,
        ] {
            assert!(!os.supported_isas().is_empty());
            assert!(!os.discovery_probes().is_empty());
            let hook = os.os_hook_backend();
            assert!(!hook.api_signature().is_empty());
        }
    }
}
