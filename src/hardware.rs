//! Hardware-based model auto-selection.
//!
//! On first run (when the user has not manually picked a model), NIC chooses a
//! model preset that fits the machine: small models on weak hardware, larger
//! ones where there's RAM/VRAM to spare. This keeps "download → run → works"
//! true out of the box instead of defaulting everyone to the medium model.

use tracing::info;

/// Detected hardware capability, used to pick a model preset.
#[derive(Debug, Clone, Copy)]
pub struct Hardware {
    /// Total system RAM in GB.
    pub ram_gb: f64,
    /// Largest dedicated GPU VRAM in GB (0.0 if no discrete GPU detected).
    pub vram_gb: f64,
}

impl Hardware {
    pub fn detect() -> Self {
        let ram_gb = total_ram_gb();
        let vram_gb = max_vram_gb();
        info!("[Hardware] RAM={:.1} GB, VRAM={:.1} GB", ram_gb, vram_gb);
        Self { ram_gb, vram_gb }
    }

    /// Picks the best-fitting preset id from the available ids.
    ///
    /// The model can run on the GPU (uses VRAM) *or* on the CPU (uses system
    /// RAM), and llama.cpp will use whichever path fits. So we compute a tier
    /// from each resource independently and take the BEST of the two — a machine
    /// with a modest GPU but plenty of RAM should still get a bigger model
    /// (it just runs on CPU), instead of being penalised for the weak GPU.
    ///
    /// Tier by VRAM: ≥6→large, ≥4→medium, else small.
    /// Tier by RAM:  ≥16→large, ≥8→medium, else small.
    /// Final tier = max(vram_tier, ram_tier).
    ///
    /// Falls back gracefully to whatever preset ids actually exist.
    pub fn recommended_model<'a>(&self, available: &'a [String]) -> Option<&'a str> {
        // Tier: 0 = small (0.5B), 1 = medium (1.5B), 2 = large (3B).
        let vram_tier = if self.vram_gb >= 6.0 { 2 } else if self.vram_gb >= 4.0 { 1 } else { 0 };
        let ram_tier  = if self.ram_gb  >= 16.0 { 2 } else if self.ram_gb  >= 8.0 { 1 } else { 0 };
        let tier = vram_tier.max(ram_tier);

        let want_large  = tier >= 2;
        let want_medium = tier >= 1;

        let large  = ["qwen-3b", "llama-3.2-3b"];
        let medium = ["qwen-1.5b"];
        let small  = ["qwen-0.5b"];

        let pick = |ids: &[&str]| -> Option<&'a str> {
            ids.iter()
                .find_map(|want| available.iter().find(|a| a.as_str() == *want))
                .map(String::as_str)
        };

        let chosen = if want_large {
            pick(&large).or_else(|| pick(&medium)).or_else(|| pick(&small))
        } else if want_medium {
            pick(&medium).or_else(|| pick(&small)).or_else(|| pick(&large))
        } else {
            pick(&small).or_else(|| pick(&medium)).or_else(|| pick(&large))
        };

        // Last resort: any available preset.
        chosen.or_else(|| available.first().map(String::as_str))
    }
}

fn total_ram_gb() -> f64 {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    // sysinfo reports bytes.
    sys.total_memory() as f64 / 1_073_741_824.0
}

/// Largest dedicated VRAM across GPU adapters, in GB. Returns 0.0 on failure
/// or when no discrete GPU is present.
#[cfg(windows)]
fn max_vram_gb() -> f64 {
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory1, IDXGIAdapter1, IDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE,
    };

    unsafe {
        let factory: IDXGIFactory1 = match CreateDXGIFactory1() {
            Ok(f) => f,
            Err(_) => return 0.0,
        };

        let mut best: u64 = 0;
        let mut i = 0u32;
        loop {
            let adapter: IDXGIAdapter1 = match factory.EnumAdapters1(i) {
                Ok(a) => a,
                Err(_) => break, // no more adapters
            };
            i += 1;

            // windows-rs 0.58: GetDesc1 takes no out-param and returns the desc.
            if let Ok(desc) = adapter.GetDesc1() {
                // Skip the software/WARP adapter — its "VRAM" is system RAM.
                let is_software =
                    (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) != 0;
                if !is_software {
                    best = best.max(desc.DedicatedVideoMemory as u64);
                }
            }
        }
        best as f64 / 1_073_741_824.0
    }
}

#[cfg(not(windows))]
fn max_vram_gb() -> f64 {
    // VRAM detection not implemented on non-Windows; rely on RAM tiering.
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hw(ram: f64, vram: f64) -> Hardware {
        Hardware { ram_gb: ram, vram_gb: vram }
    }
    fn ids() -> Vec<String> {
        vec!["qwen-0.5b".into(), "qwen-1.5b".into(), "qwen-3b".into(), "llama-3.2-3b".into()]
    }

    #[test]
    fn weak_machine_gets_small() {
        assert_eq!(hw(4.0, 0.0).recommended_model(&ids()), Some("qwen-0.5b"));
    }

    #[test]
    fn midrange_ram_gets_medium() {
        assert_eq!(hw(8.0, 0.0).recommended_model(&ids()), Some("qwen-1.5b"));
    }

    #[test]
    fn strong_ram_gets_large() {
        assert_eq!(hw(16.0, 0.0).recommended_model(&ids()), Some("qwen-3b"));
    }

    #[test]
    fn gpu_with_6gb_gets_large_even_on_low_ram() {
        assert_eq!(hw(8.0, 6.0).recommended_model(&ids()), Some("qwen-3b"));
    }

    #[test]
    fn discrete_4gb_gpu_gets_at_least_medium() {
        assert_eq!(hw(6.0, 4.0).recommended_model(&ids()), Some("qwen-1.5b"));
    }

    #[test]
    fn good_ram_modest_gpu_uses_ram_tier() {
        // Regression: 16 GB RAM + 3.9 GB VRAM must NOT collapse to small just
        // because the GPU is modest — RAM can run the bigger model on CPU.
        assert_eq!(hw(15.7, 3.9).recommended_model(&ids()), Some("qwen-1.5b"));
    }

    #[test]
    fn takes_best_of_ram_and_vram() {
        // Big GPU rescues a low-RAM machine; big RAM rescues a low-VRAM machine.
        assert_eq!(hw(4.0, 8.0).recommended_model(&ids()), Some("qwen-3b"));
        assert_eq!(hw(32.0, 1.0).recommended_model(&ids()), Some("qwen-3b"));
    }

    #[test]
    fn falls_back_to_available_when_preferred_missing() {
        let only_small = vec!["qwen-0.5b".to_string()];
        assert_eq!(hw(64.0, 24.0).recommended_model(&only_small), Some("qwen-0.5b"));
    }

    #[test]
    fn empty_list_returns_none() {
        assert_eq!(hw(16.0, 8.0).recommended_model(&[]), None);
    }
}
