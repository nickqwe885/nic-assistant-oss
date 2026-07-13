//! Local performance benchmark — model load time, VRAM, and tok/s for the
//! configured model + llama-server. Spawns llama-server with the SAME args the
//! app uses (GPU, ctx, parallel from config), measures, then always kills it.
//!
//! Run from the repo root once the model is present (first app run downloads it):
//!   cargo run --bin bench
//!
//! It sends HTTP only and shuts the server down at the end — it does not launch
//! the full NIC app and performs no OS automation.

use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

fn main() {
    let cfg = match nic_assistant_lib::config::load_config() {
        Ok(c) => c,
        Err(e) => { eprintln!("config load failed: {e}"); std::process::exit(1); }
    };
    let m = &cfg.models;
    let port: u16 = 8099; // off the app's 8090 to avoid clashing with a running app

    println!("=== NIC bench ===");
    println!("model:  {}", m.model_path.display());
    println!("server: {}", m.server_bin.display());
    println!("args:   --n-gpu-layers {} --ctx-size {} --parallel {}",
             m.n_gpu_layers, m.ctx_size, m.parallel);

    if !m.server_bin.exists() || !m.model_path.exists() {
        eprintln!("\nserver or model missing — run the app once to download them, then retry.");
        std::process::exit(1);
    }

    let load_start = Instant::now();
    let child = Command::new(&m.server_bin)
        .args([
            "--model",        &*m.model_path.to_string_lossy(),
            "--host",         "127.0.0.1",
            "--port",         &port.to_string(),
            "--n-gpu-layers", &m.n_gpu_layers.to_string(),
            "--ctx-size",     &m.ctx_size.to_string(),
            "--parallel",     &m.parallel.to_string(),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn();

    let mut child: Child = match child {
        Ok(c) => c,
        Err(e) => { eprintln!("failed to spawn llama-server: {e}"); std::process::exit(1); }
    };

    // Always kill the server, even if the measurement errors out.
    let result = run_bench(port, load_start);
    let _ = child.kill();
    let _ = child.wait();
    println!("\nllama-server stopped — VRAM freed.");

    if let Err(e) = result {
        eprintln!("bench error: {e}");
        std::process::exit(1);
    }
}

fn run_bench(port: u16, load_start: Instant) -> Result<(), String> {
    let agent  = ureq::AgentBuilder::new().timeout(Duration::from_secs(180)).build();
    let health = format!("http://127.0.0.1:{port}/health");

    // Poll until the model is loaded and the server answers /health.
    let mut ready = false;
    for _ in 0..240 {
        std::thread::sleep(Duration::from_millis(500));
        if agent.get(&health).timeout(Duration::from_secs(2)).call().is_ok() { ready = true; break; }
    }
    if !ready { return Err("llama-server never became ready (120s)".into()); }
    println!("\nMODEL LOAD → ready:  {:.1} s", load_start.elapsed().as_secs_f64());

    // VRAM snapshot (best-effort — needs nvidia-smi on PATH).
    if let Ok(out) = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used,memory.total", "--format=csv,noheader"])
        .output()
    {
        print!("VRAM used/total:     {}", String::from_utf8_lossy(&out.stdout));
    }

    // One generation pass → llama.cpp returns exact prompt/predict timings.
    let prompt = "<|im_start|>user\nExplain what a CPU does in two sentences.<|im_end|>\n<|im_start|>assistant\n";
    let body = serde_json::json!({
        "prompt": prompt, "n_predict": 128, "stream": false, "temperature": 0.3
    });
    let text = agent.post(&format!("http://127.0.0.1:{port}/completion"))
        .set("Content-Type", "application/json")
        .send_string(&body.to_string())
        .map_err(|e| format!("completion request: {e}"))?
        .into_string()
        .map_err(|e| format!("read body: {e}"))?;
    let json: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| format!("parse body: {e}"))?;

    let t = &json["timings"];
    let f = |k: &str| t[k].as_f64().unwrap_or(0.0);
    println!("prompt eval (TTFT):  {:.0} ms / {} tok  ({:.1} tok/s)",
             f("prompt_ms"), t["prompt_n"], f("prompt_per_second"));
    println!("generation:          {:.0} ms / {} tok  ({:.1} tok/s)",
             f("predicted_ms"), t["predicted_n"], f("predicted_per_second"));
    if let Some(c) = json["content"].as_str() {
        println!("answer preview:      {}", c.chars().take(140).collect::<String>());
    }
    Ok(())
}
