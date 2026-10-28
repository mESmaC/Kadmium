use kadmium_condenser::plugin::run_stdio_plugin;
use kadmium_condenser::{CondenserError, ToolResult};
use std::io::{BufRead, Write};
use std::time::Duration;

fn main() {
    match std::env::args().nth(1).as_deref() {
        Some("hang-after-response") => run_faulty_plugin(false),
        Some("oversized-line") => run_faulty_plugin(true),
        _ => run_echo_plugin(),
    }
}

fn run_echo_plugin() {
    if let Err(error) = run_stdio_plugin("condenser-echo", |name, arguments| {
        if name != "echo" {
            return Ok(ToolResult::failure(format!("unknown tool `{name}`")));
        }
        let text = arguments["text"]
            .as_str()
            .ok_or_else(|| CondenserError::Plugin("echo needs a string `text` argument".into()))?;
        Ok(ToolResult::success(text))
    }) {
        eprintln!("condenser-echo: {error}");
        std::process::exit(1);
    }
}

fn run_faulty_plugin(oversized: bool) {
    let mut request = String::new();
    std::io::stdin().lock().read_line(&mut request).unwrap();
    let mut stdout = std::io::stdout().lock();
    if oversized {
        let chunk = [b'x'; 8 * 1024];
        for _ in 0..=1024 {
            stdout.write_all(&chunk).unwrap();
        }
    } else {
        stdout
            .write_all(
                b"{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"content\":\"completed\",\"is_error\":false}}\n",
            )
            .unwrap();
    }
    stdout.flush().unwrap();
    loop {
        std::thread::sleep(Duration::from_secs(60));
    }
}
