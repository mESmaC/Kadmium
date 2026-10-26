use kadmium_condenser::plugin::run_stdio_plugin;
use kadmium_condenser::{CondenserError, ToolResult};

fn main() {
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
