# CodeBlock

Commands and source in JetBrains Mono on a `graphite-2` well, with a header naming the file or `terminal` and a copy button. The built-in highlighter knows shell (prompt, flags, comments, strings), Rust, TOML and JSON; it colours keywords `code-keyword`, strings `code-string`, numbers `code-number` and comments `code-comment`.

You provide `code` (with `$ ` prompts on shell lines — the prompt is not copied by selection), `language`, an optional `title` and `copy={false}` to hide the button. Long lines scroll; never wrap code. For a single inline command in prose, use a `code` span, not this block.
