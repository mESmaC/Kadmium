//! Real file -> bounded RAM staging -> file transfer through KIP/1.
//! cargo run --locked -p kadmium-data --example interdevice_copy -- SOURCE NEW_DESTINATION

use kadmium_data::interdevice::*;
use std::fs::{File, OpenOptions};
use std::io;
use std::time::Instant;

fn main() -> io::Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "usage: interdevice_copy SOURCE NEW_DESTINATION (destination must not exist)",
        ));
    }
    // Explicit conservative priors, not measured performance. This adapter has only one
    // supported route; a GPU adapter supplies its actual directional connectivity and costs.
    let mut backend = HostBackend::new(
        "portable-buffered-v1",
        HostCosts {
            ram_bytes_per_second: 10e9,
            read_bytes_per_second: 1e9,
            write_bytes_per_second: 0.5e9,
            io_latency_ns: 100_000,
        },
    )?;
    let source = backend.register_file(File::open(&args[0])?, 0, false)?;
    let bytes = source.region().bytes;
    let destination_file = OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(&args[1])?;
    destination_file.set_len(bytes)?;
    let destination = backend.register_file(destination_file, 1, true)?;
    let mut engine = TransferEngine::new(backend, Limits::default())?;
    engine.submit(Request {
        version: PROTOCOL_VERSION,
        source: Span {
            region: source.region().id,
            offset: 0,
        },
        destination: Span {
            region: destination.region().id,
            offset: 0,
        },
        bytes,
        priority: Priority::Demand,
    })?;
    let started = Instant::now();
    while !engine.is_idle() {
        // Host copies complete synchronously. Native async adapters should drive progress
        // from device events/the worker loop instead of copying this polling loop.
        for completion in engine.progress() {
            match completion.outcome {
                Outcome::Complete => {}
                Outcome::Failed(error) => return Err(error),
                Outcome::Cancelled => {
                    return Err(io::Error::new(io::ErrorKind::Interrupted, "copy cancelled"));
                }
            }
        }
    }
    destination.sync_all()?;
    let stats = engine.stats();
    println!(
        "Copied {bytes} bytes in {:?}; {} physical bytes, {} operations, {} peak staging bytes (buffered I/O).",
        started.elapsed(),
        stats.physical_bytes,
        stats.operations,
        stats.peak_staging_bytes
    );
    Ok(())
}
