//! Coalesced, aligned, pooled reads of many byte ranges from one file, after InferSSD's expert
//! stream (`engine/ggml/src/ggml-ssd/stream/copy-plan.h`, `io/submit-window.h`,
//! `io/pool-ring.cpp`): contiguous and near-contiguous ranges become one read, every read is
//! aligned to the drive's sector so a direct read can serve it, reads are submitted in chunks
//! sized by how many there are, and a small pool of reader threads keeps several in flight.
//!
//! The planner is pure: it maps the caller's `(offset, len)` ranges onto merged reads and
//! remembers which part of which read each range is. The reader runs the plan with one file
//! handle per thread. Nothing about tensors or `.ktb` pages lives here; the safetensors
//! manifest and the Hugging Face range slicer call the planner with their own ranges.

use std::fs::File;
use std::io;
use std::path::Path;

/// Reads are aligned to this: the sector size a direct read needs (InferSSD's `4096`).
pub const READ_ALIGNMENT_BYTES: u64 = 4096;
/// Ranges this close are merged into one read: one `.ktb` page, which is also about what a
/// NVMe drive hands over per request before latency dominates.
pub const DEFAULT_COALESCE_GAP_BYTES: u64 = 65_536;
/// InferSSD's pool reader keeps at most eight reader threads.
pub const MAX_POOL_READERS: usize = 8;
/// A merged read never grows past this, so one request's buffer stays bounded however many
/// adjacent tensors a shard packs together.
pub const MAX_MERGED_READ_BYTES: u64 = 64 << 20;

/// One caller range as a part of a merged read: `index` is its position in the caller's list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadPart {
    pub index: usize,
    pub offset: u64,
    pub len: u64,
}

/// One read to issue: an aligned span of the file and the caller ranges it serves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergedRead {
    pub offset: u64,
    pub len: u64,
    pub parts: Vec<ReadPart>,
}

impl MergedRead {
    pub fn end(&self) -> u64 {
        self.offset + self.len
    }

    /// Bytes the caller asked for inside this read.
    pub fn useful_bytes(&self) -> u64 {
        self.parts.iter().map(|p| p.len).sum()
    }
}

/// What a plan saves, for a trace line.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReadPlanStats {
    pub ranges: usize,
    pub reads: usize,
    pub bytes_requested: u64,
    pub bytes_read: u64,
}

impl ReadPlanStats {
    /// Bytes read that no range asked for: alignment padding and merged gaps.
    pub fn waste_bytes(&self) -> u64 {
        self.bytes_read.saturating_sub(self.bytes_requested)
    }
}

/// The merged, aligned reads for `ranges` (`(offset, len)` pairs in any order): ranges whose
/// aligned spans overlap or sit within `max_gap` of each other become one read, in file order,
/// as long as the read stays under [`MAX_MERGED_READ_BYTES`]. Empty ranges are kept as parts
/// of length 0 so every index has an answer.
pub fn plan_reads(ranges: &[(u64, u64)], max_gap: u64, alignment: u64) -> Vec<MergedRead> {
    let alignment = alignment.max(1);
    let align_down = |v: u64| v / alignment * alignment;
    let align_up = |v: u64| v.div_ceil(alignment) * alignment;
    let mut order: Vec<usize> = (0..ranges.len()).collect();
    order.sort_by_key(|&i| (ranges[i].0, ranges[i].1));

    let mut reads: Vec<MergedRead> = Vec::new();
    for index in order {
        let (offset, len) = ranges[index];
        let part = ReadPart { index, offset, len };
        let start = align_down(offset);
        let end = align_up(offset.saturating_add(len));
        match reads.last_mut() {
            Some(last)
                if start <= last.end().saturating_add(max_gap)
                    && end.max(last.end()) - last.offset
                        <= MAX_MERGED_READ_BYTES.max(end - start) =>
            {
                last.len = end.max(last.end()) - last.offset;
                last.parts.push(part);
            }
            _ => reads.push(MergedRead {
                offset: start,
                len: end - start,
                parts: vec![part],
            }),
        }
    }
    reads
}

pub fn plan_stats(ranges: &[(u64, u64)], reads: &[MergedRead]) -> ReadPlanStats {
    ReadPlanStats {
        ranges: ranges.len(),
        reads: reads.len(),
        bytes_requested: ranges.iter().map(|r| r.1).sum(),
        bytes_read: reads.iter().map(|r| r.len).sum(),
    }
}

/// How many reads to submit together: InferSSD measured chunks of four best for five to eight
/// misses, smaller batches together, and chunks of two beyond that.
pub fn adaptive_submit_chunk(jobs: usize) -> usize {
    if jobs <= 4 {
        jobs.max(1)
    } else if jobs <= 8 {
        4
    } else {
        2
    }
}

/// Reader threads for `reads` reads at a queue depth: `min(depth, reads, 8)`.
pub fn pool_readers(reads: usize, depth: usize) -> usize {
    depth.min(reads).clamp(1, MAX_POOL_READERS)
}

#[cfg(unix)]
fn read_at(file: &File, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
    use std::os::unix::fs::FileExt;
    file.read_at(buf, offset)
}

#[cfg(windows)]
fn read_at(file: &File, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
    use std::os::windows::fs::FileExt;
    file.seek_read(buf, offset)
}

#[cfg(not(any(unix, windows)))]
fn read_at(file: &File, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = file;
    f.seek(SeekFrom::Start(offset))?;
    f.read(buf)
}

/// Fills `buf` from `offset`, allowing a short count only at end of file (InferSSD's
/// `expected_bytes` rule: a read past the end completes with what the file has). Returns the
/// bytes read.
fn read_fully_at(file: &File, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
    let mut done = 0;
    while done < buf.len() {
        let n = read_at(file, offset + done as u64, &mut buf[done..])?;
        if n == 0 {
            break;
        }
        done += n;
    }
    Ok(done)
}

/// The outcome of one merged read.
#[derive(Debug)]
struct ReadResult {
    read: usize,
    bytes: Vec<u8>,
    got: usize,
}

/// Runs a plan against one file with a pool of reader threads, each with its own handle, and
/// returns one buffer per caller range, in the caller's order. A range that reaches past the
/// end of the file is an error; a read whose padding alone passes the end is not.
pub fn read_planned(path: &Path, reads: &[MergedRead], depth: usize) -> io::Result<Vec<Vec<u8>>> {
    let total_parts: usize = reads.iter().map(|r| r.parts.len()).sum();
    let mut out: Vec<Option<Vec<u8>>> = (0..total_parts).map(|_| None).collect();
    if reads.is_empty() {
        return Ok(Vec::new());
    }
    let readers = pool_readers(reads.len(), depth);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let chunk = adaptive_submit_chunk(reads.len());
    let results: Vec<io::Result<ReadResult>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..readers)
            .map(|_| {
                let next = &next;
                scope.spawn(move || -> io::Result<Vec<ReadResult>> {
                    let file = File::open(path)?;
                    let mut mine = Vec::new();
                    loop {
                        // Claim a chunk of reads at a time, as InferSSD submits them.
                        let first = next.fetch_add(chunk, std::sync::atomic::Ordering::Relaxed);
                        if first >= reads.len() {
                            break;
                        }
                        let last = (first + chunk).min(reads.len());
                        for (index, read) in reads.iter().enumerate().take(last).skip(first) {
                            let mut bytes: Vec<u8> = Vec::new();
                            bytes.try_reserve_exact(read.len as usize).map_err(|_| {
                                io::Error::new(
                                    io::ErrorKind::OutOfMemory,
                                    format!("cannot allocate {} bytes for a read", read.len),
                                )
                            })?;
                            bytes.resize(read.len as usize, 0);
                            let got = read_fully_at(&file, read.offset, &mut bytes)?;
                            mine.push(ReadResult {
                                read: index,
                                bytes,
                                got,
                            });
                        }
                    }
                    Ok(mine)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| match h.join() {
                Ok(Ok(results)) => results.into_iter().map(Ok).collect::<Vec<_>>(),
                Ok(Err(e)) => vec![Err(e)],
                Err(_) => vec![Err(io::Error::other("reader thread panicked"))],
            })
            .collect::<Vec<Vec<_>>>()
            .into_iter()
            .flatten()
            .collect()
    });
    let mut parts_seen = 0;
    for result in results {
        let result = result?;
        let read = &reads[result.read];
        for part in &read.parts {
            let start = (part.offset - read.offset) as usize;
            let end = start + part.len as usize;
            if end > result.got {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!(
                        "range at {} of {} bytes passes the end of {}",
                        part.offset,
                        part.len,
                        path.display()
                    ),
                ));
            }
            out[part.index] = Some(result.bytes[start..end].to_vec());
            parts_seen += 1;
        }
    }
    if parts_seen != total_parts {
        return Err(io::Error::other("a planned read produced no result"));
    }
    Ok(out.into_iter().map(|b| b.unwrap_or_default()).collect())
}

/// Plans and runs the reads for `ranges` from `path` with the default gap and alignment.
pub fn read_ranges(
    path: &Path,
    ranges: &[(u64, u64)],
    depth: usize,
) -> io::Result<(Vec<Vec<u8>>, ReadPlanStats)> {
    let reads = plan_reads(ranges, DEFAULT_COALESCE_GAP_BYTES, READ_ALIGNMENT_BYTES);
    let stats = plan_stats(ranges, &reads);
    Ok((read_planned(path, &reads, depth)?, stats))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT: AtomicU64 = AtomicU64::new(0);

    fn temp_file(bytes: &[u8]) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "kadmium-batch-read-{}-{}.bin",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn contiguous_and_near_ranges_merge_into_aligned_reads_in_file_order() {
        // Three tensors back to back, one far away, one between but within the gap.
        let ranges = [
            (100_000, 500),
            (4_100, 100),
            (4_200, 300),
            (10_000, 10),
            (4_096, 4),
        ];
        let reads = plan_reads(&ranges, 1024, 4096);
        assert_eq!(reads.len(), 2, "{reads:?}");
        assert_eq!(reads[0].offset, 4096);
        assert_eq!(reads[0].end(), 12_288);
        assert_eq!(reads[0].parts.len(), 4);
        assert_eq!(reads[0].parts[0].index, 4, "the lowest offset first");
        assert_eq!(reads[1].offset, 98_304);
        assert_eq!(reads[1].parts[0].index, 0);
        let stats = plan_stats(&ranges, &reads);
        assert_eq!(stats.ranges, 5);
        assert_eq!(stats.reads, 2);
        assert_eq!(stats.bytes_requested, 914);
        assert!(stats.waste_bytes() > 0);

        // Without a gap and without alignment, only touching ranges merge.
        let tight = plan_reads(&[(0, 10), (10, 10), (21, 10)], 0, 1);
        assert_eq!(tight.len(), 2);
        assert_eq!(tight[0].len, 20);
        assert!(plan_reads(&[], 0, 1).is_empty());
        // A merge never grows a read past the cap; a single range larger than the cap is
        // still one read.
        let half = MAX_MERGED_READ_BYTES / 2 + 1;
        let capped = plan_reads(&[(0, half), (half, half), (2 * half, 10)], 0, 1);
        assert_eq!(capped.len(), 2, "{capped:?}");
        assert_eq!(capped[0].len, half);
        let huge = plan_reads(&[(0, 3 * MAX_MERGED_READ_BYTES)], 0, 1);
        assert_eq!(huge.len(), 1);
    }

    #[test]
    fn submit_chunks_and_pool_sizes_follow_the_measured_rules() {
        assert_eq!(adaptive_submit_chunk(0), 1);
        assert_eq!(adaptive_submit_chunk(3), 3);
        assert_eq!(adaptive_submit_chunk(4), 4);
        assert_eq!(adaptive_submit_chunk(6), 4);
        assert_eq!(adaptive_submit_chunk(9), 2);
        assert_eq!(pool_readers(100, 32), 8);
        assert_eq!(pool_readers(3, 32), 3);
        assert_eq!(pool_readers(0, 32), 1);
    }

    #[test]
    fn pooled_reads_return_each_range_in_order_and_fail_past_the_end() {
        let data: Vec<u8> = (0..20_000u32).map(|i| (i % 251) as u8).collect();
        let path = temp_file(&data);
        let ranges = [
            (19_990, 10),
            (0, 5),
            (4_000, 3_000),
            (7_000, 100),
            (12_000, 1),
        ];
        let (out, stats) = read_ranges(&path, &ranges, 4).unwrap();
        assert_eq!(out.len(), ranges.len());
        for (i, (offset, len)) in ranges.iter().enumerate() {
            let (o, l) = (*offset as usize, *len as usize);
            assert_eq!(out[i], &data[o..o + l], "range {i}");
        }
        // 19_990..20_000 is padded to 20_480 by alignment: past the end, but the part fits.
        assert!(stats.bytes_read > stats.bytes_requested);
        let past = read_ranges(&path, &[(19_995, 10)], 2).unwrap_err();
        assert_eq!(past.kind(), io::ErrorKind::UnexpectedEof);
        // An empty range is fine.
        let (empty, _) = read_ranges(&path, &[(5, 0)], 1).unwrap();
        assert!(empty[0].is_empty());
        std::fs::remove_file(path).unwrap();
    }
}
