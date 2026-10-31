//! Intel AI Boost through the dynamically loaded Level Zero graph extension.
//! This module intentionally uses `Rc`: driver handles cannot escape to another
//! thread or outlive their loader, and kernel execution requires exclusive access.
mod ffi;
#[cfg(test)]
mod lifecycle_tests;

use crate::{
    Arithmetic, DeviceInfo, Error, MatMulKernel, MatMulShape, NpuBackend, Result, ir,
    validate_tensor,
};
use ffi::*;
use std::{ffi::CString, ptr, rc::Rc};

const TIMEOUT_NS: u64 = 30_000_000_000;

struct Context {
    api: Api,
    graph: GraphApi,
    build_log: Option<BuildLog>,
    initialization: Option<(GetProperties2, Destroy)>,
    device: Handle,
    handle: Handle,
    ordinal: u32,
    info: DeviceInfo,
}

impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: all kernels retain this context; the last reference drops only
        // after their queues, command lists, graphs and allocations are gone.
        unsafe {
            (self.api.context_destroy)(self.handle);
        }
    }
}

/// A real Intel NPU, never an inference from a CPU model or marketing name.
pub struct IntelNpu {
    context: Rc<Context>,
}

impl IntelNpu {
    /// Select the first Intel VPU with a compatible graph compiler and compute
    /// queue. Missing loaders, hardware and incompatible drivers are errors.
    pub fn open() -> Result<Self> {
        if !cfg!(all(target_pointer_width = "64", target_endian = "little")) {
            return Err(Error::Unsupported(
                "Intel NPU requires a 64-bit little-endian host".into(),
            ));
        }
        let api = Api::load()?;
        // SAFETY: pointers refer to initialized, correctly sized ABI structures;
        // enumeration buffers remain live for each call. Only loader-provided
        // handles are passed back, and counts are checked before indexing.
        unsafe {
            check((api.init)(2), "zeInit(VPU_ONLY)")?;
            let mut count = 0;
            check(
                (api.drivers)(&mut count, ptr::null_mut()),
                "zeDriverGet(count)",
            )?;
            let mut drivers = vec![ptr::null_mut(); count as usize];
            if drivers.is_empty() {
                return Err(Error::Unavailable("no Level Zero NPU drivers".into()));
            }
            check(
                (api.drivers)(&mut count, drivers.as_mut_ptr()),
                "zeDriverGet",
            )?;
            drivers.truncate(count as usize);
            for driver in drivers {
                let mut count = 0;
                check(
                    (api.extensions)(driver, &mut count, ptr::null_mut()),
                    "zeDriverGetExtensionProperties(count)",
                )?;
                let mut extensions = vec![
                    ExtensionProperties {
                        name: [0; 256],
                        version: 0
                    };
                    count as usize
                ];
                if extensions.is_empty() {
                    continue;
                }
                check(
                    (api.extensions)(driver, &mut count, extensions.as_mut_ptr()),
                    "zeDriverGetExtensionProperties",
                )?;
                extensions.truncate(count as usize);
                let extension = extensions
                    .iter()
                    .filter(|e| {
                        let n = name(&e.name);
                        (n == "ZE_extension_graph"
                            || n.strip_prefix("ZE_extension_graph_").is_some_and(|s| {
                                !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit() || b == b'_')
                            }))
                            && e.version >> 16 == 1
                    })
                    .max_by_key(|e| e.version);
                let Some(extension) = extension else {
                    continue;
                };
                let extension_name = CString::new(name(&extension.name)).expect("bounded C name");
                let mut address = ptr::null_mut();
                check(
                    (api.extension_address)(driver, extension_name.as_ptr(), &mut address),
                    "zeDriverGetExtensionFunctionAddress",
                )?;
                if address.is_null() {
                    return Err(Error::Unsupported("null graph extension table".into()));
                }
                // The advertised version guarantees at least the 1.0 prefix.
                let graph = (*address.cast::<GraphApi>()).validate()?;
                let build_log = if extension.version >= 0x10004 {
                    // Graph 1.4 appends BuildLogGetString at pointer slot 15.
                    *address.cast::<Option<BuildLog>>().add(15)
                } else {
                    None
                };
                let initialization = if extension.version >= 0x10008 {
                    // Version 1.8 adds GetProperties2 and GraphInitialize at
                    // slots 21 and 22. Both host and command-list stages may be
                    // required, depending on device generation and graph.
                    let properties = *address.cast::<Option<GetProperties2>>().add(21);
                    let initialize = *address.cast::<Option<Destroy>>().add(22);
                    Some(properties.zip(initialize).ok_or_else(|| {
                        Error::Unsupported("incomplete Intel graph 1.8 initialization table".into())
                    })?)
                } else {
                    None
                };
                let mut count = 0;
                check(
                    (api.devices)(driver, &mut count, ptr::null_mut()),
                    "zeDeviceGet(count)",
                )?;
                let mut devices = vec![ptr::null_mut(); count as usize];
                if devices.is_empty() {
                    continue;
                }
                check(
                    (api.devices)(driver, &mut count, devices.as_mut_ptr()),
                    "zeDeviceGet",
                )?;
                devices.truncate(count as usize);
                for device in devices {
                    // These ABI structs contain only integers, raw pointers and
                    // byte arrays, for which zero is a valid initial bit pattern.
                    let mut props: DeviceProperties = std::mem::zeroed();
                    props.header = Header::new(3);
                    check(
                        (api.device_properties)(device, &mut props),
                        "zeDeviceGetProperties",
                    )?;
                    if props.vendor != 0x8086 || props.kind != 5 {
                        continue;
                    }
                    let mut gp: DeviceGraphProperties = std::mem::zeroed();
                    gp.header = Header::new(1);
                    check(
                        graph.device_properties.unwrap()(device, &mut gp),
                        "zeDeviceGetGraphProperties",
                    )?;
                    if gp.formats & 2 == 0 || gp.max_opset < 1 {
                        continue;
                    }
                    let mut count = 0;
                    check(
                        (api.queue_properties)(device, &mut count, ptr::null_mut()),
                        "zeDeviceGetCommandQueueGroupProperties(count)",
                    )?;
                    let mut groups = vec![
                        QueueProperties {
                            header: Header::new(6),
                            flags: 0,
                            max_fill: 0,
                            count: 0
                        };
                        count as usize
                    ];
                    if groups.is_empty() {
                        continue;
                    }
                    check(
                        (api.queue_properties)(device, &mut count, groups.as_mut_ptr()),
                        "zeDeviceGetCommandQueueGroupProperties",
                    )?;
                    groups.truncate(count as usize);
                    let Some(ordinal) = groups.iter().position(|g| g.flags & 1 != 0 && g.count > 0)
                    else {
                        continue;
                    };
                    let desc = FlagsDesc {
                        header: Header::new(0xd),
                        flags: 0,
                    };
                    let mut handle = ptr::null_mut();
                    check(
                        (api.context_create)(driver, &desc, &mut handle),
                        "zeContextCreate",
                    )?;
                    return Ok(Self {
                        context: Rc::new(Context {
                            api,
                            graph,
                            build_log,
                            initialization,
                            device,
                            handle,
                            ordinal: ordinal as u32,
                            info: DeviceInfo {
                                name: name(&props.name),
                                vendor_id: props.vendor,
                                device_id: props.device,
                                graph_extension_version: extension.version,
                                compiler_version: (gp.compiler_major, gp.compiler_minor),
                                arithmetic: Arithmetic::ApproximateFp16,
                            },
                        }),
                    });
                }
            }
        }
        Err(Error::Unavailable(
            "no compatible Intel NPU graph compiler and compute queue".into(),
        ))
    }
}

impl NpuBackend for IntelNpu {
    type Kernel = IntelMatMul;
    fn device(&self) -> &DeviceInfo {
        &self.context.info
    }
    fn prepare_matmul(&self, shape: MatMulShape, weights: &[f32]) -> Result<IntelMatMul> {
        validate_tensor(weights, shape.weights_len())?;
        let mut kernel = IntelMatMul {
            context: Rc::clone(&self.context),
            shape,
            graph: ptr::null_mut(),
            queue: ptr::null_mut(),
            list: ptr::null_mut(),
            input: ptr::null_mut(),
            output: ptr::null_mut(),
            payload: ir::matmul(shape, weights, self.context.info.compiler_version),
            in_flight: false,
            poisoned: false,
        };
        kernel.prepare()?;
        Ok(kernel)
    }
}

/// Compiled MatMul with immutable weights and reusable host-visible tensors.
/// Driver resources are released in reverse dependency order, including on
/// partial preparation failure. A failed wait poisons the kernel permanently.
pub struct IntelMatMul {
    context: Rc<Context>,
    shape: MatMulShape,
    graph: Handle,
    queue: Handle,
    list: Handle,
    input: Handle,
    output: Handle,
    // Retained for the full graph lifetime, as required by older drivers.
    payload: Vec<u8>,
    in_flight: bool,
    poisoned: bool,
}

impl IntelMatMul {
    fn prepare(&mut self) -> Result<()> {
        let c = &self.context;
        let mut init_stages = 1;
        let flags = c"--inputs_precisions=\"input:FP32\" --inputs_layouts=\"input:NC\" --outputs_precisions=\"matmul:FP32\" --outputs_layouts=\"matmul:NC\" --config PERFORMANCE_HINT=\"LATENCY\"";
        let desc = GraphDesc {
            header: Header::new(2),
            format: 2,
            size: self.payload.len(),
            input: self.payload.as_ptr(),
            flags: flags.as_ptr(),
        };
        // SAFETY: descriptors and payload live through creation (payload through
        // destruction); every successful creation is immediately owned by self.
        // Graph argument metadata is validated before allocating or binding I/O.
        unsafe {
            let status = c.graph.create.unwrap()(c.handle, c.device, &desc, &mut self.graph);
            if status != 0 {
                let mut log = String::from("driver did not provide a build log");
                if let Some(get_log) = c.build_log {
                    let mut size = 0;
                    if get_log(self.graph, &mut size, ptr::null_mut()) == 0
                        && size > 0
                        && size <= 1024 * 1024
                    {
                        let mut bytes = vec![0u8; size as usize];
                        if get_log(self.graph, &mut size, bytes.as_mut_ptr().cast()) == 0 {
                            log = name(&bytes);
                        }
                    }
                }
                return Err(Error::Compilation { code: status, log });
            }
            let mut props = GraphProperties {
                header: Header::new(3),
                count: 0,
            };
            check(
                c.graph.properties.unwrap()(self.graph, &mut props),
                "zeGraphGetProperties",
            )?;
            if let Some((properties, _)) = c.initialization {
                let mut props2 = GraphProperties2 {
                    header: Header::new(0x10),
                    count: 0,
                    init_stages: 0,
                };
                check(properties(self.graph, &mut props2), "zeGraphGetProperties2")?;
                init_stages = props2.init_stages;
                if init_stages & !3 != 0 || props2.count != props.count {
                    return Err(Error::Unsupported(
                        "unsupported graph initialization metadata".into(),
                    ));
                }
            }
            if props.count != 2 {
                return Err(Error::Unsupported(format!(
                    "expected 2 graph arguments, got {}",
                    props.count
                )));
            }
            let mut indices = [None, None];
            for index in 0..props.count {
                let mut arg: ArgumentProperties = std::mem::zeroed();
                arg.header = Header::new(4);
                check(
                    c.graph.argument.unwrap()(self.graph, index, &mut arg),
                    "zeGraphGetArgumentProperties",
                )?;
                let slot = validate_argument(&arg, self.shape)?;
                if indices[slot].replace(index).is_some() {
                    return Err(Error::Unsupported("duplicate graph input or output".into()));
                }
            }
            let (Some(input_index), Some(output_index)) = (indices[0], indices[1]) else {
                return Err(Error::Unsupported("missing graph input or output".into()));
            };
            let alloc = FlagsDesc {
                header: Header::new(0x16),
                flags: 0,
            };
            check(
                (c.api.alloc_host)(
                    c.handle,
                    &alloc,
                    self.shape.input_len() * 4,
                    64,
                    &mut self.input,
                ),
                "zeMemAllocHost(input)",
            )?;
            check(
                (c.api.alloc_host)(
                    c.handle,
                    &alloc,
                    self.shape.output_len() * 4,
                    64,
                    &mut self.output,
                ),
                "zeMemAllocHost(output)",
            )?;
            if self.input.is_null() || self.output.is_null() {
                return Err(Error::Unsupported(
                    "driver returned null tensor allocation".into(),
                ));
            }
            // Initialize all output bytes so even a faulty successful driver
            // cannot expose uninitialized host memory through the safe Rust API.
            ptr::write_bytes(self.output.cast::<u8>(), 0, self.shape.output_len() * 4);
            check(
                c.graph.set_argument.unwrap()(self.graph, input_index, self.input),
                "zeGraphSetArgumentValue(input)",
            )?;
            check(
                c.graph.set_argument.unwrap()(self.graph, output_index, self.output),
                "zeGraphSetArgumentValue(output)",
            )?;
            let queue_desc = QueueDesc {
                header: Header::new(0xe),
                ordinal: c.ordinal,
                index: 0,
                flags: 0,
                mode: 2,
                priority: 0,
            };
            check(
                (c.api.queue_create)(c.handle, c.device, &queue_desc, &mut self.queue),
                "zeCommandQueueCreate",
            )?;
            let list_desc = ListDesc {
                header: Header::new(0xf),
                ordinal: c.ordinal,
                flags: 0,
            };
            check(
                (c.api.list_create)(c.handle, c.device, &list_desc, &mut self.list),
                "zeCommandListCreate",
            )?;
            if init_stages & 2 != 0 {
                check(c.initialization.unwrap().1(self.graph), "zeGraphInitialize")?;
            }
            if init_stages & 1 != 0 {
                check(
                    c.graph.initialize.unwrap()(
                        self.list,
                        self.graph,
                        ptr::null_mut(),
                        0,
                        ptr::null(),
                    ),
                    "zeCommandListAppendGraphInitialize",
                )?;
            }
        }
        if init_stages & 1 != 0 {
            self.submit_and_wait()?;
            // SAFETY: initialization completed, so this command list is idle.
            unsafe {
                check(
                    (self.context.api.list_reset)(self.list),
                    "zeCommandListReset",
                )?;
            }
        }
        // SAFETY: initialization completed, so the list may be reset and reused
        // for the immutable execution command with stable argument allocations.
        unsafe {
            check(
                self.context.graph.execute.unwrap()(
                    self.list,
                    self.graph,
                    ptr::null_mut(),
                    ptr::null_mut(),
                    0,
                    ptr::null(),
                ),
                "zeCommandListAppendGraphExecute",
            )?;
            check(
                (self.context.api.list_close)(self.list),
                "zeCommandListClose(execute)",
            )?;
        }
        Ok(())
    }

    fn submit_and_wait(&mut self) -> Result<()> {
        // SAFETY: during initialization only, the list is complete but not closed.
        unsafe {
            check(
                (self.context.api.list_close)(self.list),
                "zeCommandListClose(initialize)",
            )?;
        }
        self.run_closed_list()
    }

    fn run_closed_list(&mut self) -> Result<()> {
        self.in_flight = true;
        self.poisoned = true;
        // SAFETY: list, graph, queue, allocations and context are retained until
        // successful completion; a failed submission is treated as potentially
        // in flight and cannot lead to buffer reuse or premature deallocation.
        unsafe {
            check(
                (self.context.api.queue_execute)(self.queue, 1, &self.list, ptr::null_mut()),
                "zeCommandQueueExecuteCommandLists",
            )?;
            check(
                (self.context.api.queue_sync)(self.queue, TIMEOUT_NS),
                "zeCommandQueueSynchronize (30s)",
            )?;
        }
        self.in_flight = false;
        self.poisoned = false;
        Ok(())
    }
}

impl MatMulKernel for IntelMatMul {
    fn shape(&self) -> MatMulShape {
        self.shape
    }
    fn arithmetic(&self) -> Arithmetic {
        Arithmetic::ApproximateFp16
    }
    fn execute(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        if self.poisoned {
            return Err(Error::Poisoned);
        }
        validate_tensor(input, self.shape.input_len())?;
        // SAFETY: idle, exclusive kernel; input is validated and the allocation
        // has exactly input_len initialized f32 elements after this copy.
        unsafe {
            ptr::copy_nonoverlapping(input.as_ptr(), self.input.cast::<f32>(), input.len());
        }
        self.run_closed_list()?;
        // SAFETY: a successful queue wait makes output host-readable; allocation
        // is aligned, initialized, correctly sized and retained during the copy.
        let output = unsafe {
            std::slice::from_raw_parts(self.output.cast::<f32>(), self.shape.output_len()).to_vec()
        };
        if output.iter().any(|v| !v.is_finite()) {
            return Err(Error::NonFiniteOutput);
        }
        Ok(output)
    }
}

impl Drop for IntelMatMul {
    fn drop(&mut self) {
        let c = &self.context;
        // SAFETY: never free anything potentially referenced by device work.
        // If a final bounded wait fails, quarantine the entire native resource
        // graph for process lifetime, including the loader and IR backing bytes.
        // This deliberate leak is preferable to a DMA use-after-free or hanging
        // indefinitely in Drop. Normal and partial-build paths release resources.
        unsafe {
            if self.in_flight && (c.api.queue_sync)(self.queue, TIMEOUT_NS) != 0 {
                std::mem::forget(Rc::clone(c));
                std::mem::forget(std::mem::take(&mut self.payload));
                return;
            }
            if !self.list.is_null() {
                (c.api.list_destroy)(self.list);
            }
            if !self.queue.is_null() {
                (c.api.queue_destroy)(self.queue);
            }
            if !self.graph.is_null() {
                c.graph.destroy.unwrap()(self.graph);
            }
            if !self.input.is_null() {
                (c.api.free)(c.handle, self.input);
            }
            if !self.output.is_null() {
                (c.api.free)(c.handle, self.output);
            }
        }
    }
}

fn validate_argument(arg: &ArgumentProperties, shape: MatMulShape) -> Result<usize> {
    let columns = match arg.kind {
        0 => shape.k(),
        1 => shape.n(),
        _ => {
            return Err(Error::Unsupported(
                "unknown graph argument direction".into(),
            ));
        }
    };
    // NC is the explicitly requested contiguous row-major external ABI.
    // Trailing dimensions in old graph metadata are zero or singleton padding.
    if arg.device_precision != 1
        || arg.device_layout != 0xc1
        || arg.dims[0] as usize != shape.m()
        || arg.dims[1] as usize != columns
        || arg.dims[2..].iter().any(|&d| d > 1)
    {
        return Err(Error::Unsupported(format!(
            "graph argument {} has unsupported precision/layout/dimensions: {}/{}/{:?}",
            name(&arg.name),
            arg.device_precision,
            arg.device_layout,
            arg.dims
        )));
    }
    Ok(arg.kind as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_device_metadata_that_could_overrun_host_buffers() {
        let shape = MatMulShape::new(2, 3, 4).unwrap();
        let mut arg = ArgumentProperties {
            header: Header::new(4),
            name: [0; 256],
            kind: 0,
            dims: [2, 3, 1, 1, 1],
            network_precision: 1,
            network_layout: 0xc1,
            device_precision: 1,
            device_layout: 0xc1,
        };
        assert_eq!(validate_argument(&arg, shape).unwrap(), 0);
        arg.dims[2] = 2;
        assert!(validate_argument(&arg, shape).is_err());
        arg.dims[2] = 1;
        arg.device_precision = 2;
        assert!(validate_argument(&arg, shape).is_err());
        arg.device_precision = 1;
        arg.device_layout = 0xc2;
        assert!(validate_argument(&arg, shape).is_err());
        arg.device_layout = 0xc1;
        arg.kind = 1;
        assert!(validate_argument(&arg, shape).is_err());
        arg.dims[1] = 4;
        assert_eq!(validate_argument(&arg, shape).unwrap(), 1);
    }
}
