//! Minimal Level Zero 1.0 / Intel graph extension 1.0 ABI.
//! Sources: oneapi-src/level-zero include/ze_api.h and
//! intel/level-zero-npu-extensions ze_graph_ext.h. No generated C build step.
use crate::{Error, Result};
use libloading::Library;
use std::ffi::{c_char, c_void};

pub(super) type Handle = *mut c_void;
pub(super) type Status = u32;
pub(super) type Destroy = unsafe extern "system" fn(Handle) -> Status;
pub(super) type BuildLog = unsafe extern "system" fn(Handle, *mut u32, *mut c_char) -> Status;
pub(super) type GetProperties2 = unsafe extern "system" fn(Handle, *mut GraphProperties2) -> Status;

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct Header {
    pub stype: u32,
    pub next: *const c_void,
}
impl Header {
    pub fn new(stype: u32) -> Self {
        Self {
            stype,
            next: std::ptr::null(),
        }
    }
}
#[repr(C)]
pub(super) struct FlagsDesc {
    pub header: Header,
    pub flags: u32,
}
#[repr(C)]
pub(super) struct QueueDesc {
    pub header: Header,
    pub ordinal: u32,
    pub index: u32,
    pub flags: u32,
    pub mode: u32,
    pub priority: u32,
}
#[repr(C)]
pub(super) struct ListDesc {
    pub header: Header,
    pub ordinal: u32,
    pub flags: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct QueueProperties {
    pub header: Header,
    pub flags: u32,
    pub max_fill: usize,
    pub count: u32,
}
#[repr(C)]
pub(super) struct DeviceProperties {
    pub header: Header,
    pub kind: u32,
    pub vendor: u32,
    pub device: u32,
    pub flags: u32,
    pub subdevice: u32,
    pub clock: u32,
    pub max_alloc: u64,
    pub contexts: u32,
    pub priority: u32,
    pub threads: u32,
    pub simd: u32,
    pub eus: u32,
    pub subslices: u32,
    pub slices: u32,
    pub timer: u64,
    pub timestamp_bits: u32,
    pub kernel_timestamp_bits: u32,
    pub uuid: [u8; 16],
    pub name: [u8; 256],
}
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct ExtensionProperties {
    pub name: [u8; 256],
    pub version: u32,
}
#[repr(C)]
pub(super) struct DeviceGraphProperties {
    pub header: Header,
    pub version: u32,
    pub compiler_major: u16,
    pub compiler_minor: u16,
    pub formats: u32,
    pub max_opset: u32,
}
#[repr(C)]
pub(super) struct GraphDesc {
    pub header: Header,
    pub format: u32,
    pub size: usize,
    pub input: *const u8,
    pub flags: *const c_char,
}
#[repr(C)]
pub(super) struct GraphProperties {
    pub header: Header,
    pub count: u32,
}
#[repr(C)]
pub(super) struct GraphProperties2 {
    pub header: Header,
    pub count: u32,
    pub init_stages: u32,
}
#[repr(C)]
pub(super) struct ArgumentProperties {
    pub header: Header,
    pub name: [u8; 256],
    pub kind: u32,
    pub dims: [u32; 5],
    pub network_precision: u32,
    pub network_layout: u32,
    pub device_precision: u32,
    pub device_layout: u32,
}

// Only read the 1.0 prefix, even when the installed driver advertises a newer
// version. Option<fn> represents nullable C function pointers; validate before use.
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct GraphApi {
    pub create:
        Option<unsafe extern "system" fn(Handle, Handle, *const GraphDesc, *mut Handle) -> Status>,
    pub destroy: Option<Destroy>,
    pub properties: Option<unsafe extern "system" fn(Handle, *mut GraphProperties) -> Status>,
    pub argument: Option<unsafe extern "system" fn(Handle, u32, *mut ArgumentProperties) -> Status>,
    pub set_argument: Option<unsafe extern "system" fn(Handle, u32, *const c_void) -> Status>,
    pub initialize:
        Option<unsafe extern "system" fn(Handle, Handle, Handle, u32, *const Handle) -> Status>,
    pub execute: Option<
        unsafe extern "system" fn(Handle, Handle, Handle, Handle, u32, *const Handle) -> Status,
    >,
    pub native_binary: Option<unsafe extern "system" fn(Handle, *mut usize, *mut u8) -> Status>,
    pub device_properties:
        Option<unsafe extern "system" fn(Handle, *mut DeviceGraphProperties) -> Status>,
}
impl GraphApi {
    pub fn validate(self) -> Result<Self> {
        if self.create.is_none()
            || self.destroy.is_none()
            || self.properties.is_none()
            || self.argument.is_none()
            || self.set_argument.is_none()
            || self.initialize.is_none()
            || self.execute.is_none()
            || self.device_properties.is_none()
        {
            return Err(Error::Unsupported(
                "incomplete Intel graph 1.0 function table".into(),
            ));
        }
        Ok(self)
    }
}

macro_rules! api {
    ($($field:ident: $symbol:literal => unsafe extern "system" fn($($arg:ty),*) -> Status),* $(,)?) => {
        pub(super) struct Api { $(pub $field: unsafe extern "system" fn($($arg),*) -> Status,)* _library: Option<Library> }
        impl Api {
            #[cfg(test)]
            pub fn mock() -> Self {
                Self { $($field: {
                    unsafe extern "system" fn unsupported($(_: $arg),*) -> Status { 0x78000003 }
                    unsupported
                },)* _library: None }
            }
            pub fn load() -> Result<Self> {
                #[cfg(target_os = "windows")]
                let name = "ze_loader.dll";
                #[cfg(target_os = "linux")]
                let name = "libze_loader.so.1";
                #[cfg(not(any(target_os = "windows", target_os = "linux")))]
                { Err(Error::Unavailable("Intel Level Zero requires Windows or Linux".into())) }
                #[cfg(any(target_os = "windows", target_os = "linux"))]
                {
                    // SAFETY: loading the installed vendor loader is the system trust
                    // boundary. All symbols below use the published Level Zero ABI.
                    #[cfg(target_os = "windows")]
                    let library = unsafe { libloading::os::windows::Library::load_with_flags(
                        name, libloading::os::windows::LOAD_LIBRARY_SEARCH_SYSTEM32)
                    }.map(Library::from)
                        .map_err(|e| Error::Unavailable(format!("{name}: {e}")))?;
                    #[cfg(target_os = "linux")]
                    let library = unsafe { Library::new(name) }
                        .map_err(|e| Error::Unavailable(format!("{name}: {e}")))?;
                    // SAFETY: function pointers remain valid because this object owns
                    // the library and is retained by every device and kernel resource.
                    unsafe {
                        Ok(Self { $($field: *library.get::<unsafe extern "system" fn($($arg),*) -> Status>(concat!($symbol, "\0").as_bytes())
                            .map_err(|e| Error::Unavailable(e.to_string()))?,)* _library: Some(library) })
                    }
                }
            }
        }
    }
}
api! {
    init: "zeInit" => unsafe extern "system" fn(u32) -> Status,
    drivers: "zeDriverGet" => unsafe extern "system" fn(*mut u32, *mut Handle) -> Status,
    devices: "zeDeviceGet" => unsafe extern "system" fn(Handle, *mut u32, *mut Handle) -> Status,
    device_properties: "zeDeviceGetProperties" => unsafe extern "system" fn(Handle, *mut DeviceProperties) -> Status,
    extensions: "zeDriverGetExtensionProperties" => unsafe extern "system" fn(Handle, *mut u32, *mut ExtensionProperties) -> Status,
    extension_address: "zeDriverGetExtensionFunctionAddress" => unsafe extern "system" fn(Handle, *const c_char, *mut *mut c_void) -> Status,
    queue_properties: "zeDeviceGetCommandQueueGroupProperties" => unsafe extern "system" fn(Handle, *mut u32, *mut QueueProperties) -> Status,
    context_create: "zeContextCreate" => unsafe extern "system" fn(Handle, *const FlagsDesc, *mut Handle) -> Status,
    context_destroy: "zeContextDestroy" => unsafe extern "system" fn(Handle) -> Status,
    queue_create: "zeCommandQueueCreate" => unsafe extern "system" fn(Handle, Handle, *const QueueDesc, *mut Handle) -> Status,
    queue_destroy: "zeCommandQueueDestroy" => unsafe extern "system" fn(Handle) -> Status,
    queue_execute: "zeCommandQueueExecuteCommandLists" => unsafe extern "system" fn(Handle, u32, *const Handle, Handle) -> Status,
    queue_sync: "zeCommandQueueSynchronize" => unsafe extern "system" fn(Handle, u64) -> Status,
    list_create: "zeCommandListCreate" => unsafe extern "system" fn(Handle, Handle, *const ListDesc, *mut Handle) -> Status,
    list_close: "zeCommandListClose" => unsafe extern "system" fn(Handle) -> Status,
    list_reset: "zeCommandListReset" => unsafe extern "system" fn(Handle) -> Status,
    list_destroy: "zeCommandListDestroy" => unsafe extern "system" fn(Handle) -> Status,
    alloc_host: "zeMemAllocHost" => unsafe extern "system" fn(Handle, *const FlagsDesc, usize, usize, *mut Handle) -> Status,
    free: "zeMemFree" => unsafe extern "system" fn(Handle, Handle) -> Status,
}

pub(super) fn check(code: Status, operation: &'static str) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(Error::Driver { operation, code })
    }
}

pub(super) fn name(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

#[cfg(all(test, target_pointer_width = "64"))]
mod tests {
    use super::*;
    #[test]
    fn published_64_bit_abi_layouts() {
        use std::mem::{offset_of, size_of};
        assert_eq!(size_of::<DeviceProperties>(), 368);
        assert_eq!(offset_of!(DeviceProperties, name), 112);
        assert_eq!(size_of::<QueueDesc>(), 40);
        assert_eq!(size_of::<QueueProperties>(), 40);
        assert_eq!(size_of::<GraphDesc>(), 48);
        assert_eq!(size_of::<DeviceGraphProperties>(), 32);
        assert_eq!(size_of::<ArgumentProperties>(), 312);
        assert_eq!(offset_of!(ArgumentProperties, device_precision), 304);
        assert_eq!(size_of::<GraphApi>(), 9 * size_of::<usize>());
    }
    #[test]
    fn bounded_names_do_not_require_a_terminator() {
        assert_eq!(name(b"Intel\0garbage"), "Intel");
        assert_eq!(name(b"Intel"), "Intel");
    }
}
