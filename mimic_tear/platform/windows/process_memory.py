from __future__ import annotations

import ctypes
import os
import struct
import time
from ctypes import wintypes
from dataclasses import dataclass, field


PROCESS_VM_READ = 0x0010
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
TH32CS_SNAPPROCESS = 0x00000002
TH32CS_SNAPMODULE = 0x00000008
TH32CS_SNAPMODULE32 = 0x00000010
MAX_PATH = 260
MAX_READ_SIZE = 1_048_576
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
ANTI_CHEAT_CHECK_INTERVAL_NS = 1_000_000_000


class ProcessMemoryError(RuntimeError):
    pass


class ProcessNotFoundError(ProcessMemoryError):
    pass


class ModuleNotFoundError(ProcessMemoryError):
    pass


class MemoryReadError(ProcessMemoryError):
    pass


class AntiCheatDetectedError(ProcessMemoryError):
    pass


class PEFormatError(ProcessMemoryError):
    pass


class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.c_size_t),
        ("th32ModuleID", wintypes.DWORD),
        ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD),
        ("pcPriClassBase", wintypes.LONG),
        ("dwFlags", wintypes.DWORD),
        ("szExeFile", wintypes.WCHAR * MAX_PATH),
    ]


class MODULEENTRY32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("th32ModuleID", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("GlblcntUsage", wintypes.DWORD),
        ("ProccntUsage", wintypes.DWORD),
        ("modBaseAddr", ctypes.POINTER(ctypes.c_ubyte)),
        ("modBaseSize", wintypes.DWORD),
        ("hModule", wintypes.HMODULE),
        ("szModule", wintypes.WCHAR * 256),
        ("szExePath", wintypes.WCHAR * MAX_PATH),
    ]


def _require_windows() -> ctypes.WinDLL:
    if os.name != "nt":
        raise OSError("ReadProcessMemory capture is only available on Windows")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.Process32FirstW.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(PROCESSENTRY32W),
    ]
    kernel32.Process32FirstW.restype = wintypes.BOOL
    kernel32.Process32NextW.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(PROCESSENTRY32W),
    ]
    kernel32.Process32NextW.restype = wintypes.BOOL
    kernel32.Module32FirstW.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(MODULEENTRY32W),
    ]
    kernel32.Module32FirstW.restype = wintypes.BOOL
    kernel32.Module32NextW.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(MODULEENTRY32W),
    ]
    kernel32.Module32NextW.restype = wintypes.BOOL
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.ReadProcessMemory.argtypes = [
        wintypes.HANDLE,
        wintypes.LPCVOID,
        wintypes.LPVOID,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    kernel32.ReadProcessMemory.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    return kernel32


def _windows_error(operation: str) -> ProcessMemoryError:
    error_code = ctypes.get_last_error()
    return ProcessMemoryError(
        f"{operation} failed with Windows error {error_code}: "
        f"{ctypes.FormatError(error_code).strip()}"
    )


@dataclass(frozen=True, slots=True)
class RunningProcess:
    process_id: int
    name: str


def list_processes() -> list[RunningProcess]:
    kernel32 = _require_windows()
    snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if snapshot == INVALID_HANDLE_VALUE:
        raise _windows_error("CreateToolhelp32Snapshot(processes)")
    processes: list[RunningProcess] = []
    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(entry)
        success = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while success:
            processes.append(
                RunningProcess(int(entry.th32ProcessID), str(entry.szExeFile))
            )
            success = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    return processes


def find_process_ids(process_name: str) -> list[int]:
    expected = process_name.casefold()
    return [
        process.process_id
        for process in list_processes()
        if process.name.casefold() == expected
    ]


def is_anti_cheat_process_name(process_name: str) -> bool:
    normalized = process_name.casefold()
    return "easyanticheat" in normalized


def find_anti_cheat_processes() -> tuple[RunningProcess, ...]:
    return tuple(
        process
        for process in list_processes()
        if is_anti_cheat_process_name(process.name)
    )


def assert_anti_cheat_inactive() -> None:
    detected = find_anti_cheat_processes()
    if not detected:
        return
    details = ", ".join(
        f"{process.name} (PID {process.process_id})" for process in detected
    )
    raise AntiCheatDetectedError(
        "Refusing process-memory access because anti-cheat protection appears "
        f"to be active: {details}. Use the recorder only in a permitted offline "
        "environment where anti-cheat is already inactive."
    )


def find_process_id(process_name: str) -> int:
    process_ids = find_process_ids(process_name)
    if not process_ids:
        raise ProcessNotFoundError(f"Process is not running: {process_name}")
    if len(process_ids) > 1:
        raise ProcessMemoryError(
            f"More than one {process_name} process is running: {process_ids}"
        )
    return process_ids[0]


@dataclass(frozen=True, slots=True)
class ModuleInfo:
    name: str
    path: str
    base: int
    size: int


@dataclass(frozen=True, slots=True)
class PESection:
    name: str
    address: int
    size: int

    def contains(self, address: int) -> bool:
        return self.address <= address < self.address + self.size


def find_module(process_id: int, module_name: str) -> ModuleInfo:
    kernel32 = _require_windows()
    flags = TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32
    snapshot = kernel32.CreateToolhelp32Snapshot(flags, process_id)
    if snapshot == INVALID_HANDLE_VALUE:
        raise _windows_error("CreateToolhelp32Snapshot(modules)")
    try:
        entry = MODULEENTRY32W()
        entry.dwSize = ctypes.sizeof(entry)
        success = kernel32.Module32FirstW(snapshot, ctypes.byref(entry))
        while success:
            if entry.szModule.casefold() == module_name.casefold():
                address = ctypes.cast(entry.modBaseAddr, ctypes.c_void_p).value
                if address is None:
                    break
                return ModuleInfo(
                    name=str(entry.szModule),
                    path=str(entry.szExePath),
                    base=int(address),
                    size=int(entry.modBaseSize),
                )
            success = kernel32.Module32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    raise ModuleNotFoundError(
        f"Module {module_name!r} was not found in process {process_id}"
    )


def find_module_base(process_id: int, module_name: str) -> int:
    return find_module(process_id, module_name).base


@dataclass(slots=True)
class ProcessMemory:
    process_id: int
    module_base: int
    module_size: int
    module_name: str
    pointer_size: int
    _handle: int
    _sections: tuple[PESection, ...] | None = field(default=None, repr=False)
    _anti_cheat_guard: bool = field(default=False, repr=False)
    _next_anti_cheat_check_ns: int = field(default=0, repr=False)

    @classmethod
    def open(
        cls,
        process_name: str,
        *,
        module_name: str | None = None,
        pointer_size: int = 8,
        anti_cheat_guard: bool = False,
    ) -> "ProcessMemory":
        if pointer_size not in (4, 8):
            raise ValueError("pointer_size must be 4 or 8")
        process_id = find_process_id(process_name)
        return cls.open_process_id(
            process_id,
            module_name=module_name or process_name,
            pointer_size=pointer_size,
            anti_cheat_guard=anti_cheat_guard,
        )

    @classmethod
    def open_process_id(
        cls,
        process_id: int,
        *,
        module_name: str,
        pointer_size: int = 8,
        anti_cheat_guard: bool = False,
    ) -> "ProcessMemory":
        if process_id <= 0:
            raise ValueError("process_id must be greater than zero")
        if pointer_size not in (4, 8):
            raise ValueError("pointer_size must be 4 or 8")
        if anti_cheat_guard:
            assert_anti_cheat_inactive()
        kernel32 = _require_windows()
        access = PROCESS_VM_READ | PROCESS_QUERY_LIMITED_INFORMATION
        handle = kernel32.OpenProcess(access, False, process_id)
        if not handle:
            raise _windows_error(f"OpenProcess({process_id})")
        try:
            module = find_module(process_id, module_name)
        except Exception:
            kernel32.CloseHandle(handle)
            raise
        handle_value = ctypes.cast(handle, ctypes.c_void_p).value
        if handle_value is None:
            kernel32.CloseHandle(handle)
            raise ProcessMemoryError("OpenProcess returned an invalid handle")
        return cls(
            process_id,
            module.base,
            module.size,
            module.name,
            pointer_size,
            int(handle_value),
            None,
            anti_cheat_guard,
            time.monotonic_ns() + ANTI_CHEAT_CHECK_INTERVAL_NS,
        )

    def close(self) -> None:
        if not self._handle:
            return
        _require_windows().CloseHandle(self._handle)
        self._handle = 0

    def read(self, address: int, size: int) -> bytes:
        self._check_anti_cheat_guard()
        if not self._handle:
            raise MemoryReadError("Process handle is closed")
        if address <= 0:
            raise MemoryReadError(f"Refusing to read invalid address: {address:#x}")
        if not 0 < size <= MAX_READ_SIZE:
            raise ValueError(f"Read size must be in [1, {MAX_READ_SIZE}]")
        kernel32 = _require_windows()
        buffer = ctypes.create_string_buffer(size)
        bytes_read = ctypes.c_size_t()
        success = kernel32.ReadProcessMemory(
            self._handle,
            ctypes.c_void_p(address),
            buffer,
            size,
            ctypes.byref(bytes_read),
        )
        if not success or bytes_read.value != size:
            error_code = ctypes.get_last_error()
            raise MemoryReadError(
                f"ReadProcessMemory({address:#x}, {size}) failed with Windows "
                f"error {error_code}: {ctypes.FormatError(error_code).strip()}"
            )
        return buffer.raw

    def _check_anti_cheat_guard(self) -> None:
        if not self._anti_cheat_guard:
            return
        now_ns = time.monotonic_ns()
        if now_ns < self._next_anti_cheat_check_ns:
            return
        assert_anti_cheat_inactive()
        self._next_anti_cheat_check_ns = now_ns + ANTI_CHEAT_CHECK_INTERVAL_NS

    def read_pointer(self, address: int) -> int:
        kind = "<Q" if self.pointer_size == 8 else "<I"
        pointer = int(struct.unpack(kind, self.read(address, self.pointer_size))[0])
        if pointer == 0:
            raise MemoryReadError(f"Null pointer at {address:#x}")
        return pointer

    def read_int32(self, address: int) -> int:
        return int(struct.unpack("<i", self.read(address, 4))[0])

    def resolve_address(
        self,
        base_address: int,
        pointer_offsets: tuple[int, ...],
    ) -> int:
        address = base_address
        for offset in pointer_offsets:
            address = self.read_pointer(address) + offset
        return address

    def resolve(self, base_offset: int, pointer_offsets: tuple[int, ...]) -> int:
        return self.resolve_address(self.module_base + base_offset, pointer_offsets)

    def pe_sections(self) -> tuple[PESection, ...]:
        if self._sections is not None:
            return self._sections

        dos_header = self.read(self.module_base, 0x40)
        if dos_header[:2] != b"MZ":
            raise PEFormatError("Module does not have an MZ header")
        pe_offset = struct.unpack_from("<I", dos_header, 0x3C)[0]
        if pe_offset <= 0 or pe_offset > self.module_size - 24:
            raise PEFormatError(f"Invalid PE header offset: {pe_offset:#x}")

        nt_headers = self.read(self.module_base + pe_offset, 24)
        if nt_headers[:4] != b"PE\x00\x00":
            raise PEFormatError("Module does not have a PE signature")
        section_count = struct.unpack_from("<H", nt_headers, 6)[0]
        optional_header_size = struct.unpack_from("<H", nt_headers, 20)[0]
        if not 0 < section_count <= 96:
            raise PEFormatError(f"Invalid PE section count: {section_count}")

        table_address = self.module_base + pe_offset + 24 + optional_header_size
        table = self.read(table_address, section_count * 40)
        sections: list[PESection] = []
        for index in range(section_count):
            offset = index * 40
            raw_name = table[offset:offset + 8].split(b"\x00", 1)[0]
            name = raw_name.decode("ascii", errors="replace")
            virtual_size = struct.unpack_from("<I", table, offset + 8)[0]
            virtual_address = struct.unpack_from("<I", table, offset + 12)[0]
            raw_size = struct.unpack_from("<I", table, offset + 16)[0]
            size = min(
                max(virtual_size, raw_size),
                max(0, self.module_size - virtual_address),
            )
            if size:
                sections.append(
                    PESection(name, self.module_base + virtual_address, size)
                )
        if not sections:
            raise PEFormatError("Module contains no readable PE sections")
        self._sections = tuple(sections)
        return self._sections

    def section(self, name: str) -> PESection:
        for section in self.pe_sections():
            if section.name == name:
                return section
        raise PEFormatError(f"PE section was not found: {name}")

    def iter_memory(
        self,
        address: int,
        size: int,
        *,
        overlap: int = 0,
        chunk_size: int = MAX_READ_SIZE,
    ):
        if size < 0:
            raise ValueError("size cannot be negative")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be in [0, chunk_size)")
        offset = 0
        while offset < size:
            read_size = min(chunk_size, size - offset)
            yield address + offset, self.read(address + offset, read_size)
            if offset + read_size >= size:
                break
            offset += read_size - overlap

    def read_typed(self, address: int, value_type: str, *, length: int | None) -> object:
        scalar_formats = {
            "bool": "<?",
            "int8": "<b",
            "uint8": "<B",
            "int16": "<h",
            "uint16": "<H",
            "int32": "<i",
            "uint32": "<I",
            "int64": "<q",
            "uint64": "<Q",
            "float32": "<f",
            "float64": "<d",
        }
        if value_type in scalar_formats:
            format_string = scalar_formats[value_type]
            size = struct.calcsize(format_string)
            return struct.unpack(format_string, self.read(address, size))[0]
        if value_type not in ("utf8", "utf16"):
            raise ValueError(f"Unsupported memory value type: {value_type}")
        if length is None or length <= 0:
            raise ValueError(f"{value_type} fields require a positive length")
        unit_size = 2 if value_type == "utf16" else 1
        raw = self.read(address, length * unit_size)
        encoding = "utf-16-le" if value_type == "utf16" else "utf-8"
        terminator = b"\x00\x00" if value_type == "utf16" else b"\x00"
        if value_type == "utf16":
            end = next(
                (index for index in range(0, len(raw), 2) if raw[index:index + 2] == terminator),
                len(raw),
            )
            raw = raw[:end]
        else:
            raw = raw.split(terminator, 1)[0]
        return raw.decode(encoding, errors="replace")

    def __enter__(self) -> "ProcessMemory":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
