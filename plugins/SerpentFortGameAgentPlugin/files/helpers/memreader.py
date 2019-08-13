from ctypes import *
from ctypes.wintypes import *
from time import sleep
import win32api, win32ui, win32process
import ctypes
import ntpath

class MemoryReader:
    def __init__(self, window_title):
        OpenProcess = windll.kernel32.OpenProcess
        self.ReadProcessMemory = windll.kernel32.ReadProcessMemory
        SIZE_T = c_size_t
        self.ReadProcessMemory.argtypes = [HANDLE, LPCVOID, LPVOID, SIZE_T, POINTER(SIZE_T)]
        FindWindowA = windll.user32.FindWindowA
        GetWindowThreadProcessId = windll.user32.GetWindowThreadProcessId

        PROCESS_ALL_ACCESS = 0x1F0FFF
        HWND = win32ui.FindWindow(None, window_title).GetSafeHwnd()
        PID = win32process.GetWindowThreadProcessId(HWND)[1]
        self.processHandle = ctypes.windll.kernel32.OpenProcess(PROCESS_ALL_ACCESS,False,PID)

        values = win32process.EnumProcessModules(self.processHandle)
        self.pointer_dict = dict()
        self.address_dict = dict()
        for value in values:
            name = ntpath.basename(win32process.GetModuleFileNameEx(self.processHandle, value))
            self.pointer_dict[name] = value


    def get_address(self, offsets):
        numRead = c_size_t()
        base_pointer = self.pointer_dict[offsets[0]]
        tmp = c_int()
        print(f"{base_pointer} + {offsets[1]} = {base_pointer + offsets[1]}")
        if not self.ReadProcessMemory(self.processHandle, base_pointer + offsets[1], byref(tmp), 4, byref(numRead)):
            raise RuntimeError(f"Failed to read a location in the memory. Address: {base_pointer + offsets[1]}")
        for i in range(2, offsets.__len__() - 1):
            print(f"{tmp.value} + {offsets[i]} = {tmp.value + offsets[i]}")
            if not self.ReadProcessMemory(self.processHandle, tmp.value + offsets[i], byref(tmp), 4, byref(numRead)):
                raise RuntimeError(f"Failed to read a location in the memory. Address: {tmp.value + offsets[i]}")

        return tmp.value + offsets[offsets.__len__()-1]

    def read_address(self, path):
        address = self.get_address(path)
        numRead = c_size_t()
        tmp = c_int()
        if not self.ReadProcessMemory(self.processHandle, address, byref(tmp), 4, byref(numRead)):
            raise RuntimeError(f"Failed to read the address from memory. Address: {address}")

        return tmp.value

    def store_address(self, key, path):
        self.address_dict[key] = self.get_address(path)


    def read(self, key):
        numRead = c_size_t()
        tmp = c_int()
        if not self.ReadProcessMemory(self.processHandle, self.address_dict[key], byref(tmp), 4, byref(numRead)):
            raise RuntimeError(f"Failed to read the {key} address from memory. Address: {self.address_dict[key]}")

        return tmp.value
