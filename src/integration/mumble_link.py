import ctypes
import mmap
import json
import logging
import time
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class Link(ctypes.Structure):
    _fields_ = [
        ("uiVersion", ctypes.c_uint32),           # 4 bytes
        ("uiTick", ctypes.c_ulong),               # 4 bytes
        ("fAvatarPosition", ctypes.c_float * 3),  # 3*4 bytes
        ("fAvatarFront", ctypes.c_float * 3),     # 3*4 bytes
        ("fAvatarTop", ctypes.c_float * 3),       # 3*4 bytes
        ("name", ctypes.c_wchar * 256),           # 512 bytes
        ("fCameraPosition", ctypes.c_float * 3),  # 3*4 bytes
        ("fCameraFront", ctypes.c_float * 3),     # 3*4 bytes
        ("fCameraTop", ctypes.c_float * 3),       # 3*4 bytes
        ("identity", ctypes.c_wchar * 256),       # 512 bytes
        ("context_len", ctypes.c_uint32),         # 4 bytes
        # ("context", ctypes.c_ubyte * 256),        # 256 bytes - handled separately
        # ("description", ctypes.c_wchar * 2048),   # 4096 bytes - handled separately
    ]

class Context(ctypes.Structure):
    _fields_ = [
        ("serverAddress", ctypes.c_ubyte * 28),   # 28 bytes
        ("mapId", ctypes.c_uint32),               # 4 bytes
        ("mapType", ctypes.c_uint32),             # 4 bytes
        ("shardId", ctypes.c_uint32),             # 4 bytes
        ("instance", ctypes.c_uint32),            # 4 bytes
        ("buildId", ctypes.c_uint32),             # 4 bytes
        ("uiState", ctypes.c_uint32),             # 4 bytes
        ("compassWidth", ctypes.c_uint16),        # 2 bytes
        ("compassHeight", ctypes.c_uint16),       # 2 bytes
        ("compassRotation", ctypes.c_float),      # 4 bytes
        ("playerX", ctypes.c_float),              # 4 bytes
        ("playerY", ctypes.c_float),              # 4 bytes
        ("mapCenterX", ctypes.c_float),           # 4 bytes
        ("mapCenterY", ctypes.c_float),           # 4 bytes
        ("mapScale", ctypes.c_float),             # 4 bytes
        ("processId", ctypes.c_uint32),           # 4 bytes
        ("mountIndex", ctypes.c_uint8),           # 1 byte
    ]

# Common PvP Map IDs for convenience
PVP_MAPS = {
    549: "Battle of Kyhlo",
    554: "Forest of Niflhel",
    795: "Legacy of the Foefire",
    894: "Temple of the Silent Storm",
    900: "Skyhammer",
    984: "Courtyard",
    1163: "Revenge of the Capricorn",
    1206: "Eternal Coliseum",
    1326: "Djinn's Dominion",
}

class MumbleLink:
    """
    Interface to the Guild Wars 2 MumbleLink shared memory.
    """
    def __init__(self, map_csv_path: Optional[str] = None):
        self.size_link = ctypes.sizeof(Link)
        self.size_context = ctypes.sizeof(Context)
        self.map_names = PVP_MAPS.copy()
        
        # Load extra map names if CSV is provided
        if map_csv_path:
            self.load_map_names(map_csv_path)

        # Calculate total memory size
        # Link struct (1108) + Context (approx 80-90) + Padding
        # The total size must match exactly what GW2 allocates: 5460 bytes
        size_discarded = 256 - self.size_context + 4096 
        memfile_length = self.size_link + self.size_context + size_discarded
        
        self.memfile = None
        self.data: Optional[Link] = None
        self.context_obj: Optional[Context] = None
        self._linked = False

        try:
            # Using defaults from the wiki example: fileno=-1
            # Note regarding access: 
            # If we create it (fileno=-1), default is write.
            # But "tagname" links it to existing system object if it exists.
            self.memfile = mmap.mmap(fileno=-1, length=memfile_length, tagname="MumbleLink")
            self._linked = True
            logger.info("Successfully connected to MumbleLink shared memory.")
        except FileNotFoundError:
            logger.warning("Could not open MumbleLink shared memory. Is Guild Wars 2 running?")
            self._linked = False
        except Exception as e:
            logger.error(f"Error connecting to MumbleLink: {e}")
            self._linked = False

    def load_map_names(self, csv_path: str):
        """Loads MapID -> MapName mappings from a CSV file."""
        import csv
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'MapID' in row and 'MapName' in row:
                        try:
                            map_id = int(row['MapID'])
                            self.map_names[map_id] = row['MapName']
                        except ValueError:
                            continue
        except Exception as e:
            logger.error(f"Failed to load map names from {csv_path}: {e}")

    @property
    def is_active(self) -> bool:
        """Returns True if connected to shared memory."""
        return self._linked and self.memfile is not None

    def read(self):
        """
        Reads the latest data from shared memory.
        """
        if not self.is_active:
            # Try to reconnect
            try:
                # Recalculate length is pointless here as it's constant, but logic flow is same
                size_discarded = 256 - self.size_context + 4096 
                memfile_length = self.size_link + self.size_context + size_discarded
                self.memfile = mmap.mmap(fileno=-1, length=memfile_length, tagname="MumbleLink")
                self._linked = True
            except FileNotFoundError:
                return
            except Exception:
                return
        
        try:
            self.memfile.seek(0)
            
            # 1. Read Link struct
            link_buffer = self.memfile.read(self.size_link)
            self.data = self.unpack(Link, link_buffer)
            
            # 2. Read Context struct
            # Note: Context follows immediately after Link in the memory map?
            # In the Wiki example:
            # self.data = self.unpack(Link, self.memfile.read(self.size_link))
            # self.context = self.unpack(Context, self.memfile.read(self.size_context))
            # Yes, they read sequentially.
            
            context_buffer = self.memfile.read(self.size_context)
            self.context_obj = self.unpack(Context, context_buffer)

        except Exception as e:
            logger.error(f"Error reading MumbleLink data: {e}")

    @staticmethod
    def unpack(ctype, buf):
        cstring = ctypes.create_string_buffer(buf)
        ctype_instance = ctypes.cast(ctypes.pointer(cstring), ctypes.POINTER(ctype)).contents
        return ctype_instance

    def get_identity(self) -> Dict[str, Any]:
        """
        Returns the parsed identity JSON dictionary.
        Keys often include: 'name', 'profession', 'spec', 'race', 'map_id', 'world_id', 'team_color_id', 'commander'
        """
        if not self.data or not self.data.identity:
            return {}
        
        try:
            return json.loads(self.data.identity)
        except json.JSONDecodeError:
            return {}

    def get_player_name(self) -> Optional[str]:
        identity = self.get_identity()
        return identity.get("name")

    def get_map_id(self) -> Optional[int]:
        identity = self.get_identity()
        return identity.get("map_id")

    def get_map_name(self) -> str:
        map_id = self.get_map_id()
        if map_id in self.map_names:
            return self.map_names[map_id]
        return f"Unknown Map ({map_id})" if map_id else "Unknown"

    def close(self):
        if self.memfile:
            self.memfile.close()
            self._linked = False
