from .basic_controller import BasicMAC
from .entity_controller import EntityMAC
from .comm_controller import CommMAC
from .comm_controller_sp import CommMACSP

REGISTRY = {}
REGISTRY["basic_mac"] = BasicMAC
REGISTRY["entity_mac"] = EntityMAC
REGISTRY["comm_mac"] = CommMAC
REGISTRY["comm_mac_sp"] = CommMACSP




