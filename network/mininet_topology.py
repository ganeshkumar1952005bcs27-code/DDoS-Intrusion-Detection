"""
network/mininet_topology.py
-----------------------------
Minimal Mininet topology to test the OpenFlow-switch stage against
controller/ryu_mitigation.py.

    h1, h2, h3, attacker --- s1 (OpenFlow 1.3 switch) --- remote Ryu controller

Settings (controller IP/port, host IPs) are read from config.json /
config.example.json -> "network".

WHERE TO STORE
  network/mininet_topology.py   (this file)

HOW TO RUN  (needs a Linux host/VM with Mininet + Open vSwitch --
             requires raw sockets/root; run OUTSIDE this sandbox)
  1. sudo apt-get install mininet
  2. In one terminal:  ryu-manager controller/ryu_mitigation.py
  3. In another:        sudo python3 network/mininet_topology.py
  4. From the Mininet CLI:
       mininet> h1 ping h2                 # normal traffic, should work
       mininet> attacker ping -f h1        # simulate a flood
       mininet> attacker ping h1           # after mitigation, this
                                            # should start timing out
  5. Or simulate an alert directly (from a 3rd terminal), matching
     either edge-node mode:
       curl -X POST http://<controller-ip>:8080/api/alert \\
            -H "Content-Type: application/json" \\
            -d '{"device_id":"edge-sensor-01","src_ip":"10.0.0.5",
                 "predicted_class":1,"timestamp_ms":0}'
     Then re-run `attacker ping h1` inside Mininet and observe it drop.
"""

import json
import os

from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel
from mininet.link import TCLink

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_config():
    cfg_path = os.path.join(PROJECT_ROOT, "config.json")
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(PROJECT_ROOT, "config.example.json")
    with open(cfg_path) as fh:
        return json.load(fh)


def build_topology(cfg: dict):
    net_cfg = cfg.get("network", {})
    controller_ip = net_cfg.get("controller_ip", "127.0.0.1")
    controller_port = net_cfg.get("controller_port", 6653)
    hosts_ips = net_cfg.get("hosts", ["10.0.0.1", "10.0.0.2", "10.0.0.3"])
    attacker_ip = net_cfg.get("attacker_ip", "10.0.0.5")

    net = Mininet(switch=OVSSwitch, link=TCLink, controller=None, autoSetMacs=True)

    controller = net.addController(
        "c0", controller=RemoteController, ip=controller_ip, port=controller_port
    )

    s1 = net.addSwitch("s1", protocols="OpenFlow13")

    hosts = [net.addHost(f"h{i+1}", ip=f"{ip}/24") for i, ip in enumerate(hosts_ips)]
    attacker = net.addHost("attacker", ip=f"{attacker_ip}/24")

    for h in hosts + [attacker]:
        net.addLink(h, s1)

    net.build()
    controller.start()
    s1.start([controller])

    return net, hosts_ips, attacker_ip


if __name__ == "__main__":
    setLogLevel("info")
    cfg = _load_config()
    net, hosts_ips, attacker_ip = build_topology(cfg)
    print(f"\n[topology] Ready. Hosts: {hosts_ips}  attacker={attacker_ip}")
    print("[topology] Try:  attacker ping -f h1   (flood, to trigger "
          "packet-in-rate heuristic)")
    print("[topology] Or POST an alert to the controller's REST API "
          "(see docstring) and re-test connectivity.\n")
    CLI(net)
    net.stop()
