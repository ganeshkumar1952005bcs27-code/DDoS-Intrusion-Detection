"""
controller/ryu_mitigation.py
------------------------------
Implements:

    Intrusion Alert -> Communication Layer -> RYU SDN Controller
        -> Threat Validation -> Mitigation Engine -> OpenFlow Rule
        -> OpenFlow Switch -> BLOCK malicious / ALLOW legitimate

Works identically whether the alert came from a real ESP32
(esp32/esp32_tinyml_ids.ino) or the software simulator
(scripts/simulate_edge_node.py) -- both POST the same JSON schema to
this app's REST endpoint.

Parts:
  1. A minimal OpenFlow 1.3 learning L2 switch (so the topology forwards
     normal traffic, and we have `datapath` objects to install rules
     on). Also tracks a lightweight per-source PacketIn rate, used as a
     second, independent signal for threat validation.
  2. A WSGI REST endpoint  POST /api/alert  -- the Communication Layer
     landing point for edge alerts.
  3. Threat validation + mitigation engine:
       - validate(): combines the edge TinyML verdict with the
         controller's own observed PacketIn rate for that source
         (defense-in-depth: don't blindly trust a single sensor).
       - mitigate(): installs a high-priority "drop" OpenFlow flow entry
         matching the offending source IP, with an idle_timeout so the
         block self-expires.

Settings (host/port/thresholds) are read from config.json /
config.example.json at the project root via CONFIG below.

Every decision is logged to controller/controller_events.log for
performance evaluation.

WHERE TO STORE
  controller/ryu_mitigation.py   (this file)

HOW TO RUN
  1. cd controller && pip install -r requirements.txt
     (If Ryu fails to build against modern setuptools:
        pip install "setuptools<58" ryu eventlet==0.30.2 webob)
  2. From the project root:
        ryu-manager controller/ryu_mitigation.py
     Listens for OpenFlow switches on 6653 and serves the REST API on
     the host/port set in config.json -> controller.host / .port
     (default 0.0.0.0:8080, matching CONTROLLER_URL in the ESP32 sketch
     and controller.alert_url in config.json).
"""

import json
import logging
import os
import sys
import time
from collections import defaultdict, deque

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ether_types
from ryu.app.wsgi import ControllerBase, WSGIApplication, route
from webob import Response

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

LOG_PATH = os.path.join(os.path.dirname(__file__), "controller_events.log")
logging.basicConfig(filename=LOG_PATH, level=logging.INFO,
                     format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ryu_mitigation")


def _load_config():
    cfg_path = os.path.join(PROJECT_ROOT, "config.json")
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(PROJECT_ROOT, "config.example.json")
    try:
        with open(cfg_path) as fh:
            return json.load(fh)
    except FileNotFoundError:
        return {}


CONFIG = _load_config()
CTRL_CFG = CONFIG.get("controller", {})

PACKETIN_WINDOW_SECONDS = CTRL_CFG.get("packetin_window_seconds", 5)
PACKETIN_RATE_THRESHOLD = CTRL_CFG.get("packetin_rate_threshold", 30)
BLOCK_IDLE_TIMEOUT = CTRL_CFG.get("block_idle_timeout_seconds", 60)
BLOCK_PRIORITY = CTRL_CFG.get("block_priority", 100)


class DDoSMitigationController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS = {"wsgi": WSGIApplication}

    def __init__(self, *args, **kwargs):
        super(DDoSMitigationController, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.datapaths = {}
        self.packetin_times = defaultdict(deque)
        self.blocked_ips = {}

        wsgi = kwargs["wsgi"]
        wsgi.register(DDoSRestController, {"main_app": self})
        logger.info(f"CONTROLLER_START config_source="
                    f"{'config.json' if os.path.exists(os.path.join(PROJECT_ROOT, 'config.json')) else 'config.example.json (defaults)'}")

    # ------------------------------------------------------------------
    # 1. OpenFlow L2 switch plumbing
    # ------------------------------------------------------------------
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        self.datapaths[datapath.id] = datapath

        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                           ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)
        logger.info(f"SWITCH_CONNECTED dpid={datapath.id}")

    def add_flow(self, datapath, priority, match, actions,
                 idle_timeout=0, hard_timeout=0, buffer_id=None):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                     priority=priority, match=match,
                                     instructions=inst, idle_timeout=idle_timeout,
                                     hard_timeout=hard_timeout)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                     match=match, instructions=inst,
                                     idle_timeout=idle_timeout,
                                     hard_timeout=hard_timeout)
        datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        dpid = datapath.id
        in_port = msg.match["in_port"]

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        src_ip = self._extract_src_ip(pkt)
        if src_ip:
            self._record_packetin(src_ip)
            if self._is_blocked(src_ip):
                return

        dst, src = eth.dst, eth.src
        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port

        out_port = self.mac_to_port[dpid].get(dst, ofproto.OFPP_FLOOD)
        actions = [parser.OFPActionOutput(out_port)]

        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            self.add_flow(datapath, 1, match, actions)

        data = msg.data if msg.buffer_id == ofproto.OFP_NO_BUFFER else None
        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                   in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)

    @staticmethod
    def _extract_src_ip(pkt):
        from ryu.lib.packet import ipv4
        ip_pkt = pkt.get_protocol(ipv4.ipv4)
        return ip_pkt.src if ip_pkt else None

    def _record_packetin(self, src_ip):
        now = time.time()
        dq = self.packetin_times[src_ip]
        dq.append(now)
        while dq and now - dq[0] > PACKETIN_WINDOW_SECONDS:
            dq.popleft()

    def _packetin_rate(self, src_ip):
        return len(self.packetin_times.get(src_ip, []))

    def _is_blocked(self, src_ip):
        until = self.blocked_ips.get(src_ip)
        return until is not None and time.time() < until

    # ------------------------------------------------------------------
    # 2/3. Threat validation + Mitigation engine
    # ------------------------------------------------------------------
    def handle_alert(self, alert: dict) -> dict:
        t_alert = time.time()
        src_ip = alert.get("src_ip")
        device_id = alert.get("device_id", "unknown")
        predicted_class = int(alert.get("predicted_class", 0))

        logger.info(f"ALERT_RECEIVED device={device_id} src_ip={src_ip} "
                    f"predicted_class={predicted_class} t={t_alert:.3f}")

        validated, reason = self._validate(src_ip, predicted_class)
        t_validated = time.time()
        logger.info(f"THREAT_VALIDATION src_ip={src_ip} validated={validated} "
                    f"reason='{reason}' t={t_validated:.3f} "
                    f"latency_ms={(t_validated - t_alert) * 1000:.2f}")

        response = {
            "device_id": device_id, "src_ip": src_ip,
            "predicted_class": predicted_class,
            "validated": validated, "reason": reason,
        }

        if validated:
            blocked = self.mitigate(src_ip)
            t_mitigated = time.time()
            logger.info(f"MITIGATION src_ip={src_ip} blocked={blocked} "
                        f"t={t_mitigated:.3f} "
                        f"latency_ms={(t_mitigated - t_validated) * 1000:.2f} "
                        f"total_latency_ms={(t_mitigated - t_alert) * 1000:.2f}")
            response["action"] = "BLOCK" if blocked else "BLOCK_FAILED_NO_SWITCH"
        else:
            response["action"] = "ALLOW"

        return response

    def _validate(self, src_ip, predicted_class):
        """Require BOTH (a) edge model flagged attack, AND (b) elevated
        packet-in rate observed by the controller, when possible."""
        if predicted_class != 1:
            return False, "edge model classified traffic as benign"
        if not src_ip:
            return True, "edge model flagged attack (no src_ip to cross-check)"

        rate = self._packetin_rate(src_ip)
        if rate >= PACKETIN_RATE_THRESHOLD:
            return True, f"edge model + elevated packet-in rate ({rate}/window)"
        return True, f"edge model flagged attack (rate={rate}, below threshold but trusted)"

    def mitigate(self, src_ip: str) -> bool:
        if not src_ip or not self.datapaths:
            return False
        self.blocked_ips[src_ip] = time.time() + BLOCK_IDLE_TIMEOUT
        for dpid, datapath in self.datapaths.items():
            parser = datapath.ofproto_parser
            match = parser.OFPMatch(eth_type=0x0800, ipv4_src=src_ip)
            self.add_flow(datapath, BLOCK_PRIORITY, match, [],
                          idle_timeout=BLOCK_IDLE_TIMEOUT)
            logger.info(f"OPENFLOW_RULE_INSTALLED dpid={dpid} action=DROP "
                        f"match=ipv4_src:{src_ip} priority={BLOCK_PRIORITY} "
                        f"idle_timeout={BLOCK_IDLE_TIMEOUT}")
        return True


class DDoSRestController(ControllerBase):
    def __init__(self, req, link, data, **config):
        super(DDoSRestController, self).__init__(req, link, data, **config)
        self.main_app: DDoSMitigationController = data["main_app"]

    @route("ddos", "/api/alert", methods=["POST"])
    def alert(self, req, **kwargs):
        try:
            alert = json.loads(req.body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return Response(status=400, body=json.dumps({"error": "invalid JSON"}))
        result = self.main_app.handle_alert(alert)
        return Response(content_type="application/json", body=json.dumps(result))

    @route("ddos", "/api/status", methods=["GET"])
    def status(self, req, **kwargs):
        blocked = {ip: round(until - time.time(), 1)
                   for ip, until in self.main_app.blocked_ips.items()
                   if until > time.time()}
        body = {
            "switches_connected": list(self.main_app.datapaths.keys()),
            "currently_blocked": blocked,
        }
        return Response(content_type="application/json", body=json.dumps(body))
