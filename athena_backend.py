"""
╔══════════════════════════════════════════════════════╗
║        ATHENA CLOUD BACKEND — FastAPI                ║
║   Interpreta comandos com IA e roteia para agentes   ║
╚══════════════════════════════════════════════════════╝

Deploy gratuito no Railway:
    1. Crie conta em railway.app
    2. New Project → Deploy from GitHub → aponte este arquivo
    3. Adicione as variáveis de ambiente:
       - ANTHROPIC_API_KEY=sk-ant-...
       - SUPABASE_URL=https://xxx.supabase.co
       - SUPABASE_KEY=eyJ...
       - SECRET_KEY=uma-chave-aleatoria-longa

Instalação local:
    pip install fastapi uvicorn websockets anthropic supabase python-dotenv

Rodar local:
    uvicorn athena_backend:app --reload --port 8000
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Optional, Dict

import anthropic
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# ─── Configuração ──────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [BACKEND] %(levelname)s: %(message)s")
log = logging.getLogger("athena.backend")

# ─── Supabase (opcional — funciona sem ele em modo local) ──────────────────────
try:
    from supabase import create_client, Client
    SUPABASE_URL = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
    if SUPABASE_URL and SUPABASE_KEY:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        HAS_SUPABASE = True
        log.info("✅ Supabase conectado")
    else:
        HAS_SUPABASE = False
        log.warning("Supabase não configurado — usando memória local")
except Exception as e:
    HAS_SUPABASE = False
    log.warning(f"Supabase indisponível: {e}")

# ─── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Athena Cloud",
    description="Sistema Autônomo de Controle Remoto",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Estado em memória (substituído pelo Supabase quando disponível) ────────────
memory_store: Dict[str, dict] = {
    "commands": {},
    "results": {},
    "agents": {},
    "alerts": [],
    "learned": {}
}

# ─── Claude AI ─────────────────────────────────────────────────────────────────
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")
claude = anthropic.Anthropic(api_key=ANTHROPIC_KEY) if ANTHROPIC_KEY else None

SECRET_KEY = os.getenv("SECRET_KEY", "athena-dev-secret")


# ─── GERENCIADOR DE CONEXÕES WebSocket ────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.agents: Dict[str, WebSocket] = {}         # agent_id → ws
        self.agent_info: Dict[str, dict] = {}          # agent_id → metadata
        self.pending_results: Dict[str, asyncio.Event] = {}
        self.results_cache: Dict[str, dict] = {}

    async def connect(self, agent_id: str, ws: WebSocket, info: dict):
        await ws.accept()
        self.agents[agent_id] = ws
        self.agent_info[agent_id] = {**info, "connected_at": datetime.now().isoformat(), "online": True}
        log.info(f"✅ Agente conectado: {agent_id}")

    def disconnect(self, agent_id: str):
        self.agents.pop(agent_id, None)
        if agent_id in self.agent_info:
            self.agent_info[agent_id]["online"] = False
            self.agent_info[agent_id]["disconnected_at"] = datetime.now().isoformat()
        log.info(f"❌ Agente desconectado: {agent_id}")

    async def send_command(self, agent_id: str, payload: dict) -> bool:
        ws = self.agents.get(agent_id)
        if not ws:
            return False
        try:
            await ws.send_json(payload)
            return True
        except Exception as e:
            log.error(f"Erro ao enviar para {agent_id}: {e}")
            self.disconnect(agent_id)
            return False

    async def broadcast(self, payload: dict):
        dead = []
        for aid, ws in self.agents.items():
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(aid)
        for aid in dead:
            self.disconnect(aid)

    def get_agents(self):
        return [
            {"id": aid, **info}
            for aid, info in self.agent_info.items()
        ]


manager = ConnectionManager()


# ─── INTERPRETADOR DE COMANDOS COM IA ─────────────────────────────────────────

SYSTEM_PROMPT = """Você é Athena, a intérprete de comandos de um sistema de automação.
Converta linguagem natural em ações estruturadas JSON.

Retorne APENAS JSON válido (sem markdown, sem texto extra):
{
  "action": "<nome_da_ação>",
  "params": {<parâmetros específicos da ação>},
  "response_text": "<o que Athena deve falar em português>",
  "priority": "low | medium | high | critical",
  "confidence": 0.0-1.0
}

AÇÕES DISPONÍVEIS E SEUS PARÂMETROS:

Sistema e Terminal:
  shell         → {"cmd": "comando bash/cmd"}
  powershell    → {"cmd": "comando ps"}
  system_info   → {}
  list_processes→ {}
  kill_process  → {"name": "processo.exe"} | {"pid": 1234}
  open_app      → {"app": "nome_do_app"}

Arquivos:
  read_file     → {"path": "/caminho/arquivo.txt"}
  write_file    → {"path": "/caminho/arquivo.txt", "content": "conteúdo", "mode": "w|a"}
  delete_file   → {"path": "/caminho"}
  list_dir      → {"path": "/caminho"}
  move_file     → {"src": "/origem", "dst": "/destino"}
  copy_file     → {"src": "/origem", "dst": "/destino"}

Web:
  open_url      → {"url": "https://..."}
  web_search    → {"query": "termos de busca"}
  http_get      → {"url": "https://api...", "headers": {}}
  http_post     → {"url": "https://...", "data": {}, "headers": {}}

Mouse e Teclado:
  click         → {"x": 100, "y": 200, "button": "left|right"}
  double_click  → {"x": 100, "y": 200}
  type_text     → {"text": "texto a digitar", "interval": 0.05}
  hotkey        → {"keys": ["ctrl", "c"]}
  press_key     → {"key": "enter|tab|esc|..."}
  scroll        → {"amount": 3}
  move_mouse    → {"x": 100, "y": 200}

Captura:
  screenshot    → {"path": "~/screenshot.png"}
  screenshot_base64 → {}

Clipboard:
  clipboard_set → {"text": "texto"}
  clipboard_get → {}

Relatórios:
  create_report → {"title": "Título", "content": "Conteúdo", "path": "~/relatorio.txt", "open": true}

Outros:
  speak         → {"text": "texto para falar"}
  wait          → {"seconds": 2}

EXEMPLOS:
  "abra o chrome" → {"action":"open_app","params":{"app":"chrome"},"response_text":"Abrindo o Chrome","priority":"low","confidence":0.9}
  "pesquise clima em São Paulo" → {"action":"web_search","params":{"query":"clima São Paulo hoje"},"response_text":"Pesquisando clima em São Paulo","priority":"low","confidence":0.95}
  "qual é o uso de CPU?" → {"action":"system_info","params":{},"response_text":"Verificando informações do sistema","priority":"low","confidence":0.95}
  "crie um relatório de vendas" → {"action":"create_report","params":{"title":"Relatório de Vendas","content":"Relatório gerado em [DATA]","path":"~/relatorio_vendas.txt"},"response_text":"Criando relatório de vendas","priority":"medium","confidence":0.8}
  "ALERTA: CPU 100%" → {"action":"system_info","params":{},"response_text":"Verificando situação crítica do sistema","priority":"critical","confidence":0.99}

Se o comando for ambíguo, escolha a ação mais provável e indique confidence < 0.7.
Sempre responda em português no campo response_text."""


async def interpret_command(command: str) -> dict:
    """Interpreta comando em linguagem natural com Claude"""

    # Fallback se sem API key
    if not claude:
        return {
            "action": "shell",
            "params": {"cmd": command},
            "response_text": f"Executando: {command}",
            "priority": "low",
            "confidence": 0.5
        }

    try:
        response = claude.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": command}]
        )
        raw = response.content[0].text.strip()
        # Remove possíveis backticks
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)

    except json.JSONDecodeError as e:
        log.error(f"Claude retornou JSON inválido: {e}")
        return {
            "action": "shell",
            "params": {"cmd": command},
            "response_text": "Executando comando diretamente",
            "priority": "low",
            "confidence": 0.3
        }
    except Exception as e:
        log.error(f"Erro ao chamar Claude: {e}")
        raise HTTPException(500, f"Erro na interpretação: {str(e)}")


# ─── PERSISTÊNCIA ─────────────────────────────────────────────────────────────

async def save_command(data: dict):
    if HAS_SUPABASE:
        try:
            supabase.table("commands").insert(data).execute()
        except Exception as e:
            log.warning(f"Supabase write error: {e}")
    else:
        memory_store["commands"][data["id"]] = data


async def save_result(data: dict):
    if HAS_SUPABASE:
        try:
            supabase.table("results").insert(data).execute()
        except Exception as e:
            log.warning(f"Supabase write error: {e}")
    else:
        memory_store["results"][data["command_id"]] = data


async def get_history(limit=50):
    if HAS_SUPABASE:
        try:
            r = supabase.table("commands").select("*, results(*)").order("created_at", desc=True).limit(limit).execute()
            return r.data
        except Exception as e:
            log.warning(f"Supabase read error: {e}")
    return list(memory_store["commands"].values())[-limit:]


# ─── MODELOS ──────────────────────────────────────────────────────────────────

class CommandRequest(BaseModel):
    command: str
    agent_id: str
    source: str = "dashboard"  # dashboard | voice | api | auto
    force_ai: bool = False
    priority_override: Optional[str] = None


class AlertRequest(BaseModel):
    level: str  # info | warning | high | critical
    title: str
    message: str
    agent_id: Optional[str] = None


# ─── ENDPOINTS ────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    return """<html><body style='font-family:monospace;background:#0a0a0a;color:#00ff88;padding:40px'>
    <h1>🤖 Athena Cloud Backend</h1>
    <p>Status: <b style='color:lime'>ONLINE</b></p>
    <p>Docs: <a href='/docs' style='color:#00aaff'>/docs</a></p>
    <p>WebSocket: <code>wss://[host]/ws/[agent-id]</code></p>
    </body></html>"""


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "agents_online": len(manager.agents),
        "supabase": HAS_SUPABASE,
        "ai": bool(claude)
    }


# ─── WEBSOCKET para agentes ───────────────────────────────────────────────────

@app.websocket("/ws/{agent_id}")
async def websocket_agent(ws: WebSocket, agent_id: str):
    await ws.accept()

    # Aguarda registro inicial
    try:
        raw = await asyncio.wait_for(ws.receive_json(), timeout=10)
    except asyncio.TimeoutError:
        await ws.close(1008, "Registration timeout")
        return

    info = {
        "platform": raw.get("platform", "unknown"),
        "hostname": raw.get("hostname", "unknown"),
        "capabilities": raw.get("capabilities", []),
        "stats": raw.get("stats", {})
    }
    await manager.connect(agent_id, ws, info)

    await ws.send_json({"type": "welcome", "message": f"Athena Cloud conectado. Agente {agent_id} registrado."})

    try:
        async for raw in ws:
            data = json.loads(raw) if isinstance(raw, str) else raw

            if data.get("type") == "result":
                cmd_id = data.get("command_id")
                result_data = {
                    "command_id": cmd_id,
                    "agent_id": agent_id,
                    "success": data.get("success"),
                    "output": data.get("output"),
                    "duration": data.get("duration"),
                    "timestamp": data.get("timestamp", datetime.now().isoformat())
                }
                await save_result(result_data)
                log.info(f"📊 Resultado de {agent_id} [{cmd_id[:8]}]: {'✅' if data.get('success') else '❌'}")

            elif data.get("type") == "register":
                pass  # já registrado

    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.error(f"Erro WebSocket {agent_id}: {e}")
    finally:
        manager.disconnect(agent_id)


# ─── ENVIAR COMANDO ───────────────────────────────────────────────────────────

@app.post("/command")
async def send_command(req: CommandRequest):
    """Interpreta e envia comando para o agente"""

    # Verificar agente
    if req.agent_id not in manager.agents:
        # Verificar se existe mas offline
        if req.agent_id in manager.agent_info:
            raise HTTPException(503, f"Agente '{req.agent_id}' está offline")
        raise HTTPException(404, f"Agente '{req.agent_id}' não encontrado")

    # Interpretar com IA
    log.info(f"🧠 Interpretando: {req.command}")
    action_data = await interpret_command(req.command)

    # Override de prioridade
    if req.priority_override:
        action_data["priority"] = req.priority_override

    cmd_id = str(uuid.uuid4())
    payload = {
        "type": "command",
        "command_id": cmd_id,
        "command": req.command,
        "action": action_data["action"],
        "params": action_data.get("params", {}),
        "response_text": action_data.get("response_text", ""),
        "priority": action_data.get("priority", "low"),
        "source": req.source,
        "force_ai": req.force_ai,
        "timestamp": datetime.now().isoformat()
    }

    # Salvar
    await save_command({
        "id": cmd_id,
        "agent_id": req.agent_id,
        "command": req.command,
        "action": action_data["action"],
        "params": json.dumps(action_data.get("params", {})),
        "priority": action_data.get("priority", "low"),
        "source": req.source,
        "confidence": action_data.get("confidence", 1.0),
        "created_at": datetime.now().isoformat()
    })

    # Enviar
    sent = await manager.send_command(req.agent_id, payload)
    if not sent:
        raise HTTPException(503, "Falha ao enviar comando para o agente")

    log.info(f"📤 Comando enviado [{cmd_id[:8]}]: {action_data['action']} → {req.agent_id}")

    return {
        "command_id": cmd_id,
        "action": action_data["action"],
        "params": action_data.get("params", {}),
        "response_text": action_data.get("response_text", ""),
        "priority": action_data.get("priority", "low"),
        "confidence": action_data.get("confidence", 1.0),
        "status": "sent"
    }


@app.post("/command/broadcast")
async def broadcast_command(req: CommandRequest):
    """Envia comando para TODOS os agentes conectados"""
    action_data = await interpret_command(req.command)
    cmd_id = str(uuid.uuid4())

    payload = {
        "type": "command",
        "command_id": cmd_id,
        "command": req.command,
        "action": action_data["action"],
        "params": action_data.get("params", {}),
        "response_text": action_data.get("response_text", ""),
        "priority": action_data.get("priority", "low"),
        "source": "broadcast",
        "timestamp": datetime.now().isoformat()
    }

    await manager.broadcast(payload)
    return {"command_id": cmd_id, "sent_to": len(manager.agents), "action": action_data}


# ─── ALERTAS ─────────────────────────────────────────────────────────────────

@app.post("/alert")
async def create_alert(req: AlertRequest):
    alert = {
        "id": str(uuid.uuid4()),
        "level": req.level,
        "title": req.title,
        "message": req.message,
        "agent_id": req.agent_id,
        "seen": False,
        "timestamp": datetime.now().isoformat()
    }
    memory_store["alerts"].append(alert)

    # Notificar agente se especificado
    if req.agent_id and req.agent_id in manager.agents:
        await manager.send_command(req.agent_id, {
            "type": "alert",
            **alert
        })

    return alert


@app.get("/alerts")
async def get_alerts(unseen_only: bool = False):
    alerts = memory_store["alerts"]
    if unseen_only:
        alerts = [a for a in alerts if not a.get("seen")]
    return {"alerts": list(reversed(alerts[-100:]))}


@app.put("/alerts/{alert_id}/seen")
async def mark_alert_seen(alert_id: str):
    for a in memory_store["alerts"]:
        if a["id"] == alert_id:
            a["seen"] = True
    return {"ok": True}


# ─── AGENTES ──────────────────────────────────────────────────────────────────

@app.get("/agents")
async def list_agents():
    return {"agents": manager.get_agents(), "total": len(manager.agent_info), "online": len(manager.agents)}


@app.get("/agents/{agent_id}")
async def get_agent(agent_id: str):
    info = manager.agent_info.get(agent_id)
    if not info:
        raise HTTPException(404, "Agente não encontrado")
    return {**info, "id": agent_id, "online": agent_id in manager.agents}


# ─── HISTÓRICO ────────────────────────────────────────────────────────────────

@app.get("/history")
async def history(limit: int = 50):
    return {"commands": await get_history(limit)}


# ─── INTERPRET (sem executar) ─────────────────────────────────────────────────

@app.post("/interpret")
async def interpret_only(data: dict):
    command = data.get("command", "")
    result = await interpret_command(command)
    return result


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("athena_backend:app", host="0.0.0.0", port=8000, reload=True)
