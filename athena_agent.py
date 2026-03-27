"""
╔══════════════════════════════════════════════════════╗
║           ATHENA AGENT — Desktop Client              ║
║   Executa comandos remotos no PC em segundo plano    ║
╚══════════════════════════════════════════════════════╝

Instalação das dependências:
    pip install websockets speechrecognition pyttsx3 pyautogui
                pillow requests sqlite3 pystray Pillow selenium
                webdriver-manager psutil pyperclip

Uso:
    python athena_agent.py --id meu-pc --url wss://seu-backend.railway.app
"""

import asyncio
import websockets
import json
import subprocess
import os
import sys
import threading
import time
import sqlite3
import argparse
import logging
import uuid
import base64
from datetime import datetime
from io import BytesIO

# Configuração de log
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ATHENA] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("athena_agent.log")
    ]
)
log = logging.getLogger("athena")

# ─── Importações opcionais (não quebra se não tiver) ─────────────────────────
try:
    import pyautogui
    pyautogui.FAILSAFE = True
    HAS_PYAUTOGUI = True
except ImportError:
    HAS_PYAUTOGUI = False
    log.warning("pyautogui não instalado — ações de mouse/teclado desativadas")

try:
    import speech_recognition as sr
    HAS_SR = True
except ImportError:
    HAS_SR = False
    log.warning("SpeechRecognition não instalado — voz desativada")

try:
    import pyttsx3
    HAS_TTS = True
except ImportError:
    HAS_TTS = False
    log.warning("pyttsx3 não instalado — resposta por voz desativada")

try:
    import pyperclip
    HAS_CLIPBOARD = True
except ImportError:
    HAS_CLIPBOARD = False

try:
    import webbrowser
    HAS_BROWSER = True
except ImportError:
    HAS_BROWSER = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

import requests


# ─── BANCO DE DADOS LOCAL ─────────────────────────────────────────────────────

class AthenaDB:
    """Banco SQLite local para histórico e aprendizado"""

    def __init__(self, path="athena_local.db"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self._init_tables()

    def _init_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS command_history (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                cmd_id    TEXT UNIQUE,
                command   TEXT,
                action    TEXT,
                params    TEXT,
                result    TEXT,
                success   INTEGER,
                duration  REAL,
                timestamp TEXT
            );

            CREATE TABLE IF NOT EXISTS learned_patterns (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern       TEXT UNIQUE,
                action        TEXT,
                params        TEXT,
                success_count INTEGER DEFAULT 0,
                fail_count    INTEGER DEFAULT 0,
                avg_duration  REAL DEFAULT 0,
                last_used     TEXT,
                confidence    REAL DEFAULT 0.5
            );

            CREATE TABLE IF NOT EXISTS alerts (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                level     TEXT,
                title     TEXT,
                message   TEXT,
                seen      INTEGER DEFAULT 0,
                timestamp TEXT
            );

            CREATE TABLE IF NOT EXISTS agent_stats (
                key   TEXT PRIMARY KEY,
                value TEXT
            );
        """)
        self.conn.commit()

    def log_command(self, cmd_id, command, action, params, result, success, duration):
        self.conn.execute(
            """INSERT OR REPLACE INTO command_history
               (cmd_id, command, action, params, result, success, duration, timestamp)
               VALUES (?,?,?,?,?,?,?,?)""",
            (cmd_id, command, action, json.dumps(params), result, int(success),
             duration, datetime.now().isoformat())
        )
        self.conn.commit()

    def update_pattern(self, command, action, params, success, duration):
        """Aprende com cada execução"""
        key = command.lower().strip()[:100]
        existing = self.conn.execute(
            "SELECT * FROM learned_patterns WHERE pattern=?", (key,)
        ).fetchone()

        if existing:
            s = existing[4] + (1 if success else 0)
            f = existing[5] + (0 if success else 1)
            total = s + f
            confidence = s / total if total > 0 else 0.5
            new_avg = (existing[6] * (total - 1) + duration) / total
            self.conn.execute(
                """UPDATE learned_patterns
                   SET success_count=?, fail_count=?, confidence=?,
                       avg_duration=?, last_used=?, action=?, params=?
                   WHERE pattern=?""",
                (s, f, confidence, new_avg, datetime.now().isoformat(),
                 action, json.dumps(params), key)
            )
        else:
            self.conn.execute(
                """INSERT INTO learned_patterns
                   (pattern, action, params, success_count, fail_count, confidence, avg_duration, last_used)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (key, action, json.dumps(params),
                 1 if success else 0,
                 0 if success else 1,
                 0.8 if success else 0.2,
                 duration, datetime.now().isoformat())
            )
        self.conn.commit()

    def find_pattern(self, command):
        """Tenta encontrar padrão aprendido — retorna ação se confiança > 0.8"""
        key = command.lower().strip()[:100]
        row = self.conn.execute(
            "SELECT action, params, confidence FROM learned_patterns WHERE pattern=? AND confidence > 0.8",
            (key,)
        ).fetchone()
        if row:
            return {"action": row[0], "params": json.loads(row[1]), "confidence": row[2]}
        return None

    def add_alert(self, level, title, message):
        self.conn.execute(
            "INSERT INTO alerts (level, title, message, timestamp) VALUES (?,?,?,?)",
            (level, title, message, datetime.now().isoformat())
        )
        self.conn.commit()

    def get_stats(self):
        total = self.conn.execute("SELECT COUNT(*) FROM command_history").fetchone()[0]
        success = self.conn.execute("SELECT COUNT(*) FROM command_history WHERE success=1").fetchone()[0]
        patterns = self.conn.execute("SELECT COUNT(*) FROM learned_patterns WHERE confidence>0.8").fetchone()[0]
        return {"total_commands": total, "success_rate": (success/total*100) if total else 0, "learned_patterns": patterns}


# ─── MOTOR DE VOZ ─────────────────────────────────────────────────────────────

class VoiceEngine:
    def __init__(self):
        self.tts = None
        self.recognizer = None
        self._init()

    def _init(self):
        if HAS_TTS:
            try:
                self.tts = pyttsx3.init()
                self.tts.setProperty('rate', 175)
                voices = self.tts.getProperty('voices')
                # Prefere voz em português
                for v in voices:
                    if 'pt' in v.id.lower() or 'brazil' in v.id.lower() or 'portuguese' in v.name.lower():
                        self.tts.setProperty('voice', v.id)
                        break
                log.info("Motor de voz TTS iniciado")
            except Exception as e:
                log.error(f"Erro ao iniciar TTS: {e}")

        if HAS_SR:
            self.recognizer = sr.Recognizer()
            self.recognizer.pause_threshold = 0.8
            self.recognizer.energy_threshold = 300
            log.info("Reconhecimento de voz iniciado")

    def speak(self, text):
        if not self.tts:
            log.info(f"[ATHENA FALA]: {text}")
            return
        try:
            self.tts.say(text)
            self.tts.runAndWait()
        except Exception as e:
            log.error(f"Erro TTS: {e}")

    def listen(self, timeout=8, language="pt-BR"):
        if not self.recognizer:
            return None
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                log.info("🎤 Aguardando comando de voz...")
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=15)
                text = self.recognizer.recognize_google(audio, language=language)
                log.info(f"🎤 Reconhecido: {text}")
                return text
        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            return None
        except Exception as e:
            log.error(f"Erro no reconhecimento: {e}")
            return None


# ─── EXECUTOR DE AÇÕES ────────────────────────────────────────────────────────

class ActionExecutor:
    """Executa qualquer ação no PC"""

    def __init__(self, voice: VoiceEngine, db: AthenaDB):
        self.voice = voice
        self.db = db

    def execute(self, action: str, params: dict) -> dict:
        handler = getattr(self, f"_do_{action}", self._do_unknown)
        try:
            result = handler(params)
            return {"success": True, "output": result}
        except Exception as e:
            return {"success": False, "output": str(e)}

    # ── Shell / Terminal ──────────────────────────────────────────────────────
    def _do_shell(self, p):
        cmd = p.get("cmd", "")
        timeout = p.get("timeout", 30)
        log.info(f"Shell: {cmd}")
        proc = subprocess.run(
            cmd, shell=True, capture_output=True,
            text=True, timeout=timeout, encoding='utf-8', errors='replace'
        )
        out = proc.stdout.strip() or proc.stderr.strip()
        return out[:2000] if out else "(sem saída)"

    def _do_powershell(self, p):
        cmd = p.get("cmd", "")
        ps = f'powershell -Command "{cmd}"'
        return self._do_shell({"cmd": ps})

    # ── Arquivos ──────────────────────────────────────────────────────────────
    def _do_read_file(self, p):
        path = os.path.expanduser(p.get("path", ""))
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        return content[:5000]

    def _do_write_file(self, p):
        path = os.path.expanduser(p.get("path", ""))
        content = p.get("content", "")
        mode = p.get("mode", "w")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, mode, encoding='utf-8') as f:
            f.write(content)
        return f"Arquivo salvo: {path}"

    def _do_delete_file(self, p):
        path = os.path.expanduser(p.get("path", ""))
        if os.path.isfile(path):
            os.remove(path)
            return f"Arquivo deletado: {path}"
        elif os.path.isdir(path):
            import shutil
            shutil.rmtree(path)
            return f"Pasta deletada: {path}"
        return "Arquivo não encontrado"

    def _do_list_dir(self, p):
        path = os.path.expanduser(p.get("path", os.path.expanduser("~")))
        items = os.listdir(path)
        return "\n".join(items[:100])

    def _do_move_file(self, p):
        import shutil
        src = os.path.expanduser(p.get("src", ""))
        dst = os.path.expanduser(p.get("dst", ""))
        shutil.move(src, dst)
        return f"Movido: {src} → {dst}"

    def _do_copy_file(self, p):
        import shutil
        src = os.path.expanduser(p.get("src", ""))
        dst = os.path.expanduser(p.get("dst", ""))
        shutil.copy2(src, dst)
        return f"Copiado: {src} → {dst}"

    # ── Web / Browser ─────────────────────────────────────────────────────────
    def _do_open_url(self, p):
        url = p.get("url", "")
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        webbrowser.open(url)
        return f"Abrindo: {url}"

    def _do_web_search(self, p):
        query = p.get("query", "")
        url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        webbrowser.open(url)
        return f"Pesquisando: {query}"

    def _do_http_get(self, p):
        url = p.get("url", "")
        headers = p.get("headers", {})
        r = requests.get(url, headers=headers, timeout=15)
        return r.text[:3000]

    def _do_http_post(self, p):
        url = p.get("url", "")
        data = p.get("data", {})
        headers = p.get("headers", {})
        r = requests.post(url, json=data, headers=headers, timeout=15)
        return r.text[:3000]

    # ── Mouse e Teclado ───────────────────────────────────────────────────────
    def _do_click(self, p):
        if not HAS_PYAUTOGUI: return "pyautogui não disponível"
        x, y = p.get("x"), p.get("y")
        button = p.get("button", "left")
        if x is None or y is None:
            pyautogui.click(button=button)
        else:
            pyautogui.click(x=x, y=y, button=button)
        return f"Clique em ({x},{y})"

    def _do_double_click(self, p):
        if not HAS_PYAUTOGUI: return "pyautogui não disponível"
        pyautogui.doubleClick(p.get("x"), p.get("y"))
        return "Double click"

    def _do_type_text(self, p):
        if not HAS_PYAUTOGUI: return "pyautogui não disponível"
        text = p.get("text", "")
        interval = p.get("interval", 0.05)
        pyautogui.typewrite(text, interval=interval)
        return f"Digitado: {text[:50]}"

    def _do_hotkey(self, p):
        if not HAS_PYAUTOGUI: return "pyautogui não disponível"
        keys = p.get("keys", [])
        pyautogui.hotkey(*keys)
        return f"Atalho: {'+'.join(keys)}"

    def _do_press_key(self, p):
        if not HAS_PYAUTOGUI: return "pyautogui não disponível"
        key = p.get("key", "")
        pyautogui.press(key)
        return f"Tecla pressionada: {key}"

    def _do_scroll(self, p):
        if not HAS_PYAUTOGUI: return "pyautogui não disponível"
        amount = p.get("amount", 3)
        pyautogui.scroll(amount)
        return f"Scroll: {amount}"

    def _do_move_mouse(self, p):
        if not HAS_PYAUTOGUI: return "pyautogui não disponível"
        pyautogui.moveTo(p.get("x"), p.get("y"), duration=0.3)
        return "Mouse movido"

    # ── Screenshot ────────────────────────────────────────────────────────────
    def _do_screenshot(self, p):
        if not HAS_PYAUTOGUI: return "pyautogui não disponível"
        path = p.get("path", f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        path = os.path.expanduser(path)
        img = pyautogui.screenshot()
        img.save(path)
        return f"Screenshot salvo: {path}"

    def _do_screenshot_base64(self, p):
        if not HAS_PYAUTOGUI: return "pyautogui não disponível"
        img = pyautogui.screenshot()
        buf = BytesIO()
        img.save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode()

    # ── Sistema ───────────────────────────────────────────────────────────────
    def _do_system_info(self, p):
        info = {
            "os": sys.platform,
            "python": sys.version,
            "cwd": os.getcwd(),
            "user": os.getenv("USERNAME") or os.getenv("USER"),
        }
        if HAS_PSUTIL:
            info["cpu_percent"] = psutil.cpu_percent(interval=1)
            info["ram_percent"] = psutil.virtual_memory().percent
            info["disk_percent"] = psutil.disk_usage('/').percent
        return json.dumps(info, indent=2)

    def _do_list_processes(self, p):
        if not HAS_PSUTIL: return "psutil não disponível"
        procs = [f"{p.pid}: {p.name()}" for p in psutil.process_iter(['pid', 'name'])]
        return "\n".join(procs[:50])

    def _do_kill_process(self, p):
        if not HAS_PSUTIL: return "psutil não disponível"
        name = p.get("name", "")
        pid = p.get("pid")
        killed = []
        for proc in psutil.process_iter(['pid', 'name']):
            if (name and name.lower() in proc.name().lower()) or (pid and proc.pid == pid):
                proc.kill()
                killed.append(f"{proc.pid}: {proc.name()}")
        return f"Encerrados: {killed}" if killed else "Processo não encontrado"

    def _do_open_app(self, p):
        app = p.get("app", "")
        return self._do_shell({"cmd": f"start {app}" if sys.platform == "win32" else f"open {app}" if sys.platform == "darwin" else f"xdg-open {app}"})

    # ── Clipboard ─────────────────────────────────────────────────────────────
    def _do_clipboard_set(self, p):
        if not HAS_CLIPBOARD: return "pyperclip não disponível"
        pyperclip.copy(p.get("text", ""))
        return "Texto copiado para clipboard"

    def _do_clipboard_get(self, p):
        if not HAS_CLIPBOARD: return "pyperclip não disponível"
        return pyperclip.paste()

    # ── Voz ───────────────────────────────────────────────────────────────────
    def _do_speak(self, p):
        self.voice.speak(p.get("text", ""))
        return "Falado"

    # ── Relatório ─────────────────────────────────────────────────────────────
    def _do_create_report(self, p):
        title = p.get("title", "Relatório Athena")
        content = p.get("content", "")
        path = p.get("path", f"relatorio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        path = os.path.expanduser(path)
        report = f"{'='*60}\n{title}\nGerado em: {datetime.now()}\n{'='*60}\n\n{content}"
        with open(path, 'w', encoding='utf-8') as f:
            f.write(report)
        if p.get("open", True):
            webbrowser.open(f"file://{os.path.abspath(path)}")
        return f"Relatório criado: {path}"

    def _do_wait(self, p):
        secs = p.get("seconds", 1)
        time.sleep(secs)
        return f"Aguardado {secs}s"

    def _do_unknown(self, p):
        return "Ação desconhecida"


# ─── AGENT PRINCIPAL ──────────────────────────────────────────────────────────

class AthenaAgent:
    def __init__(self, agent_id: str, backend_url: str):
        self.agent_id = agent_id
        self.backend_url = backend_url.rstrip("/")
        self.ws_url = f"{self.backend_url}/ws/{agent_id}"
        self.db = AthenaDB()
        self.voice = VoiceEngine()
        self.executor = ActionExecutor(self.voice, self.db)
        self.running = True
        self.ws = None
        self.reconnect_delay = 3
        self.voice_thread = None
        self.voice_active = HAS_SR

        log.info(f"🤖 Athena Agent iniciado | ID: {agent_id}")
        self.voice.speak(f"Athena iniciada. Agente {agent_id} pronto e aguardando comandos.")

    # ── WebSocket ─────────────────────────────────────────────────────────────
    async def connect_loop(self):
        while self.running:
            try:
                log.info(f"Conectando ao backend: {self.ws_url}")
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=30,
                    ping_timeout=10,
                    extra_headers={"X-Agent-ID": self.agent_id}
                ) as ws:
                    self.ws = ws
                    self.reconnect_delay = 3
                    log.info("✅ Conectado ao Athena Cloud")
                    self.voice.speak("Conexão com Athena Cloud estabelecida.")

                    await ws.send(json.dumps({
                        "type": "register",
                        "agent_id": self.agent_id,
                        "platform": sys.platform,
                        "hostname": os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME", "unknown"),
                        "capabilities": self._get_capabilities(),
                        "stats": self.db.get_stats()
                    }))

                    async for raw in ws:
                        await self._on_message(ws, raw)

            except (websockets.exceptions.ConnectionClosed,
                    ConnectionRefusedError, OSError) as e:
                log.warning(f"Desconectado: {e}. Reconectando em {self.reconnect_delay}s...")
                self.ws = None
                await asyncio.sleep(self.reconnect_delay)
                self.reconnect_delay = min(self.reconnect_delay * 2, 60)
            except Exception as e:
                log.error(f"Erro inesperado: {e}")
                await asyncio.sleep(self.reconnect_delay)

    async def _on_message(self, ws, raw):
        try:
            data = json.loads(raw)
            msg_type = data.get("type")

            if msg_type == "command":
                await self._handle_command(ws, data)
            elif msg_type == "ping":
                await ws.send(json.dumps({"type": "pong", "agent_id": self.agent_id}))
            elif msg_type == "set_voice":
                self.voice_active = data.get("enabled", True)
            elif msg_type == "shutdown":
                log.info("Comando de desligamento recebido")
                self.running = False

        except json.JSONDecodeError:
            log.error("Mensagem inválida recebida")
        except Exception as e:
            log.error(f"Erro ao processar mensagem: {e}")

    async def _handle_command(self, ws, data):
        cmd_id = data.get("command_id", str(uuid.uuid4()))
        command = data.get("command", "")
        action = data.get("action", "shell")
        params = data.get("params", {})
        priority = data.get("priority", "low")
        response_text = data.get("response_text", "")

        log.info(f"📩 Comando [{priority.upper()}]: {command}")
        self.db.add_alert(priority, f"Comando: {action}", command)

        # Falar confirmação
        if self.voice_active and priority in ("high", "critical"):
            self.voice.speak(f"Executando: {command[:60]}")

        start = time.time()

        # Verificar padrão aprendido
        learned = self.db.find_pattern(command)
        if learned and not data.get("force_ai"):
            log.info(f"📚 Padrão aprendido encontrado (confiança: {learned['confidence']:.0%})")
            action = learned["action"]
            params = learned["params"]

        # Executar em thread separada (não bloqueia o event loop)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.executor.execute(action, params)
        )

        duration = time.time() - start

        # Aprender com execução
        self.db.log_command(cmd_id, command, action, params,
                            str(result["output"])[:500],
                            result["success"], duration)
        self.db.update_pattern(command, action, params, result["success"], duration)

        log.info(f"{'✅' if result['success'] else '❌'} Resultado ({duration:.2f}s): {str(result['output'])[:100]}")

        # Resposta por voz
        if self.voice_active:
            speak_text = response_text if response_text else (
                f"Concluído em {duration:.1f} segundos" if result["success"]
                else f"Falhou: {str(result['output'])[:80]}"
            )
            threading.Thread(target=self.voice.speak, args=(speak_text,), daemon=True).start()

        # Enviar resultado de volta
        if self.ws:
            await ws.send(json.dumps({
                "type": "result",
                "command_id": cmd_id,
                "agent_id": self.agent_id,
                "action": action,
                "success": result["success"],
                "output": str(result["output"])[:10000],
                "duration": duration,
                "timestamp": datetime.now().isoformat()
            }))

    # ── Loop de voz local ─────────────────────────────────────────────────────
    def _voice_loop(self):
        """Escuta voz localmente e envia para o backend"""
        wake_words = ["athena", "atena"]
        log.info("🎤 Loop de voz local ativado. Diga 'Athena' para ativar.")

        while self.running:
            try:
                text = self.voice.listen(timeout=10)
                if not text:
                    continue

                lower = text.lower()
                if any(w in lower for w in wake_words):
                    self.voice.speak("Sim, pode falar.")
                    command = self.voice.listen(timeout=8)
                    if command:
                        log.info(f"🎤 Comando de voz: {command}")
                        self._send_voice_command(command)
            except Exception as e:
                log.debug(f"Erro no loop de voz: {e}")
                time.sleep(1)

    def _send_voice_command(self, command: str):
        """Envia comando de voz para o backend via HTTP"""
        try:
            r = requests.post(
                f"{self.backend_url.replace('ws://', 'http://').replace('wss://', 'https://')}/command",
                json={"command": command, "agent_id": self.agent_id, "source": "voice"},
                timeout=10
            )
            if r.status_code != 200:
                self.voice.speak("Não consegui enviar o comando.")
        except Exception as e:
            log.error(f"Erro ao enviar comando de voz: {e}")
            self.voice.speak("Sem conexão com o servidor.")

    # ── Capacidades ───────────────────────────────────────────────────────────
    def _get_capabilities(self):
        caps = ["shell", "read_file", "write_file", "delete_file", "list_dir",
                "open_url", "web_search", "http_get", "http_post", "system_info",
                "speak", "wait", "create_report", "clipboard_set", "clipboard_get",
                "open_app", "screenshot_base64"]
        if HAS_PYAUTOGUI:
            caps += ["click", "double_click", "type_text", "hotkey", "press_key",
                     "scroll", "move_mouse", "screenshot"]
        if HAS_PSUTIL:
            caps += ["list_processes", "kill_process"]
        return caps

    # ── Start ─────────────────────────────────────────────────────────────────
    def start(self):
        # Thread de voz local
        if self.voice_active:
            self.voice_thread = threading.Thread(target=self._voice_loop, daemon=True)
            self.voice_thread.start()

        # Loop principal WebSocket
        asyncio.run(self.connect_loop())


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Athena Desktop Agent")
    parser.add_argument("--id", default="agente-1", help="ID único deste agente")
    parser.add_argument("--url", default="ws://localhost:8000",
                        help="URL do backend Athena (ex: wss://athena.railway.app)")
    args = parser.parse_args()

    print("""
    ╔══════════════════════════════════════╗
    ║   🤖  A T H E N A   A G E N T       ║
    ║   Sistema Autônomo de Controle       ║
    ╚══════════════════════════════════════╝
    """)

    agent = AthenaAgent(agent_id=args.id, backend_url=args.url)
    try:
        agent.start()
    except KeyboardInterrupt:
        log.info("Athena Agent encerrado pelo usuário.")


if __name__ == "__main__":
    main()
