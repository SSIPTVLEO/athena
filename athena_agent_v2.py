"""
╔══════════════════════════════════════════════════════════╗
║         ATHENA AGENT v2.0 — Desktop Client               ║
║                                                          ║
║  Integrações:                                            ║
║  • AthenaSeleniumEngine  — controle total de browser     ║
║  • AthenaLearningEngine  — aprendizado avançado TF-IDF   ║
║  • AthenaDB              — banco local + histórico       ║
║  • VoiceEngine           — STT + TTS em português        ║
║  • ActionExecutor        — 40+ ações nativas             ║
╚══════════════════════════════════════════════════════════╝

Instalação:
    pip install websockets speechrecognition pyttsx3 pyautogui pillow
                requests pyperclip psutil selenium webdriver-manager
                numpy scipy scikit-learn

Uso:
    python athena_agent.py --id meu-pc --url wss://athena.railway.app
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

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ATHENA] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("athena_agent.log", encoding="utf-8")
    ]
)
log = logging.getLogger("athena")

# ─── Módulos Athena ───────────────────────────────────────────────────────────
try:
    from athena_selenium import AthenaSeleniumEngine
    HAS_SELENIUM_ENGINE = True
    log.info("✅ AthenaSeleniumEngine carregado")
except ImportError as e:
    HAS_SELENIUM_ENGINE = False
    log.warning(f"AthenaSeleniumEngine não disponível: {e}")

try:
    from athena_learning import AthenaLearningEngine
    HAS_LEARNING = True
    log.info("✅ AthenaLearningEngine carregado")
except ImportError as e:
    HAS_LEARNING = False
    log.warning(f"AthenaLearningEngine não disponível: {e}")

# ─── Importações opcionais ────────────────────────────────────────────────────
try:
    import pyautogui
    pyautogui.FAILSAFE = True
    HAS_PYAUTOGUI = True
except ImportError:
    HAS_PYAUTOGUI = False
    log.warning("pyautogui não instalado")

try:
    import speech_recognition as sr
    HAS_SR = True
except ImportError:
    HAS_SR = False
    log.warning("SpeechRecognition não instalado")

try:
    import pyttsx3
    HAS_TTS = True
except ImportError:
    HAS_TTS = False
    log.warning("pyttsx3 não instalado")

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


# ════════════════════════════════════════════════════════════════════
#  BANCO DE DADOS LOCAL (compatibilidade + legacy)
# ════════════════════════════════════════════════════════════════════

class AthenaDB:
    def __init__(self, path="athena_local.db"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self._init_tables()

    def _init_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS command_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cmd_id TEXT UNIQUE, command TEXT, action TEXT, params TEXT,
                result TEXT, success INTEGER, duration REAL, timestamp TEXT
            );
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                level TEXT, title TEXT, message TEXT,
                seen INTEGER DEFAULT 0, timestamp TEXT
            );
        """)
        self.conn.commit()

    def log_command(self, cmd_id, command, action, params, result, success, duration):
        self.conn.execute(
            """INSERT OR REPLACE INTO command_history
               (cmd_id, command, action, params, result, success, duration, timestamp)
               VALUES (?,?,?,?,?,?,?,?)""",
            (cmd_id, command, action, json.dumps(params), result[:500],
             int(success), duration, datetime.now().isoformat())
        )
        self.conn.commit()

    def add_alert(self, level, title, message):
        self.conn.execute(
            "INSERT INTO alerts (level, title, message, timestamp) VALUES (?,?,?,?)",
            (level, title, message, datetime.now().isoformat())
        )
        self.conn.commit()

    def get_stats(self):
        total   = self.conn.execute("SELECT COUNT(*) FROM command_history").fetchone()[0]
        success = self.conn.execute("SELECT COUNT(*) FROM command_history WHERE success=1").fetchone()[0]
        return {"total_commands": total, "success_rate": (success/total*100) if total else 0}


# ════════════════════════════════════════════════════════════════════
#  MOTOR DE VOZ
# ════════════════════════════════════════════════════════════════════

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
                for v in voices:
                    if any(k in v.id.lower() for k in ('pt', 'brazil', 'portuguese')):
                        self.tts.setProperty('voice', v.id)
                        break
                log.info("TTS iniciado")
            except Exception as e:
                log.error(f"TTS erro: {e}")

        if HAS_SR:
            self.recognizer = sr.Recognizer()
            self.recognizer.pause_threshold = 0.8
            self.recognizer.energy_threshold = 300

    def speak(self, text: str):
        if not self.tts:
            log.info(f"[ATHENA]: {text}")
            return
        try:
            self.tts.say(text)
            self.tts.runAndWait()
        except Exception as e:
            log.error(f"TTS falar: {e}")

    def listen(self, timeout=8, language="pt-BR") -> str | None:
        if not self.recognizer:
            return None
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                log.info("🎤 Ouvindo...")
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=15)
                text  = self.recognizer.recognize_google(audio, language=language)
                log.info(f"🎤 Reconhecido: {text}")
                return text
        except (sr.WaitTimeoutError, sr.UnknownValueError):
            return None
        except Exception as e:
            log.error(f"STT erro: {e}")
            return None


# ════════════════════════════════════════════════════════════════════
#  EXECUTOR DE AÇÕES (NATIVO + SELENIUM)
# ════════════════════════════════════════════════════════════════════

class ActionExecutor:
    """
    Executa ações nativas do sistema operacional.
    As ações de browser (browser_*) são delegadas ao SeleniumEngine.
    As ações run_macro / learn_* são delegadas ao LearningEngine.
    """

    def __init__(self, voice: VoiceEngine, db: AthenaDB,
                 selenium: "AthenaSeleniumEngine | None" = None,
                 learning: "AthenaLearningEngine | None" = None):
        self.voice    = voice
        self.db       = db
        self.selenium = selenium
        self.learning = learning

    # ── Dispatcher principal ─────────────────────────────────────────────────
    def execute(self, action: str, params: dict) -> dict:
        # 1. Ações de Browser → Selenium
        if action.startswith("browser_") or action in ("google_search",):
            if not self.selenium:
                return {"success": False, "output": "Selenium não disponível"}
            result = self.selenium.dispatch(action, params)
            ok     = result.pop("ok", False)
            return {"success": ok, "output": json.dumps(result, ensure_ascii=False)}

        # 2. Macros → Learning
        if action == "run_macro":
            return self._run_macro(params)

        if action == "learning_stats":
            return self._learning_stats(params)

        if action == "list_macros":
            return self._list_macros(params)

        if action == "create_macro":
            return self._create_macro(params)

        if action == "export_knowledge":
            return self._export_knowledge(params)

        if action == "learning_dashboard":
            return self._learning_dashboard(params)

        # 3. Ações nativas
        handler = getattr(self, f"_do_{action}", self._do_unknown)
        try:
            output = handler(params)
            return {"success": True, "output": output}
        except Exception as e:
            log.error(f"Erro em {action}: {e}")
            return {"success": False, "output": str(e)}

    # ── Macro runner ─────────────────────────────────────────────────────────
    def _run_macro(self, p: dict) -> dict:
        if not self.learning:
            return {"success": False, "output": "Learning Engine não disponível"}

        seq_hash = p.get("seq_hash", "")
        macro    = self.learning.get_macro(seq_hash)
        if not macro:
            return {"success": False, "output": f"Macro não encontrado: {seq_hash}"}

        steps   = macro.get("steps", [])
        results = []
        log.info(f"▶ Executando macro: {macro['name']} ({len(steps)} passos)")

        for i, step in enumerate(steps):
            log.info(f"  Passo {i+1}/{len(steps)}: {step.get('action')}")
            r = self.execute(step["action"], step.get("params", {}))
            results.append({
                "step":    i + 1,
                "action":  step["action"],
                "success": r["success"],
                "output":  str(r["output"])[:200],
            })
            # Pausa entre passos
            delay = step.get("delay", 0.5)
            if delay > 0:
                time.sleep(delay)

        all_ok = all(r["success"] for r in results)
        return {
            "success": all_ok,
            "output": f"Macro '{macro['name']}' executado: {sum(r['success'] for r in results)}/{len(steps)} passos OK\n{json.dumps(results, ensure_ascii=False, indent=2)}"
        }

    def _learning_stats(self, p: dict) -> dict:
        if not self.learning:
            return {"success": False, "output": "Learning Engine não disponível"}
        stats = self.learning.get_full_stats()
        return {"success": True, "output": json.dumps(stats, ensure_ascii=False, indent=2)}

    def _list_macros(self, p: dict) -> dict:
        if not self.learning:
            return {"success": False, "output": "Learning Engine não disponível"}
        macros = self.learning.list_macros()
        return {"success": True, "output": json.dumps([
            {"name": m["name"], "steps": len(m.get("steps",[])),
             "frequency": m["frequency"], "trigger": m.get("trigger","")}
            for m in macros
        ], ensure_ascii=False, indent=2)}

    def _create_macro(self, p: dict) -> dict:
        if not self.learning:
            return {"success": False, "output": "Learning Engine não disponível"}
        name    = p.get("name", "Macro")
        steps   = p.get("steps", [])
        trigger = p.get("trigger", "")
        r = self.learning.create_manual_macro(name, steps, trigger)
        return {"success": r["ok"], "output": f"Macro '{name}' criado com {len(steps)} passos. Hash: {r['seq_hash']}"}

    def _export_knowledge(self, p: dict) -> dict:
        if not self.learning:
            return {"success": False, "output": "Learning Engine não disponível"}
        path = p.get("path", "athena_knowledge.json")
        r    = self.learning.export_knowledge(path)
        return {"success": r["ok"], "output": f"Conhecimento exportado: {r.get('patterns',0)} padrões, {r.get('sequences',0)} macros → {path}"}

    def _learning_dashboard(self, p: dict) -> dict:
        if not self.learning:
            return {"success": False, "output": "Learning Engine não disponível"}
        data = self.learning.get_dashboard_data()
        return {"success": True, "output": json.dumps(data, ensure_ascii=False, indent=2)}

    # ════════════════════════════════════════════════════════════════
    #  AÇÕES NATIVAS DO SISTEMA OPERACIONAL
    # ════════════════════════════════════════════════════════════════

    # ── Shell ────────────────────────────────────────────────────────────────
    def _do_shell(self, p):
        cmd = p.get("cmd", "")
        log.info(f"Shell: {cmd}")
        proc = subprocess.run(
            cmd, shell=True, capture_output=True,
            text=True, timeout=p.get("timeout", 30),
            encoding='utf-8', errors='replace'
        )
        return (proc.stdout.strip() or proc.stderr.strip() or "(sem saída)")[:3000]

    def _do_powershell(self, p):
        return self._do_shell({"cmd": f'powershell -Command "{p.get("cmd","")}"'})

    def _do_cmd(self, p):
        return self._do_shell({"cmd": p.get("cmd", "")})

    # ── Arquivos ─────────────────────────────────────────────────────────────
    def _do_read_file(self, p):
        path = os.path.expanduser(p.get("path", ""))
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()[:6000]

    def _do_write_file(self, p):
        path = os.path.expanduser(p.get("path", ""))
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, p.get("mode","w"), encoding='utf-8') as f:
            f.write(p.get("content",""))
        return f"Salvo: {path}"

    def _do_append_file(self, p):
        return self._do_write_file({**p, "mode": "a"})

    def _do_delete_file(self, p):
        path = os.path.expanduser(p.get("path",""))
        if os.path.isfile(path):
            os.remove(path)
            return f"Deletado: {path}"
        import shutil
        shutil.rmtree(path, ignore_errors=True)
        return f"Pasta deletada: {path}"

    def _do_list_dir(self, p):
        path = os.path.expanduser(p.get("path", os.path.expanduser("~")))
        items = os.listdir(path)
        return "\n".join(items[:200])

    def _do_move_file(self, p):
        import shutil
        src = os.path.expanduser(p.get("src",""))
        dst = os.path.expanduser(p.get("dst",""))
        shutil.move(src, dst)
        return f"Movido: {src} → {dst}"

    def _do_copy_file(self, p):
        import shutil
        src = os.path.expanduser(p.get("src",""))
        dst = os.path.expanduser(p.get("dst",""))
        shutil.copy2(src, dst)
        return f"Copiado: {src} → {dst}"

    def _do_file_exists(self, p):
        path = os.path.expanduser(p.get("path",""))
        return f"{'Existe' if os.path.exists(path) else 'Não existe'}: {path}"

    # ── Web / HTTP ────────────────────────────────────────────────────────────
    def _do_open_url(self, p):
        url = p.get("url","")
        if not url.startswith(("http://","https://")):
            url = "https://" + url
        webbrowser.open(url)
        return f"Aberto: {url}"

    def _do_web_search(self, p):
        q = p.get("query","")
        webbrowser.open(f"https://www.google.com/search?q={q.replace(' ','+')}")
        return f"Pesquisando: {q}"

    def _do_http_get(self, p):
        r = requests.get(p.get("url",""), headers=p.get("headers",{}), timeout=15)
        return r.text[:4000]

    def _do_http_post(self, p):
        r = requests.post(p.get("url",""), json=p.get("data",{}),
                          headers=p.get("headers",{}), timeout=15)
        return r.text[:4000]

    # ── Mouse / Teclado ───────────────────────────────────────────────────────
    def _do_click(self, p):
        if not HAS_PYAUTOGUI: return "pyautogui não disponível"
        x, y = p.get("x"), p.get("y")
        if x is not None and y is not None:
            pyautogui.click(x=x, y=y, button=p.get("button","left"))
        else:
            pyautogui.click(button=p.get("button","left"))
        return f"Clicado em ({x},{y})"

    def _do_double_click(self, p):
        if not HAS_PYAUTOGUI: return "pyautogui não disponível"
        pyautogui.doubleClick(p.get("x"), p.get("y"))
        return "Double click"

    def _do_right_click(self, p):
        if not HAS_PYAUTOGUI: return "pyautogui não disponível"
        pyautogui.click(button='right', x=p.get("x"), y=p.get("y"))
        return "Right click"

    def _do_type_text(self, p):
        if not HAS_PYAUTOGUI: return "pyautogui não disponível"
        pyautogui.typewrite(p.get("text",""), interval=p.get("interval", 0.04))
        return f"Digitado: {p.get('text','')[:50]}"

    def _do_hotkey(self, p):
        if not HAS_PYAUTOGUI: return "pyautogui não disponível"
        keys = p.get("keys",[])
        pyautogui.hotkey(*keys)
        return f"Atalho: {'+'.join(keys)}"

    def _do_press_key(self, p):
        if not HAS_PYAUTOGUI: return "pyautogui não disponível"
        pyautogui.press(p.get("key",""))
        return f"Tecla: {p.get('key','')}"

    def _do_scroll(self, p):
        if not HAS_PYAUTOGUI: return "pyautogui não disponível"
        pyautogui.scroll(p.get("amount", 3))
        return "Scroll"

    def _do_move_mouse(self, p):
        if not HAS_PYAUTOGUI: return "pyautogui não disponível"
        pyautogui.moveTo(p.get("x"), p.get("y"), duration=0.3)
        return "Mouse movido"

    def _do_drag(self, p):
        if not HAS_PYAUTOGUI: return "pyautogui não disponível"
        pyautogui.dragTo(p.get("x"), p.get("y"), duration=p.get("duration",0.5))
        return "Drag feito"

    # ── Screenshot / Captura ─────────────────────────────────────────────────
    def _do_screenshot(self, p):
        if not HAS_PYAUTOGUI: return "pyautogui não disponível"
        path = os.path.expanduser(p.get("path", f"~/screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
        pyautogui.screenshot().save(path)
        return f"Screenshot: {path}"

    def _do_screenshot_base64(self, p):
        if not HAS_PYAUTOGUI: return "pyautogui não disponível"
        buf = BytesIO()
        pyautogui.screenshot().save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode()

    # ── Sistema ───────────────────────────────────────────────────────────────
    def _do_system_info(self, p):
        info = {
            "os": sys.platform, "python": sys.version.split()[0],
            "user": os.getenv("USERNAME") or os.getenv("USER","?"),
            "cwd": os.getcwd(),
        }
        if HAS_PSUTIL:
            m = psutil.virtual_memory()
            d = psutil.disk_usage('/')
            info.update({
                "cpu_percent":    psutil.cpu_percent(interval=1),
                "cpu_cores":      psutil.cpu_count(),
                "ram_total_gb":   round(m.total / 1e9, 1),
                "ram_used_gb":    round(m.used / 1e9, 1),
                "ram_percent":    m.percent,
                "disk_total_gb":  round(d.total / 1e9, 1),
                "disk_used_gb":   round(d.used / 1e9, 1),
                "disk_percent":   d.percent,
            })
        return json.dumps(info, indent=2, ensure_ascii=False)

    def _do_list_processes(self, p):
        if not HAS_PSUTIL: return "psutil não disponível"
        procs = [(proc.pid, proc.name(), proc.cpu_percent(interval=None))
                 for proc in psutil.process_iter(['pid','name','cpu_percent'])]
        procs.sort(key=lambda x: -x[2])
        return "\n".join(f"{pid}: {name} ({cpu:.1f}%)" for pid, name, cpu in procs[:40])

    def _do_kill_process(self, p):
        if not HAS_PSUTIL: return "psutil não disponível"
        name = p.get("name","").lower()
        pid  = p.get("pid")
        killed = []
        for proc in psutil.process_iter(['pid','name']):
            try:
                if (name and name in proc.name().lower()) or (pid and proc.pid == pid):
                    proc.kill(); killed.append(f"{proc.pid}: {proc.name()}")
            except Exception:
                pass
        return f"Encerrados: {killed}" if killed else "Processo não encontrado"

    def _do_open_app(self, p):
        app = p.get("app","")
        cmd = (f"start {app}" if sys.platform == "win32"
               else f"open {app}" if sys.platform == "darwin"
               else f"xdg-open {app}")
        return self._do_shell({"cmd": cmd})

    def _do_get_env(self, p):
        key = p.get("key","")
        return os.environ.get(key, f"Variável '{key}' não encontrada")

    def _do_set_env(self, p):
        os.environ[p.get("key","")] = p.get("value","")
        return f"Variável de ambiente definida: {p.get('key')}"

    # ── Clipboard ─────────────────────────────────────────────────────────────
    def _do_clipboard_set(self, p):
        if not HAS_CLIPBOARD: return "pyperclip não disponível"
        pyperclip.copy(p.get("text",""))
        return "Copiado para clipboard"

    def _do_clipboard_get(self, p):
        if not HAS_CLIPBOARD: return "pyperclip não disponível"
        return pyperclip.paste()

    # ── Voz ───────────────────────────────────────────────────────────────────
    def _do_speak(self, p):
        self.voice.speak(p.get("text",""))
        return "Falado"

    # ── Relatório ─────────────────────────────────────────────────────────────
    def _do_create_report(self, p):
        title   = p.get("title","Relatório Athena")
        content = p.get("content","")
        path    = os.path.expanduser(p.get("path", f"~/relatorio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"))
        report  = f"{'='*60}\n{title}\nGerado em: {datetime.now()}\n{'='*60}\n\n{content}"
        with open(path, 'w', encoding='utf-8') as f:
            f.write(report)
        if p.get("open", True) and HAS_BROWSER:
            webbrowser.open(f"file://{os.path.abspath(path)}")
        return f"Relatório criado: {path}"

    # ── Utilitários ───────────────────────────────────────────────────────────
    def _do_wait(self, p):
        time.sleep(p.get("seconds", 1))
        return f"Aguardado {p.get('seconds',1)}s"

    def _do_unknown(self, p):
        return f"Ação desconhecida"

    # ── Capacidades ───────────────────────────────────────────────────────────
    def get_capabilities(self) -> list:
        caps = [
            "shell","powershell","cmd",
            "read_file","write_file","append_file","delete_file",
            "list_dir","move_file","copy_file","file_exists",
            "open_url","web_search","http_get","http_post",
            "system_info","list_processes","kill_process","open_app",
            "get_env","set_env",
            "speak","wait","create_report",
            "clipboard_set","clipboard_get",
            "run_macro","list_macros","create_macro",
            "export_knowledge","learning_stats","learning_dashboard",
        ]
        if HAS_PYAUTOGUI:
            caps += ["click","double_click","right_click","type_text","hotkey",
                     "press_key","scroll","move_mouse","drag","screenshot","screenshot_base64"]
        if HAS_PSUTIL:
            caps += ["list_processes","kill_process"]
        if self.selenium and self.selenium.is_available():
            caps += self.selenium.get_capabilities()
        return caps


# ════════════════════════════════════════════════════════════════════
#  AGENTE PRINCIPAL
# ════════════════════════════════════════════════════════════════════

class AthenaAgent:
    """
    Agente autônomo com:
    - WebSocket persistente ao backend
    - Learning Engine (TF-IDF, clustering, macros)
    - Selenium Engine (30+ ações de browser)
    - Voz local (wake word + TTS)
    - 50+ ações nativas
    """

    def __init__(self, agent_id: str, backend_url: str,
                 learning_db: str = "athena_learning.db"):
        self.agent_id    = agent_id
        self.backend_url = backend_url.rstrip("/")
        self.ws_url      = f"{self.backend_url}/ws/{agent_id}"
        self.running     = True
        self.ws          = None
        self.reconnect_delay = 3

        # ── Sub-sistemas ──────────────────────────────
        self.db      = AthenaDB()
        self.voice   = VoiceEngine()

        self.selenium = AthenaSeleniumEngine() if HAS_SELENIUM_ENGINE else None
        self.learning = AthenaLearningEngine(path=learning_db) if HAS_LEARNING else None

        self.executor = ActionExecutor(
            voice=self.voice,
            db=self.db,
            selenium=self.selenium,
            learning=self.learning,
        )

        # ── Voz ───────────────────────────────────────
        self.voice_active  = HAS_SR
        self.voice_thread  = None

        # ── Métricas em memória ───────────────────────
        self._ops_total   = 0
        self._ops_success = 0

        log.info(f"🤖 Athena Agent v2.0 | ID: {agent_id}")
        log.info(f"   Selenium: {'✅' if self.selenium else '❌'} | Learning: {'✅' if self.learning else '❌'}")

        self.voice.speak(f"Athena versão dois iniciada. Agente {agent_id} pronto.")

    # ═══════════════════════════════════════════════════
    #  WEBSOCKET
    # ═══════════════════════════════════════════════════

    async def connect_loop(self):
        while self.running:
            try:
                log.info(f"Conectando: {self.ws_url}")
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=30, ping_timeout=10,
                    extra_headers={"X-Agent-ID": self.agent_id}
                ) as ws:
                    self.ws = ws
                    self.reconnect_delay = 3
                    log.info("✅ Conectado ao Athena Cloud")
                    self.voice.speak("Conexão estabelecida.")

                    # Registro
                    await ws.send(json.dumps({
                        "type":         "register",
                        "agent_id":     self.agent_id,
                        "platform":     sys.platform,
                        "hostname":     os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME", "unknown"),
                        "capabilities": self.executor.get_capabilities(),
                        "has_selenium": bool(self.selenium and self.selenium.is_available()),
                        "has_learning": bool(self.learning),
                        "stats":        self.db.get_stats(),
                        "learning_stats": self.learning.get_full_stats() if self.learning else {},
                    }))

                    async for raw in ws:
                        await self._on_message(ws, raw)

            except (websockets.exceptions.ConnectionClosed, ConnectionRefusedError, OSError) as e:
                log.warning(f"Desconectado: {e}. Reconectando em {self.reconnect_delay}s…")
                self.ws = None
                await asyncio.sleep(self.reconnect_delay)
                self.reconnect_delay = min(self.reconnect_delay * 2, 60)
            except Exception as e:
                log.error(f"Erro: {e}")
                await asyncio.sleep(5)

    async def _on_message(self, ws, raw):
        try:
            data     = json.loads(raw)
            msg_type = data.get("type")

            if msg_type == "command":
                await self._handle_command(ws, data)
            elif msg_type == "ping":
                await ws.send(json.dumps({"type":"pong","agent_id":self.agent_id}))
            elif msg_type == "set_voice":
                self.voice_active = data.get("enabled", True)
            elif msg_type == "shutdown":
                self.running = False
            elif msg_type == "learning_query":
                await self._handle_learning_query(ws, data)
        except json.JSONDecodeError:
            log.error("Mensagem JSON inválida")
        except Exception as e:
            log.error(f"_on_message: {e}")

    # ═══════════════════════════════════════════════════
    #  HANDLER DE COMANDO
    # ═══════════════════════════════════════════════════

    async def _handle_command(self, ws, data: dict):
        cmd_id        = data.get("command_id", str(uuid.uuid4()))
        command       = data.get("command", "")
        action        = data.get("action", "shell")
        params        = data.get("params", {})
        priority      = data.get("priority", "low")
        response_text = data.get("response_text", "")
        source        = data.get("source", "dashboard")

        log.info(f"📩 [{priority.upper()}] {command[:80]}")
        self.db.add_alert(priority, f"CMD:{action}", command[:100])

        if self.voice_active and priority in ("high","critical"):
            self.voice.speak(f"Atenção: {command[:50]}")

        start = time.time()

        # ─ 1. Consultar Learning Engine primeiro ────────
        learned_prediction = None
        if self.learning and not data.get("force_ai"):
            learned_prediction = self.learning.predict(command)
            if learned_prediction:
                conf = learned_prediction["confidence"]
                log.info(f"📚 Learning: {learned_prediction['action']} (conf={conf:.0%}, uses={learned_prediction['total_uses']})")
                action = learned_prediction["action"]
                params = learned_prediction["params"]

        # ─ 2. Executar ──────────────────────────────────
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.executor.execute(action, params)
        )

        duration = time.time() - start
        self._ops_total   += 1
        self._ops_success += 1 if result["success"] else 0

        # ─ 3. Aprender com a execução ───────────────────
        learn_result = None
        if self.learning:
            learn_result = self.learning.learn(
                command  = command,
                action   = action,
                params   = params,
                success  = result["success"],
                duration = duration,
                result   = str(result["output"])[:300],
                source   = source,
                cmd_id   = cmd_id,
                agent_id = self.agent_id,
            )
            if learn_result.get("anomaly"):
                anom = learn_result["anomaly"]
                log.warning(f"⚠️ ANOMALIA: {anom['reason']}")
                self.db.add_alert("high", "Anomalia Detectada", anom["reason"])

            if learn_result.get("sequence"):
                seq = learn_result["sequence"]
                log.info(f"🔁 Macro auto-detectado: '{seq['name']}' ({seq['steps']} passos)")
                self.db.add_alert("info", "Macro Criado", f"'{seq['name']}' — trigger: '{seq['trigger']}'")

        # ─ 4. Log legado ────────────────────────────────
        self.db.log_command(cmd_id, command, action, params,
                            str(result["output"])[:300], result["success"], duration)

        # ─ 5. Resposta de voz ───────────────────────────
        if self.voice_active:
            speak = response_text or (
                f"Concluído em {duration:.1f} segundos" if result["success"]
                else f"Falhou: {str(result['output'])[:60]}"
            )
            threading.Thread(target=self.voice.speak, args=(speak,), daemon=True).start()

        # ─ 6. Enviar resultado ──────────────────────────
        if self.ws:
            await ws.send(json.dumps({
                "type":          "result",
                "command_id":    cmd_id,
                "agent_id":      self.agent_id,
                "action":        action,
                "success":       result["success"],
                "output":        str(result["output"])[:10000],
                "duration":      round(duration, 3),
                "timestamp":     datetime.now().isoformat(),
                "learned_from":  bool(learned_prediction),
                "learn_result":  learn_result,
                "ops_total":     self._ops_total,
                "success_rate":  round(self._ops_success / self._ops_total * 100, 1),
            }))

        log.info(f"{'✅' if result['success'] else '❌'} {action} → {duration:.2f}s | ops={self._ops_total}")

    # ═══════════════════════════════════════════════════
    #  LEARNING QUERY (stats ao vivo)
    # ═══════════════════════════════════════════════════

    async def _handle_learning_query(self, ws, data: dict):
        if not self.learning:
            await ws.send(json.dumps({"type":"learning_response","error":"Learning não disponível"}))
            return
        query = data.get("query","dashboard")
        if query == "dashboard":
            resp = self.learning.get_dashboard_data()
        elif query == "stats":
            resp = self.learning.get_full_stats()
        elif query == "macros":
            resp = {"macros": self.learning.list_macros()}
        elif query == "suggestions":
            resp = {"suggestions": self.learning.get_suggestions()}
        elif query == "cluster":
            resp = self.learning.cluster_commands()
        elif query == "performance":
            resp = self.learning.analyze_performance()
        else:
            resp = {"error": f"Query desconhecida: {query}"}
        await ws.send(json.dumps({"type":"learning_response","query":query,"data":resp}))

    # ═══════════════════════════════════════════════════
    #  VOZ LOCAL
    # ═══════════════════════════════════════════════════

    def _voice_loop(self):
        wake_words = ["athena","atena","atenção"]
        log.info("🎤 Loop de voz ativo. Diga 'Athena' para ativar.")
        while self.running:
            try:
                text = self.voice.listen(timeout=10)
                if not text:
                    continue
                lower = text.lower()
                if any(w in lower for w in wake_words):
                    self.voice.speak("Sim?")
                    command = self.voice.listen(timeout=8)
                    if command:
                        log.info(f"🎤 Voz: {command}")
                        self._send_voice_command(command)
            except Exception as e:
                log.debug(f"Voice loop: {e}")
                time.sleep(1)

    def _send_voice_command(self, command: str):
        http_url = self.backend_url.replace("ws://","http://").replace("wss://","https://")
        try:
            r = requests.post(
                f"{http_url}/command",
                json={"command": command, "agent_id": self.agent_id, "source": "voice"},
                timeout=10
            )
            if r.status_code != 200:
                self.voice.speak("Não consegui enviar.")
        except Exception as e:
            log.error(f"Voice send: {e}")
            self.voice.speak("Sem conexão.")

    # ═══════════════════════════════════════════════════
    #  START / STOP
    # ═══════════════════════════════════════════════════

    def start(self):
        if self.voice_active:
            self.voice_thread = threading.Thread(target=self._voice_loop, daemon=True)
            self.voice_thread.start()
        try:
            asyncio.run(self.connect_loop())
        finally:
            self._cleanup()

    def _cleanup(self):
        log.info("Limpando recursos...")
        if self.selenium:
            self.selenium.cleanup()
        log.info("Agente encerrado.")


# ════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Athena Agent v2.0")
    parser.add_argument("--id",      default="agente-1",
                        help="ID único deste agente")
    parser.add_argument("--url",     default="ws://localhost:8000",
                        help="URL WebSocket do backend")
    parser.add_argument("--learn-db",default="athena_learning.db",
                        help="Caminho para o banco de aprendizado")
    args = parser.parse_args()

    print("""
    ╔══════════════════════════════════════════════════╗
    ║      🤖  A T H E N A   A G E N T  v 2 . 0       ║
    ║   Selenium + Advanced Learning + Voice + 50 ops  ║
    ╚══════════════════════════════════════════════════╝
    """)
    print(f"  ID:        {args.id}")
    print(f"  Backend:   {args.url}")
    print(f"  Selenium:  {'✅' if HAS_SELENIUM_ENGINE else '❌ pip install selenium webdriver-manager'}")
    print(f"  Learning:  {'✅' if HAS_LEARNING else '❌ pip install scikit-learn numpy'}")
    print(f"  Voice STT: {'✅' if HAS_SR else '❌ pip install SpeechRecognition'}")
    print(f"  Voice TTS: {'✅' if HAS_TTS else '❌ pip install pyttsx3'}")
    print()

    agent = AthenaAgent(agent_id=args.id, backend_url=args.url, learning_db=args.learn_db)
    try:
        agent.start()
    except KeyboardInterrupt:
        log.info("Encerrado pelo usuário.")


if __name__ == "__main__":
    main()
