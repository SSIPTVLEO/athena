"""
╔══════════════════════════════════════════════════════════╗
║        ATHENA SELENIUM ENGINE — Browser Automation       ║
║   Controle completo de navegadores via Selenium          ║
╚══════════════════════════════════════════════════════════╝

Instalação:
    pip install selenium webdriver-manager pillow

Suporte: Chrome, Firefox, Edge (auto-download do driver)
"""

import os
import sys
import time
import json
import base64
import logging
import threading
from io import BytesIO
from typing import Optional, Dict, Any, List
from datetime import datetime

log = logging.getLogger("athena.selenium")

# ── Selenium imports ──────────────────────────────────────────────────────────
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.webdriver.support.ui import WebDriverWait, Select
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.webdriver.firefox.service import Service as FirefoxService
    from selenium.webdriver.edge.service import Service as EdgeService
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.firefox.options import Options as FirefoxOptions
    from selenium.webdriver.edge.options import Options as EdgeOptions
    from selenium.common.exceptions import (
        TimeoutException, NoSuchElementException,
        ElementNotInteractableException, WebDriverException,
        StaleElementReferenceException
    )
    HAS_SELENIUM = True
except ImportError:
    HAS_SELENIUM = False
    log.warning("Selenium não instalado. Execute: pip install selenium")

try:
    from webdriver_manager.chrome import ChromeDriverManager
    from webdriver_manager.firefox import GeckoDriverManager
    from webdriver_manager.microsoft import EdgeChromiumDriverManager
    HAS_WDM = True
except ImportError:
    HAS_WDM = False
    log.warning("webdriver-manager não instalado. Execute: pip install webdriver-manager")

# ── Seletor padrão ────────────────────────────────────────────────────────────
BY_MAP = {
    "id":           By.ID,
    "name":         By.NAME,
    "css":          By.CSS_SELECTOR,
    "xpath":        By.XPATH,
    "tag":          By.TAG_NAME,
    "class":        By.CLASS_NAME,
    "link":         By.LINK_TEXT,
    "partial_link": By.PARTIAL_LINK_TEXT,
}


class AthenaBrowserSession:
    """Sessão de browser individual com histórico e estado"""

    def __init__(self, session_id: str, driver, browser_type: str):
        self.session_id  = session_id
        self.driver      = driver
        self.browser     = browser_type
        self.created_at  = datetime.now().isoformat()
        self.last_action = datetime.now().isoformat()
        self.action_count = 0
        self.tab_handles: List[str] = []
        self._update_tabs()

    def _update_tabs(self):
        try:
            self.tab_handles = list(self.driver.window_handles)
        except Exception:
            pass

    def touch(self):
        self.last_action  = datetime.now().isoformat()
        self.action_count += 1
        self._update_tabs()

    def info(self) -> dict:
        try:
            url   = self.driver.current_url
            title = self.driver.title
            tabs  = len(self.driver.window_handles)
        except Exception:
            url, title, tabs = "N/A", "N/A", 0
        return {
            "session_id":   self.session_id,
            "browser":      self.browser,
            "url":          url,
            "title":        title,
            "tabs":         tabs,
            "action_count": self.action_count,
            "created_at":   self.created_at,
            "last_action":  self.last_action,
        }


class AthenaSeleniumEngine:
    """
    Motor completo de automação de browsers.
    Gerencia múltiplas sessões simultâneas, detecta
    elementos de forma inteligente e aprende com o uso.
    """

    def __init__(self):
        self.sessions: Dict[str, AthenaBrowserSession] = {}
        self.active_session: Optional[str]             = None
        self._lock = threading.Lock()

        if not HAS_SELENIUM:
            log.error("Selenium não disponível. Instale: pip install selenium webdriver-manager")

    # ═══════════════════════════════════════════════════════
    #  SESSÃO / DRIVER
    # ═══════════════════════════════════════════════════════

    def open_browser(self, params: dict) -> dict:
        """Abre um novo browser e cria sessão."""
        if not HAS_SELENIUM:
            return {"ok": False, "error": "Selenium não instalado"}

        browser   = params.get("browser", "chrome").lower()
        headless  = params.get("headless", False)
        url       = params.get("url", "")
        session_id = params.get("session_id", f"session_{int(time.time())}")
        maximized = params.get("maximized", True)
        incognito = params.get("incognito", False)
        proxy     = params.get("proxy", "")
        user_agent= params.get("user_agent", "")

        try:
            driver = self._create_driver(browser, headless, maximized, incognito, proxy, user_agent)
            session = AthenaBrowserSession(session_id, driver, browser)
            with self._lock:
                self.sessions[session_id] = session
                self.active_session       = session_id

            if url:
                driver.get(url)
                WebDriverWait(driver, 15).until(
                    lambda d: d.execute_script("return document.readyState") == "complete"
                )

            log.info(f"Browser {browser} aberto | Sessão: {session_id}")
            return {
                "ok": True,
                "session_id": session_id,
                "browser":    browser,
                "url":        driver.current_url,
                "title":      driver.title,
            }

        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _create_driver(self, browser, headless, maximized, incognito, proxy, user_agent):
        if browser == "chrome":
            opts = ChromeOptions()
            if headless:
                opts.add_argument("--headless=new")
            if maximized:
                opts.add_argument("--start-maximized")
            if incognito:
                opts.add_argument("--incognito")
            if proxy:
                opts.add_argument(f"--proxy-server={proxy}")
            if user_agent:
                opts.add_argument(f"--user-agent={user_agent}")
            opts.add_argument("--disable-blink-features=AutomationControlled")
            opts.add_experimental_option("excludeSwitches", ["enable-automation"])
            opts.add_experimental_option("useAutomationExtension", False)
            opts.add_argument("--no-sandbox")
            opts.add_argument("--disable-dev-shm-usage")

            if HAS_WDM:
                svc = ChromeService(ChromeDriverManager().install())
            else:
                svc = ChromeService()
            driver = webdriver.Chrome(service=svc, options=opts)
            # Anti-detection
            driver.execute_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
            )
            return driver

        elif browser == "firefox":
            opts = FirefoxOptions()
            if headless:
                opts.add_argument("--headless")
            if incognito:
                opts.add_argument("-private")
            if user_agent:
                opts.set_preference("general.useragent.override", user_agent)
            svc = FirefoxService(GeckoDriverManager().install()) if HAS_WDM else FirefoxService()
            return webdriver.Firefox(service=svc, options=opts)

        elif browser == "edge":
            opts = EdgeOptions()
            if headless:
                opts.add_argument("--headless")
            if maximized:
                opts.add_argument("--start-maximized")
            svc = EdgeService(EdgeChromiumDriverManager().install()) if HAS_WDM else EdgeService()
            return webdriver.Edge(service=svc, options=opts)

        raise ValueError(f"Browser não suportado: {browser}. Use: chrome, firefox, edge")

    def close_browser(self, params: dict) -> dict:
        sid = params.get("session_id", self.active_session)
        sess = self._get_session(sid)
        if not sess:
            return {"ok": False, "error": f"Sessão {sid} não encontrada"}
        try:
            sess.driver.quit()
        except Exception:
            pass
        with self._lock:
            del self.sessions[sid]
            if self.active_session == sid:
                self.active_session = next(iter(self.sessions), None)
        return {"ok": True, "closed_session": sid}

    def close_all_browsers(self, params: dict) -> dict:
        closed = []
        for sid, sess in list(self.sessions.items()):
            try:
                sess.driver.quit()
                closed.append(sid)
            except Exception:
                pass
        self.sessions.clear()
        self.active_session = None
        return {"ok": True, "closed": closed}

    # ═══════════════════════════════════════════════════════
    #  NAVEGAÇÃO
    # ═══════════════════════════════════════════════════════

    def navigate(self, params: dict) -> dict:
        sess = self._require_session(params)
        if isinstance(sess, dict):
            return sess
        url     = params.get("url", "")
        timeout = params.get("timeout", 20)
        wait_js = params.get("wait_js", True)

        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        try:
            sess.driver.get(url)
            if wait_js:
                WebDriverWait(sess.driver, timeout).until(
                    lambda d: d.execute_script("return document.readyState") == "complete"
                )
            sess.touch()
            return {
                "ok":    True,
                "url":   sess.driver.current_url,
                "title": sess.driver.title,
            }
        except TimeoutException:
            return {"ok": False, "error": f"Timeout ao carregar: {url}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def go_back(self, params: dict) -> dict:
        sess = self._require_session(params)
        if isinstance(sess, dict):
            return sess
        sess.driver.back()
        sess.touch()
        return {"ok": True, "url": sess.driver.current_url}

    def go_forward(self, params: dict) -> dict:
        sess = self._require_session(params)
        if isinstance(sess, dict):
            return sess
        sess.driver.forward()
        sess.touch()
        return {"ok": True, "url": sess.driver.current_url}

    def refresh(self, params: dict) -> dict:
        sess = self._require_session(params)
        if isinstance(sess, dict):
            return sess
        sess.driver.refresh()
        sess.touch()
        return {"ok": True, "url": sess.driver.current_url}

    # ═══════════════════════════════════════════════════════
    #  BUSCA DE ELEMENTOS
    # ═══════════════════════════════════════════════════════

    def _find_element(self, driver, by_str, selector, timeout=10):
        """Busca elemento com fallback inteligente."""
        by = BY_MAP.get(by_str.lower(), By.CSS_SELECTOR)
        try:
            return WebDriverWait(driver, timeout).until(
                EC.presence_of_element_located((by, selector))
            )
        except TimeoutException:
            raise NoSuchElementException(f"Elemento não encontrado: [{by_str}] {selector}")

    def _smart_find(self, driver, hint: str, timeout=10):
        """
        Busca inteligente por hint (texto, placeholder, label, aria-label, id, name).
        Tenta múltiplas estratégias automaticamente.
        """
        strategies = [
            # Exato
            (By.ID,             hint),
            (By.NAME,           hint),
            (By.CSS_SELECTOR,   f"[placeholder='{hint}']"),
            (By.CSS_SELECTOR,   f"[aria-label='{hint}']"),
            (By.CSS_SELECTOR,   f"[title='{hint}']"),
            (By.LINK_TEXT,      hint),
            # Parcial
            (By.PARTIAL_LINK_TEXT, hint),
            (By.CSS_SELECTOR,   f"[placeholder*='{hint}']"),
            (By.CSS_SELECTOR,   f"[aria-label*='{hint}']"),
            # XPath por texto visível
            (By.XPATH, f"//*[normalize-space(text())='{hint}']"),
            (By.XPATH, f"//*[contains(normalize-space(text()),'{hint}')]"),
            # XPath label
            (By.XPATH, f"//label[contains(text(),'{hint}')]/following-sibling::input"),
            (By.XPATH, f"//label[contains(text(),'{hint}')]/..//input"),
            # Botões
            (By.XPATH, f"//button[contains(.,'{hint}')]"),
            (By.XPATH, f"//input[@type='submit' and @value='{hint}']"),
        ]
        for by, sel in strategies:
            try:
                el = WebDriverWait(driver, 1).until(
                    EC.presence_of_element_located((by, sel))
                )
                return el
            except Exception:
                continue
        raise NoSuchElementException(f"Elemento não encontrado por hint: '{hint}'")

    # ═══════════════════════════════════════════════════════
    #  INTERAÇÃO COM ELEMENTOS
    # ═══════════════════════════════════════════════════════

    def click(self, params: dict) -> dict:
        sess = self._require_session(params)
        if isinstance(sess, dict):
            return sess

        selector = params.get("selector", "")
        by       = params.get("by", "css")
        hint     = params.get("hint", "")
        timeout  = params.get("timeout", 10)
        js_click = params.get("js_click", False)
        double   = params.get("double", False)

        try:
            if hint:
                el = self._smart_find(sess.driver, hint, timeout)
            else:
                el = self._find_element(sess.driver, by, selector, timeout)

            # Scroll to element
            sess.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
            time.sleep(0.2)

            if js_click:
                sess.driver.execute_script("arguments[0].click();", el)
            elif double:
                ActionChains(sess.driver).double_click(el).perform()
            else:
                WebDriverWait(sess.driver, timeout).until(EC.element_to_be_clickable(el))
                el.click()

            sess.touch()
            return {"ok": True, "clicked": selector or hint, "url": sess.driver.current_url}

        except Exception as e:
            # Fallback: JS click
            try:
                el = self._smart_find(sess.driver, hint or selector, 2)
                sess.driver.execute_script("arguments[0].click();", el)
                sess.touch()
                return {"ok": True, "clicked": selector or hint, "method": "js_fallback"}
            except Exception:
                return {"ok": False, "error": str(e)}

    def type_text(self, params: dict) -> dict:
        sess = self._require_session(params)
        if isinstance(sess, dict):
            return sess

        selector = params.get("selector", "")
        by       = params.get("by", "css")
        hint     = params.get("hint", "")
        text     = params.get("text", "")
        clear    = params.get("clear", True)
        slow     = params.get("slow", False)  # Digita letra a letra
        timeout  = params.get("timeout", 10)
        submit   = params.get("submit", False)

        try:
            el = self._smart_find(sess.driver, hint, timeout) if hint \
                 else self._find_element(sess.driver, by, selector, timeout)

            sess.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
            time.sleep(0.1)
            el.click()

            if clear:
                el.clear()
                el.send_keys(Keys.CONTROL + "a")
                el.send_keys(Keys.DELETE)

            if slow:
                for char in text:
                    el.send_keys(char)
                    time.sleep(0.05)
            else:
                el.send_keys(text)

            if submit:
                el.send_keys(Keys.RETURN)

            sess.touch()
            return {"ok": True, "typed": text[:80], "field": selector or hint}

        except Exception as e:
            return {"ok": False, "error": str(e)}

    def fill_form(self, params: dict) -> dict:
        """Preenche múltiplos campos de um formulário de uma vez."""
        sess = self._require_session(params)
        if isinstance(sess, dict):
            return sess

        fields  = params.get("fields", {})   # {"hint_or_selector": "valor"}
        submit  = params.get("submit", False)
        submit_hint = params.get("submit_hint", "")
        results = {}

        for field_hint, value in fields.items():
            r = self.type_text({
                **params,
                "hint":  field_hint,
                "text":  str(value),
                "clear": True,
                "slow":  False,
            })
            results[field_hint] = r.get("ok", False)
            time.sleep(0.3)

        if submit or submit_hint:
            time.sleep(0.5)
            if submit_hint:
                self.click({**params, "hint": submit_hint})
            else:
                # Envia o formulário pelo primeiro input ativo
                try:
                    el = sess.driver.find_element(By.CSS_SELECTOR, "input:not([type='hidden'])")
                    el.send_keys(Keys.RETURN)
                except Exception:
                    pass

        sess.touch()
        return {"ok": True, "fields_filled": results, "url": sess.driver.current_url}

    def select_option(self, params: dict) -> dict:
        """Seleciona opção em <select>."""
        sess = self._require_session(params)
        if isinstance(sess, dict):
            return sess

        selector = params.get("selector", "select")
        by       = params.get("by", "css")
        value    = params.get("value", "")
        text     = params.get("text", "")
        index    = params.get("index")

        try:
            el  = self._find_element(sess.driver, by, selector)
            sel = Select(el)
            if text:
                sel.select_by_visible_text(text)
            elif value:
                sel.select_by_value(value)
            elif index is not None:
                sel.select_by_index(index)
            sess.touch()
            return {"ok": True, "selected": text or value or str(index)}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def hover(self, params: dict) -> dict:
        sess = self._require_session(params)
        if isinstance(sess, dict):
            return sess
        selector = params.get("selector", "")
        hint     = params.get("hint", "")
        try:
            el = self._smart_find(sess.driver, hint) if hint \
                 else self._find_element(sess.driver, params.get("by","css"), selector)
            ActionChains(sess.driver).move_to_element(el).perform()
            sess.touch()
            return {"ok": True, "hovered": selector or hint}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def drag_and_drop(self, params: dict) -> dict:
        sess = self._require_session(params)
        if isinstance(sess, dict):
            return sess
        src_sel = params.get("source", "")
        dst_sel = params.get("target", "")
        try:
            src = self._find_element(sess.driver, params.get("by","css"), src_sel)
            dst = self._find_element(sess.driver, params.get("by","css"), dst_sel)
            ActionChains(sess.driver).drag_and_drop(src, dst).perform()
            sess.touch()
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def send_keys(self, params: dict) -> dict:
        """Envia teclas especiais (Enter, Tab, Escape, etc.)."""
        sess = self._require_session(params)
        if isinstance(sess, dict):
            return sess
        key_name = params.get("key", "").upper()
        key_map  = {
            "ENTER": Keys.RETURN, "TAB": Keys.TAB, "ESCAPE": Keys.ESCAPE,
            "SPACE": Keys.SPACE, "BACKSPACE": Keys.BACKSPACE, "DELETE": Keys.DELETE,
            "UP": Keys.ARROW_UP, "DOWN": Keys.ARROW_DOWN,
            "LEFT": Keys.ARROW_LEFT, "RIGHT": Keys.ARROW_RIGHT,
            "HOME": Keys.HOME, "END": Keys.END,
            "F5": Keys.F5, "F11": Keys.F11, "F12": Keys.F12,
            "CTRL_A": Keys.CONTROL + "a", "CTRL_C": Keys.CONTROL + "c",
            "CTRL_V": Keys.CONTROL + "v", "CTRL_Z": Keys.CONTROL + "z",
        }
        key = key_map.get(key_name)
        if not key:
            return {"ok": False, "error": f"Tecla desconhecida: {key_name}"}
        try:
            active = sess.driver.switch_to.active_element
            active.send_keys(key)
            sess.touch()
            return {"ok": True, "key_sent": key_name}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ═══════════════════════════════════════════════════════
    #  EXTRAÇÃO DE DADOS (SCRAPING)
    # ═══════════════════════════════════════════════════════

    def get_text(self, params: dict) -> dict:
        sess = self._require_session(params)
        if isinstance(sess, dict):
            return sess
        selector = params.get("selector", "body")
        by       = params.get("by", "css")
        hint     = params.get("hint", "")
        multiple = params.get("multiple", False)

        try:
            if hint:
                el = self._smart_find(sess.driver, hint)
                return {"ok": True, "text": el.text.strip()}

            if multiple:
                els = sess.driver.find_elements(BY_MAP.get(by, By.CSS_SELECTOR), selector)
                texts = [e.text.strip() for e in els if e.text.strip()]
                return {"ok": True, "texts": texts, "count": len(texts)}

            el = self._find_element(sess.driver, by, selector)
            return {"ok": True, "text": el.text.strip()}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def get_attribute(self, params: dict) -> dict:
        sess = self._require_session(params)
        if isinstance(sess, dict):
            return sess
        selector  = params.get("selector", "")
        by        = params.get("by", "css")
        attribute = params.get("attribute", "href")
        multiple  = params.get("multiple", False)

        try:
            if multiple:
                els   = sess.driver.find_elements(BY_MAP.get(by, By.CSS_SELECTOR), selector)
                values= [e.get_attribute(attribute) for e in els if e.get_attribute(attribute)]
                return {"ok": True, "values": values}
            el = self._find_element(sess.driver, by, selector)
            return {"ok": True, "value": el.get_attribute(attribute)}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def get_table(self, params: dict) -> dict:
        """Extrai tabela HTML em formato JSON."""
        sess = self._require_session(params)
        if isinstance(sess, dict):
            return sess
        selector = params.get("selector", "table")
        by       = params.get("by", "css")

        try:
            table = self._find_element(sess.driver, by, selector)
            headers= [th.text.strip() for th in table.find_elements(By.TAG_NAME, "th")]
            rows   = []
            for tr in table.find_elements(By.TAG_NAME, "tr"):
                cells = [td.text.strip() for td in tr.find_elements(By.TAG_NAME, "td")]
                if cells:
                    if headers and len(cells) == len(headers):
                        rows.append(dict(zip(headers, cells)))
                    else:
                        rows.append(cells)
            return {"ok": True, "headers": headers, "rows": rows, "count": len(rows)}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def get_links(self, params: dict) -> dict:
        """Extrai todos os links da página."""
        sess = self._require_session(params)
        if isinstance(sess, dict):
            return sess
        filter_text = params.get("filter", "").lower()
        try:
            els   = sess.driver.find_elements(By.TAG_NAME, "a")
            links = []
            for el in els:
                href = el.get_attribute("href")
                text = el.text.strip()
                if href and (not filter_text or filter_text in text.lower() or filter_text in href.lower()):
                    links.append({"text": text, "url": href})
            return {"ok": True, "links": links[:100], "count": len(links)}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def get_page_source(self, params: dict) -> dict:
        sess = self._require_session(params)
        if isinstance(sess, dict):
            return sess
        limit = params.get("limit", 5000)
        try:
            src = sess.driver.page_source
            return {"ok": True, "source": src[:limit], "total_length": len(src)}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def wait_for_element(self, params: dict) -> dict:
        sess = self._require_session(params)
        if isinstance(sess, dict):
            return sess
        selector  = params.get("selector", "")
        by        = params.get("by", "css")
        timeout   = params.get("timeout", 30)
        condition = params.get("condition", "visible")  # visible|clickable|present|invisible

        cond_map = {
            "visible":   EC.visibility_of_element_located,
            "clickable": EC.element_to_be_clickable,
            "present":   EC.presence_of_element_located,
            "invisible": EC.invisibility_of_element_located,
        }
        ec_fn = cond_map.get(condition, EC.presence_of_element_located)
        try:
            WebDriverWait(sess.driver, timeout).until(
                ec_fn((BY_MAP.get(by, By.CSS_SELECTOR), selector))
            )
            return {"ok": True, "element": selector, "condition": condition}
        except TimeoutException:
            return {"ok": False, "error": f"Timeout: elemento não ficou '{condition}' em {timeout}s"}

    def wait_for_url(self, params: dict) -> dict:
        sess = self._require_session(params)
        if isinstance(sess, dict):
            return sess
        url_contains = params.get("url_contains", "")
        timeout      = params.get("timeout", 30)
        try:
            WebDriverWait(sess.driver, timeout).until(
                EC.url_contains(url_contains)
            )
            return {"ok": True, "current_url": sess.driver.current_url}
        except TimeoutException:
            return {"ok": False, "error": f"URL não mudou para conter '{url_contains}'"}

    # ═══════════════════════════════════════════════════════
    #  JAVASCRIPT
    # ═══════════════════════════════════════════════════════

    def execute_js(self, params: dict) -> dict:
        sess = self._require_session(params)
        if isinstance(sess, dict):
            return sess
        script = params.get("script", "")
        args   = params.get("args", [])
        try:
            result = sess.driver.execute_script(script, *args)
            sess.touch()
            return {"ok": True, "result": str(result)[:2000] if result is not None else None}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def scroll(self, params: dict) -> dict:
        sess = self._require_session(params)
        if isinstance(sess, dict):
            return sess
        direction = params.get("direction", "down")  # up|down|top|bottom
        amount    = params.get("amount", 500)
        selector  = params.get("selector", "")

        try:
            if selector:
                el = self._find_element(sess.driver, params.get("by","css"), selector)
                sess.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
            elif direction == "top":
                sess.driver.execute_script("window.scrollTo(0, 0);")
            elif direction == "bottom":
                sess.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            elif direction == "down":
                sess.driver.execute_script(f"window.scrollBy(0, {amount});")
            elif direction == "up":
                sess.driver.execute_script(f"window.scrollBy(0, -{amount});")
            sess.touch()
            return {"ok": True, "scrolled": direction}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ═══════════════════════════════════════════════════════
    #  ABAS / JANELAS
    # ═══════════════════════════════════════════════════════

    def new_tab(self, params: dict) -> dict:
        sess = self._require_session(params)
        if isinstance(sess, dict):
            return sess
        url = params.get("url", "")
        try:
            sess.driver.execute_script("window.open('');")
            sess.driver.switch_to.window(sess.driver.window_handles[-1])
            if url:
                sess.driver.get(url)
            sess.touch()
            return {"ok": True, "handles": sess.driver.window_handles, "url": sess.driver.current_url}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def switch_tab(self, params: dict) -> dict:
        sess = self._require_session(params)
        if isinstance(sess, dict):
            return sess
        index = params.get("index", 0)
        handle= params.get("handle", "")
        try:
            handles = sess.driver.window_handles
            target  = handle if handle else handles[index]
            sess.driver.switch_to.window(target)
            sess.touch()
            return {"ok": True, "tab": index, "url": sess.driver.current_url, "title": sess.driver.title}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def close_tab(self, params: dict) -> dict:
        sess = self._require_session(params)
        if isinstance(sess, dict):
            return sess
        try:
            sess.driver.close()
            if sess.driver.window_handles:
                sess.driver.switch_to.window(sess.driver.window_handles[-1])
            sess.touch()
            return {"ok": True, "remaining_tabs": len(sess.driver.window_handles)}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ═══════════════════════════════════════════════════════
    #  SCREENSHOT / CAPTURA
    # ═══════════════════════════════════════════════════════

    def screenshot_browser(self, params: dict) -> dict:
        sess = self._require_session(params)
        if isinstance(sess, dict):
            return sess
        path     = params.get("path", f"browser_screenshot_{int(time.time())}.png")
        as_b64   = params.get("base64", False)
        selector = params.get("selector", "")   # Captura só um elemento

        try:
            if selector:
                el  = self._find_element(sess.driver, params.get("by","css"), selector)
                png = el.screenshot_as_png
            else:
                png = sess.driver.get_screenshot_as_png()

            if as_b64:
                return {"ok": True, "base64": base64.b64encode(png).decode()}

            with open(path, "wb") as f:
                f.write(png)
            sess.touch()
            return {"ok": True, "path": path, "size": len(png)}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ═══════════════════════════════════════════════════════
    #  COOKIES / SESSÃO
    # ═══════════════════════════════════════════════════════

    def get_cookies(self, params: dict) -> dict:
        sess = self._require_session(params)
        if isinstance(sess, dict):
            return sess
        try:
            cookies = sess.driver.get_cookies()
            return {"ok": True, "cookies": cookies}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def set_cookie(self, params: dict) -> dict:
        sess = self._require_session(params)
        if isinstance(sess, dict):
            return sess
        cookie = {
            "name":  params.get("name", ""),
            "value": params.get("value", ""),
            "domain":params.get("domain", ""),
        }
        try:
            sess.driver.add_cookie(cookie)
            return {"ok": True, "cookie_set": cookie["name"]}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def clear_cookies(self, params: dict) -> dict:
        sess = self._require_session(params)
        if isinstance(sess, dict):
            return sess
        try:
            sess.driver.delete_all_cookies()
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ═══════════════════════════════════════════════════════
    #  ALERTAS / MODAIS
    # ═══════════════════════════════════════════════════════

    def handle_alert(self, params: dict) -> dict:
        sess = self._require_session(params)
        if isinstance(sess, dict):
            return sess
        action = params.get("action", "accept")   # accept|dismiss
        text   = params.get("text", "")

        try:
            alert = WebDriverWait(sess.driver, 5).until(EC.alert_is_present())
            alert_text = alert.text
            if text:
                alert.send_keys(text)
            if action == "accept":
                alert.accept()
            else:
                alert.dismiss()
            sess.touch()
            return {"ok": True, "alert_text": alert_text, "action": action}
        except TimeoutException:
            return {"ok": False, "error": "Nenhum alerta encontrado"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ═══════════════════════════════════════════════════════
    #  UPLOAD DE ARQUIVO
    # ═══════════════════════════════════════════════════════

    def upload_file(self, params: dict) -> dict:
        sess = self._require_session(params)
        if isinstance(sess, dict):
            return sess
        selector = params.get("selector", "input[type='file']")
        by       = params.get("by", "css")
        file_path= os.path.abspath(params.get("path", ""))

        if not os.path.exists(file_path):
            return {"ok": False, "error": f"Arquivo não encontrado: {file_path}"}
        try:
            el = self._find_element(sess.driver, by, selector)
            el.send_keys(file_path)
            sess.touch()
            return {"ok": True, "uploaded": file_path}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ═══════════════════════════════════════════════════════
    #  AUTOMAÇÕES PRONTAS (HIGH-LEVEL)
    # ═══════════════════════════════════════════════════════

    def google_search(self, params: dict) -> dict:
        """Pesquisa no Google e retorna resultados."""
        query = params.get("query", "")
        if not query:
            return {"ok": False, "error": "query é obrigatório"}

        # Garante sessão
        if not self.active_session:
            r = self.open_browser({"browser": params.get("browser","chrome"), "url": f"https://www.google.com/search?q={query}"})
            if not r["ok"]:
                return r
        else:
            self.navigate({**params, "url": f"https://www.google.com/search?q={query}"})

        time.sleep(1.5)
        sess = self._get_session(self.active_session)
        if not sess:
            return {"ok": False, "error": "Sessão perdida"}

        try:
            results = []
            items = sess.driver.find_elements(By.CSS_SELECTOR, "div.g")
            for item in items[:8]:
                try:
                    title_el = item.find_element(By.TAG_NAME, "h3")
                    link_el  = item.find_element(By.TAG_NAME, "a")
                    snip_els = item.find_elements(By.CSS_SELECTOR, "div[data-sncf], .IsZvec span")
                    snippet  = snip_els[0].text if snip_els else ""
                    results.append({
                        "title":   title_el.text,
                        "url":     link_el.get_attribute("href"),
                        "snippet": snippet[:200],
                    })
                except Exception:
                    continue
            return {"ok": True, "query": query, "results": results, "count": len(results)}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def login(self, params: dict) -> dict:
        """Login genérico em qualquer site."""
        url         = params.get("url", "")
        user_field  = params.get("user_field", "email")
        pass_field  = params.get("pass_field", "password")
        username    = params.get("username", "")
        password    = params.get("password", "")
        submit_hint = params.get("submit_hint", "")

        if not self.active_session:
            r = self.open_browser({"browser": params.get("browser","chrome"), "url": url})
            if not r["ok"]:
                return r
        elif url:
            self.navigate({**params, "url": url})

        time.sleep(1)
        r = self.fill_form({
            **params,
            "fields":      {user_field: username, pass_field: password},
            "submit":      not submit_hint,
            "submit_hint": submit_hint,
        })
        time.sleep(2)
        sess = self._get_session(self.active_session)
        return {**r, "final_url": sess.driver.current_url if sess else ""}

    def scrape_data(self, params: dict) -> dict:
        """Extrai dados estruturados de uma página."""
        url      = params.get("url", "")
        extract  = params.get("extract", {})  # {"campo": "css_selector"}
        multiple = params.get("multiple", False)
        container= params.get("container", "")

        if url:
            self.navigate({**params, "url": url})
            time.sleep(2)

        sess = self._require_session(params)
        if isinstance(sess, dict):
            return sess

        try:
            result = {}
            for field, selector in extract.items():
                try:
                    if multiple:
                        els = sess.driver.find_elements(By.CSS_SELECTOR, selector)
                        result[field] = [e.text.strip() for e in els]
                    else:
                        el = sess.driver.find_element(By.CSS_SELECTOR, selector)
                        result[field] = el.text.strip()
                except Exception:
                    result[field] = None

            return {"ok": True, "data": result, "url": sess.driver.current_url}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ═══════════════════════════════════════════════════════
    #  UTILITÁRIOS
    # ═══════════════════════════════════════════════════════

    def get_session_info(self, params: dict) -> dict:
        sid  = params.get("session_id", self.active_session)
        sess = self._get_session(sid)
        if not sess:
            return {"ok": False, "error": "Nenhuma sessão ativa"}
        return {"ok": True, **sess.info()}

    def list_sessions(self, params: dict) -> dict:
        return {
            "ok": True,
            "active": self.active_session,
            "sessions": [s.info() for s in self.sessions.values()],
            "count": len(self.sessions),
        }

    def maximize(self, params: dict) -> dict:
        sess = self._require_session(params)
        if isinstance(sess, dict):
            return sess
        sess.driver.maximize_window()
        return {"ok": True}

    def set_window_size(self, params: dict) -> dict:
        sess = self._require_session(params)
        if isinstance(sess, dict):
            return sess
        w = params.get("width", 1280)
        h = params.get("height", 720)
        sess.driver.set_window_size(w, h)
        return {"ok": True, "size": f"{w}x{h}"}

    # ═══════════════════════════════════════════════════════
    #  HELPERS INTERNOS
    # ═══════════════════════════════════════════════════════

    def _get_session(self, sid: Optional[str]) -> Optional[AthenaBrowserSession]:
        if not sid:
            sid = self.active_session
        return self.sessions.get(sid)

    def _require_session(self, params: dict):
        sid  = params.get("session_id", self.active_session)
        sess = self._get_session(sid)
        if not sess:
            return {"ok": False, "error": "Nenhuma sessão de browser ativa. Use open_browser primeiro."}
        return sess

    # ═══════════════════════════════════════════════════════
    #  DISPATCH — mapeamento ação → método
    # ═══════════════════════════════════════════════════════

    ACTION_MAP = {
        "browser_open":         "open_browser",
        "browser_close":        "close_browser",
        "browser_close_all":    "close_all_browsers",
        "browser_navigate":     "navigate",
        "browser_back":         "go_back",
        "browser_forward":      "go_forward",
        "browser_refresh":      "refresh",
        "browser_click":        "click",
        "browser_type":         "type_text",
        "browser_fill_form":    "fill_form",
        "browser_select":       "select_option",
        "browser_hover":        "hover",
        "browser_drag_drop":    "drag_and_drop",
        "browser_send_key":     "send_keys",
        "browser_get_text":     "get_text",
        "browser_get_attr":     "get_attribute",
        "browser_get_table":    "get_table",
        "browser_get_links":    "get_links",
        "browser_get_source":   "get_page_source",
        "browser_wait_element": "wait_for_element",
        "browser_wait_url":     "wait_for_url",
        "browser_execute_js":   "execute_js",
        "browser_scroll":       "scroll",
        "browser_new_tab":      "new_tab",
        "browser_switch_tab":   "switch_tab",
        "browser_close_tab":    "close_tab",
        "browser_screenshot":   "screenshot_browser",
        "browser_get_cookies":  "get_cookies",
        "browser_set_cookie":   "set_cookie",
        "browser_clear_cookies":"clear_cookies",
        "browser_handle_alert": "handle_alert",
        "browser_upload_file":  "upload_file",
        "browser_google_search":"google_search",
        "browser_login":        "login",
        "browser_scrape":       "scrape_data",
        "browser_session_info": "get_session_info",
        "browser_list_sessions":"list_sessions",
        "browser_maximize":     "maximize",
        "browser_set_size":     "set_window_size",
    }

    def dispatch(self, action: str, params: dict) -> dict:
        method_name = self.ACTION_MAP.get(action)
        if not method_name:
            return {"ok": False, "error": f"Ação Selenium desconhecida: {action}"}
        method = getattr(self, method_name, None)
        if not method:
            return {"ok": False, "error": f"Método não implementado: {method_name}"}
        try:
            return method(params)
        except Exception as e:
            return {"ok": False, "error": f"Erro em {method_name}: {e}"}

    def get_capabilities(self) -> List[str]:
        return list(self.ACTION_MAP.keys())

    def is_available(self) -> bool:
        return HAS_SELENIUM

    def cleanup(self):
        for sess in list(self.sessions.values()):
            try:
                sess.driver.quit()
            except Exception:
                pass
        self.sessions.clear()
        self.active_session = None
