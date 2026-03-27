"""
╔══════════════════════════════════════════════════════════╗
║      ATHENA LEARNING ENGINE — Motor de Aprendizado       ║
║                                                          ║
║  • TF-IDF + Similaridade coseno para matching            ║
║  • Clustering de comandos similares (K-Means)            ║
║  • Detecção de sequências → Auto-macros                  ║
║  • Padrões temporais (horários de uso)                   ║
║  • Detecção de anomalias                                 ║
║  • Scoring Bayesiano de confiança                        ║
║  • Sugestões proativas                                   ║
║  • Exportação/importação de conhecimento                 ║
╚══════════════════════════════════════════════════════════╝

Instalação:
    pip install numpy scipy scikit-learn
    (Funciona sem sklearn usando implementação pura Python como fallback)
"""

import os
import re
import json
import math
import time
import sqlite3
import hashlib
import logging
import threading
import statistics
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

log = logging.getLogger("athena.learning")

# Importações opcionais — fallback puro Python se não tiver
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    log.info("NumPy não disponível — usando math puro (funcional, mais lento)")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import normalize
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    log.info("scikit-learn não disponível — usando TF-IDF puro Python")

# ─── STOPWORDS PT/EN ─────────────────────────────────────────────────────────
STOPWORDS = {
    "a","o","e","de","do","da","no","na","para","com","que","em","é","um","uma",
    "me","por","se","ao","dos","das","os","as","mais","mas","ou","como","esse",
    "esta","este","isso","aqui","ali","meu","minha","seu","sua","foi","ser",
    "the","is","to","of","in","and","for","on","at","by","an","be","it","this",
    "that","with","from","are","was","has","have","will","can","do","did","not",
}

# ─── TOKENS IMPORTANTES (peso extra) ─────────────────────────────────────────
IMPORTANT_TOKENS = {
    "abrir","abra","fechar","feche","criar","crie","deletar","delete","copiar",
    "mover","pesquisar","buscar","acessar","navegar","preencher","clicar","digitar",
    "screenshot","relatório","sistema","processo","arquivo","pasta","janela",
    "open","close","create","delete","move","search","click","type","fill",
    "browser","chrome","firefox","excel","word","notepad","teams","outlook",
}


# ════════════════════════════════════════════════════════════════════
#  TF-IDF PURO PYTHON (fallback quando sklearn não disponível)
# ════════════════════════════════════════════════════════════════════

class PureTFIDF:
    """Implementação leve de TF-IDF sem dependências."""

    def __init__(self):
        self.vocab:     Dict[str, int] = {}
        self.idf:       Dict[str, float] = {}
        self.docs:      List[List[str]] = []
        self.doc_vecs:  List[Dict[str, float]] = []
        self._fitted    = False

    def _tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r'\b[a-záéíóúâêîôûãõç]{2,}\b', text.lower())
        return [t for t in tokens if t not in STOPWORDS]

    def fit(self, texts: List[str]):
        tokenized = [self._tokenize(t) for t in texts]
        self.docs  = tokenized

        # Build vocab
        all_tokens = set(t for doc in tokenized for t in doc)
        self.vocab = {t: i for i, t in enumerate(sorted(all_tokens))}

        # IDF
        N = len(tokenized)
        df: Dict[str, int] = defaultdict(int)
        for doc in tokenized:
            for t in set(doc):
                df[t] += 1
        self.idf = {t: math.log((N + 1) / (df[t] + 1)) + 1 for t in self.vocab}

        # TF-IDF vectors
        self.doc_vecs = [self._tfidf_vec(doc) for doc in tokenized]
        self._fitted   = True

    def _tfidf_vec(self, tokens: List[str]) -> Dict[str, float]:
        tf: Dict[str, float] = Counter(tokens)
        total = len(tokens) if tokens else 1
        vec   = {}
        for t, count in tf.items():
            if t in self.idf:
                # Boost important tokens
                boost = 1.5 if t in IMPORTANT_TOKENS else 1.0
                vec[t] = (count / total) * self.idf[t] * boost
        # Normalize
        norm = math.sqrt(sum(v*v for v in vec.values())) or 1.0
        return {t: v/norm for t, v in vec.items()}

    def transform(self, text: str) -> Dict[str, float]:
        tokens = self._tokenize(text)
        return self._tfidf_vec(tokens)

    def cosine_sim(self, v1: Dict[str, float], v2: Dict[str, float]) -> float:
        keys = set(v1) & set(v2)
        if not keys:
            return 0.0
        dot = sum(v1[k] * v2[k] for k in keys)
        # Already normalized
        return dot

    def find_similar(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        if not self._fitted:
            return []
        qvec = self.transform(query)
        sims = [(i, self.cosine_sim(qvec, dvec)) for i, dvec in enumerate(self.doc_vecs)]
        sims.sort(key=lambda x: -x[1])
        return [(i, s) for i, s in sims if s > 0.05][:top_k]


# ════════════════════════════════════════════════════════════════════
#  BANCO DE DADOS DE APRENDIZADO
# ════════════════════════════════════════════════════════════════════

class LearningDB:
    """SQLite estendido para aprendizado avançado."""

    def __init__(self, path="athena_learning.db"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init()

    def _init(self):
        self.conn.executescript("""
            -- Histórico completo de execuções
            CREATE TABLE IF NOT EXISTS executions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                cmd_id      TEXT UNIQUE,
                raw_command TEXT NOT NULL,
                normalized  TEXT,
                action      TEXT,
                params      TEXT,
                result      TEXT,
                success     INTEGER,
                duration    REAL,
                source      TEXT DEFAULT 'user',
                agent_id    TEXT,
                hour_of_day INTEGER,
                day_of_week INTEGER,
                timestamp   TEXT,
                embedding   TEXT        -- JSON vetor TF-IDF
            );

            -- Padrões aprendidos com scoring bayesiano
            CREATE TABLE IF NOT EXISTS patterns (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_hash  TEXT UNIQUE,
                canonical     TEXT,
                action        TEXT,
                params        TEXT,
                variants      TEXT,   -- JSON lista de variações
                alpha         REAL DEFAULT 1.0,   -- sucessos (Bayesian)
                beta          REAL DEFAULT 1.0,   -- falhas   (Bayesian)
                confidence    REAL DEFAULT 0.5,
                avg_duration  REAL DEFAULT 0.0,
                total_uses    INTEGER DEFAULT 0,
                last_used     TEXT,
                created_at    TEXT,
                cluster_id    INTEGER DEFAULT -1,
                auto_learned  INTEGER DEFAULT 0,
                tags          TEXT DEFAULT '[]'
            );

            -- Sequências detectadas → macros automáticos
            CREATE TABLE IF NOT EXISTS sequences (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                seq_hash    TEXT UNIQUE,
                name        TEXT,
                description TEXT,
                steps       TEXT,    -- JSON lista de {action, params, command}
                trigger     TEXT,    -- Palavra-chave de ativação
                frequency   INTEGER DEFAULT 0,
                last_seen   TEXT,
                auto_created INTEGER DEFAULT 1,
                enabled     INTEGER DEFAULT 1,
                created_at  TEXT
            );

            -- Clusters de comandos
            CREATE TABLE IF NOT EXISTS clusters (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                label       TEXT,
                centroid    TEXT,    -- JSON vetor médio
                action      TEXT,   -- Ação dominante
                member_count INTEGER DEFAULT 0,
                created_at  TEXT
            );

            -- Padrões temporais
            CREATE TABLE IF NOT EXISTS temporal_patterns (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                hour        INTEGER,
                day_of_week INTEGER,
                action      TEXT,
                count       INTEGER DEFAULT 0,
                last_seen   TEXT
            );

            -- Anomalias detectadas
            CREATE TABLE IF NOT EXISTS anomalies (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                command     TEXT,
                action      TEXT,
                reason      TEXT,
                score       REAL,
                reviewed    INTEGER DEFAULT 0,
                timestamp   TEXT
            );

            -- Sugestões proativas
            CREATE TABLE IF NOT EXISTS suggestions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                suggestion  TEXT,
                action      TEXT,
                params      TEXT,
                reason      TEXT,
                score       REAL,
                shown       INTEGER DEFAULT 0,
                accepted    INTEGER DEFAULT 0,
                created_at  TEXT
            );

            -- Métricas de aprendizado
            CREATE TABLE IF NOT EXISTS learning_metrics (
                key   TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_exec_ts    ON executions(timestamp);
            CREATE INDEX IF NOT EXISTS idx_exec_action ON executions(action);
            CREATE INDEX IF NOT EXISTS idx_pat_conf   ON patterns(confidence);
            CREATE INDEX IF NOT EXISTS idx_seq_freq   ON sequences(frequency);
        """)
        self.conn.commit()

    # ── Execuções ─────────────────────────────────────────
    def log_execution(self, data: dict):
        n = datetime.now()
        self.conn.execute("""
            INSERT OR REPLACE INTO executions
            (cmd_id, raw_command, normalized, action, params, result,
             success, duration, source, agent_id, hour_of_day, day_of_week, timestamp, embedding)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            data.get("cmd_id", ""),
            data.get("command", ""),
            data.get("normalized", ""),
            data.get("action", ""),
            json.dumps(data.get("params", {})),
            str(data.get("result", ""))[:500],
            int(data.get("success", 0)),
            data.get("duration", 0.0),
            data.get("source", "user"),
            data.get("agent_id", ""),
            n.hour,
            n.weekday(),
            n.isoformat(),
            json.dumps(data.get("embedding", {})),
        ))
        self.conn.commit()

    def get_recent_executions(self, limit=200, action=None, success_only=False):
        q = "SELECT * FROM executions"
        conds = []
        args  = []
        if action:
            conds.append("action=?"); args.append(action)
        if success_only:
            conds.append("success=1")
        if conds:
            q += " WHERE " + " AND ".join(conds)
        q += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        return [dict(r) for r in self.conn.execute(q, args).fetchall()]

    # ── Padrões ───────────────────────────────────────────
    def upsert_pattern(self, data: dict):
        ph    = data["pattern_hash"]
        existing = self.conn.execute(
            "SELECT alpha, beta, total_uses, avg_duration, variants FROM patterns WHERE pattern_hash=?", (ph,)
        ).fetchone()

        success  = int(data.get("success", 1))
        duration = data.get("duration", 0.0)

        if existing:
            alpha = existing[0] + (1 if success else 0)
            beta  = existing[1] + (0 if success else 1)
            total = existing[2] + 1
            conf  = alpha / (alpha + beta)
            avg_d = (existing[3] * (total - 1) + duration) / total

            # Merge variants
            variants = json.loads(existing[4] or "[]")
            new_v    = data.get("canonical", "")
            if new_v and new_v not in variants:
                variants.append(new_v)
                if len(variants) > 20:
                    variants = variants[-20:]

            self.conn.execute("""
                UPDATE patterns SET alpha=?, beta=?, confidence=?, avg_duration=?,
                    total_uses=?, last_used=?, variants=?, action=?, params=?
                WHERE pattern_hash=?
            """, (alpha, beta, conf, avg_d, total,
                  datetime.now().isoformat(), json.dumps(variants),
                  data.get("action",""), json.dumps(data.get("params",{})), ph))
        else:
            alpha = 2.0 if success else 0.5
            beta  = 0.5 if success else 2.0
            self.conn.execute("""
                INSERT INTO patterns
                (pattern_hash, canonical, action, params, variants, alpha, beta,
                 confidence, avg_duration, total_uses, last_used, created_at, auto_learned)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (ph, data.get("canonical",""), data.get("action",""),
                  json.dumps(data.get("params",{})),
                  json.dumps([data.get("canonical","")]),
                  alpha, beta, alpha/(alpha+beta), duration, 1,
                  datetime.now().isoformat(), datetime.now().isoformat(),
                  int(data.get("auto_learned", 0))))
        self.conn.commit()

    def find_patterns(self, min_confidence=0.7, limit=500):
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM patterns WHERE confidence>=? ORDER BY confidence DESC, total_uses DESC LIMIT ?",
            (min_confidence, limit)
        ).fetchall()]

    def get_pattern_by_hash(self, ph: str):
        r = self.conn.execute("SELECT * FROM patterns WHERE pattern_hash=?", (ph,)).fetchone()
        return dict(r) if r else None

    # ── Sequências ────────────────────────────────────────
    def upsert_sequence(self, data: dict):
        sh = data["seq_hash"]
        ex = self.conn.execute("SELECT frequency FROM sequences WHERE seq_hash=?", (sh,)).fetchone()
        if ex:
            self.conn.execute(
                "UPDATE sequences SET frequency=?, last_seen=?, steps=? WHERE seq_hash=?",
                (ex[0]+1, datetime.now().isoformat(), json.dumps(data.get("steps",[])), sh)
            )
        else:
            self.conn.execute("""
                INSERT INTO sequences (seq_hash, name, description, steps, trigger, frequency, last_seen, created_at)
                VALUES (?,?,?,?,?,?,?,?)
            """, (sh, data.get("name",""), data.get("description",""),
                  json.dumps(data.get("steps",[])), data.get("trigger",""),
                  1, datetime.now().isoformat(), datetime.now().isoformat()))
        self.conn.commit()

    def get_sequences(self, enabled_only=True, min_frequency=2):
        q = "SELECT * FROM sequences WHERE frequency>=?"
        args = [min_frequency]
        if enabled_only:
            q += " AND enabled=1"
        q += " ORDER BY frequency DESC"
        return [dict(r) for r in self.conn.execute(q, args).fetchall()]

    # ── Temporal ──────────────────────────────────────────
    def update_temporal(self, hour: int, dow: int, action: str):
        ex = self.conn.execute(
            "SELECT count FROM temporal_patterns WHERE hour=? AND day_of_week=? AND action=?",
            (hour, dow, action)
        ).fetchone()
        if ex:
            self.conn.execute(
                "UPDATE temporal_patterns SET count=?, last_seen=? WHERE hour=? AND day_of_week=? AND action=?",
                (ex[0]+1, datetime.now().isoformat(), hour, dow, action)
            )
        else:
            self.conn.execute(
                "INSERT INTO temporal_patterns (hour, day_of_week, action, count, last_seen) VALUES (?,?,?,?,?)",
                (hour, dow, action, 1, datetime.now().isoformat())
            )
        self.conn.commit()

    def get_temporal_suggestions(self, hour: int, dow: int, top_k=3):
        return [dict(r) for r in self.conn.execute(
            """SELECT action, SUM(count) as total FROM temporal_patterns
               WHERE ABS(hour - ?) <= 1 AND day_of_week=?
               GROUP BY action ORDER BY total DESC LIMIT ?""",
            (hour, dow, top_k)
        ).fetchall()]

    # ── Anomalias ─────────────────────────────────────────
    def log_anomaly(self, command, action, reason, score):
        self.conn.execute(
            "INSERT INTO anomalies (command, action, reason, score, timestamp) VALUES (?,?,?,?,?)",
            (command, action, reason, score, datetime.now().isoformat())
        )
        self.conn.commit()

    def get_anomalies(self, reviewed=False, limit=20):
        q = "SELECT * FROM anomalies"
        if not reviewed:
            q += " WHERE reviewed=0"
        q += " ORDER BY id DESC LIMIT ?"
        return [dict(r) for r in self.conn.execute(q, [limit]).fetchall()]

    # ── Sugestões ─────────────────────────────────────────
    def save_suggestion(self, data: dict):
        self.conn.execute("""
            INSERT INTO suggestions (suggestion, action, params, reason, score, created_at)
            VALUES (?,?,?,?,?,?)
        """, (data["suggestion"], data.get("action",""), json.dumps(data.get("params",{})),
              data.get("reason",""), data.get("score",0.5), datetime.now().isoformat()))
        self.conn.commit()

    def get_suggestions(self, shown=False, limit=10):
        q = "SELECT * FROM suggestions"
        if not shown:
            q += " WHERE shown=0"
        q += " ORDER BY score DESC LIMIT ?"
        return [dict(r) for r in self.conn.execute(q, [limit]).fetchall()]

    # ── Métricas ─────────────────────────────────────────
    def get_stats(self) -> dict:
        def q(sql, *args):
            r = self.conn.execute(sql, args).fetchone()
            return r[0] if r else 0

        return {
            "total_executions":   q("SELECT COUNT(*) FROM executions"),
            "success_rate":       q("SELECT AVG(success)*100 FROM executions"),
            "total_patterns":     q("SELECT COUNT(*) FROM patterns"),
            "high_conf_patterns": q("SELECT COUNT(*) FROM patterns WHERE confidence>=0.8"),
            "total_sequences":    q("SELECT COUNT(*) FROM sequences"),
            "active_sequences":   q("SELECT COUNT(*) FROM sequences WHERE enabled=1 AND frequency>=2"),
            "anomalies_pending":  q("SELECT COUNT(*) FROM anomalies WHERE reviewed=0"),
            "total_suggestions":  q("SELECT COUNT(*) FROM suggestions"),
            "unique_actions":     q("SELECT COUNT(DISTINCT action) FROM executions"),
            "avg_duration":       q("SELECT AVG(duration) FROM executions WHERE success=1"),
        }

    def set_metric(self, key, value):
        self.conn.execute(
            "INSERT OR REPLACE INTO learning_metrics (key, value, updated_at) VALUES (?,?,?)",
            (key, json.dumps(value), datetime.now().isoformat())
        )
        self.conn.commit()

    def get_metric(self, key):
        r = self.conn.execute("SELECT value FROM learning_metrics WHERE key=?", (key,)).fetchone()
        return json.loads(r[0]) if r else None


# ════════════════════════════════════════════════════════════════════
#  MOTOR DE APRENDIZADO PRINCIPAL
# ════════════════════════════════════════════════════════════════════

class AthenaLearningEngine:
    """
    Motor completo de aprendizado adaptativo.
    Aprende com cada comando executado e melhora continuamente.
    """

    def __init__(self, db_path="athena_learning.db"):
        self.db       = LearningDB(db_path)
        self.tfidf    = None          # Será instanciado após fit
        self._texts   = []            # Corpus para TF-IDF
        self._pattern_map = []        # Mapeia índice → pattern dict
        self._lock    = threading.Lock()
        self._rebuild_scheduled = False

        # Detectador de sequências
        self._recent_commands: List[dict] = []
        self._seq_window = 8           # Janela de sequência
        self._seq_min_len = 2          # Comprimento mínimo de sequência
        self._seq_max_gap = 120        # Segundos máximos entre comandos na sequência

        # Parâmetros de aprendizado
        self.SIMILARITY_THRESHOLD = 0.72
        self.CONFIDENCE_THRESHOLD = 0.78
        self.ANOMALY_THRESHOLD    = 0.15  # Similaridade mínima; abaixo = anomalia
        self.SEQUENCE_MIN_FREQ    = 2     # Vezes para virar macro

        # Rebuild inicial
        self._rebuild_index()
        log.info("Learning Engine inicializado")

    # ═══════════════════════════════════════════════════
    #  NORMALIZAÇÃO DE TEXTO
    # ═══════════════════════════════════════════════════

    def normalize(self, text: str) -> str:
        """Normaliza comando para matching robusto."""
        t = text.lower().strip()
        # Remove pontuação excessiva
        t = re.sub(r'[!?.,;:]+', ' ', t)
        # Normaliza acentos (simplificado)
        subs = {
            'á':'a','à':'a','â':'a','ã':'a','ä':'a',
            'é':'e','è':'e','ê':'e','ë':'e',
            'í':'i','ì':'i','î':'i','ï':'i',
            'ó':'o','ò':'o','ô':'o','õ':'o','ö':'o',
            'ú':'u','ù':'u','û':'u','ü':'u',
            'ç':'c','ñ':'n',
        }
        for a, b in subs.items():
            t = t.replace(a, b)
        # Colapsa espaços
        t = re.sub(r'\s+', ' ', t).strip()
        return t

    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()[:16]

    # ═══════════════════════════════════════════════════
    #  REBUILD DO ÍNDICE TF-IDF
    # ═══════════════════════════════════════════════════

    def _rebuild_index(self):
        with self._lock:
            patterns = self.db.find_patterns(min_confidence=0.3, limit=2000)
            if not patterns:
                self.tfidf        = PureTFIDF() if not HAS_SKLEARN else None
                self._texts       = []
                self._pattern_map = []
                return

            # Textos = canonical + todas as variantes
            texts = []
            pmap  = []
            for p in patterns:
                variants = json.loads(p.get("variants") or "[]")
                all_texts = [p["canonical"]] + variants
                for t in all_texts:
                    if t:
                        texts.append(self.normalize(t))
                        pmap.append(p)

            self._texts       = texts
            self._pattern_map = pmap

            # Fit TF-IDF
            if HAS_SKLEARN:
                self.tfidf = TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                    stop_words=None,   # Controlamos nós
                )
                try:
                    self._matrix = self.tfidf.fit_transform(texts)
                except Exception as e:
                    log.warning(f"sklearn fit failed: {e}")
                    HAS_SKLEARN_LOCAL = False
                    self.tfidf = None
            
            if not HAS_SKLEARN or self.tfidf is None:
                self.tfidf = PureTFIDF()
                self.tfidf.fit(texts)

            log.info(f"Índice TF-IDF rebuilt: {len(texts)} textos, {len(set(p['pattern_hash'] for p in pmap))} padrões")

    def _schedule_rebuild(self, delay=5):
        if not self._rebuild_scheduled:
            self._rebuild_scheduled = True
            def _do():
                time.sleep(delay)
                self._rebuild_index()
                self._rebuild_scheduled = False
            threading.Thread(target=_do, daemon=True).start()

    # ═══════════════════════════════════════════════════
    #  BUSCA DE PADRÃO SEMELHANTE
    # ═══════════════════════════════════════════════════

    def find_similar(self, command: str, top_k=5) -> List[dict]:
        """Busca padrões similares ao comando. Retorna lista ranqueada."""
        if not self._texts or self.tfidf is None:
            return []

        norm = self.normalize(command)

        try:
            if HAS_SKLEARN and hasattr(self, '_matrix') and hasattr(self.tfidf, 'transform'):
                qvec = self.tfidf.transform([norm])
                sims = cosine_similarity(qvec, self._matrix)[0]
                top_idx = sims.argsort()[-top_k*3:][::-1]
                results = []
                seen_hashes = set()
                for i in top_idx:
                    s = float(sims[i])
                    if s < 0.05:
                        break
                    p = self._pattern_map[i]
                    ph = p["pattern_hash"]
                    if ph not in seen_hashes:
                        seen_hashes.add(ph)
                        results.append({**p, "similarity": s})
                    if len(results) >= top_k:
                        break
                return results
            else:
                # Pure Python TF-IDF
                raw = self.tfidf.find_similar(norm, top_k=top_k*3)
                results = []
                seen_hashes = set()
                for i, s in raw:
                    p = self._pattern_map[i]
                    ph = p["pattern_hash"]
                    if ph not in seen_hashes:
                        seen_hashes.add(ph)
                        results.append({**p, "similarity": s})
                    if len(results) >= top_k:
                        break
                return results
        except Exception as e:
            log.warning(f"find_similar error: {e}")
            return []

    def predict(self, command: str) -> Optional[dict]:
        """
        Prediz a melhor ação para o comando.
        Retorna None se não houver confiança suficiente.
        """
        matches = self.find_similar(command, top_k=3)
        if not matches:
            return None

        best = matches[0]
        combined_conf = best["similarity"] * best["confidence"]

        if combined_conf < self.CONFIDENCE_THRESHOLD:
            return None

        return {
            "action":          best["action"],
            "params":          json.loads(best.get("params") or "{}"),
            "confidence":      combined_conf,
            "similarity":      best["similarity"],
            "pattern_conf":    best["confidence"],
            "canonical":       best["canonical"],
            "total_uses":      best.get("total_uses", 0),
            "avg_duration":    best.get("avg_duration", 0),
            "source":          "learned",
        }

    # ═══════════════════════════════════════════════════
    #  APRENDER COM EXECUÇÃO
    # ═══════════════════════════════════════════════════

    def learn(self, command: str, action: str, params: dict,
              success: bool, duration: float,
              result: str = "", source: str = "user",
              cmd_id: str = "", agent_id: str = "") -> dict:
        """
        Aprende com uma execução. Atualiza padrões, detecta sequências,
        verifica anomalias, atualiza padrões temporais.
        """
        n    = datetime.now()
        norm = self.normalize(command)
        ph   = self._hash(norm)
        emb  = {}

        # ─ 1. Log execução ─────────────────────────────
        self.db.log_execution({
            "cmd_id":     cmd_id or str(time.time()),
            "command":    command,
            "normalized": norm,
            "action":     action,
            "params":     params,
            "result":     result,
            "success":    success,
            "duration":   duration,
            "source":     source,
            "agent_id":   agent_id,
            "embedding":  emb,
        })

        # ─ 2. Atualizar padrão ─────────────────────────
        self.db.upsert_pattern({
            "pattern_hash":  ph,
            "canonical":     command,
            "action":        action,
            "params":        params,
            "success":       success,
            "duration":      duration,
            "auto_learned":  0,
        })

        # ─ 3. Padrão temporal ──────────────────────────
        self.db.update_temporal(n.hour, n.weekday(), action)

        # ─ 4. Detecção de anomalia ─────────────────────
        anomaly = self._detect_anomaly(command, action, params, norm)

        # ─ 5. Detecção de sequência ────────────────────
        seq_result = self._detect_sequence(command, action, params, duration, n)

        # ─ 6. Rebuild assíncrono do índice ─────────────
        self._schedule_rebuild(delay=3)

        stats = self.db.get_stats()
        return {
            "learned":        True,
            "pattern_hash":   ph,
            "anomaly":        anomaly,
            "sequence":       seq_result,
            "total_patterns": stats["total_patterns"],
            "high_conf":      stats["high_conf_patterns"],
        }

    # ═══════════════════════════════════════════════════
    #  DETECÇÃO DE ANOMALIAS
    # ═══════════════════════════════════════════════════

    def _detect_anomaly(self, command, action, params, norm) -> Optional[dict]:
        """
        Um comando é anômalo se for muito diferente de qualquer padrão
        conhecido mas tiver uma ação de risco.
        """
        RISKY_ACTIONS = {"shell", "delete_file", "kill_process", "powershell",
                         "http_post", "write_file"}
        if action not in RISKY_ACTIONS:
            return None

        matches = self.find_similar(command, top_k=1)
        if matches and matches[0]["similarity"] > self.ANOMALY_THRESHOLD:
            return None  # Suficientemente similar a algo conhecido

        # Anômalo: ação de risco nunca vista antes
        reason = f"Ação de risco '{action}' com padrão nunca visto"
        score  = 1.0 - (matches[0]["similarity"] if matches else 0.0)

        self.db.log_anomaly(command, action, reason, score)
        log.warning(f"ANOMALIA detectada: {command[:60]} | {reason}")

        return {"detected": True, "reason": reason, "score": score}

    # ═══════════════════════════════════════════════════
    #  DETECÇÃO DE SEQUÊNCIAS → MACROS AUTOMÁTICOS
    # ═══════════════════════════════════════════════════

    def _detect_sequence(self, command, action, params, duration, now) -> Optional[dict]:
        """
        Mantém janela deslizante de comandos recentes.
        Se detectar sequência repetida → cria macro automático.
        """
        entry = {
            "command":   command,
            "action":    action,
            "params":    params,
            "duration":  duration,
            "ts":        now.timestamp(),
        }
        self._recent_commands.append(entry)

        # Remove comandos fora da janela temporal
        cutoff = now.timestamp() - self._seq_max_gap * self._seq_window
        self._recent_commands = [c for c in self._recent_commands if c["ts"] > cutoff]

        # Mantém somente a janela
        if len(self._recent_commands) > self._seq_window * 3:
            self._recent_commands = self._recent_commands[-self._seq_window*3:]

        # Detectar sub-sequências repetidas
        cmds = self._recent_commands
        if len(cmds) < self._seq_min_len * 2:
            return None

        # Verifica sequências de tamanho 2..4
        found = None
        for seq_len in range(2, min(5, len(cmds)//2 + 1)):
            for start in range(len(cmds) - seq_len*2 + 1):
                window1 = cmds[start:start+seq_len]
                window2 = cmds[start+seq_len:start+seq_len*2]
                if self._sequences_match(window1, window2):
                    seq_hash  = self._hash("|".join(c["action"] for c in window1))
                    steps     = [{"action": c["action"], "params": c["params"],
                                  "command": c["command"]} for c in window1]
                    trigger   = self._extract_trigger(window1)
                    seq_name  = self._name_sequence(window1)

                    self.db.upsert_sequence({
                        "seq_hash":    seq_hash,
                        "name":        seq_name,
                        "description": f"Auto-macro: {seq_len} passos",
                        "steps":       steps,
                        "trigger":     trigger,
                    })

                    # Verificar se atingiu frequência mínima
                    seqs = self.db.get_sequences(min_frequency=self.SEQUENCE_MIN_FREQ)
                    if any(s["seq_hash"] == seq_hash for s in seqs):
                        found = {"detected": True, "name": seq_name, "steps": seq_len, "trigger": trigger}
                        log.info(f"Macro automático detectado: '{seq_name}' ({seq_len} passos) → trigger: '{trigger}'")
                    break
            if found:
                break

        return found

    def _sequences_match(self, s1: List[dict], s2: List[dict]) -> bool:
        """Verifica se duas sequências são semanticamente similares."""
        if len(s1) != len(s2):
            return False
        for a, b in zip(s1, s2):
            if a["action"] != b["action"]:
                return False
            # Tolerância de 70% de similaridade textual
            na = self.normalize(a["command"])
            nb = self.normalize(b["command"])
            if na and nb:
                common = len(set(na.split()) & set(nb.split()))
                total  = len(set(na.split()) | set(nb.split()))
                if total and common/total < 0.6:
                    return False
        return True

    def _extract_trigger(self, steps: List[dict]) -> str:
        """Extrai palavra-chave de ativação da sequência."""
        words = []
        for step in steps:
            words.extend(self.normalize(step["command"]).split())
        common = [w for w, c in Counter(words).most_common(5)
                  if w not in STOPWORDS and len(w) > 3]
        return common[0] if common else "macro"

    def _name_sequence(self, steps: List[dict]) -> str:
        """Nomeia uma sequência automaticamente."""
        actions = [s["action"] for s in steps]
        action_labels = {
            "shell":"Terminal","open_url":"Web","web_search":"Busca",
            "read_file":"Leitura","write_file":"Escrita","screenshot":"Captura",
            "browser_open":"Browser","browser_click":"Click","browser_fill_form":"Formulário",
            "system_info":"Sistema","create_report":"Relatório",
        }
        labels = [action_labels.get(a, a) for a in actions]
        return f"Macro: {' → '.join(labels[:3])}"

    # ═══════════════════════════════════════════════════
    #  CLUSTERING
    # ═══════════════════════════════════════════════════

    def cluster_commands(self, n_clusters: int = 8) -> dict:
        """
        Agrupa comandos em clusters temáticos.
        Identifica ações dominantes por cluster.
        """
        execs = self.db.get_recent_executions(limit=500, success_only=True)
        if len(execs) < n_clusters * 2:
            return {"ok": False, "reason": "Dados insuficientes para clustering"}

        texts   = [self.normalize(e["raw_command"]) for e in execs]
        actions = [e["action"] for e in execs]

        if HAS_SKLEARN:
            try:
                vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True)
                X   = vec.fit_transform(texts)
                k   = min(n_clusters, len(texts)//3)
                km  = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = km.fit_predict(X)

                clusters = defaultdict(list)
                for i, label in enumerate(labels):
                    clusters[int(label)].append({"action": actions[i], "text": texts[i]})

                result = []
                for cid, members in clusters.items():
                    action_counts = Counter(m["action"] for m in members)
                    dominant      = action_counts.most_common(1)[0][0]
                    result.append({
                        "cluster_id": cid,
                        "dominant_action": dominant,
                        "member_count": len(members),
                        "top_actions": dict(action_counts.most_common(5)),
                        "sample_commands": [m["text"] for m in members[:3]],
                    })
                result.sort(key=lambda x: -x["member_count"])
                return {"ok": True, "clusters": result, "n_clusters": k}
            except Exception as e:
                return {"ok": False, "reason": str(e)}
        else:
            # Fallback: agrupar por ação dominante
            by_action = defaultdict(list)
            for e in execs:
                by_action[e["action"]].append(e["raw_command"])
            result = [{
                "cluster_id": i,
                "dominant_action": action,
                "member_count": len(cmds),
                "sample_commands": cmds[:3],
            } for i, (action, cmds) in enumerate(
                sorted(by_action.items(), key=lambda x: -len(x[1]))
            )]
            return {"ok": True, "clusters": result, "method": "by_action"}

    # ═══════════════════════════════════════════════════
    #  ANÁLISE DE PERFORMANCE
    # ═══════════════════════════════════════════════════

    def analyze_performance(self) -> dict:
        """Análise completa de performance e uso."""
        execs   = self.db.get_recent_executions(limit=1000)
        if not execs:
            return {"ok": False, "reason": "Sem dados"}

        total       = len(execs)
        successes   = sum(1 for e in execs if e["success"])
        durations   = [e["duration"] for e in execs if e["success"] and e["duration"]]

        # Por ação
        by_action = defaultdict(lambda: {"total":0,"success":0,"duration":[]})
        for e in execs:
            a = e["action"]
            by_action[a]["total"] += 1
            if e["success"]:
                by_action[a]["success"] += 1
                if e["duration"]:
                    by_action[a]["duration"].append(e["duration"])

        action_stats = []
        for action, d in sorted(by_action.items(), key=lambda x: -x[1]["total"]):
            action_stats.append({
                "action":       action,
                "total":        d["total"],
                "success_rate": round(d["success"]/d["total"]*100, 1) if d["total"] else 0,
                "avg_duration": round(statistics.mean(d["duration"]), 2) if d["duration"] else 0,
            })

        # Por hora do dia
        hour_counts = Counter(e["hour_of_day"] for e in execs if e.get("hour_of_day") is not None)
        peak_hour   = hour_counts.most_common(1)[0] if hour_counts else (0, 0)

        # Tendência (últimos 7 dias)
        trend = []
        for d in range(6, -1, -1):
            day = (datetime.now() - timedelta(days=d)).date()
            count = sum(1 for e in execs
                       if e.get("timestamp","")[:10] == str(day))
            trend.append({"date": str(day), "count": count})

        return {
            "ok":           True,
            "total":        total,
            "success_rate": round(successes/total*100, 1) if total else 0,
            "avg_duration": round(statistics.mean(durations), 3) if durations else 0,
            "median_duration": round(statistics.median(durations), 3) if durations else 0,
            "action_stats": action_stats[:10],
            "peak_hour":    peak_hour[0],
            "peak_hour_count": peak_hour[1],
            "daily_trend":  trend,
            "unique_actions": len(by_action),
        }

    # ═══════════════════════════════════════════════════
    #  SUGESTÕES PROATIVAS
    # ═══════════════════════════════════════════════════

    def get_suggestions(self, context: dict = None) -> List[dict]:
        """
        Gera sugestões proativas baseadas em:
        - Padrões temporais (hora do dia)
        - Últimos comandos executados
        - Sequências frequentes
        - Padrões não explorados
        """
        suggestions = []
        now = datetime.now()

        # 1. Sugestões temporais
        temporal = self.db.get_temporal_suggestions(now.hour, now.weekday(), top_k=3)
        for t in temporal:
            suggestions.append({
                "type":       "temporal",
                "suggestion": f"Executar '{t['action']}' (frequente neste horário)",
                "action":     t["action"],
                "params":     {},
                "reason":     f"Usado {t['total']}x neste horário",
                "score":      min(0.9, t["total"] / 20),
            })

        # 2. Sequências / macros prontos
        seqs = self.db.get_sequences(min_frequency=self.SEQUENCE_MIN_FREQ)[:3]
        for seq in seqs:
            suggestions.append({
                "type":       "macro",
                "suggestion": f"Executar macro: {seq['name']}",
                "action":     "run_macro",
                "params":     {"seq_hash": seq["seq_hash"]},
                "reason":     f"Sequência detectada {seq['frequency']}x",
                "score":      min(0.95, seq["frequency"] / 10),
            })

        # 3. Padrões de alta confiança não usados recentemente
        patterns = self.db.find_patterns(min_confidence=0.9, limit=10)
        recently_used = {e["action"] for e in self.db.get_recent_executions(limit=20)}
        for p in patterns:
            if p["action"] not in recently_used and p["total_uses"] > 3:
                suggestions.append({
                    "type":       "pattern",
                    "suggestion": p["canonical"],
                    "action":     p["action"],
                    "params":     json.loads(p.get("params") or "{}"),
                    "reason":     f"Confiança {round(p['confidence']*100)}% — usado {p['total_uses']}x",
                    "score":      p["confidence"] * 0.8,
                })

        # Ordena por score
        suggestions.sort(key=lambda x: -x["score"])
        return suggestions[:8]

    # ═══════════════════════════════════════════════════
    #  EXECUÇÃO DE MACROS
    # ═══════════════════════════════════════════════════

    def get_macro(self, seq_hash: str) -> Optional[dict]:
        seqs = self.db.get_sequences(min_frequency=1)
        for s in seqs:
            if s["seq_hash"] == seq_hash:
                s["steps"] = json.loads(s.get("steps") or "[]")
                return s
        return None

    def list_macros(self) -> List[dict]:
        seqs = self.db.get_sequences(min_frequency=self.SEQUENCE_MIN_FREQ)
        for s in seqs:
            s["steps"] = json.loads(s.get("steps") or "[]")
        return seqs

    def create_manual_macro(self, name: str, steps: List[dict], trigger: str = "") -> dict:
        """Cria macro manualmente."""
        sh = self._hash(name + str(time.time()))
        self.db.upsert_sequence({
            "seq_hash":    sh,
            "name":        name,
            "description": "Macro criado manualmente",
            "steps":       steps,
            "trigger":     trigger or name.lower().split()[0],
        })
        return {"ok": True, "seq_hash": sh, "name": name, "steps": len(steps)}

    def delete_macro(self, seq_hash: str) -> dict:
        self.db.conn.execute("UPDATE sequences SET enabled=0 WHERE seq_hash=?", (seq_hash,))
        self.db.conn.commit()
        return {"ok": True}

    # ═══════════════════════════════════════════════════
    #  EXPORTAR / IMPORTAR CONHECIMENTO
    # ═══════════════════════════════════════════════════

    def export_knowledge(self, path: str = "athena_knowledge.json") -> dict:
        """Exporta todo o conhecimento aprendido para JSON."""
        patterns  = self.db.find_patterns(min_confidence=0.5)
        sequences = self.db.get_sequences(min_frequency=1)
        stats     = self.db.get_stats()

        knowledge = {
            "exported_at": datetime.now().isoformat(),
            "version":     "1.0",
            "stats":       stats,
            "patterns":    [
                {k: v for k, v in p.items() if k not in ("id","cluster_id")}
                for p in patterns
            ],
            "sequences":   [
                {k: v for k, v in s.items() if k not in ("id",)}
                for s in sequences
            ],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(knowledge, f, ensure_ascii=False, indent=2)

        return {"ok": True, "path": path, "patterns": len(patterns), "sequences": len(sequences)}

    def import_knowledge(self, path: str) -> dict:
        """Importa conhecimento de outro agente."""
        if not os.path.exists(path):
            return {"ok": False, "error": f"Arquivo não encontrado: {path}"}

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        imported_patterns  = 0
        imported_sequences = 0

        for p in data.get("patterns", []):
            try:
                self.db.upsert_pattern({**p, "auto_learned": 1})
                imported_patterns += 1
            except Exception:
                pass

        for s in data.get("sequences", []):
            try:
                self.db.upsert_sequence(s)
                imported_sequences += 1
            except Exception:
                pass

        self._schedule_rebuild(delay=1)
        return {
            "ok": True,
            "imported_patterns":  imported_patterns,
            "imported_sequences": imported_sequences,
        }

    # ═══════════════════════════════════════════════════
    #  API PÚBLICA — STATS PARA DASHBOARD
    # ═══════════════════════════════════════════════════

    def get_dashboard_data(self) -> dict:
        stats    = self.db.get_stats()
        perf     = self.analyze_performance()
        macros   = self.list_macros()
        anomalies= self.db.get_anomalies(limit=5)
        suggests = self.get_suggestions()
        clusters = self.cluster_commands(n_clusters=6)

        return {
            "stats":      stats,
            "performance":perf,
            "macros":     macros[:10],
            "anomalies":  anomalies,
            "suggestions":suggests,
            "clusters":   clusters,
            "index_size": len(self._texts),
        }

    def get_full_stats(self) -> dict:
        return self.db.get_stats()
