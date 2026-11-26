"""Ponte em Python para a Esteira Seletora (Câmera Camo/Webcam + API Render + ESP32).

Mantém o vídeo ao vivo rodando na tela (OpenCV) com leitura contínua ao vivo
(sem encavalar requisições via thread worker) e monitora o gatilho da ESP32
(GET /status -> peca_detectada == true -> POST /predict -> POST /decisao).

Atalhos na janela de vídeo:
  [L] Liga / desliga a leitura contínua ao vivo na tela
  [E] Liga / desliga o monitoramento automático da ESP32 (GET /status)
  [P] Classifica o frame atual na tela (sem acionar a esteira)
  [D] Força captura -> POST /predict -> POST /decisao na ESP32
  [I] Permite digitar um novo IP da ESP32 no terminal (útil para hotspot)
  [Q] ou [ESC] Encerra a ponte
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from typing import Any, Dict, Optional

import cv2
import requests

DEFAULT_API_URL = os.getenv(
    "API_URL", "https://projeto-harduino-ia.onrender.com"
)
DEFAULT_ESP32_IP = os.getenv("ESP32_IP", "http://192.168.43.100")


def normalize_url(url: str, default_scheme: str = "http://") -> str:
    url = url.strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        url = default_scheme + url
    return url


class BridgeState:
    """Estado compartilhado com locks para evitar encavalar requisições."""

    def __init__(
        self,
        api_url: str,
        esp32_url: str,
        live_enabled: bool = True,
        esp_enabled: bool = True,
        live_interval_s: float = 0.75,
        esp_poll_s: float = 0.40,
        esp_cooldown_s: float = 1.80,
    ) -> None:
        self.api_url = normalize_url(api_url, "https://")
        self.esp32_url = normalize_url(esp32_url, "http://")
        self.live_enabled = live_enabled
        self.esp_enabled = esp_enabled
        self.live_interval_s = live_interval_s
        self.esp_poll_s = esp_poll_s
        self.esp_cooldown_s = esp_cooldown_s

        self.running = True
        self.frame_lock = threading.Lock()
        self.predict_lock = threading.Lock()
        self.latest_frame = None

        self.is_predicting = False
        self.is_processing_esp_trigger = False
        self.last_live_ts = 0.0
        self.last_esp_poll_ts = 0.0
        self.last_decisao_ts = 0.0

        self.last_prediction: Optional[Dict[str, Any]] = None
        self.last_latency_ms: int = 0
        self.last_source: str = "AGUARDANDO..."
        self.esp_status_text: str = "Aguardando GET /status"
        self.esp_peca_detectada: bool = False
        self.esp_distancia: Optional[float] = None
        self.pieces_processed: int = 0

    def set_frame(self, frame) -> None:
        with self.frame_lock:
            self.latest_frame = frame.copy()

    def get_jpeg_320x240(self) -> Optional[bytes]:
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            frame_copy = self.latest_frame.copy()

        resized = cv2.resize(frame_copy, (320, 240), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return None
        return buf.tobytes()


def check_api_health(api_url: str) -> None:
    try:
        res = requests.get(f"{api_url}/health", timeout=15)
        if res.ok:
            data = res.json()
            print(
                f"[API /health] status={data.get('status')} | "
                f"modelo_carregado={data.get('modelo_carregado')} | "
                f"num_classes={data.get('num_classes')} | "
                f"threshold={data.get('confidence_threshold')}"
            )
        else:
            print(f"[API /health] HTTP {res.status_code}: {res.text}")
    except Exception as exc:
        print(f"[API /health] Falha ao consultar {api_url}/health: {exc}")


def call_predict(
    state: BridgeState,
    source_label: str = "LEITURA AO VIVO",
    wait_if_busy: bool = False,
) -> Optional[Dict[str, Any]]:
    """Envia frame 320x240 JPEG para POST /predict com controle de concorrência."""
    acquired = state.predict_lock.acquire(blocking=wait_if_busy, timeout=5.0 if wait_if_busy else -1)
    if not acquired:
        return None

    state.is_predicting = True
    t0 = time.perf_counter()
    try:
        jpeg_bytes = state.get_jpeg_320x240()
        if not jpeg_bytes:
            return None

        files = {"file": ("frame.jpg", jpeg_bytes, "image/jpeg")}
        res = requests.post(f"{state.api_url}/predict", files=files, timeout=12)
        res.raise_for_status()
        data = res.json()

        latency_ms = int((time.perf_counter() - t0) * 1000)
        state.last_prediction = data
        state.last_latency_ms = latency_ms
        state.last_source = source_label
        return data
    except Exception as exc:
        print(f"[ERRO /predict] {exc}")
        return None
    finally:
        state.is_predicting = False
        state.predict_lock.release()


def send_decisao_to_esp32(state: BridgeState, pred: Dict[str, Any]) -> bool:
    """Envia o JSON da IA para POST /decisao da ESP32 para liberar a esteira."""
    payload = {
        "classe": pred.get("classe", "nao_reconhecido"),
        "y": int(pred.get("y", -1)),
        "acao": pred.get("acao", "passar_reto"),
        "confianca": float(pred.get("confianca", 0.0)),
    }
    try:
        res = requests.post(
            f"{state.esp32_url}/decisao",
            json=payload,
            timeout=5,
        )
        res.raise_for_status()
        state.last_decisao_ts = time.time()
        state.pieces_processed += 1
        state.esp_peca_detectada = False
        state.esp_status_text = (
            f"Decisao enviada: {payload['classe']} (y={payload['y']})"
        )
        print(
            f"[ESP32 /decisao] Sucesso -> y={payload['y']} ({payload['classe']}, "
            f"conf={payload['confianca']:.2%}) | Esteira liberada!"
        )
        return True
    except Exception as exc:
        state.esp_status_text = f"Erro POST /decisao: {exc}"
        print(f"[ERRO ESP32 /decisao] {exc}")
        return False


def process_esp32_trigger(state: BridgeState, source_label: str = "GATILHO ESP32") -> None:
    """Executa leitura prioritária quando peca_detectada == true e envia POST /decisao."""
    if state.is_processing_esp_trigger:
        return
    state.is_processing_esp_trigger = True
    try:
        state.esp_status_text = "PECA DETECTADA (Esteira pausada -> Lendo IA...)"
        print("[ESTEIRA] Peça detectada (<= 15cm)! Capturando frame 320x240 e classificando...")
        pred = call_predict(state, source_label=source_label, wait_if_busy=True)
        if pred is not None:
            send_decisao_to_esp32(state, pred)
        else:
            print("[ESTEIRA] Não foi possível obter resposta da IA. Tentando novamente no próximo ciclo.")
    finally:
        state.is_processing_esp_trigger = False


def background_worker(state: BridgeState) -> None:
    """Thread de segundo plano para não travar a renderização de vídeo ao vivo."""
    while state.running:
        now = time.time()

        # 1. Verifica gatilho da ESP32 (prioridade máxima)
        if (
            state.esp_enabled
            and not state.is_processing_esp_trigger
            and (now - state.last_esp_poll_ts >= state.esp_poll_s)
            and (now - state.last_decisao_ts >= state.esp_cooldown_s)
        ):
            state.last_esp_poll_ts = now
            try:
                res = requests.get(f"{state.esp32_url}/status", timeout=2.2)
                if res.ok:
                    st = res.json()
                    peca = bool(st.get("peca_detectada", False))
                    state.esp_peca_detectada = peca
                    dist = st.get("distancia_cm", st.get("distancia"))
                    state.esp_distancia = float(dist) if dist is not None else None

                    if peca:
                        process_esp32_trigger(state, "GATILHO ESP32 (PECA NA ESTEIRA)")
                    else:
                        state.esp_status_text = "Conectada (Esteira Livre)"
                else:
                    state.esp_status_text = f"HTTP {res.status_code} em /status"
            except Exception:
                state.esp_status_text = f"Sem resposta ({state.esp32_url})"

        # 2. Leitura contínua ao vivo (não bloqueia nem dispara POST /decisao sem peça)
        now = time.time()
        if (
            state.live_enabled
            and not state.is_processing_esp_trigger
            and not state.is_predicting
            and (now - state.last_live_ts >= state.live_interval_s)
        ):
            state.last_live_ts = now
            threading.Thread(
                target=call_predict,
                args=(state, "LEITURA AO VIVO (CONTINUA)", False),
                daemon=True,
            ).start()

        time.sleep(0.04)


def draw_hud(frame, state: BridgeState):
    """Desenha as informações de leitura ao vivo e estado da esteira no vídeo."""
    display = frame.copy()
    h, w = display.shape[:2]

    # Cabeçalho escuro
    cv2.rectangle(display, (0, 0), (w, 68), (18, 24, 42), -1)

    live_txt = "ON" if state.live_enabled else "OFF"
    esp_txt = "ON" if state.esp_enabled else "OFF"
    cv2.putText(
        display,
        f"Leitura Ao Vivo [L]: {live_txt} | ESP32 Monitor [E]: {esp_txt} | Latencia: {state.last_latency_ms}ms",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (220, 230, 255),
        1,
        cv2.LINE_AA,
    )

    dist_str = f" | Dist: {state.esp_distancia:.1f}cm" if state.esp_distancia is not None else ""
    cv2.putText(
        display,
        f"ESP32 ({state.esp32_url}): {state.esp_status_text}{dist_str} | Pecas: {state.pieces_processed}",
        (12, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (150, 220, 255) if not state.esp_peca_detectada else (0, 200, 255),
        1,
        cv2.LINE_AA,
    )

    # Rodapé com a classificação atual
    cv2.rectangle(display, (0, h - 78), (w, h), (18, 24, 42), -1)

    pred = state.last_prediction
    if pred:
        y = int(pred.get("y", -1))
        classe = str(pred.get("classe", "nao_reconhecido")).upper()
        acao = str(pred.get("acao", "passar_reto"))
        conf = float(pred.get("confianca", 0.0)) * 100.0

        # Cores BGR: Pirâmide (Laranja), Cubo (Verde), Outros (Cinza)
        if y == 0:
            color = (0, 170, 255)
        elif y == 1:
            color = (100, 220, 40)
        else:
            color = (180, 180, 180)

        cv2.putText(
            display,
            f"{classe} (y={y}) | Confianca: {conf:.1f}%",
            (12, h - 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            f"Acao: {acao} | Origem: {state.last_source} | [P] Ler Tela [D] Enviar Decisao [I] Trocar IP [Q] Sair",
            (12, h - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (210, 215, 230),
            1,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            display,
            "Aguardando primeira leitura da IA... ([P] Ler agora | [D] Forcar envio ESP32 | [Q] Sair)",
            (12, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    return display


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ponte Esteira Seletora IA — Vídeo Ao Vivo + API Render + ESP32"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=DEFAULT_API_URL,
        help=f"URL da API FastAPI no Render ou local (padrão: {DEFAULT_API_URL})",
    )
    parser.add_argument(
        "--esp32-ip",
        type=str,
        default=DEFAULT_ESP32_IP,
        help=f"IP ou URL HTTP da ESP32 no hotspot (padrão: {DEFAULT_ESP32_IP})",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="0",
        help="Índice da câmera (0, 1, 2 para Camo/Webcam) ou URL de stream MJPEG.",
    )
    parser.add_argument(
        "--live-interval",
        type=float,
        default=0.75,
        help="Intervalo em segundos entre leituras contínuas ao vivo (padrão: 0.75s).",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.40,
        help="Intervalo em segundos para consultar GET /status da ESP32 (padrão: 0.40s).",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=1.80,
        help="Cooldown em segundos após enviar POST /decisao (padrão: 1.80s).",
    )
    parser.add_argument(
        "--no-esp",
        action="store_true",
        help="Inicia com o monitoramento da ESP32 desativado (apenas leitura ao vivo na tela).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    esp_ip = args.esp32_ip
    if esp_ip == DEFAULT_ESP32_IP and not os.getenv("ESP32_IP"):
        print("=" * 68)
        print("PONTE ESTEIRA SELETORA IA — CONFIGURAÇÃO RÁPIDA DE HOTSPOT")
        print("=" * 68)
        user_ip = input(
            f"Digite o IP da ESP32 no Hotspot (ou ENTER para usar {DEFAULT_ESP32_IP}): "
        ).strip()
        if user_ip:
            esp_ip = user_ip

    state = BridgeState(
        api_url=args.api_url,
        esp32_url=esp_ip,
        live_enabled=True,
        esp_enabled=not args.no_esp,
        live_interval_s=max(0.25, args.live_interval),
        esp_poll_s=max(0.15, args.poll_interval),
        esp_cooldown_s=max(0.50, args.cooldown),
    )

    print(f"-> API FastAPI : {state.api_url}")
    print(f"-> ESP32 URL   : {state.esp32_url}")
    check_api_health(state.api_url)

    cam_source: Any = int(args.camera) if args.camera.isdigit() else args.camera
    cap = cv2.VideoCapture(cam_source)
    if not cap.isOpened():
        raise RuntimeError(
            f"Não foi possível abrir a câmera '{args.camera}'. "
            "Tente --camera 1 ou --camera 2 para selecionar a câmera do Camo."
        )

    worker = threading.Thread(target=background_worker, args=(state,), daemon=True)
    worker.start()

    window_title = "Esteira Seletora IA - Visualizacao Ao Vivo (Camo / ESP32)"
    print("\nJanela iniciada! Atalhos: [L] Ao Vivo | [E] ESP32 | [P] Ler Tela | [D] Enviar Decisão | [I] Trocar IP | [Q] Sair\n")

    try:
        while state.running:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            state.set_frame(frame)
            hud_frame = draw_hud(frame, state)
            cv2.imshow(window_title, hud_frame)

            key = cv2.waitKey(15) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                state.running = False
                break
            elif key in (ord("l"), ord("L")):
                state.live_enabled = not state.live_enabled
                print(f"[CONTROLE] Leitura contínua ao vivo: {'ATIVADA' if state.live_enabled else 'PAUSADA'}")
            elif key in (ord("e"), ord("E")):
                state.esp_enabled = not state.esp_enabled
                print(f"[CONTROLE] Monitoramento ESP32: {'ATIVADO' if state.esp_enabled else 'PAUSADO'}")
            elif key in (ord("p"), ord("P")):
                threading.Thread(
                    target=call_predict,
                    args=(state, "LEITURA MANUAL (TELA)", True),
                    daemon=True,
                ).start()
            elif key in (ord("d"), ord("D")):
                threading.Thread(
                    target=process_esp32_trigger,
                    args=(state, "DISPARO MANUAL -> ESP32"),
                    daemon=True,
                ).start()
            elif key in (ord("i"), ord("I")):
                new_ip = input("\nNovo IP da ESP32 (ex.: 192.168.43.105): ").strip()
                if new_ip:
                    state.esp32_url = normalize_url(new_ip, "http://")
                    print(f"[CONFIG] Novo endereço da ESP32 configurado: {state.esp32_url}")
    finally:
        state.running = False
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
