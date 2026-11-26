/*
 * Esteira Seletora - ESP32 (PlatformIO / src/main.cpp)
 * Integrada com Camo + Ponte (Python / Web no PC e Celular) + API FastAPI no Render
 *
 * Padrão adotado (100% preservado):
 * 1. PINO_SERVO_CUBO no GPIO 14 (evita pino de strapping GPIO 12 no boot)
 * 2. Ultrassônico com timeout curto (5000 us) e leitura periódica (100 ms)
 *    para manter o WebServer responsivo
 * 3. Esteira pausa ao detectar peça, aguarda decisão e retoma (com timeout
 *    de segurança de 12s para nunca travar)
 * 4. Execução da decisão fora do handler HTTP + cabeçalhos CORS
 * 5. Aviso mútuo de conexão (Monitor Serial <-> HTML) + Página HTML embutida
 *    em GET / para abrir direto pelo IP da ESP32 no Celular ou em qualquer PC!
 */

#include <Arduino.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <WebServer.h>
#include <ArduinoJson.h>
#include <ESP32Servo.h>

// ===================== WI-FI =====================
const char* WIFI_SSID = "XtremeFibra@Ricardo&Flavia";
const char* WIFI_PASS = "22061978";

// ===================== API DE IA (RENDER) =====================
const char* API_HOST = "projeto-harduino-ia.onrender.com";
const int   API_PORT = 443;

WebServer server(80);

// ===================== PINOS (SEGUROS NO ESP32) =====================
const int PINO_TRIG           = 5;
const int PINO_ECHO           = 18;
const int PINO_SERVO_ESTEIRA  = 23; // Servo 360° (rotação contínua)
const int PINO_SERVO_PIRAMIDE = 13; // SG90 -> y == 0 (piramide)
const int PINO_SERVO_CUBO     = 14; // SG90 -> y == 1 (cubo) [GPIO 14]

// ===================== PARÂMETROS DA ESTEIRA =====================
const float DISTANCIA_DETECCAO_CM = 15.0f;
const int   VEL_ESTEIRA_FRENTE    = 110;
const int   VEL_ESTEIRA_PARAR     = 90;

const int   ANGULO_REPOUSO        = 20;
const int   ANGULO_CHUTE          = 85;

const unsigned long INTERVALO_ULTRASSONICO_MS = 100;
const unsigned long COOLDOWN_DETECCAO_MS      = 2500;
const unsigned long TIMEOUT_DECISAO_MS        = 12000;

Servo servoEsteira;
Servo servoPiramide;
Servo servoCubo;

volatile bool pecaDetectada = false;
float ultimaDistanciaCM = -1.0f;
bool ponteConectada = false;
unsigned long ultimoPingPonte = 0;

unsigned long ultimaDeteccao = 0;
unsigned long ultimaLeituraSensor = 0;
unsigned long ultimaLeituraPrint = 0;

bool temDecisaoPendente = false;
int  decisaoPendenteY   = -1;

// ===================== HTML EMBUTIDO (ACESSO DIRETO PELO CELULAR OU PC) =====================
const char PAGINA_PONTE_HTML[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Esteira Seletora IA — Ponte PC & Celular</title>
  <style>
    :root { --bg:#0b1020; --panel:#131b31; --panel2:#1a2442; --border:#2a3760; --text:#eef2ff; --muted:#9aa7c7; --pir:#f59e0b; --cub:#10b981; --out:#94a3b8; --ok:#22c55e; --err:#ef4444; }
    * { box-sizing:border-box; margin:0; padding:0; }
    body { font-family:system-ui,-apple-system,sans-serif; background:var(--bg); color:var(--text); padding:12px; }
    .wrap { max-width:960px; margin:0 auto; display:grid; grid-template-columns:1.15fr 0.85fr; gap:12px; }
    @media(max-width:820px){ .wrap { grid-template-columns:1fr; } }
    .card { background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:14px; display:flex; flex-direction:column; gap:10px; }
    h1 { font-size:1.1rem; }
    .banner-conn { padding:10px 12px; border-radius:8px; font-weight:700; font-size:0.86rem; border:2px solid var(--border); background:var(--panel2); display:flex; justify-content:space-between; }
    .banner-conn.ok { border-color:var(--ok); color:#4ade80; background:rgba(34,197,94,0.14); }
    .banner-conn.err { border-color:var(--err); color:#f87171; background:rgba(239,68,68,0.14); }
    .vbox { position:relative; width:100%; aspect-ratio:4/3; background:#000; border-radius:10px; overflow:hidden; border:2px solid var(--border); }
    video { width:100%; height:100%; object-fit:contain; }
    .hud { background:var(--panel2); border:2px solid var(--border); border-radius:10px; padding:10px 14px; display:flex; justify-content:space-between; align-items:center; }
    .hud.y-0 { border-color:var(--pir); } .hud.y-1 { border-color:var(--cub); }
    .hud-cls { font-size:1.3rem; font-weight:800; }
    .y-0 .hud-cls { color:var(--pir); } .y-1 .hud-cls { color:var(--cub); }
    label { font-size:0.8rem; color:var(--muted); display:flex; flex-direction:column; gap:4px; font-weight:600; }
    input, select { background:var(--panel2); color:var(--text); border:1px solid var(--border); border-radius:8px; padding:9px 10px; font-size:0.92rem; }
    button { cursor:pointer; background:#16a34a; color:#fff; border:none; border-radius:8px; padding:11px; font-size:0.92rem; font-weight:700; }
    button.sec { background:#2563eb; }
    .log { background:#060913; border:1px solid var(--border); border-radius:8px; padding:8px; height:170px; overflow-y:auto; font-family:monospace; font-size:0.76rem; display:flex; flex-direction:column; gap:4px; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Esteira Seletora IA — Vídeo & Leitura Ao Vivo</h1>
      <div class="vbox">
        <video id="video" autoplay playsinline muted></video>
        <canvas id="canvas" width="320" height="240" style="display:none;"></canvas>
      </div>
      <div id="hud" class="hud y--1">
        <div>
          <div id="hudSrc" style="font-size:0.73rem;color:var(--muted);">LEITURA AO VIVO</div>
          <div id="hudCls" class="hud-cls">AGUARDANDO...</div>
          <div id="hudAct" style="font-size:0.8rem;color:var(--muted);">Ação: --</div>
        </div>
        <div style="text-align:right;">
          <div id="hudConf" style="font-size:1.3rem;font-weight:800;">--%</div>
          <div id="hudY" style="font-size:0.76rem;color:var(--muted);">y = --</div>
        </div>
      </div>
      <label>Escolher Câmera (Camo / Celular / Webcam)
        <select id="camSel"></select>
      </label>
      <button id="btnStart">▶ Iniciar Câmera + Leitura Ao Vivo + Esteira</button>
    </div>

    <div class="card">
      <h1>Conexão com a ESP32</h1>
      <label>IP da ESP32
        <div style="display:flex;gap:6px;">
          <input type="text" id="espIp" style="flex:1;" placeholder="Ex.: 192.168.1.100" />
          <button id="btnConn" class="sec" style="padding:9px 12px;">Conectar</button>
        </div>
      </label>
      <div id="connBanner" class="banner-conn">
        <span id="connTxt">⏳ Conectando à ESP32...</span>
        <span id="distTxt">-- cm</span>
      </div>
      <div id="log" class="log"></div>
    </div>
  </div>
  <script>
    const API_URL = "https://projeto-harduino-ia.onrender.com";
    const video = document.getElementById("video");
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    const camSel = document.getElementById("camSel");
    const espInput = document.getElementById("espIp");
    const logEl = document.getElementById("log");

    // Preenche automaticamente o IP se aberto direto pela ESP32
    if (location.hostname && location.hostname !== "localhost" && !location.protocol.startsWith("file")) {
      espInput.value = location.hostname;
    } else {
      espInput.value = localStorage.getItem("harduino_esp_ip") || "192.168.43.100";
    }
    espInput.oninput = () => localStorage.setItem("harduino_esp_ip", espInput.value.trim());

    let stream = null, active = false, busyPred = false, busyTrig = false, busyPoll = false, isConn = false, lastDec = 0;

    function espUrl() {
      let u = espInput.value.trim().replace(/\/+$/, "");
      return /^https?:\/\//i.test(u) ? u : "http://" + u;
    }
    function log(m, c="#93c5fd") {
      const d = document.createElement("div");
      d.style.color = c;
      d.textContent = `[${new Date().toLocaleTimeString("pt-BR",{hour12:false})}] ${m}`;
      logEl.prepend(d);
    }
    async function checkEsp(manual=false) {
      try {
        const r = await fetch(`${espUrl()}/status${manual ? "?connect=1" : ""}`);
        if (r.ok) {
          const st = await r.json();
          const b = document.getElementById("connBanner");
          b.className = "banner-conn ok";
          document.getElementById("connTxt").textContent = `✔ ESP32 CONECTADA (${espInput.value.trim()})`;
          document.getElementById("distTxt").textContent = st.distancia_cm > 0 ? `${Number(st.distancia_cm).toFixed(1)} cm` : "Livre";
          if (!isConn || manual) { isConn = true; log(`Conectado à ESP32 em ${espUrl()}!`, "#4ade80"); }
          return st;
        }
      } catch (e) {
        const b = document.getElementById("connBanner");
        b.className = "banner-conn err";
        document.getElementById("connTxt").textContent = `✖ Sem resposta (${espInput.value.trim()})`;
        isConn = false;
      }
      return null;
    }
    async function loadCams() {
      try {
        const devs = await navigator.mediaDevices.enumerateDevices();
        const vids = devs.filter(d => d.kind === "videoinput");
        camSel.innerHTML = "";
        vids.forEach((d,i) => {
          const o = document.createElement("option");
          o.value = d.deviceId; o.textContent = d.label || `Câmera ${i+1}`;
          camSel.appendChild(o);
        });
        const camo = vids.find(d => /camo/i.test(d.label));
        if (camo) camSel.value = camo.deviceId;
      } catch(e){}
    }
    async function startCam() {
      if (stream) stream.getTracks().forEach(t => t.stop());
      const id = camSel.value;
      stream = await navigator.mediaDevices.getUserMedia({
        audio:false,
        video: id ? { deviceId:{exact:id} } : { facingMode:"environment" }
      });
      video.srcObject = stream;
      await video.play();
      await loadCams();
      log("Câmera ao vivo iniciada!", "#4ade80");
    }
    function grabBlob() {
      return new Promise((res,rej) => {
        if (!stream || video.readyState < 2) return rej(new Error("Sem câmera"));
        ctx.drawImage(video, 0, 0, 320, 240);
        canvas.toBlob(b => b ? res(b) : rej(new Error("Erro JPEG")), "image/jpeg", 0.85);
      });
    }
    async function predict(src="LEITURA AO VIVO", wait=false) {
      if (busyPred) {
        if (!wait) return null;
        const t0 = Date.now();
        while (busyPred && Date.now() - t0 < 4000) await new Promise(r => setTimeout(r, 50));
      }
      busyPred = true;
      try {
        const blob = await grabBlob();
        const fd = new FormData();
        fd.append("file", blob, "frame.jpg");
        const r = await fetch(`${API_URL}/predict`, { method:"POST", body:fd });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const d = await r.json();
        document.getElementById("hud").className = `hud y-${d.y}`;
        document.getElementById("hudSrc").textContent = src;
        document.getElementById("hudCls").textContent = (d.classe || "").toUpperCase();
        document.getElementById("hudAct").textContent = `Ação: ${d.acao}`;
        document.getElementById("hudConf").textContent = `${((d.confianca||0)*100).toFixed(1)}%`;
        document.getElementById("hudY").textContent = `y = ${d.y}`;
        return d;
      } catch(e) { return null; } finally { busyPred = false; }
    }
    setInterval(async () => {
      if (active && stream && !busyTrig && !busyPred) await predict("LEITURA AO VIVO (CONTÍNUA)", false);
    }, 700);
    setInterval(async () => {
      if (busyPoll || busyTrig || (Date.now() - lastDec < 1800)) return;
      busyPoll = true;
      try {
        const st = await checkEsp(false);
        if (st && st.peca_detectada === true && active) {
          busyTrig = true;
          log("Peça detectada na esteira! Classificando...", "#fbbf24");
          const pred = await predict("PEÇA NA ESTEIRA -> ESP32", true);
          if (pred) {
            await fetch(`${espUrl()}/decisao`, {
              method:"POST",
              headers:{"Content-Type":"application/json"},
              body:JSON.stringify({classe:pred.classe, y:Number(pred.y), acao:pred.acao, confianca:Number(pred.confianca??1)})
            });
            lastDec = Date.now();
            log(`Decisão enviada à ESP32: ${pred.classe.toUpperCase()} (y=${pred.y})`, "#4ade80");
          }
        }
      } finally { busyPoll = false; busyTrig = false; }
    }, 450);

    document.getElementById("btnConn").onclick = () => checkEsp(true);
    document.getElementById("btnStart").onclick = async (e) => {
      await startCam();
      active = true;
      checkEsp(true);
      e.target.textContent = "✔ Sistema Ativo (Ao Vivo + Esteira)";
    };
    camSel.onchange = () => { if (stream) startCam(); };
    loadCams();
    checkEsp(true);
  </script>
</body>
</html>
)rawliteral";

// ===================== ESTEIRA E SERVOS =====================
void moverEsteira(bool ligada) {
  servoEsteira.write(ligada ? VEL_ESTEIRA_FRENTE : VEL_ESTEIRA_PARAR);
}

void acionarSG90(Servo &servo, int pino, int atrasoAntesMs) {
  if (atrasoAntesMs > 0) {
    delay(atrasoAntesMs);
  }
  servo.attach(pino, 544, 2400);
  servo.write(ANGULO_CHUTE);
  delay(400);
  servo.write(ANGULO_REPOUSO);
  delay(300);
  servo.detach(); // Solta PWM para evitar tremor contínuo
}

// ===================== ULTRASSÔNICO RÁPIDO =====================
float lerDistanciaUnicaCM() {
  digitalWrite(PINO_TRIG, LOW);
  delayMicroseconds(4);
  digitalWrite(PINO_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(PINO_TRIG, LOW);

  long duracao = pulseIn(PINO_ECHO, HIGH, 5000);
  if (duracao <= 150) {
    return -1.0f;
  }

  float dist = (duracao * 0.0343f) / 2.0f;
  if (dist < 2.0f || dist > 85.0f) {
    return -1.0f;
  }
  return dist;
}

float lerDistanciaFiltradaCM() {
  const int AMOSTRAS = 3;
  float soma = 0.0f;
  int validas = 0;

  for (int i = 0; i < AMOSTRAS; i++) {
    float d = lerDistanciaUnicaCM();
    if (d > 0.0f) {
      soma += d;
      validas++;
    }
    delay(3);
  }
  return (validas == 0) ? -1.0f : (soma / validas);
}

// ===================== WARM-UP RENDER =====================
void acordarAPI() {
  WiFiClientSecure client;
  client.setInsecure();
  client.setTimeout(60000);

  Serial.println("[IA] Chamando /health para acordar a API...");
  if (!client.connect(API_HOST, API_PORT)) {
    Serial.println("[IA] Falha ao conectar no /health.");
    return;
  }

  client.println("GET /health HTTP/1.1");
  client.printf("Host: %s\r\n", API_HOST);
  client.println("Connection: close");
  client.println();

  unsigned long inicio = millis();
  while (client.connected() && !client.available() && (millis() - inicio < 60000)) {
    delay(50);
  }

  String resp = "";
  while (client.available()) {
    resp += client.readStringUntil('\n');
  }
  client.stop();

  int ini = resp.indexOf('{');
  int fim = resp.lastIndexOf('}');
  if (ini != -1 && fim != -1 && fim > ini) {
    JsonDocument doc;
    if (!deserializeJson(doc, resp.substring(ini, fim + 1))) {
      bool carregado = doc["modelo_carregado"] | false;
      Serial.printf("[IA] API online! modelo_carregado = %s\n", carregado ? "true" : "false");
      return;
    }
  }
  Serial.println("[IA] Resposta /health recebida.");
}

// ===================== EXECUÇÃO DA DECISÃO =====================
void executarDecisao(int y) {
  moverEsteira(true);

  switch (y) {
    case 0:
      Serial.println("[ESTEIRA] Desviando PIRAMIDE (y = 0)");
      acionarSG90(servoPiramide, PINO_SERVO_PIRAMIDE, 200);
      break;
    case 1:
      Serial.println("[ESTEIRA] Desviando CUBO (y = 1)");
      acionarSG90(servoCubo, PINO_SERVO_CUBO, 500);
      break;
    case -1:
    default:
      Serial.println("[ESTEIRA] Objeto NAO RECONHECIDO (y = -1) -> Passar reto!");
      break;
  }

  pecaDetectada = false;
  ultimaDeteccao = millis();
}

bool agendarDecisaoJSON(const String &payload) {
  JsonDocument doc;
  DeserializationError erro = deserializeJson(doc, payload);
  if (erro) {
    Serial.println("[ERRO] JSON invalido: " + payload);
    return false;
  }

  int y = doc["y"] | -1;
  const char* classe = doc["classe"] | "nao_reconhecido";
  const char* acao = doc["acao"] | "passar_reto";
  float confianca = doc["confianca"] | 0.0f;

  Serial.printf("[IA] classe=%s | y=%d | acao=%s | confianca=%.2f%%\n",
                classe, y, acao, confianca * 100.0f);

  decisaoPendenteY = y;
  temDecisaoPendente = true;
  return true;
}

// ===================== ENDPOINTS HTTP (COM CORS) =====================
void enviarCORS() {
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.sendHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  server.sendHeader("Access-Control-Allow-Headers", "Content-Type");
}

void handleOptions() {
  enviarCORS();
  server.send(204);
}

void handleRoot() {
  enviarCORS();
  IPAddress ipCliente = server.client().remoteIP();
  Serial.printf("[PONTE] Painel HTML aberto diretamente pelo navegador (IP: %s)\n",
                ipCliente.toString().c_str());
  server.send_P(200, "text/html; charset=utf-8", PAGINA_PONTE_HTML);
}

void handleStatus() {
  enviarCORS();

  // Avisa no Monitor Serial quando a ponte (no PC ou no Celular) conecta
  unsigned long agora = millis();
  if (!ponteConectada || server.hasArg("connect") || (agora - ultimoPingPonte > 5000)) {
    ponteConectada = true;
    IPAddress ipCliente = server.client().remoteIP();
    Serial.printf("[PONTE] Ponte HTML/Camera CONECTADA com a ESP32! (IP da Ponte: %s)\n",
                  ipCliente.toString().c_str());
  }
  ultimoPingPonte = agora;

  String json = "{\"peca_detectada\":";
  json += (pecaDetectada ? "true" : "false");
  json += ",\"ponte_conectada\":true";
  json += ",\"distancia_cm\":";
  json += String(ultimaDistanciaCM, 1);
  json += "}";
  server.send(200, "application/json", json);
}

void handleDecisao() {
  enviarCORS();

  if (server.hasArg("plain") && server.arg("plain").length() > 0) {
    if (agendarDecisaoJSON(server.arg("plain"))) {
      server.send(200, "application/json", "{\"status\":\"ok\"}");
    } else {
      server.send(400, "application/json", "{\"erro\":\"json_invalido\"}");
    }
    return;
  }

  if (server.hasArg("y")) {
    int y = server.arg("y").toInt();
    String simulado = "{\"y\":" + String(y) + ",\"classe\":\"via_http\",\"acao\":\"decisao\",\"confianca\":1.0}";
    agendarDecisaoJSON(simulado);
    server.send(200, "application/json", "{\"status\":\"ok\"}");
    return;
  }

  server.send(400, "text/plain", "Envie JSON no body ou parametro ?y=0|1|-1");
}

// ===================== SETUP E LOOP =====================
void setup() {
  Serial.begin(115200);
  delay(300);

  pinMode(PINO_TRIG, OUTPUT);
  pinMode(PINO_ECHO, INPUT);

  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);

  servoEsteira.attach(PINO_SERVO_ESTEIRA, 544, 2400);
  moverEsteira(true);

  acionarSG90(servoPiramide, PINO_SERVO_PIRAMIDE, 0);
  acionarSG90(servoCubo, PINO_SERVO_CUBO, 0);

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("Conectando ao Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.printf("\n[OK] Wi-Fi conectado! IP da ESP32: http://%s\n", WiFi.localIP().toString().c_str());

  acordarAPI();

  server.on("/",        HTTP_GET,     handleRoot);
  server.on("/ponte",   HTTP_GET,     handleRoot);
  server.on("/status",  HTTP_OPTIONS, handleOptions);
  server.on("/decisao", HTTP_OPTIONS, handleOptions);
  server.on("/status",  HTTP_GET,     handleStatus);
  server.on("/decisao", HTTP_POST,    handleDecisao);
  server.on("/decisao", HTTP_GET,     handleDecisao);
  server.begin();

  Serial.printf("[OK] WebServer iniciado! Acesse http://%s no celular ou PC!\n",
                WiFi.localIP().toString().c_str());
}

void loop() {
  server.handleClient();

  if (temDecisaoPendente) {
    temDecisaoPendente = false;
    executarDecisao(decisaoPendenteY);
  }

  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd == "0" || cmd == "1" || cmd == "-1") {
      executarDecisao(cmd.toInt());
    }
  }

  unsigned long agora = millis();

  if (ponteConectada && (agora - ultimoPingPonte > 6000)) {
    ponteConectada = false;
    Serial.println("[PONTE] Aviso: Ponte HTML desconectada (sem comunicacao ha 6s).");
  }

  if (pecaDetectada && (agora - ultimaDeteccao > TIMEOUT_DECISAO_MS)) {
    Serial.println("[AVISO] Timeout esperando decisao da IA -> retomando esteira (passar reto).");
    executarDecisao(-1);
  }

  if (!pecaDetectada && (agora - ultimaLeituraSensor >= INTERVALO_ULTRASSONICO_MS)) {
    ultimaLeituraSensor = agora;
    float dist = lerDistanciaFiltradaCM();
    ultimaDistanciaCM = dist;

    if (agora - ultimaLeituraPrint >= 500) {
      ultimaLeituraPrint = agora;
      if (dist < 0.0f) {
        Serial.println("[ULTRASSONICO] Sem objeto proximo");
      } else {
        Serial.printf("[ULTRASSONICO] Distancia: %.1f cm\n", dist);
      }
    }

    if (dist > 2.0f && dist <= DISTANCIA_DETECCAO_CM && (agora - ultimaDeteccao > COOLDOWN_DETECCAO_MS)) {
      pecaDetectada = true;
      ultimaDeteccao = agora;
      moverEsteira(false);
      Serial.printf("[ULTRASSONICO] Peca detectada a %.1f cm! Esteira pausada aguardando IA...\n", dist);
    }
  }
}
