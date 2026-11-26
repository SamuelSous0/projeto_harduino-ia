/*
 * Esteira Seletora - ESP32 (PlatformIO / src/main.cpp)
 * Integrada com Camo + Ponte (Python / Web) + API FastAPI no Render
 *
 * Padrão adotado:
 * 1. PINO_SERVO_CUBO no GPIO 14 (evita pino de strapping GPIO 12 no boot)
 * 2. Ultrassônico com timeout curto (5000 us) e leitura periódica (100 ms)
 *    para manter o WebServer responsivo
 * 3. Esteira pausa ao detectar peça, aguarda decisão e retoma (com timeout
 *    de segurança de 12s para nunca travar)
 * 4. Execução da decisão fora do handler HTTP + cabeçalhos CORS
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

void handleStatus() {
  enviarCORS();

  // Avisa no Monitor Serial quando a ponte (HTML / Python) conecta ou reconecta
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

  server.on("/status",  HTTP_OPTIONS, handleOptions);
  server.on("/decisao", HTTP_OPTIONS, handleOptions);
  server.on("/status",  HTTP_GET,     handleStatus);
  server.on("/decisao", HTTP_POST,    handleDecisao);
  server.on("/decisao", HTTP_GET,     handleDecisao);
  server.begin();

  Serial.println("[OK] WebServer iniciado! Endpoints: /status e /decisao");
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

  // Avisa no Monitor Serial se a ponte HTML ficar mais de 6 segundos sem responder
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
