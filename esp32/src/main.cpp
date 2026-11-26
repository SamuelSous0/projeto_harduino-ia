/*
 * Esteira Seletora - ESP32 (PlatformIO / src/main.cpp)
 * Integrada com Broker MQTT Público (HiveMQ) + API FastAPI no Render + HTTP Local
 *
 * Calibração de Precisão do Ultrassônico (HC-SR04) para a velocidade da esteira:
 * - Disparo único limpo a cada 45 ms (~22 Hz): respeita o tempo de dissipação acústica
 *   do HC-SR04 (evita eco fantasma que acontecia com delay(3) entre amostras).
 * - Filtro de Mediana de 3 amostras + confirmação de 2 leituras consecutivas <= 15 cm:
 *   detecta pirâmides (faces inclinadas) e cubos em ~45-90 ms com precisão e zero falso disparo.
 */

#include <Arduino.h>
#include <WiFi.h>
#include <WiFiClient.h>
#include <WiFiClientSecure.h>
#include <WebServer.h>
#include <ArduinoJson.h>
#include <ESP32Servo.h>
#include <PubSubClient.h>

// ===================== WI-FI =====================
const char* WIFI_SSID = "XtremeFibra@Ricardo&Flavia";
const char* WIFI_PASS = "22061978";

// ===================== BROKER MQTT PÚBLICO =====================
const char* MQTT_BROKER       = "broker.hivemq.com";
const int   MQTT_PORT         = 1883;
const char* MQTT_TOPIC_STATUS = "esteira-esp32-projeto/status";
const char* MQTT_TOPIC_CMD    = "esteira-esp32-projeto/comando";

// ===================== API DE IA (RENDER) =====================
const char* API_HOST = "projeto-harduino-ia.onrender.com";
const int   API_PORT = 443;

WebServer server(80);
WiFiClient espWifiClient;
PubSubClient mqttClient(espWifiClient);

// ===================== PINOS (SEGUROS NO ESP32) =====================
const int PINO_TRIG           = 5;
const int PINO_ECHO           = 18;
const int PINO_SERVO_ESTEIRA  = 23; // Servo 360° (rotação contínua)
const int PINO_SERVO_PIRAMIDE = 13; // SG90 -> y == 0 (piramide)
const int PINO_SERVO_CUBO     = 14; // SG90 -> y == 1 (cubo) [GPIO 14]

// ===================== PARÂMETROS DA ESTEIRA E ULTRASSÔNICO =====================
const float DISTANCIA_DETECCAO_CM = 15.0f;
const float DISTANCIA_MIN_CM      = 1.8f;
const float DISTANCIA_MAX_CM      = 120.0f;

const int   VEL_ESTEIRA_FRENTE    = 110;
const int   VEL_ESTEIRA_PARAR     = 90;

const int   ANGULO_REPOUSO        = 20;
const int   ANGULO_CHUTE          = 85;

// 45 ms entre pulsos é o ponto ideal do HC-SR04: o eco anterior já morreu completamente
// e ainda fazemos ~22 leituras por segundo (acompanha com folga a velocidade da esteira)
const unsigned long INTERVALO_ULTRASSONICO_MS = 45;
const int           CONFIRMACOES_NECESSARIAS  = 2; // 2 leituras seguidas <= 15cm (~45-90ms)
const unsigned long COOLDOWN_DETECCAO_MS      = 2500;
const unsigned long TIMEOUT_DECISAO_MS        = 12000;
const unsigned long INTERVALO_MQTT_STATUS_MS  = 1000;
const unsigned long INTERVALO_MQTT_RECONN_MS  = 3000;

Servo servoEsteira;
Servo servoPiramide;
Servo servoCubo;

volatile bool pecaDetectada = false;
bool esteiraRodando = true;
float ultimaDistanciaCM = -1.0f;
int totalPecasClassificadas = 0;
String ultimaClasseStr = "nenhuma";
int ultimoY = -1;

// Histórico da janela deslizante do ultrassônico para filtro de mediana
float janelaDist[3] = {-1.0f, -1.0f, -1.0f};
int   indiceJanela = 0;
int   contagemPecaConsecutiva = 0;

unsigned long ultimaDeteccao = 0;
unsigned long ultimaLeituraSensor = 0;
unsigned long ultimaLeituraPrint = 0;
unsigned long ultimoEnvioMqtt = 0;
unsigned long ultimaTentativaMqtt = 0;

bool temDecisaoPendente = false;
int  decisaoPendenteY   = -1;

void publicarStatusMQTT(const char* evento = "status");

// ===================== ESTEIRA E SERVOS =====================
void moverEsteira(bool ligada) {
  esteiraRodando = ligada;
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

// ===================== ULTRASSÔNICO PRECISO (FILTRO DE MEDIANA + DEBOUNCE) =====================
void resetarFiltroUltrassonico() {
  janelaDist[0] = -1.0f;
  janelaDist[1] = -1.0f;
  janelaDist[2] = -1.0f;
  contagemPecaConsecutiva = 0;
}

float lerPulsoUltrassonicoCM() {
  digitalWrite(PINO_TRIG, LOW);
  delayMicroseconds(4);
  digitalWrite(PINO_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(PINO_TRIG, LOW);

  // Timeout de 12000 us (~2m) evita perda de eco em superfícies inclinadas (pirâmide)
  unsigned long duracao = pulseIn(PINO_ECHO, HIGH, 12000);
  if (duracao == 0) {
    return -1.0f;
  }

  float dist = (duracao * 0.0343f) / 2.0f;
  if (dist < DISTANCIA_MIN_CM || dist > DISTANCIA_MAX_CM) {
    return -1.0f;
  }
  return dist;
}

float atualizarUltrassonicoPrecisoCM() {
  float leituraAtual = lerPulsoUltrassonicoCM();
  janelaDist[indiceJanela] = leituraAtual;
  indiceJanela = (indiceJanela + 1) % 3;

  // Coleta as leituras válidas recentes na janela de 3 amostras
  float validas[3];
  int n = 0;
  for (int i = 0; i < 3; i++) {
    if (janelaDist[i] > 0.0f) {
      validas[n++] = janelaDist[i];
    }
  }

  if (n == 0) {
    contagemPecaConsecutiva = 0;
    return -1.0f;
  }

  // Ordena para pegar a Mediana (elimina picos espúrios de ruído acústico)
  for (int i = 0; i < n - 1; i++) {
    for (int j = i + 1; j < n; j++) {
      if (validas[j] < validas[i]) {
        float tmp = validas[i];
        validas[i] = validas[j];
        validas[j] = tmp;
      }
    }
  }

  float distEstavel = (n == 3) ? validas[1] : ((n == 2) ? (validas[0] + validas[1]) * 0.5f : validas[0]);

  // Verifica confirmação consecutiva dentro da zona da peça (<= 15 cm)
  if (leituraAtual > 0.0f && leituraAtual <= DISTANCIA_DETECCAO_CM) {
    contagemPecaConsecutiva++;
  } else if (distEstavel > DISTANCIA_DETECCAO_CM + 1.5f || leituraAtual < 0.0f) {
    contagemPecaConsecutiva = 0;
  }

  return distEstavel;
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
  resetarFiltroUltrassonico();
  ultimaDeteccao = millis();
  publicarStatusMQTT("decisao_executada");
}

bool agendarDecisaoJSON(const String &payload) {
  JsonDocument doc;
  DeserializationError erro = deserializeJson(doc, payload);
  if (erro) {
    Serial.println("[ERRO] JSON invalido: " + payload);
    return false;
  }

  if (doc["tipo"].is<const char*>()) {
    const char* tipo = doc["tipo"];
    if (strcmp(tipo, "ping_ponte") == 0) {
      const char* origem = doc["origem"] | "Painel Web Vercel";
      Serial.printf("[MQTT] Painel Web CONECTADO via MQTT! (Origem: %s)\n", origem);
      publicarStatusMQTT("ponte_conectada");
      return true;
    }
  }

  int y = doc["y"] | -1;
  const char* classe = doc["classe"] | (y == 0 ? "piramide" : (y == 1 ? "cubo" : "nao_reconhecido"));
  const char* acao = doc["acao"] | (y == 0 ? "desviar_piramide" : (y == 1 ? "desviar_cubo" : "passar_reto"));
  float confianca = doc["confianca"] | 1.0f;

  Serial.printf("[IA/DECISAO] classe=%s | y=%d | acao=%s | confianca=%.2f%%\n",
                classe, y, acao, confianca * 100.0f);

  ultimaClasseStr = String(classe);
  ultimoY = y;
  totalPecasClassificadas++;

  decisaoPendenteY = y;
  temDecisaoPendente = true;
  return true;
}

// ===================== MQTT (PUBSUBCLIENT) =====================
String montarJsonStatus(const char* evento = "status") {
  JsonDocument doc;
  doc["peca_detectada"]      = pecaDetectada;
  doc["status_esteira"]      = pecaDetectada ? "pausada_aguardando_ia" : "rodando";
  doc["distancia_cm"]        = serialized(String(ultimaDistanciaCM, 1));
  doc["ultima_decisao"]      = ultimaClasseStr;
  doc["ultimo_y"]            = ultimoY;
  doc["pecas_classificadas"] = totalPecasClassificadas;
  doc["ip"]                  = WiFi.localIP().toString();
  doc["evento"]              = evento;

  String out;
  serializeJson(doc, out);
  return out;
}

void publicarStatusMQTT(const char* evento) {
  if (!mqttClient.connected()) return;
  String json = montarJsonStatus(evento);
  mqttClient.publish(MQTT_TOPIC_STATUS, json.c_str(), true);
}

void mqttCallback(char* topic, byte* payload, unsigned int length) {
  String msg;
  msg.reserve(length + 1);
  for (unsigned int i = 0; i < length; i++) {
    msg += (char)payload[i];
  }
  msg.trim();

  Serial.printf("[MQTT] Mensagem recebida em [%s]: %s\n", topic, msg.c_str());

  if (msg == "0" || msg == "1" || msg == "-1") {
    int y = msg.toInt();
    String simulado = "{\"y\":" + String(y) + ",\"classe\":\"" +
                      (y == 0 ? "piramide" : (y == 1 ? "cubo" : "nao_reconhecido")) +
                      "\",\"confianca\":1.0}";
    agendarDecisaoJSON(simulado);
    return;
  }

  agendarDecisaoJSON(msg);
}

void verificarConexaoMQTT() {
  if (WiFi.status() != WL_CONNECTED) return;
  if (mqttClient.connected()) return;

  unsigned long agora = millis();
  if (agora - ultimaTentativaMqtt < INTERVALO_MQTT_RECONN_MS) return;
  ultimaTentativaMqtt = agora;

  String clientId = "ESP32-Esteira-" + String((uint32_t)ESP.getEfuseMac(), HEX);
  Serial.printf("[MQTT] Conectando ao broker %s:%d como %s...\n",
                MQTT_BROKER, MQTT_PORT, clientId.c_str());

  if (mqttClient.connect(clientId.c_str())) {
    Serial.println("[MQTT] CONECTADO ao Broker MQTT com sucesso!");
    mqttClient.subscribe(MQTT_TOPIC_CMD);
    Serial.printf("[MQTT] Inscrito em: %s\n", MQTT_TOPIC_CMD);
    Serial.printf("[MQTT] Publicando status em: %s\n", MQTT_TOPIC_STATUS);
    publicarStatusMQTT("esp32_online");
  } else {
    Serial.printf("[MQTT] Falha na conexao (rc=%d). Tentando novamente em 3s...\n",
                  mqttClient.state());
  }
}

// ===================== ENDPOINTS HTTP LOCAIS (MANTIDOS EM PARALELO) =====================
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
  server.send(200, "application/json", montarJsonStatus("http_status"));
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
  digitalWrite(PINO_TRIG, LOW);

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

  // Configura MQTT
  mqttClient.setServer(MQTT_BROKER, MQTT_PORT);
  mqttClient.setBufferSize(512);
  mqttClient.setCallback(mqttCallback);
  verificarConexaoMQTT();

  acordarAPI();

  server.on("/status",  HTTP_OPTIONS, handleOptions);
  server.on("/decisao", HTTP_OPTIONS, handleOptions);
  server.on("/status",  HTTP_GET,     handleStatus);
  server.on("/decisao", HTTP_POST,    handleDecisao);
  server.on("/decisao", HTTP_GET,     handleDecisao);
  server.begin();

  Serial.println("[OK] WebServer HTTP + Cliente MQTT prontos!");
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    WiFi.reconnect();
  }

  verificarConexaoMQTT();
  mqttClient.loop();
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

  if (pecaDetectada && (agora - ultimaDeteccao > TIMEOUT_DECISAO_MS)) {
    Serial.println("[AVISO] Timeout esperando decisao da IA -> retomando esteira (passar reto).");
    executarDecisao(-1);
  }

  // Leitura precisa do ultrassônico a cada 45 ms (sem eco residual e sem travar o loop)
  if (!pecaDetectada && (agora - ultimaLeituraSensor >= INTERVALO_ULTRASSONICO_MS)) {
    ultimaLeituraSensor = agora;
    float dist = atualizarUltrassonicoPrecisoCM();
    ultimaDistanciaCM = dist;

    if (agora - ultimaLeituraPrint >= 450) {
      ultimaLeituraPrint = agora;
      if (dist < 0.0f) {
        Serial.println("[ULTRASSONICO] Sem objeto proximo");
      } else {
        Serial.printf("[ULTRASSONICO] Distancia estavel: %.1f cm (conf=%d/%d)\n",
                      dist, contagemPecaConsecutiva, CONFIRMACOES_NECESSARIAS);
      }
    }

    // Dispara apenas quando 2 leituras consecutivas confirmam a peça <= 15 cm
    if (contagemPecaConsecutiva >= CONFIRMACOES_NECESSARIAS &&
        dist > DISTANCIA_MIN_CM &&
        dist <= DISTANCIA_DETECCAO_CM &&
        (agora - ultimaDeteccao > COOLDOWN_DETECCAO_MS)) {
      pecaDetectada = true;
      ultimaDeteccao = agora;
      moverEsteira(false);
      Serial.printf("[ULTRASSONICO] Peca CONFIRMADA a %.1f cm! Esteira pausada aguardando IA...\n", dist);
      publicarStatusMQTT("peca_detectada");
      ultimoEnvioMqtt = agora;
    }
  }

  if (agora - ultimoEnvioMqtt >= INTERVALO_MQTT_STATUS_MS) {
    ultimoEnvioMqtt = agora;
    publicarStatusMQTT("heartbeat");
  }
}
